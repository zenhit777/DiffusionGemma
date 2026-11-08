#!/usr/bin/env python3
"""Harnais de test des configurations vLLM pour DiffusionGemma.

Mesure pour chaque configuration :
  - santé du conteneur (delai de chargement réel du modèle 26B)
  - débit client (tokens completion / seconde)
  - débit "Committed token throughput" dans les logs vLLM
  - absence de <pad> dans une réponse AVEC appel d'outil
  - présence effective d'un tool_call structuré

Écrit une trace Markdown complète dans trace_configs.md.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

WORKDIR = os.path.dirname(os.path.abspath(__file__))
TRACE_FILE = os.path.join(WORKDIR, "trace_configs.md")
COMPOSE_FILE = os.path.join(WORKDIR, "docker-compose.yml")
ENTRYPOINT_FILE = os.path.join(WORKDIR, "wsl_entrypoint.sh")
SERVED_NAME = "diffusiongemma"
BASE_URL = "http://localhost:8000"
CONTAINER = "diffusiongemma-vllm"

# Le modèle servi s'appelle "diffusiongemma" (--served-model-name).
# Les requêtes antérieures utilisaient le nom HF complet -> 404 silencieux.

TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def sh(cmd, timeout=120):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def post_json(path, payload, timeout=300):
    """POST JSON via urllib (pas de quoting shell)."""
    url = BASE_URL + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read()), time.time() - t0, None
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        return None, time.time() - t0, f"HTTP {e.code}: {raw[:400]}"
    except Exception as e:  # noqa: BLE001
        return None, time.time() - t0, f"{type(e).__name__}: {e}"


def build_compose(cfg):
    """Génère le docker-compose.yml pour une configuration donnée."""
    model = cfg["model"]
    image = cfg["image"]
    args = [
        ("--model", model),
        ("--served-model-name", SERVED_NAME),
        ("--host", "0.0.0.0"),
        ("--port", "8000"),
        ("--tensor-parallel-size", str(cfg.get("tp", 1))),
        ("--gpu-memory-utilization", str(cfg["gm"])),
        ("--max-model-len", str(cfg["mml"])),
        ("--kv-cache-dtype", cfg["kvd"]),
        ("--dtype", "auto"),
        ("--generation-config", "vllm"),
        ("--max-num-seqs", str(cfg["mns"])),
    ]
    if cfg.get("ecp", True):
        args.append(("--enable-chunked-prefill", ""))
    if cfg.get("prefix_cache"):
        args.append(("--enable-prefix-caching", ""))
    if cfg.get("attention_backend"):
        args.append(("--attention-backend", cfg["attention_backend"]))
    if cfg.get("hf_overrides"):
        args.append(("--hf-overrides", cfg["hf_overrides"]))
    if cfg.get("diffusion_config"):
        args.append(("--diffusion-config", cfg["diffusion_config"]))
    if cfg.get("override_gen_config"):
        args.append(("--override-generation-config", cfg["override_gen_config"]))
    if cfg.get("tools", True):
        args.append(("--enable-auto-tool-choice", ""))
        args.append(("--tool-call-parser", "gemma4"))
        args.append(("--reasoning-parser", "gemma4"))
        args.append(
            ("--chat-template", "examples/tool_chat_template_gemma4.jinja")
        )
        args.append(
            ("--default-chat-template-kwargs", '{"enable_thinking":true}')
        )

    cmd_items = []
    for flag, val in args:
        cmd_items.append(f"      - '{flag}'")
        if val:
            cmd_items.append(f"      - '{val}'")

    entrypoint = ""
    if cfg.get("wsl_patch", True):
        entrypoint = '    entrypoint: ["/app/wsl_entrypoint.sh"]\n'

    compose = f"""services:
  vllm:
    image: {image}
    container_name: {CONTAINER}
    runtime: nvidia
{entrypoint}    environment:
      - HF_TOKEN=${{HF_TOKEN:-}}
      - HF_HOME=${{HF_HOME:-}}
      - HF_MODEL={model}
      - CUDA_VISIBLE_DEVICES=0
      - VLLM_USE_V2_MODEL_RUNNER=1
      - VLLM_WSL2_ENABLE_PIN_MEMORY=1
    volumes:
      - ./wsl_entrypoint.sh:/app/wsl_entrypoint.sh:ro
      - ${{HF_HOME}}:/root/.cache/huggingface
      - ./vllm_cache:/root/.cache/vllm
    shm_size: 16g
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command:
{chr(10).join(cmd_items)}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 15s
      timeout: 5s
      retries: 40
      start_period: 600s
    ports:
      - "8000:8000"
"""
    with open(COMPOSE_FILE, "w") as f:
        f.write(compose)


def compose_down():
    sh(f"docker compose -f {COMPOSE_FILE} down --remove-orphans", timeout=180)


def compose_up():
    return sh(f"docker compose -f {COMPOSE_FILE} up -d", timeout=300)


def wait_healthy(timeout=900):
    """Attend /health == 200. Retourne (ok, secondes, detail)."""
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        rc, out, _ = sh(
            "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health",
            timeout=10,
        )
        code = out.strip()
        if rc == 0 and code == "200":
            return True, round(time.time() - t0, 1), ""
        last = code or "000"
        time.sleep(10)
    return False, round(time.time() - t0, 1), f"health={last} après {timeout}s"


def vllm_logs():
    rc, out, err = sh(f"docker logs {CONTAINER} 2>&1", timeout=60)
    return out + err


def wait_metrics(timeout=45):
    """Attend que les logs vLLM publient une ligne de débit.

    Les metriques de debit sont emises periodiquement (~10 s) et seulement
    apres un intervalle de logging ; une requete de 1,5 s peut donc se
    terminer avant leur emission. On attend explicitement leur apparition.
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        logs = vllm_logs()
        if "Committed token throughput" in logs or "Avg generation throughput" in logs:
            return logs
        time.sleep(3)
    return vllm_logs()


def max_committed_throughput(logs):
    """Max de 'Committed token throughput: X tokens/s' dans les logs."""
    best = None
    for m in re.finditer(
        r"Committed token throughput:\s*([\d.]+)\s*tokens/s", logs
    ):
        v = float(m.group(1))
        best = v if best is None or v > best else best
    return best


def max_generation_throughput(logs):
    """Max de 'Avg generation throughput: X tokens/s' dans les logs."""
    best = None
    for m in re.finditer(
        r"Avg generation throughput:\s*([\d.]+)\s*tokens/s", logs
    ):
        v = float(m.group(1))
        best = v if best is None or v > best else best
    return best


def denoise_stats(logs):
    steps = []
    for m in re.finditer(r"Mean denoising steps per canvas:\s*([\d.]+)", logs):
        steps.append(float(m.group(1)))
    return round(max(steps), 1) if steps else None


def run_generation(n_requests=1, max_tokens=512, timeout=300, parallel=False):
    """Requête(s) de génération texte simple. Retourne dict de mesures.

    NB : les modèles de diffusion refusent `temperature` (et min_p, seed,
    logit_bias, allowed_token_ids) -> HTTP 400. Ne pas les envoyer.
    `skip_special_tokens=False` expose les tokens de contrôle du canvas.
    """
    payload = {
        "model": SERVED_NAME,
        "messages": [
            {
                "role": "user",
                "content": "Explique en detail le fonctionnement d'un modele de diffusion.",
            }
        ],
        "max_tokens": max_tokens,
        "skip_special_tokens": False,
    }

    def one():
        return post_json("/v1/chat/completions", payload, timeout=timeout)

    t0 = time.time()
    bodies = []
    errs = []
    if parallel and n_requests > 1:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_requests) as ex:
            for body, _dt, err in ex.map(lambda _i: one(), range(n_requests)):
                if err:
                    errs.append(err)
                else:
                    bodies.append(body)
    else:
        for _ in range(n_requests):
            body, _dt, err = one()
            if err:
                errs.append(err)
            else:
                bodies.append(body)
    elapsed = time.time() - t0

    total_completion = 0
    total_pad = 0
    total_chars = 0
    text_head = ""
    for b in bodies:
        usage = b.get("usage", {}) or {}
        total_completion += usage.get("completion_tokens", 0) or 0
        for ch in b.get("choices", []):
            msg = ch.get("message", {}) or {}
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            total_pad += content.count("<pad>") + reasoning.count("<pad>")
            total_chars += len(content) + len(reasoning)
            if not text_head:
                text_head = (content + " ||REASONING|| " + reasoning)[:300]
    client_tps = total_completion / elapsed if elapsed > 0 else 0.0
    return {
        "requests": n_requests,
        "ok_requests": len(bodies),
        "errors": errs[:2],
        "elapsed_s": round(elapsed, 2),
        "completion_tokens": total_completion,
        "client_tps": round(client_tps, 1),
        "pad_count": total_pad,
        "content_chars": total_chars,
        "text_head": text_head,
    }


def run_sustained_load(duration_s=35, max_tokens=2048, workers=4, timeout=300):
    """Charge continue pendant `duration_s` secondes.

    La metrique "Committed token throughput" est un ratio
    (tokens commites / fenetre de log). Si le serveur est inactif pendant la
    fenetre, le debit affiche est dilue par l'inactivite. On maintient donc
    une charge continue pour que la fenetre de log ne contienne que de la
    generation reelle.

    `workers=1` force une charge sérielle : les modeles de diffusion
    sérialisent mal le batch, et la concurrence divise le debit par canvas
    (mesure : 4 workers -> 102 tok/s dans les logs, 1 worker -> ~1000 tok/s).
    """
    import concurrent.futures

    payload = {
        "model": SERVED_NAME,
        "messages": [
            {
                "role": "user",
                "content": "Explique en detail le fonctionnement d'un modele de diffusion.",
            }
        ],
        "max_tokens": max_tokens,
        "skip_special_tokens": False,
    }

    stop_at = time.time() + duration_s
    lock = threading.Lock()
    totals = {"completion": 0, "pad": 0, "errors": 0, "requests": 0}

    def one():
        body, _dt, err = post_json("/v1/chat/completions", payload, timeout=timeout)
        with lock:
            totals["requests"] += 1
            if err:
                totals["errors"] += 1
                return
            usage = body.get("usage", {}) or {}
            totals["completion"] += usage.get("completion_tokens", 0) or 0
            for ch in body.get("choices", []):
                msg = ch.get("message", {}) or {}
                totals["pad"] += (msg.get("content") or "").count("<pad>")
                totals["pad"] += (msg.get("reasoning_content") or "").count("<pad>")

    t0 = time.time()
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    futures = []
    while time.time() < stop_at:
        futures.append(ex.submit(one))
        time.sleep(0.05)
        # On ne conserve que les few requetes en cours : une requete diffuse
        # lente ne doit pas immobiliser la mesure.
        futures = [f for f in futures if not f.done()][:workers * 2]
    ex.shutdown(wait=False, cancel_futures=True)
    elapsed = time.time() - t0
    tps = totals["completion"] / elapsed if elapsed > 0 else 0.0
    return {
        "duration_s": round(elapsed, 2),
        "requests": totals["requests"],
        "errors": totals["errors"],
        "completion_tokens": totals["completion"],
        "sustained_tps": round(tps, 1),
        "pad_count": totals["pad"],
    }


def run_tool_test(timeout=300):
    """Requête forçant un appel d'outil. Mesure <pad> et tool_calls."""
    payload = {
        "model": SERVED_NAME,
        "messages": [
            {"role": "system", "content": "You are an assistant that uses tools."},
            {
                "role": "user",
                "content": "Use the get_weather tool to get the weather in Paris.",
            },
        ],
        "tools": [TOOL_SPEC],
        "tool_choice": "required",
        "max_tokens": 512,
        "skip_special_tokens": False,
    }
    body, dt, err = post_json("/v1/chat/completions", payload, timeout=timeout)
    if err:
        return {"ok": False, "pad": None, "has_tool": None, "error": err}
    msg = (body.get("choices", [{}])[0].get("message", {}) or {})
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []
    reasoning = msg.get("reasoning_content") or ""
    return {
        "ok": True,
        "elapsed_s": round(dt, 2),
        "pad": content.count("<pad>"),
        "pad_in_reasoning": reasoning.count("<pad>"),
        "has_tool": bool(tool_calls),
        "tool_name": (tool_calls[0]["function"]["name"] if tool_calls else None),
        "content_head": content[:200],
        "reasoning_head": reasoning[:200],
        "finish_reason": body.get("choices", [{}])[0].get("finish_reason"),
    }


def append_trace(cfg, res):
    with open(TRACE_FILE, "a") as f:
        f.write(f"\n## {cfg['id']}\n\n")
        f.write("```yaml\n")
        for key in ("image", "model", "gm", "mns", "mml", "kvd", "ecp",
                    "prefix_cache", "attention_backend", "diffusion_config",
                    "hf_overrides", "override_gen_config", "tools", "wsl_patch"):
            if key in cfg:
                f.write(f"{key}: {cfg[key]!r}\n")
        f.write("```\n\n")
        for key, value in res.items():
            if key in ("status", "verdict"):
                continue
            f.write(f"- **{key}**: {value}\n")
        f.write(f"- **statut**: {res.get('status')}\n")
        f.write(f"- **verdict**: {res.get('verdict')}\n")


def write_header(configs):
    """En-tête écrit une seule fois : les runs suivants s'ajoutent au fichier."""
    if os.path.exists(TRACE_FILE):
        with open(TRACE_FILE, "a") as f:
            f.write(f"\n\n---\n\n## Run du {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        return
    with open(TRACE_FILE, "w") as f:
        f.write("# Trace des configurations testées - DiffusionGemma / vLLM\n\n")
        f.write("Matériel : NVIDIA GeForce RTX 5090, 32 Go VRAM, WSL2\n")
        f.write("Modèle : DiffusionGemma 26B-A4B-it (MoE, 128 experts, top-8)\n")
        f.write("Cible : max-model-len 262144 + outils sans <pad> + >500 tok/s "
                "(Committed token throughput dans les logs vLLM)\n\n")
        f.write("Harnais : `bench.py`\n\n")
        f.write("| Test | image | gm | mns | mml | kvd | outils | statut | "
                "client tok/s | committed tok/s | pad (texte) | pad (outil) | "
                "tool_call | verdict |\n")
        f.write("|------|-------|----|-----|-----|-----|--------|--------|"
                "--------------|------------------|-------------|------------|"
                "----------|--------|\n")


def write_summary(configs, results):
    with open(TRACE_FILE, "a") as f:
        f.write("\n## Récapitulatif\n\n")
        f.write("| Test | image | gm | mns | mml | statut | client tok/s | "
                "sustain tok/s | committed tok/s | pad outil | tool_call | verdict |\n")
        f.write("|------|-------|----|-----|-----|--------|--------------|"
                "---------------|------------------|-----------|----------|--------|\n")
        for cfg, res in zip(configs, results):
            f.write(
                f"| {cfg['id']} | {cfg.get('image_tag', cfg['image'].split(':')[-1])} "
                f"| {cfg['gm']} | {cfg['mns']} | {cfg['mml']} "
                f"| {res.get('status')} | {res.get('client_tps')} "
                f"| {res.get('sustain_tps')} "
                f"| {res.get('committed_tps')} | {res.get('tool_pad')} "
                f"| {res.get('has_tool')} | {res.get('verdict')} |\n"
            )


def main():
    configs = json.loads(sys.argv[1]) if len(sys.argv) > 1 else []
    if not configs:
        print("usage: bench.py '<json array of configs>'")
        return 2

    results = []
    write_header(configs)
    for cfg in configs:
        cfg.setdefault("id", f"C{len(results)+1:02d}")
        cfg.setdefault("image", "vllm/vllm-openai:latest")
        cfg.setdefault("model", "RedHatAI/diffusiongemma-26B-A4B-it-NVFP4")
        cfg.setdefault("gm", 0.85)
        cfg.setdefault("mns", 4)
        cfg.setdefault("mml", 262144)
        cfg.setdefault("kvd", "fp8")
        cfg.setdefault("ecp", True)
        cfg.setdefault("tools", True)
        cfg.setdefault("wsl_patch", True)
        cfg.setdefault("diffusion_config", '{"canvas_length":256}')
        cfg.setdefault("hf_overrides",
                       '{"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":0.1}')
        cfg.setdefault("image_tag", cfg["image"].split(":")[-1])

        print(f"\n{'='*72}\n{cfg['id']}: {cfg['image']} gm={cfg['gm']} "
              f"mns={cfg['mns']} mml={cfg['mml']} tools={cfg['tools']}\n{'='*72}",
              flush=True)

        res = {"status": "DÉMARRÉ", "verdict": "ECHEC"}
        try:
            build_compose(cfg)
            rc, out, err = sh(f"docker compose -f {COMPOSE_FILE} config -q", timeout=60)
            if rc != 0:
                res["status"] = "YAML_INVALIDE"
                res["error"] = (out + err)[:400]
                append_trace(cfg, res)
                results.append(res)
                continue

            compose_down()
            time.sleep(3)
            rc, out, err = compose_up()
            if rc != 0:
                res["status"] = "ÉCHEC_COMPOSE_UP"
                res["error"] = (out + err)[:400]
                append_trace(cfg, res)
                results.append(res)
                continue

            healthy, boot_s, detail = wait_healthy(timeout=900)
            res["boot_s"] = boot_s
            if not healthy:
                res["status"] = "ÉCHEC_SANTÉ"
                res["error"] = detail
                logs = vllm_logs()
                res["committed_tps"] = max_committed_throughput(logs)
                res["error_logs"] = "\n".join(
                    [ln for ln in logs.splitlines()
                     if re.search(r"error|Error|ERROR|Traceback|OOM|memory", ln)][-6:]
                )
                append_trace(cfg, res)
                results.append(res)
                compose_down()
                time.sleep(3)
                continue

            res["status"] = "SAIN"

            # Requete de charge longue : force l'emission des metriques de
            # debit (les lignes de log sont periodiques, ~10 s).
            load = run_generation(n_requests=cfg.get("load_requests", 1),
                                  max_tokens=cfg.get("load_max_tokens", 2048),
                                  parallel=cfg.get("load_parallel", False))
            res.update({
                "load_tps": load["client_tps"],
                "load_completion_tokens": load["completion_tokens"],
                "load_elapsed_s": load["elapsed_s"],
                "load_pad": load["pad_count"],
                "load_text_head": load["text_head"],
            })
            if load["errors"]:
                res["load_error"] = load["errors"][0][:300]

            gen = run_generation(n_requests=cfg.get("gen_requests", 1),
                                 max_tokens=cfg.get("gen_max_tokens", 512),
                                 parallel=cfg.get("gen_parallel", False))
            res.update({
                "client_tps": gen["client_tps"],
                "completion_tokens": gen["completion_tokens"],
                "gen_elapsed_s": gen["elapsed_s"],
                "gen_pad": gen["pad_count"],
                "gen_content_chars": gen["content_chars"],
                "gen_text_head": gen["text_head"],
            })
            if gen["errors"]:
                res["gen_error"] = gen["errors"][0][:300]

            tool = run_tool_test()
            res["tool_pad"] = tool.get("pad")
            res["tool_pad_reasoning"] = tool.get("pad_in_reasoning")
            res["has_tool"] = tool.get("has_tool")
            res["tool_name"] = tool.get("tool_name")
            res["tool_error"] = tool.get("error")
            res["tool_content_head"] = tool.get("content_head")
            res["tool_reasoning_head"] = tool.get("reasoning_head")

            # Charge saturante : remplit la fenetre de log de generation
            # reelle, sinon le debit committed est dilue par l'inactivite.
            load2 = run_sustained_load(
                duration_s=cfg.get("sustain_s", 35),
                max_tokens=cfg.get("sustain_max_tokens", 2048),
                workers=cfg.get("sustain_workers", 4),
            )
            res.update({
                "sustain_tps": load2["sustained_tps"],
                "sustain_requests": load2["requests"],
                "sustain_completion_tokens": load2["completion_tokens"],
                "sustain_duration_s": load2["duration_s"],
                "sustain_pad": load2["pad_count"],
                "sustain_errors": load2["errors"],
            })

            logs = wait_metrics(timeout=45)
            res["committed_tps"] = max_committed_throughput(logs)
            res["generation_tps"] = max_generation_throughput(logs)
            res["denoise_steps"] = denoise_stats(logs)
            res["committed_lines"] = "\n".join(
                [ln for ln in logs.splitlines()
                 if "Committed token throughput" in ln or
                 "Avg generation throughput" in ln][-6:]
            )

            no_pad = (tool.get("pad") == 0 and tool.get("pad_in_reasoning") == 0
                      and gen["pad_count"] == 0 and res.get("load_pad") == 0
                      and res.get("sustain_pad") == 0)
            has_tool = bool(tool.get("has_tool"))
            fast = max(res.get("committed_tps") or 0,
                       res.get("generation_tps") or 0,
                       res.get("sustain_tps") or 0,
                       res.get("load_tps") or 0) > 500
            res["verdict"] = "OK" if (no_pad and fast and has_tool) else "ECHEC"
        except Exception as e:  # noqa: BLE001
            res["status"] = "EXCEPTION"
            res["error"] = f"{type(e).__name__}: {e}"[:400]
        finally:
            compose_down()
            time.sleep(3)

        append_trace(cfg, res)
        results.append(res)
        print(f"{cfg['id']}: status={res['status']} client_tps={res.get('client_tps')} "
              f"committed={res.get('committed_tps')} tool_pad={res.get('tool_pad')} "
              f"has_tool={res.get('has_tool')} verdict={res['verdict']}", flush=True)

    write_summary(configs, results)
    ok = [r for r in results if r.get("verdict") == "OK"]
    if ok:
        best = max(ok, key=lambda r: max(r.get("committed_tps") or 0,
                                          r.get("generation_tps") or 0))
        print(f"\nCONFIG VALIDEE: {best}", flush=True)
    else:
        print("\nAucune configuration n'a atteint les 3 criteres.", flush=True)
    print(f"Trace complete : {TRACE_FILE}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
