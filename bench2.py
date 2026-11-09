#!/usr/bin/env python3
"""Harnais de test v2 pour DiffusionGemma / vLLM.

Mesures par configuration :
  - memoire GPU Windows : dedicated / shared (compteurs \\GPU Process Memory)
  - memoire GPU vue par WSL (nvidia-smi)
  - budget KV cache annonce par vLLM et concurrence obtenue
  - debit client serial (mediane) a prompt court et a prompt long (64k tokens)
  - TTFT (premier octet de reponse)
  - compteurs Prometheus de diffusion (denoising steps, canvas positions, committed)
  - metriques loguees vLLM (Avg generation throughput, Committed token throughput)
  - qualite : <pad>, appel d'outil structure

Le compose de production est sauvegarde puis restaure a la fin.
"""
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

WORKDIR = os.path.dirname(os.path.abspath(__file__))
COMPOSE_FILE = os.path.join(WORKDIR, "docker-compose.yml")
COMPOSE_BACKUP = os.path.join(WORKDIR, "state", "docker-compose.production.yml")
RESULT_DIR = os.path.join(WORKDIR, "state", "results")
CONTAINER = "diffusiongemma-vllm"
SERVED_NAME = "diffusiongemma"
BASE_URL = "http://localhost:8000"
METRICS_URL = BASE_URL + "/metrics"
GPU_COUNTER_PS = os.path.join(WORKDIR, "state", "gpu_counters.ps1")

TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Recupere la meteo d une ville",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

FILLER_WORDS = (
    " alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
).split()


# ---------------------------------------------------------------- utilitaires


def sh(cmd, timeout=180):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def post_json(path, payload, timeout=900):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + path, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        return None, time.time() - t0, f"HTTP {e.code}: {body[:300]}"
    except Exception as e:  # noqa: BLE001
        return None, time.time() - t0, f"{type(e).__name__}: {e}"
    dt = time.time() - t0
    try:
        return json.loads(raw), dt, None
    except Exception:
        return None, dt, "reponse non-JSON"


def prometheus_value(name, label_filter=""):
    rc, out, _ = sh(f"curl -s {METRICS_URL}", timeout=60)
    for line in out.splitlines():
        if line.startswith("#") or not line.startswith(name):
            continue
        if label_filter and label_filter not in line:
            continue
        try:
            return float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
    return None


def prometheus_counter(name):
    return prometheus_value(name)


# ------------------------------------------------------------- mesure GPU


def gpu_windows():
    """Compteurs Windows \\GPU Process Memory (dedicated/shared)."""
    if not os.path.exists(GPU_COUNTER_PS):
        return None
    wsl_path = subprocess.run(
        ["wslpath", "-w", GPU_COUNTER_PS], capture_output=True, text=True
    ).stdout.strip()
    rc, out, err = sh(
        f'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "{wsl_path}"',
        timeout=120,
    )
    if rc != 0 or not out.strip():
        return {"error": (out + err)[:200]}
    try:
        return json.loads(out.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001
        return {"error": f"parse: {e}: {out[:200]}"}


def gpu_wsl():
    rc, out, _ = sh(
        "nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu "
        "--format=csv,noheader,nounits",
        timeout=60,
    )
    if rc != 0 or not out.strip():
        return None
    parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
    return {
        "used_mib": int(parts[0]),
        "free_mib": int(parts[1]),
        "util_pct": int(parts[2]),
    }


# --------------------------------------------------------- construction compose


def build_compose(cfg):
    args = [
        ("--model", cfg["model"]),
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
    if cfg.get("diffusion_config"):
        args.append(("--diffusion-config", cfg["diffusion_config"]))
    if cfg.get("tools", True):
        args.append(("--enable-auto-tool-choice", ""))
        args.append(("--tool-call-parser", "gemma4"))
        args.append(("--reasoning-parser", "gemma4"))
        args.append(("--chat-template", "examples/tool_chat_template_gemma4.jinja"))
        args.append(("--default-chat-template-kwargs", '{"enable_thinking":true}'))

    cmd_items = []
    for flag, val in args:
        cmd_items.append(f"      - '{flag}'")
        if val:
            cmd_items.append(f"      - '{val}'")

    entrypoint = ""
    if cfg.get("wsl_patch", True):
        entrypoint = '    entrypoint: ["/app/wsl_entrypoint.sh"]\n'

    # HF_HOME par configuration : permet de tester plusieurs caches de modeles
    # sans modifier le .env de production.
    hf_home = cfg.get("hf_home") or "${HF_HOME:-}"

    compose = f"""services:
  vllm:
    image: {cfg["image"]}
    container_name: {CONTAINER}
    runtime: nvidia
{entrypoint}    environment:
      - HF_TOKEN=${{HF_TOKEN:-}}
      - HF_HOME={hf_home}
      - HF_MODEL={cfg["model"]}
      - CUDA_VISIBLE_DEVICES=0
      - VLLM_USE_V2_MODEL_RUNNER=1
      - VLLM_WSL2_ENABLE_PIN_MEMORY=1
      - VLLM_LOG_STATS_INTERVAL=10
    volumes:
      - ./wsl_entrypoint.sh:/app/wsl_entrypoint.sh:ro
      - {hf_home}:/root/.cache/huggingface
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


# ------------------------------------------------------------------ cycle de vie


def compose_down():
    sh(f"docker compose -f {COMPOSE_FILE} down --remove-orphans", timeout=300)


def compose_up():
    return sh(f"docker compose -f {COMPOSE_FILE} up -d", timeout=300)


def wait_healthy(timeout=900):
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
        # Detection rapide d'un conteneur mort : inutile d'attendre le timeout
        # complet quand le EngineCore a deja echoue (OOM, cache KV negatif...).
        rc2, out2, _ = sh(
            f"docker inspect -f '{{{{.State.Running}}}}' {CONTAINER}", timeout=20
        )
        if rc2 == 0 and out2.strip() == "false":
            rc3, out3, _ = sh(
                f"docker inspect -f '{{{{.State.ExitCode}}}}' {CONTAINER}", timeout=20
            )
            exit_code = out3.strip() or "?"
            # Laisse le temps au processus de consigner la cause racine.
            time.sleep(5)
            return False, round(time.time() - t0, 1), \
                f"conteneur arrete (exit={exit_code}) apres {round(time.time()-t0,1)}s"
        time.sleep(5)
    return False, round(time.time() - t0, 1), f"health={last} apres {timeout}s"


def vllm_logs():
    rc, out, err = sh(f"docker logs {CONTAINER} 2>&1", timeout=120)
    return out + err


def parse_startup_logs(logs):
    info = {}
    m = re.search(r"Model loading took ([\d.]+) GiB and ([\d.]+) seconds", logs)
    if m:
        info["model_load_gib"] = float(m.group(1))
        info["model_load_s"] = float(m.group(2))
    m = re.search(r"Available KV cache memory: ([\d.]+) GiB", logs)
    if m:
        info["kv_cache_gib"] = float(m.group(1))
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", logs)
    if m:
        info["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", logs)
    if m:
        info["max_model_len_logged"] = int(m.group(1).replace(",", ""))
        info["max_concurrency"] = float(m.group(2))
    m = re.search(r"Graph capturing finished in ([\d.]+) secs, took ([\d.]+) GiB", logs)
    if m:
        info["cudagraph_gib"] = float(m.group(2))
    m = re.search(r"init engine .* took ([\d.]+) s", logs)
    if m:
        info["init_engine_s"] = float(m.group(1))
    m = re.search(r"torch.compile took ([\d.]+) s", logs)
    if m:
        info["torch_compile_s"] = float(m.group(1))
    return info


def parse_throughput_logs(logs):
    out = {"gen_throughput_max": None, "committed_throughput_max": None,
           "prompt_throughput_max": None}
    for m in re.finditer(r"Avg generation throughput: ([\d.]+) tokens/s", logs):
        v = float(m.group(1))
        out["gen_throughput_max"] = max(out["gen_throughput_max"] or 0, v)
    for m in re.finditer(r"Committed token throughput: ([\d.]+) tokens/s", logs):
        v = float(m.group(1))
        out["committed_throughput_max"] = max(out["committed_throughput_max"] or 0, v)
    for m in re.finditer(r"Avg prompt throughput: ([\d.]+) tokens/s", logs):
        v = float(m.group(1))
        out["prompt_throughput_max"] = max(out["prompt_throughput_max"] or 0, v)
    return out


# ------------------------------------------------------------------ mesures API


def make_prompt(n_tokens):
    reps = n_tokens // len(FILLER_WORDS) + 1
    filler = " ".join((FILLER_WORDS * reps)[:n_tokens])
    return filler + "\n\nReponds en une phrase : que font les modeles de diffusion ?"


def text_quality(text):
    """Mesure de degradation du texte genere par un modele de diffusion.

    Quand le nombre de passes de debruitage est insuffisant, la degradation
    ne prend pas la forme d'une repetition de phrases, mais de :
      -uites de ponctuation parasite ("*****", "....,,") ;
      - mots dupliques adjacent ("ForwardForward") ;
      - mots tronques ("Denoisingising").

    On mesure deux signaux orthogonaux :
      - `junk_ratio`   : part des caracteres appartenant a une suite de 3
                         caracteres non alphanumeriques identiques ou quasi
                         identiques ;
      - `dup_word_ratio`: part des mots adjacents identiques (insensible a la
                         casse).

    Sur un texte sain les deux valent 0 ; sur un texte degrade ils montent
    simultanement.
    """
    if not text:
        return {"degradation": None, "junk_ratio": None, "dup_word_ratio": None}

    # Suites de >= 3 caracteres non alphanumeriques identiques.
    junk = 0
    run_len = 0
    prev = None
    for ch in text:
        if not ch.isalnum() and ch not in " \n\r\t":
            if prev == ch:
                run_len += 1
            else:
                run_len = 1
            if run_len >= 3:
                junk += 1
        else:
            run_len = 0
        prev = ch
    junk_ratio = round(junk / max(1, len(text)), 4)

    # Mots adjacents identiques.
    words = [w.strip(".,;:!?()[]{}\"'*").lower() for w in text.split()]
    words = [w for w in words if w]
    dup = 0
    for a, b in zip(words, words[1:]):
        if a == b:
            dup += 1
    dup_word_ratio = round(dup / max(1, len(words) - 1), 4) if len(words) > 1 else 0.0

    degradation = round(max(junk_ratio, dup_word_ratio), 4)
    return {
        "degradation": degradation,
        "junk_ratio": junk_ratio,
        "dup_word_ratio": dup_word_ratio,
        "chars": len(text),
        "words": len(words),
    }


def run_generation(prompt, max_tokens=512, timeout=900):
    payload = {
        "model": SERVED_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "skip_special_tokens": False,
    }
    body, dt, err = post_json("/v1/chat/completions", payload, timeout=timeout)
    if err:
        return {"ok": False, "error": err, "elapsed_s": round(dt, 3)}
    usage = body.get("usage") or {}
    choice = (body.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    # vLLM expose le raisonnement sous "reasoning" (et "reasoning_content"
    # selon la version). Les deux doivent etre lus, sinon le texte genere
    # passe pour vide alors qu'il est present.
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    ct = usage.get("completion_tokens") or 0
    return {
        "ok": True,
        "elapsed_s": round(dt, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": ct,
        "tps": round(ct / dt, 2) if dt else 0.0,
        "finish_reason": choice.get("finish_reason"),
        "pad": content.count("<pad>") + reasoning.count("<pad>"),
        "content_chars": len(content),
        "reasoning_chars": len(reasoning),
        # Extrait brut : verifie que les tokens generes sont du texte reel et
        # non des tokens spe specialsdegeneres.
        "content_preview": content[:400],
        "reasoning_preview": reasoning[:400],
        "quality": text_quality(reasoning or content),
    }


def measure_ttft(max_tokens=512, timeout=900):
    """TTFT : temps jusqu'au premier fragment de la reponse stream."""
    payload = {
        "model": SERVED_NAME,
        "messages": [{"role": "user", "content": "Explique brievement un modele de diffusion."}],
        "max_tokens": max_tokens,
        "skip_special_tokens": False,
        "stream": True,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE_URL + "/v1/chat/completions", data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                if line.strip():
                    return {"ttft_s": round(time.time() - t0, 3)}
    except Exception as e:  # noqa: BLE001
        return {"ttft_error": f"{type(e).__name__}: {e}"}
    return {"ttft_error": "aucun fragment"}


def measure_tool():
    payload = {
        "model": SERVED_NAME,
        "messages": [
            {"role": "system", "content": "You are an assistant that uses tools."},
            {"role": "user", "content": "Use the get_weather tool to get the weather in Paris."},
        ],
        "tools": [TOOL_SPEC],
        "tool_choice": "required",
        "max_tokens": 512,
        "skip_special_tokens": False,
    }
    body, dt, err = post_json("/v1/chat/completions", payload)
    if err:
        return {"ok": False, "error": err}
    choice = (body.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    tool_calls = msg.get("tool_calls") or []
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    return {
        "ok": True,
        "elapsed_s": round(dt, 3),
        "has_tool": bool(tool_calls),
        "tool_name": tool_calls[0]["function"]["name"] if tool_calls else None,
        "pad": content.count("<pad>") + reasoning.count("<pad>"),
        "finish_reason": choice.get("finish_reason"),
        "tool_args": (tool_calls[0]["function"].get("arguments") if tool_calls else None),
        "content_preview": content[:400],
        "reasoning_preview": reasoning[:400],
    }


def measure_config(cfg):
    """Mesures de debit et de qualite pour une configuration demarree."""
    res = {}

    short_prompt = "Explique en detail le fonctionnement d'un modele de diffusion."
    long_prompt = make_prompt(60000)

    # Chauffage (compile CUDA / autotuner)
    warm = run_generation(short_prompt, max_tokens=64)
    res["warmup"] = {"ok": warm.get("ok"), "elapsed_s": warm.get("elapsed_s")}

    # Debit prompt court : 3 repetitions
    shorts = [run_generation(short_prompt, max_tokens=512) for _ in range(3)]
    ok_shorts = [r for r in shorts if r.get("ok")]
    res["short_prompt"] = {
        "runs": shorts,
        "tps_median": round(statistics.median(r["tps"] for r in ok_shorts), 2) if ok_shorts else None,
        "tps_max": round(max((r["tps"] for r in ok_shorts), default=0), 2),
        "elapsed_median_s": round(statistics.median(r["elapsed_s"] for r in ok_shorts), 3) if ok_shorts else None,
        "pad_total": sum(r.get("pad", 0) for r in ok_shorts),
    }

    # Debit prompt long (~65k tokens) : 3 repetitions
    # (le contexte long est la metrique pertinente pour OpenCode, qui
    # re-envoie l'historique complet a chaque tour)
    longs = [run_generation(long_prompt, max_tokens=512) for _ in range(3)]
    ok_longs = [r for r in longs if r.get("ok")]
    res["long_prompt"] = {
        "runs": longs,
        "tps_median": round(statistics.median(r["tps"] for r in ok_longs), 2) if ok_longs else None,
        "tps_max": round(max((r["tps"] for r in ok_longs), default=0), 2),
        "elapsed_median_s": round(statistics.median(r["elapsed_s"] for r in ok_longs), 3) if ok_longs else None,
        "prompt_tokens": ok_longs[0].get("prompt_tokens") if ok_longs else None,
        "pad_total": sum(r.get("pad", 0) for r in ok_longs),
    }

    res["ttft"] = measure_ttft()
    res["tool"] = measure_tool()

    # Memoire et compteurs
    res["gpu_windows"] = gpu_windows()
    res["gpu_wsl"] = gpu_wsl()
    res["prometheus"] = {
        "committed_tokens_total": prometheus_counter("vllm:diffusion_num_committed_tokens_total"),
        "denoising_steps_total": prometheus_counter("vllm:diffusion_num_denoising_steps_total"),
        "canvas_positions_total": prometheus_counter("vllm:diffusion_num_canvas_positions_total"),
    }
    logs = vllm_logs()
    res["startup"] = parse_startup_logs(logs)
    res["log_throughput"] = parse_throughput_logs(logs)
    return res


# ------------------------------------------------------------------- execution


def save_production_compose():
    if os.path.exists(COMPOSE_FILE) and not os.path.exists(COMPOSE_BACKUP):
        shutil.copy2(COMPOSE_FILE, COMPOSE_BACKUP)


def restore_production_compose():
    if os.path.exists(COMPOSE_BACKUP):
        shutil.copy2(COMPOSE_BACKUP, COMPOSE_FILE)


def main():
    configs = json.loads(sys.argv[1]) if len(sys.argv) > 1 else []
    if not configs:
        print("usage: bench2.py '<json array of configs>'")
        return 2

    os.makedirs(RESULT_DIR, exist_ok=True)
    save_production_compose()

    results = []
    try:
        for cfg in configs:
            cid = cfg.get("id", f"C{len(results)+1:02d}")
            print(f"\n{'='*76}\n{cid}: gm={cfg['gm']} mns={cfg['mns']} mml={cfg['mml']} "
                  f"kvd={cfg['kvd']} diff={cfg.get('diffusion_config')}\n{'='*76}", flush=True)
            rec = {"id": cid, "config": cfg, "status": "DEMARRE", "verdict": "ECHEC"}
            try:
                build_compose(cfg)
                rc, out, err = sh(f"docker compose -f {COMPOSE_FILE} config -q", timeout=60)
                if rc != 0:
                    rec["status"] = "YAML_INVALIDE"
                    rec["error"] = (out + err)[:400]
                    results.append(rec)
                    continue

                compose_down()
                time.sleep(3)
                rc, out, err = compose_up()
                if rc != 0:
                    rec["status"] = "ECHEC_COMPOSE_UP"
                    rec["error"] = (out + err)[:400]
                    results.append(rec)
                    continue

                healthy, boot_s, detail = wait_healthy(timeout=900)
                rec["boot_s"] = boot_s
                if not healthy:
                    rec["status"] = "ECHEC_SANTE"
                    rec["error"] = detail
                    logs = vllm_logs()
                    rec["error_logs"] = "\n".join(
                        [ln for ln in logs.splitlines()
                         if re.search(r"error|Error|ERROR|Traceback|OOM|memory", ln)][-8:])
                    rec["startup"] = parse_startup_logs(logs)
                    rec["gpu_windows"] = gpu_windows()
                    rec["gpu_wsl"] = gpu_wsl()
                    rec["verdict"] = "ECHEC"
                    # Sauvegarde immediate : un echec de demarrage est un
                    # resultat de mesure, pas une absence de resultat.
                    with open(os.path.join(RESULT_DIR, f"{cid}.json"), "w") as f:
                        json.dump(rec, f, indent=2)
                    results.append(rec)
                    print(f"{cid}: status={rec['status']} "
                          f"error={str(rec.get('error'))[:60]} verdict={rec['verdict']}",
                          flush=True)
                    compose_down()
                    time.sleep(3)
                    continue

                rec["status"] = "SAIN"
                rec.update(measure_config(cfg))

                short_tps = rec["short_prompt"].get("tps_median") or 0
                long_tps = rec["long_prompt"].get("tps_median") or 0
                tool_ok = rec["tool"].get("has_tool")
                pads = (rec["short_prompt"].get("pad_total", 0)
                        + rec["long_prompt"].get("pad_total", 0)
                        + rec["tool"].get("pad", 0))
                # Un test n'est OK que si toutes les requetes de mesure ont
                # reussi : un prompt long refuse (HTTP 400) est un echec de
                # la configuration, meme si le prompt court est rapide.
                req_errors = sum(
                    1 for label in ("short_prompt", "long_prompt")
                    for run in rec[label]["runs"] if not run.get("ok")
                )
                rec["request_errors"] = req_errors
                # Degradation du texte : un nombre de passes de debruitage
                # insuffisant produit des suites de ponctuation parasite et
                # des mots dupliques. Mesure sur le prompt court uniquement :
                # sur le prompt long, le modele repete legitinement les mots
                # du remplissage, ce qui fausserait le signal.
                degs = [
                    (run.get("quality") or {}).get("degradation")
                    for run in rec["short_prompt"]["runs"] if run.get("ok")
                ]
                degs = [v for v in degs if v is not None]
                rec["max_degradation"] = round(max(degs), 4) if degs else None
                rec["verdict"] = "OK" if (
                    short_tps >= 400 and long_tps > 0 and tool_ok
                    and pads == 0 and req_errors == 0
                    and (rec["max_degradation"] or 0) <= 0.03
                ) else "ECHEC"
            except Exception as e:  # noqa: BLE001
                rec["status"] = "EXCEPTION"
                rec["error"] = f"{type(e).__name__}: {e}"[:400]
            finally:
                compose_down()
                time.sleep(3)

            with open(os.path.join(RESULT_DIR, f"{cid}.json"), "w") as f:
                json.dump(rec, f, indent=2)
            results.append(rec)
            print(f"{cid}: status={rec['status']} short_tps={rec.get('short_prompt', {}).get('tps_median')} "
                  f"long_tps={rec.get('long_prompt', {}).get('tps_median')} "
                  f"tool={rec.get('tool', {}).get('has_tool')} "
                  f"gpu_shared={(rec.get('gpu_windows') or {}).get('adapter_sha_mib')} "
                  f"verdict={rec['verdict']}", flush=True)
    finally:
        restore_production_compose()

    with open(os.path.join(RESULT_DIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 100)
    print(f"{'Test':28} {'Status':10} {'Court t/s':>10} {'Long t/s':>10} "
          f"{'TTFT':>7} {'Ded MiB':>9} {'Shared MiB':>11} {'Tool':>5} {'KV GiB':>7} {'Verdict':>7}")
    for r in results:
        gw = r.get("gpu_windows") or {}
        st = r.get("startup") or {}
        print(f"{r['id']:28} {r['status']:10} "
              f"{str((r.get('short_prompt') or {}).get('tps_median')):>10} "
              f"{str((r.get('long_prompt') or {}).get('tps_median')):>10} "
              f"{str((r.get('ttft') or {}).get('ttft_s')):>7} "
              f"{str(gw.get('adapter_ded_mib')):>9} "
              f"{str(gw.get('adapter_sha_mib')):>11} "
              f"{str((r.get('tool') or {}).get('has_tool')):>5} "
              f"{str(st.get('kv_cache_gib')):>7} {r['verdict']:>7}")
    print(f"\nResultats : {RESULT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
