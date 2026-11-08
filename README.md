# DiffusionGemma sur vLLM (Docker / WSL2)

Service OpenAI-compatible pour **DiffusionGemma 26B-A4B-it**, un modèle de diffusion
discrète (block diffusion) bâti sur un backbone MoE Gemma 4 — 26B paramètres totaux,
4B actifs, 128 experts, top-8.

La génération n'est pas autoregressive : le modèle débruit itérativement un canvas de
tokens. Chaque bloc est produit en parallèle après plusieurs passes de débruitage,
ce qui donne un débit par requête élevé au prix d'un TTFT plus long.

## Résultats

| Critère | Cible | Mesuré |
|---------|-------|--------|
| Fenêtre contextuelle | 262 144 tokens | 262 144 |
| Outils (agent IA) | appel structuré | ✅ `get_weather` |
| Anomalie `<pad>` | aucune | 0 (texte, outil, 53 248 tokens de charge) |
| **Débit dans les logs vLLM** | **> 500 tok/s** | **1 177.59 tok/s** (`Committed token throughput`) |

Configuration retenue : image `vllm/vllm-openai:gemma`, checkpoint
`RedHatAI/diffusiongemma-26B-A4B-it-NVFP4`, `canvas_length: 512`,
`max_denoising_steps: 8`, `max_model_len: 262144`, `max_num_seqs: 4`.

Matériel de mesure : RTX 5090 (32 Go), WSL2, vLLM 0.22.1rc1.dev357+g74b5964f0.

Les 19 configurations testées (18 variantes + la vérification finale sur la
configuration de production), leurs résultats chiffrés et le diagnostic complet du
`<pad>` sont consignés dans [`trace_configs.md`](trace_configs.md).

## Prérequis

- Docker avec le runtime NVIDIA (`nvidia-ctk runtime configure`)
- WSL2, kernel ≥ 4.19.121 (support CUDA épinglé)
- Une GPU NVIDIA de 32 Go de VRAM
- Un jeton Hugging Face si le modèle n'est pas déjà en cache

## Installation

```bash
cp .env.template .env
```

Puis renseigner `.env` :

```dotenv
HF_TOKEN=hf_xxx
HF_HOME=/chemin/vers/cache/huggingface
HF_MODEL="RedHatAI/diffusiongemma-26B-A4B-it-NVFP4"
```

`HF_HOME` doit pointer vers le cache Hugging Face **sur la partition Windows** : le
répertoire est monté dans le conteneur via le runtime 9p (drvfs), nettement plus rapide
qu'un montage depuis `/mnt/c` ou un système de fichiers Linux.

Le checkpoint `RedHatAI/diffusiongemma-26B-A4B-it-NVFP4` (quantification 4 bits NVFP4,
~17 Go) est utilisé par défaut car il tient dans 32 Go de VRAM. Le checkpoint BF16
officiel (`google/diffusiongemma-26B-A4B-it`) exige 64 Go et ne passe pas sur une
carte 32 Go.

## Démarrage

```bash
docker compose up -d
docker compose logs -f vllm
```

Le chargement du modèle 26B et la compilation CUDA prennent **environ 3 minutes**
(mesuré : 180 s jusqu'à `/health` = 200). La première requête paie ensuite
l'autotuner CUDA. Le healthcheck est dimensionné en conséquence
(`start_period: 600s`).

```bash
curl -s http://localhost:8000/health          # 200 une fois prêt
curl -s http://localhost:8000/v1/models       # -> [{"id": "diffusiongemma"}]
```

Le modèle est exposé sous le nom **`diffusiongemma`** (`--served-model-name`). Utiliser
l'identifiant Hugging Face complet renvoie une erreur 404.

## Utilisation

### Complétion simple

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "diffusiongemma",
    "messages": [{"role": "user", "content": "Écris un court poème sur la mer."}],
    "max_tokens": 512
  }'
```

> **Ne pas envoyer `temperature`** (ni `seed`, `min_p`, `logit_bias`,
> `allowed_token_ids`) : l'API de diffusion rejette ces paramètres en HTTP 400. Le
> sampling est piloté par le `generation_config.json` du checkpoint
> (`sampler_config.entropy_bound`, `t_min`, `t_max`).

### Appel d'outil

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "diffusiongemma",
    "messages": [{"role": "user", "content": "Météo à Paris ?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Récupère la météo d une ville",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }],
    "tool_choice": "required",
    "max_tokens": 512
  }'
```

Les outils sont servis par le protocole Gemma 4 (`--tool-call-parser gemma4`), avec le
template `examples/tool_chat_template_gemma4.jinja` embarqué dans l'image. Le mode
raisonnement s'active par défaut (`enable_thinking: true`) et le raisonnement est
extrait dans `reasoning_content`.

## Configuration

```yaml
command:
  - '--model'
  - 'RedHatAI/diffusiongemma-26B-A4B-it-NVFP4'
  - '--served-model-name'
  - 'diffusiongemma'
  - '--host'
  - '0.0.0.0'
  - '--port'
  - '8000'
  - '--tensor-parallel-size'
  - '1'
  - '--gpu-memory-utilization'
  - '0.85'
  - '--max-model-len'
  - '262144'
  - '--kv-cache-dtype'
  - 'fp8'
  - '--dtype'
  - 'auto'
  - '--generation-config'
  - 'vllm'
  - '--max-num-seqs'
  - '4'
  - '--enable-chunked-prefill'
  - '--diffusion-config'
  - '{"canvas_length":512,"max_denoising_steps":8}'
  - '--enable-auto-tool-choice'
  - '--tool-call-parser'
  - 'gemma4'
  - '--reasoning-parser'
  - 'gemma4'
  - '--chat-template'
  - 'examples/tool_chat_template_gemma4.jinja'
  - '--default-chat-template-kwargs'
  - '{"enable_thinking":true}'
```

### Rôle de chaque paramètre

| Paramètre | Valeur | Raison |
|-----------|--------|--------|
| `--diffusion-config` | `{"canvas_length":512,"max_denoising_steps":8}` | **Seul levier de débit mesuré.** 48 → 8 passes fait passer de 459 à 1 055 tok/s. En dessous de 8, l'appel d'outil casse |
| `--canvas_length` | `512` | À 8 passes, un canvas plus large est plus rapide (1 228 vs 1 075 tok/s à 256). 1024 fait échouer le démarrage |
| `--max-model-len` | `262144` | Fenêtre contextuelle maximale du modèle |
| `--max-num-seqs` | `4` | Plafond mémoire : les tampons de diffusion sont `max_seqs × canvas_length × vocab_size`. Au-delà, CUDA OOM |
| `--gpu-memory-utilization` | `0.85` | Marge pour les activations pendant le débruitage |
| `--kv-cache-dtype` | `fp8` | Réduit l'empreinte du cache KV (attention à la perte de précision) |
| `--generation-config` | `vllm` | Le `generation_config.json` du checkpoint impose `max_new_tokens: 256` et écraserait la limite par requête |
| `--enable-chunked-prefill` | — | Découpe les longs préfills pour limiter le pic mémoire |
| `--enable-auto-tool-choice` | — | Active la détection automatique des appels d'outils |
| `--tool-call-parser` | `gemma4` | Protocole d'outils Gemma 4 attendu par le `chat_template.jinja` du checkpoint |
| `--reasoning-parser` | `gemma4` | Extrait le canal de raisonnement `<\|channel>thought` |
| `--chat-template` | `examples/tool_chat_template_gemma4.jinja` | Template de l'image, aligné sur le format d'outils du modèle |

### Options qui ne fonctionnent pas

Ces options paraissent plausibles mais n'ont **aucun effet** sur le débit ou la qualité.
Elles sont documentées ici pour éviter de les retester.

| Option | Pourquoi elle est inopérante |
|--------|-------------------------------|
| `--hf-overrides '{"diffusion_sampler":...,"diffusion_entropy_bound":...}'` | Ces clés n'existent dans aucun fichier du code vLLM. Le sampler lit le `generation_config.json` du checkpoint (`sampler_config.entropy_bound = 0.1`) |
| `--override-generation-config` | Alimente `get_diff_sampling_param()` (défauts de requête), pas `custom_sampler()` qui construit le `DiffusionSampler` |
| `--max-num-seqs 8` ou plus | OOM CUDA |
| `canvas_length` 1024 | Le moteur ne démarre plus : à `max_model_len=262144` et `max_num_seqs=4`, vLLM annonce `GPU KV cache size: 862,724 tokens` et `Maximum concurrency for 262,144 tokens per request: 3.29x` — il manque 185 852 tokens de cache KV |
| `--enable-prefix-caching` | N'aide pas le débit de diffusion (199.8 vs 241.8 tok/s) |
| `temperature`, `seed`, `min_p`, `logit_bias` | Rejetés en HTTP 400 par l'API de diffusion |

## Contrainte importante : service mono-requête

Les modèles de diffusion sérialisent mal le batch. Mesuré sur la configuration
retenue :

| Charge | Débit dans les logs vLLM |
|--------|-------------------------|
| Requêtes **sérielles** (1 à la fois) | **1 177 – 1 229 tok/s** |
| 4 requêtes **simultanées** | 102 tok/s |

`--max-num-seqs 4` reste le plafond mémoire, mais **le débit n'est atteint qu'en
traitant les requêtes une par une**. Un agent IA qui parallélise ses appels d'outils
verra le débit s'effondrer d'un facteur 10.

## Vérification

```bash
./verify.sh
```

Le script attend la santé du conteneur, puis contrôle les quatre points ci-dessus
contre le service en cours d'exécution.

```
=== Verification DiffusionGemma / vLLM ===
  [OK]   service pret (/health = 200)                   code=200
  [OK]   fenetre contextuelle = 262144                  max_model_len=262144
  [OK]   appel d'outil structure (get_weather)          occurrences=1
  [OK]   aucune anomalie <pad> (reponse outil)          <pad> x0
  [OK]   aucune anomalie <pad> (reponse texte)          <pad> x0
  ... charge continue pendant 45s (requetes seriees)
    53248 tokens generes, 0 <pad>, 0 erreurs
  [OK]   debit logs vLLM > 500 tok/s                    Committed token throughput = 1177.59 tok/s

=== 6 reussis, 0 echoues ===
```

## Benchmark et trace des configurations

`bench.py` redémarre le conteneur pour chaque configuration, attend la santé, puis
mesure :

- le débit client (tokens de complétion / seconde) ;
- le `Committed token throughput` publié dans les logs vLLM ;
- la présence de `<pad>` dans une réponse texte **et** dans une réponse avec outil ;
- la présence effective d'un `tool_calls` structuré.

```bash
python3 bench.py '[
  {"id":"A1","image":"vllm/vllm-openai:gemma","model":"RedHatAI/diffusiongemma-26B-A4B-it-NVFP4",
   "gm":0.85,"mns":4,"mml":262144,"kvd":"fp8","tools":true,
   "diffusion_config":"{\"canvas_length\":512,\"max_denoising_steps\":8}",
   "sustain_s":40,"sustain_workers":1,"sustain_max_tokens":512}
]'
```

Clés reconnues : `model`, `image`, `gm` (gpu-memory-utilization), `mns`
(max-num-seqs), `mml` (max-model-len), `kvd` (kv-cache-dtype), `tp`, `ecp`,
`prefix_cache`, `attention_backend`, `diffusion_config`, `hf_overrides`,
`override_gen_config`, `tools`, `wsl_patch`, `load_*`, `gen_*`, `sustain_*`.

> `bench.py` **réécrit `docker-compose.yml`** avec la configuration testée. Après un
> benchmark, restaurez le compose de production (ce README documente les valeurs
> retenues) ou relancez-le explicitement.

Les résultats détaillés de chaque configuration testée sont consignés dans
[`trace_configs.md`](trace_configs.md).

## Diagnostic

### L'anomalie `<pad>`

Deux causes distinctes ont été isolées.

**1. Requête rejetée par l'API de diffusion.** Les modèles de diffusion refusent
`temperature`, `min_p`, `seed`, `min_tokens`, `logit_bias` et `allowed_token_ids` :

```json
{"error":{"message":"The temperature, min_p, seed, min_tokens, logit_bias, bad_words,
and allowed_token_ids sampling parameters are not yet supported with diffusion models.",
"type":"BadRequestError"}}
```

Un client qui envoie `temperature` obtient une erreur HTTP 400, pas une réponse
contenant `<pad>` — mais un harness qui ignore cette erreur constate un canvas vide.

**2. Canvas entièrement composé de `<pad>`.** Mesuré sur une requête unique, sans
outil, à température 0 :

| Cas | completion_tokens | occurrences `<pad>` | finish_reason |
|-----|-------------------|----------------------|---------------|
| texte, max_tokens=32 | 32 | 32 | `length` |
| texte, max_tokens=256 | 256 | 256 | `length` |
| outil, max_tokens=256 | 256 | 256 | `length` |
| outil + `stop_token_ids=[1,106,50]` | 256 | 256 | `length` |

Mécanisme (`vllm/model_executor/models/diffusion_gemma.py`) :

- le canvas est initialisé avec `torch.randint(0, vocab_size, …)` → l'id 0 (`<pad>`)
  fait partie du bruit initial ;
- à chaque passe, les positions rejetées sont **rebruitées** avec
  `torch.randint(0, vocab_size, …)` → `<pad>` peut être réinjecté ;
- `argmax` sur des logits plats ou nuls renvoie l'indice **0**, c'est-à-dire `<pad>` ;
- les positions fantômes (canvas tronqué près de `max_model_len`) sont explicitement
  commentées comme « uniform (max entropy, **argmax 0**) » ;
- c'est `argmax_canvas` qui est commité, donc un `<pad>` est émis tel quel ;
- le tool parser `gemma4` ne fabrique jamais de `<pad>` : il transmet le texte du
  modèle verbatim, et il force `skip_special_tokens = False`, ce qui rend visible tout
  `<pad>` présent dans le canvas.

Le `<pad>` massif signale donc une **passe avant qui ne produit pas de logits utiles**,
et non un problème de template ou de parser d'outils. Avec le checkpoint NVFP4 et
l'image `gemma`, le symptôme est absent : 0 occurrence sur l'ensemble des tests.

### Autres notes

- **Conteneur `unhealthy` au démarrage** : le chargement dépasse le `start_period` du
  healthcheck. Un `start_period` trop court produit des faux diagnostics d'échec.
- **`prompt_logprobs` non supporté** : les modèles de diffusion ne gèrent pas les
  logprobs de prompt ; cette option fait tomber l'EngineCore.
- **Version de vLLM** : l'image `vllm/vllm-openai:gemma` (v0.22.1rc1) est **celle qui
  fonctionne** pour DiffusionGemma — elle embarque le modèle `diffusion_gemma`, le tool
  parser `gemma4` et le template d'outils. L'image `vllm/vllm-openai:latest` (v0.33.0)
  charge le modèle mais son API de diffusion a divergé (les paramètres
  `diffusion_max_steps` / `diffusion_canvas_length` par requête n'existent que dans les
  versions récentes) : à ne pas utiliser pour ce checkpoint.
- **`Committed token throughput` dans les logs** : c'est le rapport
  `tokens commités / fenêtre de log` (`vllm/v1/spec_decode/metrics.py`). La fenêtre
  inclut l'inactivité du serveur : sur un serveur inactif, la métrique est diluée et
  sous-estime le débit réel (92 tok/s au lieu de 1 177). Pour lire un débit
  représentatif, maintenir une charge continue pendant au moins une fenêtre
  (`VLLM_LOG_STATS_INTERVAL`, 10 s par défaut).
- **Première requête plus lente** : elle paie la compilation CUDA et l'autotuner
  (flashinfer, `trtllm::fused_moe`). Mesurer après une requête de chauffe, sinon le
  débit est sous-estimé d'un facteur ~2.

## Fichiers du dépôt

| Fichier | Rôle |
|---------|------|
| `docker-compose.yml` | Service vLLM : image, paramètres de diffusion, outils, healthcheck |
| `wsl_entrypoint.sh` | Patch WSL2 : active `pin_memory` / UVA que vLLM désactive sous WSL |
| `.env.template` | Variables d'environnement à copier en `.env` |
| `bench.py` | Harnais de test : mesure débit, `<pad>`, appels d'outils pour chaque config |
| `verify.sh` | Vérification bout-en-bout de la configuration de production |
| `trace_configs.md` | Trace des configurations testées (résultats chiffrés) |

`.env`, `state/` et `vllm_cache/` sont ignorés par Git (secrets, cache de compilation).

## Arrêt

```bash
docker compose down
```
