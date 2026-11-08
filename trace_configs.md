# Trace des configurations testées — DiffusionGemma / vLLM

Matériel : NVIDIA GeForce RTX 5090, 32 Go VRAM, WSL2
Modèle : `RedHatAI/diffusiongemma-26B-A4B-it-NVFP4` (MoE, 26B total / 4B actifs, 128 experts, top-8)
Cible : `max-model-len 262144` + outils sans `<pad>` + **> 500 tok/s** (Committed token throughput dans les logs vLLM)

Harnais : `bench.py` — chaque test redémarre le conteneur, attend la santé, puis mesure débit,
`<pad>` et appels d'outils. Les runs s'ajoutent à ce fichier.

---

## Synthèse des runs

| Test | image | gm | mns | mml | canvas | steps | statut | client tok/s | committed tok/s | `<pad>` outil | tool_call | verdict |
|------|-------|----|-----|-----|--------|-------|--------|--------------|------------------|-------------|----------|--------|
| A1 | latest (0.33.0) | 0.85 | 4 | 262144 | 256 | 48 | SAIN | — | — | — | — | INVALIDE (harnais : `temperature` interdit) |
| A2 | gemma (0.22.1rc1) | 0.85 | 4 | 262144 | 256 | 48 | SAIN | 40.0 | — | 0 | ✅ `get_weather` | partiel |
| B1 | gemma (0.22.1rc1) | 0.85 | 4 | 262144 | 256 | 48 | SAIN | **241.8** | — | 0 | ✅ `get_weather` | débit < 500 |
| B2 | gemma (0.22.1rc1) | 0.85 | 4 | 262144 | 256 | 48 | SAIN | 8.2 (4 req. concur.) | 76.8 | 0 | ✅ `get_weather` | débit < 500 |
| C1 | gemma | 0.85 | 4 | 262144 | **512** | 48 | SAIN | 220.1 | — | 0 | ✅ | débit < 500 |
| C2 | gemma | 0.85 | 4 | 262144 | 256 | **8** | SAIN | **299.3** | — | 0 | ✅ | débit < 500 |
| C3 | gemma | 0.85 | 4 | 262144 | 256 | 48 + prefix cache | SAIN | 199.8 | — | 0 | ✅ | débit < 500 |
| C4 | gemma | 0.85 | 4 | 262144 | **64** | 48 | SAIN | 216.1 | 8.52 | 0 | ✅ | débit < 500 |
| D1 | gemma | 0.85 | 4 | 262144 | 256 | **4** | SAIN | 311.5 | — | 0 | ✅ | débit < 500 (mesure à froid) |
| D2 | gemma | 0.85 | 4 | 262144 | 256 | **2** | SAIN | 342.5 | — | 0 | ❌ `has_tool=False` | qualité outil cassée |
| E1 | gemma | 0.85 | 4 | 262144 | 256 | 48 | SAIN | 459.5 | 53.11 | 0 | ✅ | débit < 500 (mesure à chaud) |
| E2 | gemma | 0.85 | 4 | 262144 | 256 | **8** | SAIN | **1055.7** | 92.22 | 0 | ✅ | débit log dilué par l'inactivité |
| E3 | gemma | 0.85 | 4 | 262144 | 256 | **4** | SAIN | **1788.8** | 93.02 | 0 | ✅ | débit log dilué par l'inactivité |
| F1 | gemma | 0.85 | 4 | 262144 | 256 | 8, **charge 4 workers** | SAIN | — | 102.40 | 0 | ✅ | concurrence : débit effondré |
| G1 | gemma | 0.85 | 4 | 262144 | 256 | **8, charge sérielle** | SAIN | 1020.0 | **1075.19** | 0 | ✅ `get_weather` | **OK** |
| G2 | gemma | 0.85 | 4 | 262144 | 256 | 4, charge sérielle | SAIN | 1850.7 | 1894.37 | 0 | ❌ `has_tool=False` | ECHEC (outil cassé) |
| H1 | gemma | 0.85 | 4 | 262144 | **128** | 8, charge sérielle | SAIN | 1082.1 | 1049.61 | 0 | ✅ `get_weather` | OK |
| H2 | gemma | 0.85 | 4 | 262144 | **512** | 8, charge sérielle | SAIN | 1176.2 | **1228.66** | 0 | ✅ `get_weather` | **OK** |
| I1 | gemma | 0.85 | 4 | 262144 | **1024** | 8 | ÉCHEC | — | — | — | — | EngineCore ne démarre pas (mémoire) |
| J1 | gemma | 0.85 | 4 | 262144 | **512** | 8, charge sérielle | SAIN | 1266.4 | **1177.50** | 0 | ✅ `get_weather` | **OK** (reproductible) |
| J2 | gemma | 0.85 | 4 | 262144 | **512** | 12, charge sérielle | SAIN | 853.7 | 870.38 | 0 | ✅ `get_weather` | OK (moins bon que steps=8) |
| **V** | gemma (compose de production) | 0.85 | 4 | 262144 | **512** | **8** | SAIN | — | **1177.59** | 0 | ✅ `get_weather` | **OK** (`verify.sh`) |

### Vérification bout-en-bout (`verify.sh`, configuration de production)

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

Le service a été relancé depuis le `docker-compose.yml` de production (hors harnais
`bench.py`) avant cette vérification.

### Configuration retenue

**`canvas_length: 512`, `max_denoising_steps: 8`**, service mono-requête (charge sérielle) —
tests H2 et J1 (reproductibilité : 1228.66 puis 1177.50 tok/s).

| Critère | Cible | Mesuré (H2 / J1) |
|---------|-------|------------------|
| Fenêtre contextuelle | 262 144 | 262 144 |
| Outils | appel structuré | ✅ `get_weather` |
| `<pad>` | aucun | 0 (texte, outil, 47 104 tokens de charge) |
| **Débit dans les logs vLLM** | **> 500 tok/s** | **1228.66 / 1177.50 tok/s** (`Committed token throughput`) |

G2 atteint 1894 tok/s mais le modèle n'émet plus d'appel d'outil valide à 4 étapes :
la qualité de l'outil impose un plancher de 8 étapes de débruitage.

### Plafonds identifiés

- **canvas 1024** : le moteur ne démarre plus. À `max_model_len=262144` et
  `max_num_seqs=4`, vLLM annonce `GPU KV cache size: 862,724 tokens` et
  `Maximum concurrency for 262,144 tokens per request: 3.29x` — il manque
  185 852 tokens de cache KV. **Canvas 512 est le maximum viable** sur 32 Go.
- **Concurrence** : les modèles de diffusion sérialisent mal le batch. À 4 workers
  simultanés, le débit dans les logs tombe à 102 tok/s contre 1075 en série, alors
  que `Canvas positions evaluated` montre bien 4 × 512 positions traitées. Le
  service doit donc traiter les requêtes **une par une** (`max_num_seqs=4` reste le
  plafond mémoire, pas un objectif de débit).

**Constats établis par ces mesures**

1. **`--max-num-seqs 4` + 262 144 tokens + outils + 0 `<pad>`** fonctionne de façon
   reproductible (B1, B2, C1–C4) : `tool_call` structuré nommé `get_weather`, zéro `<pad>`
   dans le contenu et dans le raisonnement.
2. **`max_denoising_steps` est le seul levier de débit efficace** : 48 → 8 fait passer
   241.8 → 299.3 tok/s (+24 %). `canvas_length` (64 / 256 / 512) et le prefix caching
   dégradent ou n'aident pas le débit.
3. **La concurrence dégrade fortement le débit** : 4 requêtes simultanées font chuter
   241.8 → 8.2 tok/s côté client (76.8 tok/s committed). À 262k de contexte, l'attention
   des 5 couches `full_attention` domine le coût par requête.
4. **`--hf-overrides '{"diffusion_sampler":...,"diffusion_entropy_bound":...}'` est un
   no-op** : ces clés n existent nulle part dans le code vLLM. Le sampler lit
   exclusivement le `generation_config.json` du checkpoint
   (`sampler_config.entropy_bound = 0.1`, `t_min = 0.4`, `t_max = 0.8`).
5. **`--override-generation-config` n'atteint pas le sampler de diffusion** : il alimente
   `get_diff_sampling_param()` (défauts de requête), pas `custom_sampler()`, qui lit
   `try_get_generation_config()`.
6. **La ligne `Committed token throughput` n'apparaît pas de façon systématique** : le
   compteur de métriques de diffusion n'est alimenté que sur certains builds/chemins
   (`spec_decode_stats`). Sur ce build elle est présente par intermittence (B2 : 76.8,
   C4 : 8.52). Le débit client reste la mesure de référence fiable.

---

## Diagnostic du `<pad>`

Deux causes distinctes ont été isolées.

### 1. Requête rejetée par l'API de diffusion

Les modèles de diffusion refusent `temperature`, `min_p`, `seed`, `min_tokens`,
`logit_bias` et `allowed_token_ids` :

```json
{"error":{"message":"The temperature, min_p, seed, min_tokens, logit_bias, bad_words,
and allowed_token_ids sampling parameters are not yet supported with diffusion models.",
"type":"BadRequestError"}}
```

Un client qui envoie `temperature` obtient une erreur HTTP 400, pas une réponse contenant
`<pad>` — mais un harness qui ignore cette erreur peut constater un canvas vide.

### 2. Canvas entièrement composed de `<pad>` (avant correction)

Mesuré sur une requête unique, sans outil, à température 0 :

| Cas | completion_tokens | occurrences `<pad>` | finish_reason |
|-----|-------------------|----------------------|---------------|
| texte, max_tokens=32 | 32 | 32 | `length` |
| texte, max_tokens=256 | 256 | 256 | `length` |
| outil, max_tokens=256 | 256 | 256 | `length` |
| outil + `stop_token_ids=[1,106,50]` | 256 | 256 | `length` |

**Mécanisme** (`vllm/model_executor/models/diffusion_gemma.py`) :

- le canvas est initialisé avec `torch.randint(0, vocab_size, …)` → l'id 0 (`<pad>`)
  fait partie du bruit initial ;
- à chaque étape de débruitage, les positions rejetées sont **rebruitées** avec
  `torch.randint(0, vocab_size, …)` → `<pad>` peut être réinjecté ;
- `argmax` sur des logits plats ou nuls renvoie l'indice **0**, c'est-à-dire `<pad>` ;
- les positions fantômes (canvas tronqué près de `max_model_len`) sont explicitement
  commentées comme « uniform (max entropy, **argmax 0**) » ;
- c'est `argmax_canvas` qui est commité, donc un `<pad>` est émis tel quel ;
- le tool parser `gemma4` ne fabrique jamais de `<pad>` : il transmet le texte du modèle
  verbatim, et il force `skip_special_tokens = False`, ce qui rend visible tout `<pad>`
  présent dans le canvas.

Le `<pad>` massif signale donc une **passe avant qui ne produit pas de logits utiles**,
et non un problème de template ou de parser d'outils. La configuration retenue
(`.env` → `HF_HOME` sur la partition Windows, image `gemma`, NVFP4) supprime ce
symptôme : 0 occurrence sur l'ensemble des tests B et C.

---

## Leviers de débit disponibles (code vLLM)

| Levier | Mécanisme | Effet mesuré |
|--------|-----------|--------------|
| `--diffusion-config '{"max_denoising_steps":N}'` | `converged = … | (step_after >= max_denoising_steps)` : chaque canvas est commité après N passes avant | **+24 % (48 → 8)** |
| `--diffusion-config '{"canvas_length":N}'` | Nombre de tokens commités par passe avant vs. travail de débruitage | Négatif (512 : −9 %, 64 : −11 %) |
| `--max-num-seqs` | Tampons d'état de diffusion `max_seqs × canvas_length × vocab_size` ; > 4 → CUDA OOM | Plafondé à 4 |
| `generation_config.json` du checkpoint | `sampler_config.entropy_bound`, `t_min`, `t_max`, `confidence_threshold`, `stability_threshold` | Non modifiable via CLI |
| `--override-generation-config` | Alimente `get_diff_sampling_param()`, **pas** `custom_sampler()` | Sans effet sur le sampler |
| `--hf-overrides` | Mute le `PretrainedConfig` HF ; le sampler ne lit pas ces clés | No-op |

Sur les versions vLLM plus récentes que celle déployée existent en plus des overrides
par requête (`diffusion_max_steps`, `diffusion_canvas_length` via `extra_args`), absents
de l'image `gemma` utilisée ici.

---

## Détail des exécutions

Les sections « ## <Test ID> » detaillees (paramètres YAML complets, temps de démarrage,
métriques, extraits de texte) sont ajoutées ci-dessous par chaque exécution de `bench.py`.

## D1_steps4

```yaml
image: 'vllm/vllm-openai:gemma'
model: 'RedHatAI/diffusiongemma-26B-A4B-it-NVFP4'
gm: 0.85
mns: 4
mml: 262144
kvd: 'fp8'
ecp: True
diffusion_config: '{"canvas_length":256,"max_denoising_steps":4}'
hf_overrides: '{"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":0.1}'
tools: True
wsl_patch: True
```

- **boot_s**: 260.9
- **committed_tps**: None
- **generation_tps**: None
- **denoise_steps**: None
- **client_tps**: 311.5
- **completion_tokens**: 512
- **gen_elapsed_s**: 1.64
- **gen_pad**: 0
- **gen_content_chars**: 0
- **gen_text_head**:  ||REASONING|| 
- **tool_pad**: 0
- **tool_pad_reasoning**: 0
- **has_tool**: True
- **tool_name**: get_weather
- **tool_error**: None
- **tool_content_head**: 
- **tool_reasoning_head**: 
- **statut**: SAIN
- **verdict**: ECHEC

## D2_steps2

```yaml
image: 'vllm/vllm-openai:gemma'
model: 'RedHatAI/diffusiongemma-26B-A4B-it-NVFP4'
gm: 0.85
mns: 4
mml: 262144
kvd: 'fp8'
ecp: True
diffusion_config: '{"canvas_length":256,"max_denoising_steps":2}'
hf_overrides: '{"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":0.1}'
tools: True
wsl_patch: True
```

- **boot_s**: 260.9
- **committed_tps**: None
- **generation_tps**: None
- **denoise_steps**: None
- **client_tps**: 342.5
- **completion_tokens**: 512
- **gen_elapsed_s**: 1.49
- **gen_pad**: 0
- **gen_content_chars**: 0
- **gen_text_head**:  ||REASONING|| 
- **tool_pad**: 0
- **tool_pad_reasoning**: 0
- **has_tool**: False
- **tool_name**: None
- **tool_error**: None
- **tool_content_head**: 
- **tool_reasoning_head**: 
- **statut**: SAIN
- **verdict**: ECHEC


---

## Run du 2026-11-08 06:51:19


## E1_steps48

```yaml
image: 'vllm/vllm-openai:gemma'
model: 'RedHatAI/diffusiongemma-26B-A4B-it-NVFP4'
gm: 0.85
mns: 4
mml: 262144
kvd: 'fp8'
ecp: True
diffusion_config: '{"canvas_length":256}'
hf_overrides: '{"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":0.1}'
tools: True
wsl_patch: True
```

- **boot_s**: 260.9
- **load_tps**: 343.0
- **load_completion_tokens**: 1812
- **load_elapsed_s**: 5.28
- **load_pad**: 0
- **load_text_head**: Les modèles de diffusion sont la technologie de derrière les outils d'intelligence artificielle générative de célèbres comme **Midjourney, DALL-E 3 ou Stable Diffusion**.

Contrairement aux anciennes approches (comme les GAN) qui essayaient de générer une image d'un coup, les modèles de diffusion ap
- **client_tps**: 459.5
- **completion_tokens**: 512
- **gen_elapsed_s**: 1.11
- **gen_pad**: 0
- **gen_content_chars**: 0
- **gen_text_head**:  ||REASONING|| 
- **tool_pad**: 0
- **tool_pad_reasoning**: 0
- **has_tool**: True
- **tool_name**: get_weather
- **tool_error**: None
- **tool_content_head**: 
- **tool_reasoning_head**: 
- **committed_tps**: 53.11
- **generation_tps**: 45.3
- **denoise_steps**: 19.2
- **committed_lines**: (APIServer pid=1) INFO 11-08 05:55:56 [loggers.py:273] Engine 000: Avg prompt throughput: 2.6 tokens/s, Avg generation throughput: 45.3 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
(APIServer pid=1) INFO 11-08 05:55:56 [metrics.py:156] DiffusionDecoding metrics: Committed token throughput: 53.11 tokens/s, Mean denoising steps per canvas: 19.18, Mean tokens committed per denoising step: 13.35, Committed: 2816 tokens, Denoising steps: 211, Canvas positions evaluated: 54016
- **statut**: SAIN
- **verdict**: ECHEC

## E2_steps8

```yaml
image: 'vllm/vllm-openai:gemma'
model: 'RedHatAI/diffusiongemma-26B-A4B-it-NVFP4'
gm: 0.85
mns: 4
mml: 262144
kvd: 'fp8'
ecp: True
diffusion_config: '{"canvas_length":256,"max_denoising_steps":8}'
hf_overrides: '{"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":0.1}'
tools: True
wsl_patch: True
```

- **boot_s**: 260.9
- **load_tps**: 873.4
- **load_completion_tokens**: 4096
- **load_elapsed_s**: 4.69
- **load_pad**: 0
- **load_text_head**:  ||REASONING|| 
- **client_tps**: 1055.7
- **completion_tokens**: 512
- **gen_elapsed_s**: 0.48
- **gen_pad**: 0
- **gen_content_chars**: 0
- **gen_text_head**:  ||REASONING|| 
- **tool_pad**: 0
- **tool_pad_reasoning**: 0
- **has_tool**: True
- **tool_name**: get_weather
- **tool_error**: None
- **tool_content_head**: -ing thethought:<channel|><|channel>thought
<channel|>
- **tool_reasoning_head**: 
- **committed_tps**: 92.22
- **generation_tps**: 88.8
- **denoise_steps**: 9.0
- **committed_lines**: (APIServer pid=1) INFO 11-08 06:00:39 [loggers.py:273] Engine 000: Avg prompt throughput: 2.7 tokens/s, Avg generation throughput: 88.8 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
(APIServer pid=1) INFO 11-08 06:00:39 [metrics.py:156] DiffusionDecoding metrics: Committed token throughput: 92.22 tokens/s, Mean denoising steps per canvas: 9.00, Mean tokens committed per denoising step: 28.44, Committed: 4864 tokens, Denoising steps: 171, Canvas positions evaluated: 43776
- **statut**: SAIN
- **verdict**: OK

## E3_steps4

```yaml
image: 'vllm/vllm-openai:gemma'
model: 'RedHatAI/diffusiongemma-26B-A4B-it-NVFP4'
gm: 0.85
mns: 4
mml: 262144
kvd: 'fp8'
ecp: True
diffusion_config: '{"canvas_length":256,"max_denoising_steps":4}'
hf_overrides: '{"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":0.1}'
tools: True
wsl_patch: True
```

- **boot_s**: 260.9
- **load_tps**: 1348.4
- **load_completion_tokens**: 4096
- **load_elapsed_s**: 3.04
- **load_pad**: 0
- **load_text_head**:  ||REASONING|| 
- **client_tps**: 1788.8
- **completion_tokens**: 512
- **gen_elapsed_s**: 0.29
- **gen_pad**: 0
- **gen_content_chars**: 0
- **gen_text_head**:  ||REASONING|| 
- **tool_pad**: 0
- **tool_pad_reasoning**: 0
- **has_tool**: True
- **tool_name**: get_weather
- **tool_error**: None
- **tool_content_head**: 
- **tool_reasoning_head**: 
- **committed_tps**: 93.02
- **generation_tps**: 89.1
- **denoise_steps**: 5.0
- **committed_lines**: (APIServer pid=1) INFO 11-08 06:05:20 [loggers.py:273] Engine 000: Avg prompt throughput: 2.7 tokens/s, Avg generation throughput: 89.1 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
(APIServer pid=1) INFO 11-08 06:05:20 [metrics.py:156] DiffusionDecoding metrics: Committed token throughput: 93.02 tokens/s, Mean denoising steps per canvas: 5.00, Mean tokens committed per denoising step: 51.20, Committed: 4864 tokens, Denoising steps: 95, Canvas positions evaluated: 24320
- **statut**: SAIN
- **verdict**: OK

## Récapitulatif

| Test | image | gm | mns | mml | statut | client tok/s | committed tok/s | pad outil | tool_call | verdict |
|------|-------|----|-----|-----|--------|--------------|------------------|-----------|----------|--------|
| E1_steps48 | gemma | 0.85 | 4 | 262144 | SAIN | 459.5 | 53.11 | 0 | True | ECHEC |
| E2_steps8 | gemma | 0.85 | 4 | 262144 | SAIN | 1055.7 | 92.22 | 0 | True | OK |
| E3_steps4 | gemma | 0.85 | 4 | 262144 | SAIN | 1788.8 | 93.02 | 0 | True | OK |


---

## Run du 2026-11-08 07:09:05



---

## Run du 2026-11-08 07:35:27


## G1_steps8_serial

```yaml
image: 'vllm/vllm-openai:gemma'
model: 'RedHatAI/diffusiongemma-26B-A4B-it-NVFP4'
gm: 0.85
mns: 4
mml: 262144
kvd: 'fp8'
ecp: True
diffusion_config: '{"canvas_length":256,"max_denoising_steps":8}'
hf_overrides: '{"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":0.1}'
tools: True
wsl_patch: True
```

- **boot_s**: 260.9
- **load_tps**: 683.7
- **load_completion_tokens**: 2048
- **load_elapsed_s**: 3.0
- **load_pad**: 0
- **load_text_head**:  ||REASONING|| 
- **client_tps**: 1020.0
- **completion_tokens**: 512
- **gen_elapsed_s**: 0.5
- **gen_pad**: 0
- **gen_content_chars**: 0
- **gen_text_head**:  ||REASONING|| 
- **tool_pad**: 0
- **tool_pad_reasoning**: 0
- **has_tool**: True
- **tool_name**: get_weather
- **tool_error**: None
- **tool_content_head**: 
- **tool_reasoning_head**: 
- **sustain_tps**: 1061.6
- **sustain_requests**: 83
- **sustain_completion_tokens**: 42496
- **sustain_duration_s**: 40.03
- **sustain_pad**: 0
- **sustain_errors**: 0
- **committed_tps**: 1075.19
- **generation_tps**: 1075.2
- **denoise_steps**: 9.1
- **committed_lines**: (APIServer pid=1) INFO 11-08 06:40:15 [loggers.py:273] Engine 000: Avg prompt throughput: 60.9 tokens/s, Avg generation throughput: 1075.2 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.9%, Prefix cache hit rate: 0.0%
(APIServer pid=1) INFO 11-08 06:40:15 [metrics.py:156] DiffusionDecoding metrics: Committed token throughput: 1075.17 tokens/s, Mean denoising steps per canvas: 9.12, Mean tokens committed per denoising step: 28.07, Committed: 10752 tokens, Denoising steps: 383, Canvas positions evaluated: 98048
(APIServer pid=1) INFO 11-08 06:40:25 [loggers.py:273] Engine 000: Avg prompt throughput: 58.0 tokens/s, Avg generation throughput: 1049.5 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.6%, Prefix cache hit rate: 0.0%
(APIServer pid=1) INFO 11-08 06:40:25 [metrics.py:156] DiffusionDecoding metrics: Committed token throughput: 1049.56 tokens/s, Mean denoising steps per canvas: 8.98, Mean tokens committed per denoising step: 28.52, Committed: 10496 tokens, Denoising steps: 368, Canvas positions evaluated: 94208
(APIServer pid=1) INFO 11-08 06:40:35 [loggers.py:273] Engine 000: Avg prompt throughput: 60.9 tokens/s, Avg generation throughput: 1075.2 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.6%, Prefix cache hit rate: 0.0%
(APIServer pid=1) INFO 11-08 06:40:35 [metrics.py:156] DiffusionDecoding metrics: Committed token throughput: 1075.19 tokens/s, Mean denoising steps per canvas: 8.88, Mean tokens committed per denoising step: 28.83, Committed: 10752 tokens, Denoising steps: 373, Canvas positions evaluated: 95488
- **statut**: SAIN
- **verdict**: OK

## G2_steps4_serial

```yaml
image: 'vllm/vllm-openai:gemma'
model: 'RedHatAI/diffusiongemma-26B-A4B-it-NVFP4'
gm: 0.85
mns: 4
mml: 262144
kvd: 'fp8'
ecp: True
diffusion_config: '{"canvas_length":256,"max_denoising_steps":4}'
hf_overrides: '{"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":0.1}'
tools: True
wsl_patch: True
```

- **boot_s**: 260.9
- **load_tps**: 889.0
- **load_completion_tokens**: 2048
- **load_elapsed_s**: 2.3
- **load_pad**: 0
- **load_text_head**:  ||REASONING|| 
- **client_tps**: 1850.7
- **completion_tokens**: 512
- **gen_elapsed_s**: 0.28
- **gen_pad**: 0
- **gen_content_chars**: 0
- **gen_text_head**:  ||REASONING|| 
- **tool_pad**: 0
- **tool_pad_reasoning**: 0
- **has_tool**: False
- **tool_name**: None
- **tool_error**: None
- **tool_content_head**: <|tool_call>call:get_weather{city:<|"|>Paris<|"|>}}
- **tool_reasoning_head**: 
- **sustain_tps**: 1854.8
- **sustain_requests**: 145
- **sustain_completion_tokens**: 74240
- **sustain_duration_s**: 40.02
- **sustain_pad**: 0
- **sustain_errors**: 0
- **committed_tps**: 1894.37
- **generation_tps**: 1894.3
- **denoise_steps**: 5.0
- **committed_lines**: (APIServer pid=1) INFO 11-08 06:45:31 [loggers.py:273] Engine 000: Avg prompt throughput: 107.3 tokens/s, Avg generation throughput: 1894.3 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.9%, Prefix cache hit rate: 0.0%
(APIServer pid=1) INFO 11-08 06:45:31 [metrics.py:156] DiffusionDecoding metrics: Committed token throughput: 1894.37 tokens/s, Mean denoising steps per canvas: 5.04, Mean tokens committed per denoising step: 50.79, Committed: 18944 tokens, Denoising steps: 373, Canvas positions evaluated: 95488
(APIServer pid=1) INFO 11-08 06:45:41 [loggers.py:273] Engine 000: Avg prompt throughput: 104.4 tokens/s, Avg generation throughput: 1868.5 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.6%, Prefix cache hit rate: 0.0%
(APIServer pid=1) INFO 11-08 06:45:41 [metrics.py:156] DiffusionDecoding metrics: Committed token throughput: 1868.57 tokens/s, Mean denoising steps per canvas: 5.00, Mean tokens committed per denoising step: 51.20, Committed: 18688 tokens, Denoising steps: 365, Canvas positions evaluated: 93440
(APIServer pid=1) INFO 11-08 06:45:51 [loggers.py:273] Engine 000: Avg prompt throughput: 104.4 tokens/s, Avg generation throughput: 1843.1 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.3%, Prefix cache hit rate: 0.0%
(APIServer pid=1) INFO 11-08 06:45:51 [metrics.py:156] DiffusionDecoding metrics: Committed token throughput: 1843.13 tokens/s, Mean denoising steps per canvas: 4.96, Mean tokens committed per denoising step: 51.63, Committed: 18432 tokens, Denoising steps: 357, Canvas positions evaluated: 91392
- **statut**: SAIN
- **verdict**: ECHEC

## Récapitulatif

| Test | image | gm | mns | mml | statut | client tok/s | sustain tok/s | committed tok/s | pad outil | tool_call | verdict |
|------|-------|----|-----|-----|--------|--------------|---------------|------------------|-----------|----------|--------|
| G1_steps8_serial | gemma | 0.85 | 4 | 262144 | SAIN | 1020.0 | 1061.6 | 1075.19 | 0 | True | OK |
| G2_steps4_serial | gemma | 0.85 | 4 | 262144 | SAIN | 1850.7 | 1854.8 | 1894.37 | 0 | False | ECHEC |


---

## Run du 2026-11-08 07:47:11


## H1_canvas128_s8

```yaml
image: 'vllm/vllm-openai:gemma'
model: 'RedHatAI/diffusiongemma-26B-A4B-it-NVFP4'
gm: 0.85
mns: 4
mml: 262144
kvd: 'fp8'
ecp: True
diffusion_config: '{"canvas_length":128,"max_denoising_steps":8}'
hf_overrides: '{"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":0.1}'
tools: True
wsl_patch: True
```

- **boot_s**: 260.9
- **load_tps**: 602.6
- **load_completion_tokens**: 1959
- **load_elapsed_s**: 3.25
- **load_pad**: 0
- **load_text_head**: Les modèles de diffusion sont une classe de modèles d'apprentissage générérative qui aujourd aujourd les les les plus populaires (comme **Midjourney, DALL-E 3 ou Stable Diffusion**).

Contrairement aux anciens modèles (Generative Adversarial Networks)) s'affaffent, les modèles de diffusion fonctionn
- **client_tps**: 1082.1
- **completion_tokens**: 512
- **gen_elapsed_s**: 0.47
- **gen_pad**: 0
- **gen_content_chars**: 0
- **gen_text_head**:  ||REASONING|| 
- **tool_pad**: 0
- **tool_pad_reasoning**: 0
- **has_tool**: True
- **tool_name**: get_weather
- **tool_error**: None
- **tool_content_head**: 
- **tool_reasoning_head**: 
- **sustain_tps**: 1036.2
- **sustain_requests**: 81
- **sustain_completion_tokens**: 41472
- **sustain_duration_s**: 40.02
- **sustain_pad**: 0
- **sustain_errors**: 0
- **committed_tps**: 1049.61
- **generation_tps**: 1049.6
- **denoise_steps**: 9.1
- **committed_lines**: (APIServer pid=1) INFO 11-08 06:51:59 [loggers.py:273] Engine 000: Avg prompt throughput: 58.0 tokens/s, Avg generation throughput: 1049.6 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.3%, Prefix cache hit rate: 0.0%
(APIServer pid=1) INFO 11-08 06:51:59 [metrics.py:156] DiffusionDecoding metrics: Committed token throughput: 1049.60 tokens/s, Mean denoising steps per canvas: 8.95, Mean tokens committed per denoising step: 14.30, Committed: 10496 tokens, Denoising steps: 734, Canvas positions evaluated: 93952
(APIServer pid=1) INFO 11-08 06:52:09 [loggers.py:273] Engine 000: Avg prompt throughput: 60.9 tokens/s, Avg generation throughput: 1049.5 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.6%, Prefix cache hit rate: 0.0%
(APIServer pid=1) INFO 11-08 06:52:09 [metrics.py:156] DiffusionDecoding metrics: Committed token throughput: 1049.56 tokens/s, Mean denoising steps per canvas: 8.99, Mean tokens committed per denoising step: 14.24, Committed: 10496 tokens, Denoising steps: 737, Canvas positions evaluated: 94336
(APIServer pid=1) INFO 11-08 06:52:19 [loggers.py:273] Engine 000: Avg prompt throughput: 58.0 tokens/s, Avg generation throughput: 1049.6 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
(APIServer pid=1) INFO 11-08 06:52:19 [metrics.py:156] DiffusionDecoding metrics: Committed token throughput: 1049.61 tokens/s, Mean denoising steps per canvas: 8.94, Mean tokens committed per denoising step: 14.32, Committed: 10496 tokens, Denoising steps: 733, Canvas positions evaluated: 93824
- **statut**: SAIN
- **verdict**: OK

## H2_canvas512_s8

```yaml
image: 'vllm/vllm-openai:gemma'
model: 'RedHatAI/diffusiongemma-26B-A4B-it-NVFP4'
gm: 0.85
mns: 4
mml: 262144
kvd: 'fp8'
ecp: True
diffusion_config: '{"canvas_length":512,"max_denoising_steps":8}'
hf_overrides: '{"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":0.1}'
tools: True
wsl_patch: True
```

- **boot_s**: 270.9
- **load_tps**: 748.2
- **load_completion_tokens**: 2048
- **load_elapsed_s**: 2.74
- **load_pad**: 0
- **load_text_head**:  ||REASONING|| 
- **client_tps**: 1176.2
- **completion_tokens**: 512
- **gen_elapsed_s**: 0.44
- **gen_pad**: 0
- **gen_content_chars**: 0
- **gen_text_head**:  ||REASONING|| 
- **tool_pad**: 0
- **tool_pad_reasoning**: 0
- **has_tool**: True
- **tool_name**: get_weather
- **tool_error**: None
- **tool_content_head**: 
- **tool_reasoning_head**: 
- **sustain_tps**: 1176.9
- **sustain_requests**: 92
- **sustain_completion_tokens**: 47104
- **sustain_duration_s**: 40.02
- **sustain_pad**: 0
- **sustain_errors**: 0
- **committed_tps**: 1228.66
- **generation_tps**: 1184.1
- **denoise_steps**: 9.2
- **committed_lines**: (APIServer pid=1) INFO 11-08 06:57:25 [loggers.py:273] Engine 000: Avg prompt throughput: 66.7 tokens/s, Avg generation throughput: 1177.5 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.6%, Prefix cache hit rate: 0.0%
(APIServer pid=1) INFO 11-08 06:57:25 [metrics.py:156] DiffusionDecoding metrics: Committed token throughput: 1177.57 tokens/s, Mean denoising steps per canvas: 8.96, Mean tokens committed per denoising step: 57.17, Committed: 11776 tokens, Denoising steps: 206, Canvas positions evaluated: 105472
(APIServer pid=1) INFO 11-08 06:57:35 [loggers.py:273] Engine 000: Avg prompt throughput: 66.7 tokens/s, Avg generation throughput: 1177.5 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.2%, Prefix cache hit rate: 0.0%
(APIServer pid=1) INFO 11-08 06:57:35 [metrics.py:156] DiffusionDecoding metrics: Committed token throughput: 1177.54 tokens/s, Mean denoising steps per canvas: 9.17, Mean tokens committed per denoising step: 55.81, Committed: 11776 tokens, Denoising steps: 211, Canvas positions evaluated: 108032
(APIServer pid=1) INFO 11-08 06:57:45 [loggers.py:273] Engine 000: Avg prompt throughput: 66.7 tokens/s, Avg generation throughput: 1177.5 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.2%, Prefix cache hit rate: 0.0%
(APIServer pid=1) INFO 11-08 06:57:45 [metrics.py:156] DiffusionDecoding metrics: Committed token throughput: 1177.54 tokens/s, Mean denoising steps per canvas: 9.00, Mean tokens committed per denoising step: 56.89, Committed: 11776 tokens, Denoising steps: 207, Canvas positions evaluated: 105984
- **statut**: SAIN
- **verdict**: OK

## Récapitulatif

| Test | image | gm | mns | mml | statut | client tok/s | sustain tok/s | committed tok/s | pad outil | tool_call | verdict |
|------|-------|----|-----|-----|--------|--------------|---------------|------------------|-----------|----------|--------|
| H1_canvas128_s8 | gemma | 0.85 | 4 | 262144 | SAIN | 1082.1 | 1036.2 | 1049.61 | 0 | True | OK |
| H2_canvas512_s8 | gemma | 0.85 | 4 | 262144 | SAIN | 1176.2 | 1176.9 | 1228.66 | 0 | True | OK |


---

## Run du 2026-11-08 07:58:13



---

## Run du 2026-11-08 08:13:30


## J1_c512_s8_confirm

```yaml
image: 'vllm/vllm-openai:gemma'
model: 'RedHatAI/diffusiongemma-26B-A4B-it-NVFP4'
gm: 0.85
mns: 4
mml: 262144
kvd: 'fp8'
ecp: True
diffusion_config: '{"canvas_length":512,"max_denoising_steps":8}'
hf_overrides: '{"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":0.1}'
tools: True
wsl_patch: True
```

- **boot_s**: 270.9
- **load_tps**: 735.9
- **load_completion_tokens**: 2048
- **load_elapsed_s**: 2.78
- **load_pad**: 0
- **load_text_head**:  ||REASONING|| 
- **client_tps**: 1266.4
- **completion_tokens**: 512
- **gen_elapsed_s**: 0.4
- **gen_pad**: 0
- **gen_content_chars**: 0
- **gen_text_head**:  ||REASONING|| 
- **tool_pad**: 0
- **tool_pad_reasoning**: 0
- **has_tool**: True
- **tool_name**: get_weather
- **tool_error**: None
- **tool_content_head**: 
- **tool_reasoning_head**: 
- **sustain_tps**: 1176.7
- **sustain_requests**: 92
- **sustain_completion_tokens**: 47104
- **sustain_duration_s**: 40.03
- **sustain_pad**: 0
- **sustain_errors**: 0
- **committed_tps**: 1177.5
- **generation_tps**: 1177.5
- **denoise_steps**: 9.1
- **committed_lines**: (APIServer pid=1) INFO 11-08 07:18:28 [loggers.py:273] Engine 000: Avg prompt throughput: 66.7 tokens/s, Avg generation throughput: 1177.5 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.2%, Prefix cache hit rate: 0.0%
(APIServer pid=1) INFO 11-08 07:18:28 [metrics.py:156] DiffusionDecoding metrics: Committed token throughput: 1177.50 tokens/s, Mean denoising steps per canvas: 9.09, Mean tokens committed per denoising step: 56.34, Committed: 11776 tokens, Denoising steps: 209, Canvas positions evaluated: 107008
(APIServer pid=1) INFO 11-08 07:18:38 [loggers.py:273] Engine 000: Avg prompt throughput: 66.7 tokens/s, Avg generation throughput: 1177.5 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.2%, Prefix cache hit rate: 0.0%
(APIServer pid=1) INFO 11-08 07:18:38 [metrics.py:156] DiffusionDecoding metrics: Committed token throughput: 1177.48 tokens/s, Mean denoising steps per canvas: 9.13, Mean tokens committed per denoising step: 56.08, Committed: 11776 tokens, Denoising steps: 210, Canvas positions evaluated: 107520
(APIServer pid=1) INFO 11-08 07:18:48 [loggers.py:273] Engine 000: Avg prompt throughput: 66.7 tokens/s, Avg generation throughput: 1177.5 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.2%, Prefix cache hit rate: 0.0%
(APIServer pid=1) INFO 11-08 07:18:48 [metrics.py:156] DiffusionDecoding metrics: Committed token throughput: 1177.49 tokens/s, Mean denoising steps per canvas: 8.91, Mean tokens committed per denoising step: 57.44, Committed: 11776 tokens, Denoising steps: 205, Canvas positions evaluated: 104960
- **statut**: SAIN
- **verdict**: OK

## J2_c512_s12

```yaml
image: 'vllm/vllm-openai:gemma'
model: 'RedHatAI/diffusiongemma-26B-A4B-it-NVFP4'
gm: 0.85
mns: 4
mml: 262144
kvd: 'fp8'
ecp: True
diffusion_config: '{"canvas_length":512,"max_denoising_steps":12}'
hf_overrides: '{"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":0.1}'
tools: True
wsl_patch: True
```

- **boot_s**: 260.9
- **load_tps**: 600.0
- **load_completion_tokens**: 2048
- **load_elapsed_s**: 3.41
- **load_pad**: 0
- **load_text_head**: Les models de diffusion (diffusion models) are een type de de generative de une sculpture de
        *   * de de de de de de de de         de de de         de de        
         de de de de de                         de         de de de de de de de- de de de         de de de de de de        .      
- **client_tps**: 853.7
- **completion_tokens**: 512
- **gen_elapsed_s**: 0.6
- **gen_pad**: 0
- **gen_content_chars**: 0
- **gen_text_head**:  ||REASONING|| 
- **tool_pad**: 0
- **tool_pad_reasoning**: 0
- **has_tool**: True
- **tool_name**: get_weather
- **tool_error**: None
- **tool_content_head**: 
- **tool_reasoning_head**: 
- **sustain_tps**: 857.1
- **sustain_requests**: 67
- **sustain_completion_tokens**: 34304
- **sustain_duration_s**: 40.02
- **sustain_pad**: 0
- **sustain_errors**: 0
- **committed_tps**: 870.38
- **generation_tps**: 870.4
- **denoise_steps**: 13.6
- **committed_lines**: (APIServer pid=1) INFO 11-08 07:23:42 [loggers.py:273] Engine 000: Avg prompt throughput: 46.4 tokens/s, Avg generation throughput: 819.2 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.2%, Prefix cache hit rate: 0.0%
(APIServer pid=1) INFO 11-08 07:23:42 [metrics.py:156] DiffusionDecoding metrics: Committed token throughput: 819.18 tokens/s, Mean denoising steps per canvas: 13.56, Mean tokens committed per denoising step: 37.75, Committed: 8192 tokens, Denoising steps: 217, Canvas positions evaluated: 111104
(APIServer pid=1) INFO 11-08 07:23:52 [loggers.py:273] Engine 000: Avg prompt throughput: 49.3 tokens/s, Avg generation throughput: 870.4 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.2%, Prefix cache hit rate: 0.0%
(APIServer pid=1) INFO 11-08 07:23:52 [metrics.py:156] DiffusionDecoding metrics: Committed token throughput: 870.37 tokens/s, Mean denoising steps per canvas: 12.82, Mean tokens committed per denoising step: 39.93, Committed: 8704 tokens, Denoising steps: 218, Canvas positions evaluated: 111616
(APIServer pid=1) INFO 11-08 07:24:02 [loggers.py:273] Engine 000: Avg prompt throughput: 49.3 tokens/s, Avg generation throughput: 870.4 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.2%, Prefix cache hit rate: 0.0%
(APIServer pid=1) INFO 11-08 07:24:02 [metrics.py:156] DiffusionDecoding metrics: Committed token throughput: 870.38 tokens/s, Mean denoising steps per canvas: 13.00, Mean tokens committed per denoising step: 39.38, Committed: 8704 tokens, Denoising steps: 221, Canvas positions evaluated: 113152
- **statut**: SAIN
- **verdict**: OK

## Récapitulatif

| Test | image | gm | mns | mml | statut | client tok/s | sustain tok/s | committed tok/s | pad outil | tool_call | verdict |
|------|-------|----|-----|-----|--------|--------------|---------------|------------------|-----------|----------|--------|
| J1_c512_s8_confirm | gemma | 0.85 | 4 | 262144 | SAIN | 1266.4 | 1176.7 | 1177.5 | 0 | True | OK |
| J2_c512_s12 | gemma | 0.85 | 4 | 262144 | SAIN | 853.7 | 857.1 | 870.38 | 0 | True | OK |
