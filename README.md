# DiffusionGemma sur vLLM (Docker / WSL2)

Service OpenAI-compatible pour **DiffusionGemma 26B-A4B-it**, un modèle de diffusion
discrète (block diffusion) bâti sur un backbone MoE Gemma 4 — 26B paramètres totaux,
4B actifs, 128 experts, top-8.

La génération n'est pas autoregressive : le modèle débruit itérativement un canvas de
tokens. Chaque bloc est produit en parallèle après plusieurs passes de débruitage,
ce qui donne un débit par requête élevé au prix d'un TTFT plus long.

## Résultats

Mesures du 2026-11-09 sur la configuration de production (RTX 5090 32 Go, WSL2,
vLLM 0.22.1rc1.dev357+g74b5964f0).

| Critère | Cible | Mesuré |
|---------|-------|--------|
| Fenêtre contextuelle | 196 608 tokens | 196 608 (`Maximum concurrency: 2.04x`) |
| Outils (agent IA) | appel structuré | ✅ `get_weather` → `{"city": "Paris"}` |
| Anomalie `<pad>` | aucune | 0 (texte, outil) |
| **Débit client, prompt court** | **> 400 tok/s** | **1 365 tok/s** (médiane de 3) |
| **Débit client, prompt 65 k tokens** | — | **410 tok/s** (médiane de 3) |
| **Débit session OpenCode (8 tours)** | — | **1 411 tok/s** (médiane), min 333, max 1 435 |
| **Qualité (banc à 8 critères)** | — | **7/8** — seul échec : rappel au-delà de 100 k tokens |
| **Rappel en contexte de 65 k tokens** | — | **16/20 réponses exactes** (raisononnement correct 20/20) |
| TTFT | — | 0,21 s |
| Mémoire partagée GPU (débitement) | 0 | **127 Mio** (contre 2 650 Mio avant) |
| VRAM libre | — | ~2 800 Mio (contre 300 Mio avant) |

Configuration retenue : image `vllm/vllm-openai:gemma`, checkpoint
`RedHatAI/diffusiongemma-26B-A4B-it-NVFP4`, **`canvas_length: 256`**,
`max_denoising_steps: 8`, `max_model_len: 196608`, `max_num_seqs: 4`,
**`gpu_memory_utilization: 0.70`**, `kv_cache_dtype: fp8`.

### Le paramètre critique : `--gpu-memory-utilization 0.70`

Avec `0.85`, vLLM réserve 27,7 Gio sur une carte de 32,6 Gio sans tenir compte des
~5,5 Gio déjà occupés par Windows (4,37 Gio pour le compositeur de bureau, ~1 Go pour
les autres applications). Le surplus est serviced par la **mémoire partagée** : 2 650
Mio de backing-store, des accès PCIe via le CPU, et un débit effondré à ~140 tok/s
(8 tok/s en contexte long OpenCode).

À `0.70`, le budget vLLM tient dans la VRAM réellement disponible : plus aucun
débordement (127 Mio partagés), et le débit passe à 1 365 tok/s. Le gain est donc
d'un facteur ~10, et non lié à la quantification ni à la longueur du canvas.

### Le second levier : `--diffusion-config '{"canvas_length":256}'`

Le canvas est la taille du bloc de tokens que le modèle débruitit en parallèle. Il
détermine directement le coût d'une passe de débruitage, donc le débit.

| `canvas_length` | Débit court | Débit 65 k tokens | Dégradation du texte |
|-----------------|-------------|-------------------|----------------------|
| 64 | 611 tok/s | 195 tok/s | — |
| 128 | 1 017 tok/s | 345 tok/s | — |
| **256** | **1 365 tok/s** | **410 tok/s** | **0 %** |
| 512 | 1 525 tok/s | 121 tok/s | 0 % |

Le débit court augmente marginalement avec le canvas, mais le débit en contexte long
— le cas d'usage OpenCode, qui ré-envoie tout l'historique — est **3,4× supérieur** en
`canvas 256`. La validation sur une session agent réelle de 8 tours confirme le gain
(1 411 tok/s de médiane contre 1 092 tok/s) et surtout la **régularité** : en
`canvas 512`, le débit oscille entre 110 et 1 545 tok/s selon le tour, en `canvas 256`
il reste entre 333 et 1 435 tok/s.

`canvas_length` n'est pas un paramètre de vLLM mais un réglage du modèle de diffusion :
il est passé via `--diffusion-config` (JSON).

### La mémoire n'est PAS multipliée par le nombre de passes

Une intuition naturelle : « le débruitage passe 8 fois sur le contexte, donc il faut
8 fois plus de mémoire ». **C'est faux, et c'est mesuré.**

Le cache KV est **partagé** entre les passes : chacune recalcule les logits du
canvas mais réutilise les KV du contexte. Échantillonnage de
`vllm:kv_cache_usage_perc` pendant une génération longue :

| Tokens du prompt | Usage KV | Tokens occupés | Ratio |
|------------------|----------|----------------|-------|
| 49 174 | 17,2 % | 69 035 | 1,40× |
| 98 327 | 27,8 % | 111 518 | 1,13× |

Le ratio reste constant (~1,1×) : le cache est alloué **une fois**, pas huit fois.
Si chaque passe dupliquait le contexte, 98 327 tokens occuperaient 786 616 tokens,
soit 196 % du cache disponible (400 959 tokens) — la requête serait refusée.

Corroboration : 196 608 tokens de fenêtre sont servis avec 4,42 Gio de cache KV et
`Maximum concurrency: 2.04x`. Si le contexte était multiplié par 8, la fenêtre utile
serait 24 576 tokens.

Ce qui coûte 8 fois, c'est le **temps de calcul**, pas la mémoire. D'où le débit de
121 tok/s à `canvas 512` contre 410 tok/s en `canvas 256` sur 65 k tokens.

### Qualité des réponses

La métrique de « dégradation » de `bench2.py` ne mesure que deux choses : la
ponctuation parasite (`***`, `....,,`) et les mots adjacents dupliqués. Elle détecte
une **dégénérescence** du modèle de diffusion (quand le nombre de passes est
insuffisant), et rien d'autre. Elle ne dit rien de la justesse du propos.

`state/quality_bench.py` complète avec 8 critères **objectifs et vérifiables
automatiquement** — rappel en contexte long, respect d'un nombre exact d'éléments,
phrase unique, mot interdit, JSON strict, appel d'outil structuré, code Python qui
compile, absence de référence à un fichier inexistant.

| Critère | `canvas 256` | `canvas 512` |
|---------|--------------|--------------|
| `exact_5_lines` | OK | OK |
| `one_sentence` | OK | OK |
| `forbidden_word_absent` | OK | OK |
| `strict_json` | OK | OK |
| `tool_use` | OK | OK |
| `code_compiles` | OK | OK |
| `no_hallucinated_file` | OK | OK |
| `long_context_recall` (153 k tokens) | ÉCHEC | ÉCHEC |
| **Score** | **7/8** | **7/8** |

Le canvas ne dégrade pas la qualité du propos. Sur le rappel en contexte long (64 953
tokens, 20 trials, jugement sur la réponse finale) :

| Configuration | Réponses exactes | Vides | Raisonnement correct | Débit |
|---------------|------------------|-------|-----------------------|-------|
| **`canvas 256`** | **16/20 (80 %)** | 2/20 | 20/20 | 249 tok/s |
| `canvas 512` | 13/20 (65 %) | 5/20 | 20/20 | 121 tok/s |

**Le point faible réel n'est pas le canvas mais le commit final.** Dans 100 % des
trials, le raisonnement contient la bonne réponse : le modèle *localise* toujours
l'information. Ce qui varie, c'est le texte effectivement commité — il peut être vide,
tronqué, ou approchant (`7429831` → `74299811`). C'est la mécanique de la diffusion :
le canvas est débruité stochastiquement (Gumbel-max, renoise aléatoire).

Au-delà de ~100 k tokens de contexte, le rappel s'effondre dans les deux
configurations. La fenêtre utile réelle se situe donc autour de **65 k–100 k tokens**,
indépendamment du réglage vLLM.

### Transparence sur la métrique de débit

Trois grandeurs distinctes coexistent, et il faut les nommer explicitement :

| Grandeur | Valeur | Ce qu'elle mesure |
|---|---|---|
| **Débit client** | 1 365 tok/s | tokens de complétion ÷ durée réelle de la requête (inclut le TTFT) — **c'est la grandeur affichée par OpenCode** |
| `Committed token throughput` (log), **sous charge continue** | 1 382 tok/s | tokens ÷ fenêtre glissante de 10 s, sur une charge qui remplit la fenêtre |
| `Committed token throughput` (log), **serveur inactif** | ~38 tok/s | la même métrique, mais la fenêtre contient surtout des secondes sans requête : elle est alors diluée et ne décrit plus rien |

Les 1 177 tok/s historiquement cités dans les logs vLLM provenaient d'un intervalle de
log saturé par une charge continue — c'est le seul cas où la métrique des logs
représente le débit réel. Sur une requête isolée, elle affiche **bien moins** que le
débit client. Toute comparaison doit donc préciser la grandeur utilisée **et** l'état
de charge du serveur.

### Dépendance à la longueur du contexte

Le coût d'une passe de débruitage est proportionnel à la longueur du prompt, et il est
payé `max_denoising_steps` fois. Le débit décroît donc avec la taille de l'historique :

| Prompt | Débit client (`canvas 256`) |
|---|---|
| 29 tokens | 1 365 tok/s |
| 65 031 tokens | 410 tok/s |

OpenCode ré-envoie l'historique complet à chaque tour : une session de 30 k à 60 k
tokens de contexte se situe dans cette zone de déplombage. C'est la limite structurelle
du modèle de diffusion, indépendante de la configuration vLLM.

Les 38 configurations testées et leurs résultats chiffrés sont consignés dans
[`JOURNAL.md`](JOURNAL.md) ; l'historique des runs antérieurs dans
[`trace_configs.md`](trace_configs.md).

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

> ⚠️ Le cache doit contenir **le checkpoint configuré dans `HF_MODEL`**. Un bind-mount
> ne protège pas les fichiers téléchargés dans la couche overlay du conteneur : si le
> modèle n'est pas dans le dossier monté, il est perdu au premier `docker compose down`.
> Après un téléchargement, vérifiez que le dossier du modèle existe bien sous
> `$HF_HOME/hub/`.

Le checkpoint `RedHatAI/diffusiongemma-26B-A4B-it-NVFP4` (quantification 4 bits NVFP4,
18 Go) est utilisé par défaut : il tient dans 32 Go de VRAM et s'est révélé plus rapide
que l'alternative AWQ-INT4 testée (`cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4`,
17,2 Go) — 1 215 tok/s contre 1 525 tok/s à configuration identique, et 31 tok/s
contre 121 tok/s en contexte long. Le checkpoint BF16 officiel
(`google/diffusiongemma-26B-A4B-it`) exige 64 Go et ne passe pas sur une carte 32 Go.

Pour changer de checkpoint, modifier `HF_MODEL` **et** placer les poids correspondants
dans `$HF_HOME/hub/`.

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
extrait dans le champ **`reasoning`** de la réponse.

> ⚠️ **Le texte généré se trouve dans `message.reasoning`, pas dans `message.content`.**
> En mode `thinking`, `content` vaut `null` et toute la réponse est dans `reasoning`.
> Le champ `reasoning_content` n'existe pas dans cette version de vLLM (0.22.1rc1) :
> le lire revient à conclure, à tort, que le modèle ne produit rien.
>
> ```json
> {"choices":[{"message":{"role":"assistant","content":null,
>   "reasoning":"Diffusion Models (Modèles de diffusion)..."}}]}
> ```

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
  - '0.70'
  - '--max-model-len'
  - '196608'
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
  - '{"canvas_length":256,"max_denoising_steps":8}'
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
| **`--gpu-memory-utilization`** | **`0.70`** | **Levier n°1.** À `0.85`, vLLM réserve 27,7 Gio sur 32,6 Gio sans tenir compte des ~5,5 Gio pris par Windows : 2 650 Mio de débordement en RAM partagée et le débit tombe à ~140 tok/s. À `0.70` : 136 Mio partagés, 1 361 tok/s. Fenêtre utile mesurée : `[0,65 ; 0,78]` |
| **`--diffusion-config`** | **`{"canvas_length":256,"max_denoising_steps":8}`** | **Levier n°2.** `canvas_length` : 410 tok/s en contexte long en 256 contre 121 tok/s en 512 (×3,4). `max_denoising_steps` : en dessous de 8, le texte se dégrade (42,6 % à 4 passes) et l'appel d'outil casse |
| `--canvas_length` | `256` | Réglage du modèle de diffusion, passé via `--diffusion-config`. `1024` fait échouer le démarrage |
| `--max-model-len` | `196608` | Fenêtre contextuelle. **N'a aucun effet sur le débit** : mesuré à 1 507–1 566 tok/s pour 131 072, 196 608 et 262 144. Il fixe la concurrence (2,81× / 2,04× / 1,58×) et la longueur acceptée |
| `--max-num-seqs` | `4` | Aucun gain mesuré : `mns=1/2/4/8` donnent 1 529–1 539 tok/s. Le modèle sérialise les requêtes |
| `--kv-cache-dtype` | `fp8` | **Obligatoire** : en bf16, servir une requête à 196 608 tokens exige 4,34 Gio de cache KV, il n'y en a que 4,28 → le moteur ne démarre pas |
| `--generation-config` | `vllm` | Le `generation_config.json` du checkpoint impose `max_new_tokens: 256` et écraserait la limite par requête |
| `--enable-chunked-prefill` | — | Découpe les longs préfills pour limiter le pic mémoire |
| `--enable-auto-tool-choice` | — | Active la détection automatique des appels d'outils |
| `--tool-call-parser` | `gemma4` | Protocole d'outils Gemma 4 attendu par le `chat_template.jinja` du checkpoint |
| `--reasoning-parser` | `gemma4` | Extrait le canal de raisonnement `<\|channel>thought` |
| `--chat-template` | `examples/tool_chat_template_gemma4.jinja` | Template de l'image, aligné sur le format d'outils du modèle |

### Ce que `--max-model-len` fait — et ne fait pas

Le paramètre est **respecté** : la valeur annoncée dans les logs
(`Maximum concurrency for N tokens per request`) correspond toujours à la valeur
demandée. Mais il **n'améliore pas le débit**, et ce n'est pas un défaut :

- le débit ne dépend que de la longueur **réellement transmise** dans la requête ;
- `max_model_len` alimente `CommonAttentionMetadata.max_seq_len`, qui sert à
  dimensionner les buffers de l'attention — pas à borner la requête ;
- ce qu'il change réellement : le nombre de requêtes concurrentes possibles à pleine
  fenêtre, et la longueur maximale acceptée (au-delà, la requête est refusée).

Un prompt de 65 k tokens coûte le même à `max_model_len=131072` qu'à `262144`.
C'est la longueur du prompt, pas la fenêtre autorisée, qui dégrade le débit.

### Options qui ne fonctionnent pas

Ces options paraissent plausibles mais n'ont **aucun effet** sur le débit ou la qualité.
Elles sont documentées ici pour éviter de les retester.

| Option | Pourquoi elle est inopérante |
|--------|-------------------------------|
| `--hf-overrides '{"diffusion_sampler":...,"diffusion_entropy_bound":...}'` | Ces clés n'existent dans aucun fichier du code vLLM. Le sampler lit le `generation_config.json` du checkpoint (`sampler_config.entropy_bound = 0.1`) |
| `--override-generation-config` | Alimente `get_diff_sampling_param()` (défauts de requête), pas `custom_sampler()` qui construit le `DiffusionSampler` |
| `--enable-prefix-caching` | **Déjà actif par défaut** dans vLLM V1 (`CacheConfig.enable_prefix_caching = True`). L'option est donc inutile ; le préfixe de contexte est bien réutilisé. Mesuré : 1 524 t/s avec, 1 525 t/s sans — écart nul |
| `--attention-backend TRITON_ATTN` | Aucun gain (1 516 t/s contre 1 525 t/s en `FLASH_ATTN`, le défaut) |
| `--attention-backend FLASHINFER` | Le moteur ne démarre pas : impose `cudagraph_mode=PIECEWISE` (`FULL_AND_PIECEWISE` non supporté avec FlashInfer) |
| `--attention-backend TORCH_SDPA` | `ValueError: Backend TORCH_SDPA must be registered before use` — absent de cette image vLLM |
| `temperature`, `seed`, `min_p`, `logit_bias` | Rejetés en HTTP 400 par l'API de diffusion |
| `canvas_length` 1024 | Le moteur ne démarre plus : à `max_model_len=262144` et `max_num_seqs=4`, vLLM annonce `GPU KV cache size: 862,724 tokens` et `Maximum concurrency for 262,144 tokens per request: 3.29x` — il manque 185 852 tokens de cache KV |

> **Attention — `--max-num-seqs 8` n'est plus un cas d'échec.** Il OOM à
> `gpu_memory_utilization=0.85`, mais fonctionne à `0.70` (mesuré : 1 529 t/s, 4,42 Gio
> de KV, 2 695 Mio de VRAM libre). Le débit reste identique : le modèle sérialise les
> requêtes de toute façon, la concurrence ne l'améliore pas.

## Contrainte importante : service mono-requête

Les modèles de diffusion sérialisent mal le batch. Mesuré sur la configuration
retenue :

| Charge | Débit client |
|---------|--------------|
| Requêtes **sérielles** (1 à la fois) | **1 361 – 1 435 tok/s** |
| 4 requêtes **simultanées** | 102 tok/s |

`--max-num-seqs 4` reste le plafond mémoire, mais **le débit n'est atteint qu'en
traitant les requêtes une par une**. Un agent IA qui parallélise ses appels d'outils
verra le débit s'effondrer d'un facteur 10. Mesuré : `max_num_seqs` de 1 à 8 ne change
rien au débit sériel (1 529–1 539 tok/s) — le moteur sérialise de lui-même.

## Vérification

```bash
./verify.sh
```

Le script attend la santé du conteneur, puis contrôle la configuration de production
contre le service en cours d'exécution : fenêtre contextuelle, budget mémoire GPU,
absence de débordement en RAM partagée, appel d'outil structuré, absence de `<pad>`
et **débit client**.

```
=== Verification DiffusionGemma / vLLM ===
  [OK]   service pret (/health = 200)                   code=200
  [OK]   fenetre contextuelle = 196608                  max_model_len=196608
  [OK]   gpu_memory_utilization = 0.70                  gpu_memory_utilization=0.7
  [OK]   pas de debordement GPU (< 500 Mio partages)    memoire partagee = 136.0 Mio
  [OK]   appel d'outil structure (get_weather)          occurrences=1
  [OK]   aucune anomalie <pad> (reponse outil)          <pad> x0
  [OK]   aucune anomalie <pad> (reponse texte)          <pad> x0
  ... charge continue pendant 40s (requetes seriees)
  [OK]   debit client > 400 tok/s                      debit client median = 1361.26 tok/s
  [INFO]  Committed token throughput (logs vLLM, dilue) = 1382.32 tok/s

=== 8 reussis, 0 echoues ===
```

Le compteur `Committed token throughput` des logs vLLM est affiché pour information
seulement : il est moyenné sur une fenêtre de 10 s et reste très inférieur au débit
client dès que le serveur n'est pas saturé en permanence.

## Benchmark et trace des configurations

`bench.py` redémarre le conteneur pour chaque configuration, attend la santé, puis
mesure :

- le débit client (tokens de complétion / seconde) ;
- le `Committed token throughput` publié dans les logs vLLM ;
- la présence de `<pad>` dans une réponse texte **et** dans une réponse avec outil ;
- la présence effective d'un `tool_calls` structuré.

```bash
python3 bench2.py '[
  {"id":"A1","image":"vllm/vllm-openai:gemma","model":"RedHatAI/diffusiongemma-26B-A4B-it-NVFP4",
   "gm":0.70,"mns":4,"mml":196608,"kvd":"fp8","tools":true,
   "diffusion_config":"{\"canvas_length\":256,\"max_denoising_steps\":8}",
   "sustain_s":40,"sustain_workers":1,"sustain_max_tokens":512}
]'
```

Clés reconnues : `model`, `image`, `gm` (gpu-memory-utilization), `mns`
(max-num-seqs), `mml` (max-model-len), `kvd` (kv-cache-dtype), `tp`, `ecp`,
`prefix_cache`, `attention_backend`, `diffusion_config`, `hf_overrides`,
`override_gen_config`, `tools`, `wsl_patch`, `load_*`, `gen_*`, `sustain_*`.

> `bench2.py` **réécrit `docker-compose.yml`** avec la configuration testée. Après un
> benchmark, restaurez le compose de production (ce README documente les valeurs
> retenues) ou relancez-le explicitement. La matrice complète des 38 configurations
> testées est dans `state/final_matrix.json`, les résultats dans `state/results/`, et
> la synthèse par `python3 state/summarize.py`.

Les résultats détaillés de chaque configuration testée sont consignés dans
[`JOURNAL.md`](JOURNAL.md) et [`trace_configs.md`](trace_configs.md).

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
  sous-estime le débit réel — sur la configuration de production, 37,6 tok/s dans les
  logs contre 1 521 tok/s côté client pour la même charge. Pour lire un débit
  représentatif, maintenir une charge continue pendant au moins une fenêtre
  (`VLLM_LOG_STATS_INTERVAL`, 10 s par défaut).
- **Première requête plus lente** : elle paie la compilation CUDA et l'autotuner
  (flashinfer, `trtllm::fused_moe`). Mesurer après une requête de chauffe, sinon le
  débit est sous-estimé d'un facteur ~2.

## Fichiers du dépôt

| Fichier | Rôle |
|---------|------|
| `docker-compose.yml` | Service vLLM : image, paramètres de diffusion, outils, healthcheck |
| `wsl_entrypoint.sh` | **Patch WSL2 obligatoire** : active `pin_memory` / UVA que vLLM désactive sous WSL. Sans lui, le moteur ne démarre pas (`RuntimeError: UVA is not available`) |
| `.env.template` | Variables d'environnement à copier en `.env` |
| `bench.py` | Harnais de test historique (mesure débit, `<pad>`, appels d'outils) |
| `bench2.py` | Harnais de test v2 : ajoute mémoire GPU Windows/WSL, TTFT, débit client, qualité du texte, test de scenario OpenCode |
| `state/quality_bench.py` | Banc de qualité : 8 critères objectifs (rappel long, format, JSON, outil, code, hallucination) |
| `state/recall_trial.py` | Caractérisation du rappel en contexte long, avec répétitions |
| `state/kv_probe.py` | Mesure de l'usage du cache KV pendant une génération longue |
| `verify.sh` | Vérification bout-en-bout de la configuration de production |
| `JOURNAL.md` | Journal d'investigation : causes, mesures, arbitrages |
| `trace_configs.md` | Trace des configurations testées (résultats chiffrés) |

`.env`, `state/` et `vllm_cache/` sont ignorés par Git (secrets, cache de compilation).

## Arrêt

```bash
docker compose down
```
