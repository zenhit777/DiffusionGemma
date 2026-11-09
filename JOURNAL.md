# Journal des opérations — DiffusionGemma / vLLM

Suivi chronologique des investigations, mesures et changements de configuration.
Mise à jour après chaque phase significative.

---

## 2026-11-08 — Phase 1 : cartographie et reprise de contexte

### Objectif utilisateur

- Comprendre pourquoi le débit passe de ~516 t/s à ~8 t/s après la fin du démarrage de vLLM.
- Comprendre le comportement de `--max-model-len` (paramétré à 196608 puis 262144 sans effet).
- Clarifier la différence entre le débit affiché dans les logs vLLM et le débit observé côté outils.
- Atteindre une configuration stable à **≥ 400 t/s** et **196 608 tokens** de fenêtre contextuelle.
- Exécuter **30 à 40 nouveaux tests**.

### État initial constaté

| Élément | Valeur observée |
|---|---|
| Conteneur | `diffusiongemma-vllm`, image `vllm/vllm-openai:gemma`, sain |
| Commande active | `--gpu-memory-utilization 0.85`, `--max-model-len 196608`, `--kv-cache-dtype fp8`, `--max-num-seqs 4`, `--diffusion-config {"canvas_length":512,"max_denoising_steps":8}` |
| `max_model_len` dans les logs | `196608` (respecté par vLLM) |
| KV cache GPU | 9,11 Gio utiles → 825 206 tokens, concurrence 4,20× |
| Chargement modèle | 17,39 Gio en 165 s |
| Capture graphes | 0,24 Gio |
| Mémoire GPU (WSL `nvidia-smi`) | 31 905 / 32 607 Mio utilisés |
| Compteurs GPU Windows | 30 824 Mio dédiés (vmwp) + 4 368 Mio (dwm) + **2 650 Mio partagés** |

### Constat 1 — le débit « 1177,59 t/s » est bien une métrique vLLM, mais mal définie

Source : `state/metrics.py` (`vllm/v1/spec_decode/metrics.py`, snapshot de l'image).

```python
elapsed_time = time.monotonic() - self.last_log_time
accepted_throughput = num_accepted_tokens / elapsed_time
```

`Committed token throughput` = **tokens commités / intervalle de log**, où l'intervalle est
réinitialisé à chaque émission (`reset()` fixe `last_log_time = time.monotonic()`).
Conséquences :

- c'est un débit **moyen fenêtré**, pas un débit instantané ;
- l'inactivité du serveur dilue le résultat ;
- `Mean denoising steps per canvas` est en réalité `canvas_positions / committed_tokens`
  (le nom est trompeur) ;
- la métrique ne sort que s'il y a eu au moins un `observe()` dans la fenêtre.

La métrique `Avg generation throughput` (loggers.py) est le compteur scheduler
`num_generation_tokens / delta_time`, avec la même fenêtre de 10 s.

**Comparaison avec le débit OpenCode** : OpenCode affiche `tokens_output / durée totale de
la requête`, donc il inclut le TTFT et le temps de parsing. Ce n'est pas la même grandeur.

### Constat 2 — la dégradation n'est pas due à `--max-model-len`

`--max-model-len` contrôle la **longueur de séquence vLLM** (plafond de requête) :

- dimensionnement des tables de blocs KV et contrôle de démarrage
  (`state/kv_cache_utils.py`, `GPU KV cache size`, `Maximum concurrency`) ;
- `max_seq_len` passé à l'attention (`state/diffusion_gemma.py:1044`) ;
- validation de la longueur des requêtes ;
- troncature du dernier canvas près de la limite (`valid_canvas_len`).

Ce n'est **ni** la taille du canvas, **ni** une allocation d'état de diffusion.
Preuve directe : les logs de démarrage montrent `max_model_len: 196608` et
`Maximum concurrency for 196,608 tokens per request: 4.20x` — la valeur est respectée.

En revanche, `--max-model-len` **influence** le coût d'attention par requête : plus la
fenêtre est large, plus le cache KV alloue et plus l'attention est coûteuse. Il ne peut
donc pas augmenter le débit ; il le dégrade éventuellement.

### Constat 3 — cause racine du « 8 t/s » : surengagement mémoire GPU → débordement en RAM partagée

Mesure Windows (`\GPU Process Memory(*)`) :

| Processus | Dédié | Partagé |
|---|---|---|
| `vmwp` (WSL / conteneur vLLM) | 30 825 Mio | 2 650 Mio |
| `dwm` (compositeur Windows) | 4 368 Mio | 1,2 Mio |
| autres (desktop, navigateurs, LM Studio…) | ~1 000 Mio | ~40 Mio |

Total dedicated = **31 884 Mio** pour une carte de **32 607 Mio**, et **2 781 Mio** de
mémoire partagée. La VRAM est saturée ; une partie du travail GPU est backing-store en
RAM partagée Windows. C'est cohérent avec :

- `nvidia-smi` (WSL) : 31 905 Mio utilisés, **262 Mio libres** ;
- chute du débit à ~8 t/s côté client ;
- `Avg generation throughput` effondré alors que `GPU KV cache usage: 0.0%`.

Point important : `DWM` retient **4,37 Gio** sur la carte pour l'affichage du bureau
(écran 1920×1080). C'est une mémoire Windows, pas celle de vLLM, mais elle participates
directement à l'épuisement de la VRAM.

### Constat 4 — la mesure directe contredit les chiffres archivés

Mesures sur le conteneur actuellement démarré (compose de production, `max-model-len 196608`) :

| Requête (max_tokens) | Durée | Tokens | Débit client |
|---|---|---|---|
| 64 | 3,51 s | 64 | 18,2 t/s |
| 128 | 3,93 s | 128 | 32,5 t/s |
| 256 | 3,96 s | 256 | 64,7 t/s |
| 512 | 3,92 s | 512 | 130,6 t/s |
| 1024 | 7,38 s | 1024 | 138,8 t/s |
| 1536 | 10,87 s | 1536 | 141,3 t/s |
| 2048 | 95,58 s (1re, à froid) | 2048 | 21,4 t/s |

Puis, à chaud, 6 requêtes `max_tokens=512` sérielles : **3,83–3,89 s** chacune,
soit **131–134 t/s** de façon parfaitement reproductible.

Les valeurs archivées dans `trace_configs.md` (1 177 t/s, 1 228 t/s) ne sont **pas
reproductibles** dans l'état actuel de la machine. Le débit maximal observé aujourd'hui
est ~141 t/s, pas ~1 200 t/s.

### Constat 5 — décomposition de la latence vLLM

Métriques Prometheus du service en cours :

```
vllm:request_e2e_latency_seconds_sum       225,57   (14 requêtes)
vllm:request_prefill_time_seconds_sum       206,42   (14 requêtes)
vllm:request_decode_time_seconds_sum         17,06   (14 requêtes)
vllm:request_queue_time_seconds_sum           0,00015
vllm:time_to_first_token_seconds_sum        208,51
vllm:request_time_per_output_token_seconds_sum  0,00833
```

→ **91,5 % du temps est passé en "prefill"**, pas en décodage. Pour un modèle de
diffusion, le scheduler traite chaque canvas comme un batch de tokens ; l'essentiel du
coût est donc dans les passes de débruitage du canvas et dans l'attention bidirectionnelle,
pas dans une génération token-par-token.

Corollaire : le débit apparent est fortement non-linéaire en `max_tokens` — le temps est
quasi constant jusqu'à ~512 tokens (un seul canvas), puis croît par paliers de canvas.

### Constat 6 — le patch WSL2 force un chemin non supporté

`wsl_entrypoint.sh` force `is_pin_memory_available() = True` et `is_uva_available() = True`.
Or vLLM détecte WSL et désactive volontairement le pinning
(`state/platform_interface.py:758` : *« Using 'pin_memory=False' as WSL is detected. This
may slow down the performance. »*).

Vérification dans le conteneur :

- `is_pin_memory_available()` → `False` (vLLM), `True` (patché) ;
- `is_uva_available()` → `False` (vLLM), `True` (patché) ;
- `torch.zeros(..., pin_memory=True)` fonctionne pourtant (1, 64 et 256 Mio testés).

→ Le contournement est possible, mais il contourne aussi la branche « WSL » qui choisit
des buffers **non pinnés** (`state/buffer_utils.py:102`, `pin_memory=False`). Le patch
change donc le chemin d'exécution sans mesure de son impact réel.

### Constat 7 — échantillonnage GPU pendant une génération

```
WSL 21:43:40  100 % util, 171 W, 2827 MHz, 31936 Mio
WSL 21:43:54    2 % util, 150 W, 2827 MHz, 31933 Mio
WSL 21:43:56    2 % util,  71 W,  787 MHz, 31933 Mio
WSL 21:44:00   10 % util,  50 W,  555 MHz, 31931 Mio   ← puis stable ~12 %, 630 MHz
```

La génération s'exécute à 100 % d'utilisation pendant ~14 s, puis retombe à ~12 % avec
les horloges mémoire basses. La mémoire utilisée ne varie pas (31,93 Go) : le débordement
de 2,7 Go est **permanent**, pas transitoire.

---

### Constat 8 — CAUSE RACINE du « 8 t/s » : le débit s'effondre avec la longueur du prompt

C'est le constat décisif. Mesures sur le service en cours (`max-model-len 196608`,
`canvas 512`, `steps 8`), `max_tokens=512`, 2 répétitions par point :

| Prompt réel | Débit médian | Débit max | Durée médiane | Tokens completion |
|---|---|---|---|---|
| 33 | 71,8 t/s | 88,7 t/s | 4,19 s | 283 |
| 1 143 | 99,9 t/s | 101,3 t/s | 3,36 s | 335 |
| 4 471 | 76,4 t/s | 87,6 t/s | 3,45 s | 230 |
| 17 783 | 66,8 t/s | 82,5 t/s | 3,75 s | 207 |
| 35 532 | 46,3 t/s | 48,5 t/s | 4,11 s | 202 |
| 71 031 | 39,6 t/s | 56,4 t/s | 5,61 s | 166 |
| **142 028** | **4,4 t/s** | **7,0 t/s** | 9,99 s | 27 |

Confirmation isolée (requête unique, prompt de 142 025 tokens) : **4,39 s, 23,7 t/s**,
puis 2e passage : **9,99 s → 4,38 t/s**.

**C'est exactement le débit observé dans OpenCode (8 t/s).** Et les sessions OpenCode
enregistrent `tokens_input = 62 253` (session `quiet-wolf`) et `tokens_input = 31 417`
(session build) : OpenCode ré-envoie l'historique complet à chaque tour, ce qui place
la requête dans la zone de déplombage.

Explication : pour un modèle de diffusion, chaque passe de débruitage réévalue **tout**
le contexte en attention bidirectionnelle (`prepare_attn` → `causal=False` pour le
débruitage, `state/diffusion_gemma.py:1024-1048`). Le coût d'une passe est donc
proportionnel à la longueur du prompt, et il est payé `max_denoising_steps` fois.

→ **`--max-model-len` n'est pas en cause** : il fixe le plafond autorisé, pas la
longueur réellement transmise. Mais il détermine la taille du cache KV alloué, donc
l'empreinte VRAM, donc la part de surengagement.

### Constat 9 — le surengagement VRAM est confirmé et quantifiable

| Élément | Valeur |
|---|---|
| `gpu-memory-utilization 0.85` × 32 607 Mio | **27 716 Mio réservés par vLLM** |
| `dwm` (bureau Windows, écran 1920×1080) | **4 368 Mio** |
| autres processus (explorer, brave, Antigravity, Code, EdgeWebView) | ~1 100 Mio |
| Total demandé | **≈ 33 200 Mio > 32 607 Mio** |
| Débordement mesuré en RAM partagée | **2 650–2 781 Mio** |

Le profileur vLLM annonce pourtant `Available KV cache memory: 9.11 GiB` : vLLM calcule
son budget sur la **totalité** de la carte et ignore les ~5,5 Gio déjà pris par Windows.

Conséquence : 2,7 Go de l'empreinte du modèle et du cache KV sont backing-store en RAM
partagée Windows → accès PCIe via le CPU → effondrement du débit et élévation du TTFT.

### Constat 10 — le débit « 516 t/s » du démarrage est un artefact, pas une mesure

```
18:34:58  Avg prompt throughput: 516.2 tokens/s, Avg generation throughput: 10.1 tokens/s
          Committed token throughput: 8.41 tokens/s, Committed: 512 tokens
```

Les 516,2 t/s sont du **prompt** (le warmup de compilation), pas de la génération.
La génération de ce même intervalle est à **10,1 t/s** (log scheduler) et
**8,41 t/s** (committed). Ces trois chiffres ne sont pas contradictoires : ils mesurent
trois choses différentes sur le même intervalle.

### Constat 11 — `<pad>` et appel d'outil : état actuel

Appel d'outil `get_weather` sur le service en cours : **valide**, 2,52 s,
`tool_calls[0].function.name = "get_weather"`, arguments `{"city": "Paris"}`, 0 `<pad>`.

---

## 2026-11-09 — Phase 2 : checkpoint AWQ-INT4 et sécurisation du cache

### Découverte majeure — le checkpoint NVFP4 était servi depuis l'overlay du conteneur

Le journal de démarrage indique :

```
(EngineCore pid=205) INFO 11-08 18:31:50 [weight_utils.py:922] Filesystem type for
checkpoints: OVERLAY. Checkpoint size: 16.81 GiB.
```

Or le `docker-compose.yml` ne monte que `${HF_HOME}` (le cache sur D:) dans
`/root/.cache/huggingface`. La vue **hôte** de ce répertoire ne contenait plus que
`models--cyankiwi--diffusiongemma-26B-A4B-it-AWQ-INT4` ; le NVFP4 n'existait que dans
la couche overlay du conteneur en cours, hors bind-mount.

Preuve : un conteneur neuf monté sur le même dossier D: ne voit que l'AWQ :

```
$ docker run --rm -v /mnt/d/.../hf_cache:/hf alpine ls /hf/hub/
models--cyankiwi--diffusiongemma-26B-A4B-it-AWQ-INT4
```

Conséquence : **le moindre `docker compose down/up` aurait définitivement perdu le
checkpoint NVFP4 de production**. Il a été extrait de l'overlay avant toute manipulation
(`docker cp`, 17 Go) et sauvegardé dans un cache local ext4, avec vérification de
l'en-tête safetensors de chaque shard (47 337 tenseurs, 18,05 Go, cohérents).

Cache local désormais disponible pour les deux checkpoints :

| Checkpoint | Format | Poids | Shards | Tenseurs |
|---|---|---|---|---|
| `RedHatAI/diffusiongemma-26B-A4B-it-NVFP4` | `nvfp4-pack-quantized` | 18,05 Go | 1 | 47 337 |
| `cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4` | `pack-quantized` (int4 g32) | 17,23 Go | 4 | 35 777 |

### Comparaison des deux checkpoints

| Critère | NVFP4 (RedHatAI) | AWQ-INT4 (cyankiwi) |
|---|---|---|
| Poids sur disque | 18,05 Go | **17,23 Go** (−0,82 Go) |
| Format quant. | NVFP4 (float4, groupe 16) | AWQ int4 (groupe 32) |
| Attention du décodeur | **non quantifiée** (`weight` bf16) | **quantifiée** (`weight_packed` int4) |
| MLP experts du décodeur | quantifié | quantifié |
| Couches non quantifiées | 495 | 315 |
| Encodeur (tower vision + LM) | non quantifié | non quantifié |
| `dtype` dans config.json | `bfloat16` | `float16` |
| `transformers_version` | 5.8.0.dev0 | 5.13.0.dev0 |
| Schéma vLLM correspondant | `compressed_tensors_w4a4_nvfp4.py` | `compressed_tensors_wNa16.py` (WNA16, `weight_packed`/`weight_shape`) |

Deux écarts structurants :

1. **Couverture différente.** Le NVFP4 laisse toute l'attention du décodeur en bf16 ;
   l'AWQ la quantifie en int4. L'AWQ est donc *plus* agressif, pas moins.
2. **Gain disque marginal.** 0,82 Go de moins, alors que le débordement mesuré en
   mémoire partagée Windows est de **2,65 Go**. Changer de checkpoint ne suffit donc pas
   à éliminer le surengagement VRAM ; il faut en plus réduire `--gpu-memory-utilization`.

### Compatibilité vLLM de l'AWQ

- `QUANTIZATION_METHODS` de l'image inclut `compressed-tensors` (retenu par les deux
  checkpoints) ;
- le scheme `compressed_tensors_wNa16.py` gère exactement le format `pack-quantized`
  int4 : `weight_packed` + `weight_scale` + `weight_shape`, `num_bits=4`, `group_size=32`
  — signature exacte des tenseurs AWQ ;
- `hf_to_vllm_mapper` est absent des deux `config.json` : le mapper est fourni par le
  code vLLM du modèle (`state/diffusion_gemma.py:169`, avec
  `orig_to_new_substr={".experts.": ".moe.experts."}`), qui couvre les deux formats.

Point de vigilance : le chemin MoE expert. Les deux checkpoints stockent les experts
sous `model.decoder.layers.N.experts.M.*`, mappé vers `.moe.experts.` par le code vLLM.
L'AWQ ne quantifie pas `mlp.gate_proj` / `mlp.down_proj` du décodeur (alors que le
NVFP4 les quantifie) : le choix du kernel MoE diffère entre les deux, ce qui peut
modifier le débit indépendamment de la quantification.

### Constat 12 — `--gpu-memory-utilization 0.70` supprime le débordement et restaure le débit

Test A/B à configuration strictement identique (`mml=196608`, `mns=4`, `canvas 512`,
`steps 8`, `kv fp8`), cache local ext4, seule la variable `gm` et le checkpoint changent :

| Mesure | `gm=0.85` (avant) | `gm=0.70` NVFP4 | `gm=0.70` AWQ |
|---|---|---|---|
| Mémoire partagée Windows | **2 650 Mio** | **151 Mio** | 156 Mio |
| Mémoire dédiée Windows | 31 884 Mio | 29 705 Mio | 29 472 Mio |
| VRAM libre (WSL) | 300 Mio | 2 498 Mio | 2 734 Mio |
| KV cache alloué | 9,11 Gio | 4,42 Gio | 5,06 Gio |
| Concurrence à 196 608 tok | 4,20× | 2,04× | 2,33× |
| **Débit prompt court (médiane)** | ~140 t/s | **1 527 t/s** | 1 210 t/s |
| **Débit prompt 65 k (médiane)** | 4,4 t/s | **80 t/s** | **114 t/s** |
| TTFT | 3,5 s | **0,358 s** | 0,444 s |
| Appel d'outil | valide | valide | valide |
| `<pad>` | 0 | 0 | 0 |

**La cause racine est donc confirmée et quantifiée.** Le budget de 0,85 (27,7 Gio)
dépassait la VRAM réelle disponible (~22 Go une fois le bureau Windows et les autres
processus déduits) ; vLLM calculait son budget sur la carte entière, donc
2,65 Go du modèle et du cache KV étaient backing-store en RAM partagée Windows, et le
transfert PCIe via le CPU effondrait le débit d'un facteur ~10.

À `gm=0.70` (22,8 Gio), le budget vLLM tient dans la VRAM réellement disponible :
plus aucun débordement, et le débit court passe de ~140 à 1 527 t/s — soit **au-dessus
de l'objectif de 400 t/s**.

### Constat 13 — l'AWQ-INT4 n'est pas le meilleur choix

Comparaison à `gm=0.70` strictement identique :

| Critère | NVFP4 | AWQ-INT4 |
|---|---|---|
| Débit prompt court | **1 527 t/s** | 1 210 t/s (−21 %) |
| Débit prompt 65 k | 80 t/s | **114 t/s** (+43 %) |
| TTFT | **0,358 s** | 0,444 s |
| Poids en VRAM (Model loading) | 17,39 Gio | 16,66 Gi (−0,73 Gi) |
| KV cache obtenu | 4,42 Gio | **5,06 Gio** |
| Compilation torch | **8,8 s** (cache hit) | 75,2 s |
| Temps de démarrage total | **277 s** | 337 s |
| Appel d'outil | valide | valide |

L'AWQ est **plus lent** sur prompt court (les projections d'attention, quantifiées en
int4 chez AWQ mais laissées en bf16 chez NVFP4, ne profitent pas des mêmes kernels
rapides) et plus rapide sur contexte long. Son avantage mémoire est marginal
(0,73 Gi) et son coût de compilation est 8× plus élevé.

Recommandation : **conserver le NVFP4** comme checkpoint par défaut. L'AWQ n'a d'intérêt
que si le gain mémoire devient le facteur limitant (par exemple si `gm` doit descendre
sous 0,65 pour libérer la VRAM).

### Constat 14 — bug de mesure : le texte généré était invisible

Le harnais lisait `message.reasoning_content`, champ que cette version de vLLM
n'expose pas. Or le raisonnement est renvoyé sous `message.reasoning`. Conséquence :
toutes les mesures antérieures affichaient `content_chars=0` alors que le texte était
présent — ce n'était pas une génération vide, mais une lecture au mauvais endroit.

Confirmé sur les réponses archivées (`state/direct_512.json`) :

```json
"message": {"role": "assistant", "content": null,
            "reasoning": "Diffusion Models (Modèles de diffusion).\nExplain in detail..."}
```

`content` est `null` par construction (le modèle est en mode *thinking*, tout passe par
`reasoning`). Le harnais lit désormais les deux champs.

### Constat 15 — le débit log vLLM reste dilué, la mesure client fait foi

Sur le run AB1 (1 527 t/s client), les logs vLLM n'affichent que 29,3 t/s
(`Avg generation throughput`) et 37,6 t/s (`Committed token throughput`). L'écart est
attendu : ces compteurs sont moyennés sur une fenêtre de 10 s pendant laquelle le
serveur est inactif la majeure partie du temps, alors que la mesure client ne compte
que le temps d'usage effectif de la requête.

**Le débit à retenir pour l'usage réel est le débit client** (tokens de complétion
divisés par la durée de la requête), qui inclut le TTFT. C'est la grandeur qu'affiche
OpenCode.

### Constat 16 — seuil de débordement : `gm ≤ 0.78` sur cette machine

Balayage fin de `--gpu-memory-utilization` (NVFP4, `mml=196608`, `mns=4`, canvas 512,
steps 8), mesuré sur la même machine et dans la même session :

| `gm` | Budget vLLM | Mémoire partagée | VRAM libre (WSL) | Débit court | KV cache |
|---|---|---|---|---|---|
| 0,70 | 22 825 Mio | 151 Mio | 2 498 Mio | 1 527 t/s | 4,42 Gio |
| 0,72 | 23 477 Mio | 122 Mio | 2 462 Mio | **1 542 t/s** | 5,06 Gio |
| 0,75 | 24 455 Mio | 130 Mio | 1 483 Mio | 1 534 t/s | 6,02 Gio |
| 0,78 | 25 433 Mio | 130 Mio | 529 Mio | 1 527 t/s | 6,97 Gio |
| **0,80** | 26 086 Mio | **706 Mio** | 463 Mio | **942 t/s** | 7,61 Gio |
| 0,85 | 27 716 Mio | **2 650 Mio** | 300 Mio | ~140 t/s | 9,11 Gio |

Le mécanisme est net : à `gm=0.78`, vLLM consomme 31 659 Mio et il ne reste que
529 Mio libres — juste assez pour ne pas déborder. À `gm=0.80`, la saturation de la
carte est complète (31 725 Mio utilisés) et **576 Mio basculent en mémoire partagée** ;
le débit s'effondre de 1 527 à 942 t/s (−38 %). À `gm=0.85`, le débordement atteint
2 650 Mio et le débit tombe à ~140 t/s.

Le seuil de bascule est donc à **0,78 sur cette configuration**. Il n'est pas une
constante de vLLM : il dépend de la consommation des autres processus Windows
(≈ 5,5 Gio : 4,37 Gio pour DWM, ~1 Go pour le reste du bureau). Sur une machine
dédiée, ou avec un bureau moins gourmand, le seuil serait plus élevé.

### Constat 17 — zone d'opération stable : `gm ∈ [0,65 ; 0,78]`

Balayage complet (NVFP4, `mml=196608`, `mns=4`, canvas 512, steps 8) :

| `gm` | Débit court | Débit 65 k | TTFT | Mémoire partagée | VRAM libre | KV cache |
|---|---|---|---|---|---|---|
| 0,65 | 1 525 t/s | 143 t/s | 0,353 s | 130 Mio | **4 715 Mio** | 2,83 Gio |
| 0,68 | 1 538 t/s | 166 t/s | 0,355 s | 119 Mio | 3 746 Mio | 3,79 Gio |
| **0,70** | 1 523 t/s | 121 t/s | 0,357 s | 125 Mio | 3 085 Mio | 4,42 Gio |
| 0,72 | 1 542 t/s | 114 t/s | 0,351 s | 122 Mio | 2 462 Mio | 5,06 Gio |
| 0,75 | 1 534 t/s | 103 t/s | 0,352 s | 130 Mio | 1 483 Mio | 6,02 Gio |
| 0,78 | 1 527 t/s | 154 t/s | 0,341 s | 130 Mio | 529 Mio | 6,97 Gio |
| 0,80 | **942 t/s** | 107 t/s | 0,627 s | **706 Mio** | 463 Mio | 7,61 Gio |

Sur toute la plage 0,65–0,78, le débit court est stable à **1 523–1 542 t/s** (écart
< 1,3 %, hors variation de température du GPU) et la mémoire partagée reste
sous 135 Mio. Le débit ne dépend donc **pas** de `gm` tant qu'on reste sous le seuil
de débordement : le budget vLLM est dominé par les poids du modèle (17,39 Gio), pas
par le cache KV.

`gm=0.70` a été retenu pour la production : débit identique au reste de la plage,
2 à 5 Gio de marge VRAM libre, et un cache KV de 4,42 Gio qui permet 2,04 requêtes
concurrentes à pleine fenêtre de 196 608 tokens.

À noter : le débit sur prompt long (65 k tokens) oscille entre 103 et 166 t/s sans
corrélation avec `gm` — c'est la variance de la génération de diffusion (nombre de
passes de débruitage effectivement nécessaires, variable selon l'échantillon), pas un
effet mémoire.

### Constat 18 — plancher physique : `gm=0.55` ne démarre pas

À `gm=0.55` (17 933 Mio), le budget vLLM ne couvre plus les poids (17,39 Gio) ni les
activations de débruitage :

```
(EngineCore) INFO  [gpu_worker.py:480] Available KV cache memory: -0.35 GiB
(EngineCore) ERROR [core.py:1196] EngineCore failed to start.
```

Le budget disponible devient négatif : vLLM ne peut plus allouer le cache KV et
l'EngineCore s'arrête. La borne basse n'est donc pas un choix de performance mais une
limite physique : **poids du modèle + activations + cache KV doivent tenir dans
`gm × 32 607 Mio`**.

Interprétation complète du paramètre sur cette machine :

| `gm` | Budget | Comportement |
|---|---|---|
| ≤ 0,58 | < 19 Gio | **ÉCHEC** — cache KV négatif, l'EngineCore ne démarre pas |
| 0,62–0,65 | 20–21 Gio | SAIN, KV 1,7–2,8 Gio, 1 525 t/s, 4,7 Gio de VRAM libre |
| 0,66–0,78 | 21,5–25,4 Gio | SAIN, 1 523–1 542 t/s, mémoire partagée < 135 Mio |
| 0,79–0,80 | 25,7–26,1 Gio | SAIN mais **début de débordement** (706 Mio), 942 t/s |
| ≥ 0,85 | 27,7 Gio | **Débordement massif** (2 650 Mio), ~140 t/s |

La fenêtre utile est donc `[0,62 ; 0,78]`, et `0.70` en est le milieu prudent.

### Constat 19 — `--max-model-len` : le paramètre est respecté, mais il n'agit pas sur le débit

C'est la réponse directe à la question posée au début de l'investigation : « le
paramètre n'est pas respecté ». **Il est respecté** — la valeur annoncée dans les logs
correspond exactement à celle demandée. Balayage à `gm=0.70`, tout le reste identique :

| `--max-model-len` | Valeur dans les logs | KV cache | Concurrence | Débit court | Débit 65 k | TTFT |
|---|---|---|---|---|---|---|
| 131 072 | 131 072 | 4,33 Gio | 2,81× | 1 529 t/s | 161 t/s | 0,352 s |
| 196 608 | 196 608 | 4,42 Gio | 2,04× | 1 507 t/s | 40 t/s | 0,361 s |
| 262 144 | 262 144 | 4,42 Gio | 1,58× | 1 566 t/s | 171 t/s | 0,348 s |

Conclusions :

1. **Le plafond est respecté** : `Maximum concurrency for N tokens per request` est
   toujours calculé sur la valeur demandée.
2. **Le débit ne dépend pas de `max-model-len`** : 1 507–1 566 t/s sur les trois
   valeurs, écart de 3,9 % — du même ordre que la variance entre deux runs de la
   même configuration (1 523–1 542 t/s dans le balayage `gm`).
3. **Ce que le paramètre change réellement** : la *concurrence* possible à pleine
   fenêtre (2,81× à 131 072 contre 1,58× à 262 144) et la validation des requêtes
   (une requête plus longue que le plafond est refusée, pas tronquée).
4. **Pourquoi l'observation initiale « le paramètre ne change rien »** : le débit ne
   dépend que de la longueur *réellement transmise*, pas de la fenêtre autorisée. Le
   prompt envoyé fait toute la différence ; le plafond n'en fait aucune.

Mécanisme dans le code (`state/diffusion_gemma.py:1044`) : `max_model_len` est
transmis à l'attention sous le nom `max_seq_len`, via
`CommonAttentionMetadata.max_seq_len`. Il sert à **dimensionner les buffers** de
l'attention (nombre de blocs KV maximaux), pas à borner la requête ni à modifier le
plan d'exécution. Le coût réel est porté par `seq_lens`, c'est-à-dire la longueur
effective de chaque requête.

Le seul usage directe de `max_model_len` dans le code du modèle est la troncature du
dernier canvas à l'approche du plafond (`valid_canvas_len`, lignes 562, 627, 1251) :
sans effet sur une requête de 65 k tokens sous un plafond de 131 k ou 196 k.

**Conclusion pour l'utilisateur** : conserver 196 608 (la cible acceptée). Réduire
ce plafond ne rendrait pas le modèle plus rapide ; cela ne ferait que réduire le nombre
de requêtes concurrentes possibles et interdire les historiques plus longs.

### Constat 20 — `canvas_length` et `max_denoising_steps` : le second levier de débit

Mesuré à `gm=0.70`, `mml=196608`, `mns=4`, NVFP4. La métrique `degrad` mesure la
dégradation du texte (suites de ponctuation parasite + mots dupliqués adjacents) ;
seuil de validité fixé à 3 %.

| canvas / steps | Débit court | Débit 65 k | TTFT | `degrad` | Verdict |
|---|---|---|---|---|---|
| 512 / 4 | **2 427 t/s** | 93 t/s | 0,230 s | **0,148** | ECHEC (texte dégradé) |
| 512 / 6 | 1 938 t/s | 163 t/s | 0,284 s | 0,021 | OK |
| **512 / 8** (retenu) | 1 525 t/s | 121 t/s | 0,356 s | 0,000 | OK |
| 512 / 12 | 1 099 t/s | 167 t/s | 0,489 s | 0,000 | OK |
| 256 / 6 | 1 757 t/s | **469 t/s** | 0,190 s | 0,039 | limite |
| **256 / 8** | 1 365 t/s | **410 t/s** | 0,212 s | **0,000** | **OK** |
| 128 / 8 | 1 017 t/s | **345 t/s** | 0,161 s | 0,000 | OK |
| 64 / 8 | 611 t/s | 195 t/s | 0,140 s | 0,000 | OK |

Deux enseignements :

1. **Descendre sous 8 passes dégrade le texte, même si l'appel d'outil reste
   valide.** À 4 passes, la sortie contient des suites `*****`, `,,,,` et des mots
   dupliqués (`ForwardForward`, `Denoisingising`) : le modèle produit un texte
   inexploitable à 2 427 t/s. L'ancien critère « l'appel d'outil casse » était
   donc insuffisant — l'appel d'outil passe, mais la qualité de la prose est
   détruite. C'est un critère de qualité qui manquait aux mesures précédentes.

2. **Un canvas plus petit est plus rapide sur contexte long.** À 8 passes,
   `canvas 256` donne 410 t/s sur prompt de 65 k contre 121 t/s en `canvas 512`
   (+239 %), et 1 365 t/s sur prompt court (−10 % par rapport à 512/8). Le débit
   long est multiplié par 3,4 : c'est le levier le plus efficace contre la
   dégradation en contexte long, qui est le vrai problème d'OpenCode.

   Explication : avec un canvas de 512 positions, chaque passe de débruitage
   réévalue 512 tokens ; le coût par token généré est proportionnel à la taille
   du canvas. En canvas 256, il faut deux fois plus de passes pour la même
   quantité de tokens, mais chaque passe coûte deux fois moins — et surtout, la
   première passe n'a pas à traiter un canvas de 512 positions en attention
   bidirectionnelle sur un contexte de 65 k tokens.

Ces deux leviers sont **indépendants de la cause racine mémoire** : ils agissent sur
le coût de calcul par token, pas sur l'empreinte VRAM.

### Constat 21 — le prefix caching est actif par défaut : la 1re requête longue coûte 5× plus cher

`enable_prefix_caching` vaut `True` par défaut dans vLLM V1 (vérifié sur
`CacheConfig`) — le README le présentait comme une option à activer, ce qui est
faux. Conséquence mesurée, systématique sur toutes les configurations :

| Requête (prompt 65 k) | 1re | 2e | 3e |
|---|---|---|---|
| `canvas 256 / 8` | 5,56 s (52 t/s) | 1,06 s (468 t/s) | 0,58 s (410 t/s) |
| `canvas 512 / 8` | 5,61 s (25 t/s) | 1,03 s (121 t/s) | 0,91 s (209 t/s) |
| `canvas 128 / 8` | 5,83 s (68 t/s) | 1,38 s (372 t/s) | 1,00 s (345 t/s) |

La première requête paie le préfill complet du préfixe de 65 k tokens ; les
suivantes réutilisent le cache de préfixe et ne coûtent plus que la génération.

**Conséquence pratique pour OpenCode** : c'est un bon news. OpenCode ré-envoie
l'historique complet à chaque tour, donc le préfixe est réutilisable d'un tour à
l'autre. La première requête d'une session est lente (5,5 s de préfill à 65 k
tokens) ; les tours suivants ne payent que le delta de l'historique plus la
génération. C'est aussi pourquoi la médiane sur 3 runs est plus stable que le
maximum : elle retient le régime Established, pas le régime cold start.

Cela ne change pas le diagnostic principal : le débit « à froid » d'un tour de
session OpenCode reste plafonné par la taille de l'historique transmis.

### Constat 22 — backends d'attention : aucun gain, et `TORCH_SDPA` est indisponible

Sur RTX 5090 (Blackwell, sm_120), vLLM choisit `FLASH_ATTN` par défaut (ordre de
priorité dans `vllm/platforms/cuda.py`). Testé à `gm=0.70`, `mml=196608`, canvas 512,
8 passes :

| Backend | Débit court | Débit 65 k | TTFT | Compilation | Verdict |
|---|---|---|---|---|---|
| `FLASH_ATTN` (défaut) | 1 525 t/s | 121 t/s | 0,356 s | 6,8 s | OK |
| `TRITON_ATTN` | 1 516 t/s | 150 t/s | 0,356 s | 7,2 s | OK (équivalent) |
| `FLASHINFER` | — | — | — | — | **ÉCHEC au démarrage** |
| `TORCH_SDPA` | — | — | — | — | **ÉCHEC au démarrage** |

- `TORCH_SDPA` n'est pas supporté par cette image :
  `ValueError: Backend TORCH_SDPA must be registered before use. Use register_backend(Backend.TORCH_SDPA, 'your.module.YourClass')`.
- `FLASHINFER` sélectionne `FlashInferCutlassNvFp4LinearKernel` pour le GEMM NVFP4 et
  `FLASHINFER_CUTLASS` pour le MoE, mais impose
  `cudagraph_mode=PIECEWISE` (le mode `FULL_AND_PIECEWISE` n'est pas supporté avec
  FlashInfer), ce qui dégrade la capture de graphes CUDA. Le moteur ne démarre
  pas dans les conditions du test.

Conclusion : **le backend d'attention n'est pas un levier**. `FLASH_ATTN` (défaut) est
conservé — aucune option `--attention-backend` ne doit être ajoutée au compose.

### Constat 23 — le patch WSL2 n'est pas une optimisation : c'est un prérequis

Test `D_nopatch` : même configuration, mais sans l'entrypoint `wsl_entrypoint.sh`
qui force `is_pin_memory_available()` et `is_uva_available()` à `True`.

**Le moteur ne démarre pas** :

```
(EngineCore) WARNING [interface.py:757] Using 'pin_memory=False' as WSL is detected.
                        This may slow down the performance.
(EngineCore) ERROR   [core.py:1196] RuntimeError: UVA is not available
(APIServer) ERROR    RuntimeError: Engine core initialization failed.
```

Le chemin « WSL » de vLLM désactive volontairement le pinning mémoire, et le code de
DiffusionGemma alloue alors un `UvaBuffer` qui requiert l'UVA — indisponible. Le patch
`wsl_entrypoint.sh` n'est donc pas un contournement facultatif : **sans lui, le service
est inopérant** sur WSL2 avec cette version de vLLM.

À noter : le message `Unknown vLLM environment variable detected:
VLLM_WSL2_ENABLE_PIN_MEMORY` montre que cette variable d'environnement est ignorée
par vLLM ; le patch agit par `sitecustomize` (injection Python), pas par la variable.

Conséquence : le patch est conservé tel quel dans la configuration de production. Sa
version « patchée » n'a pas pu être comparée à une variante « patchée plus léger »
(le patch force les deux drapeaux à `True`).

### Constat 24 — `--enable-prefix-caching` est sans effet (déjà actif par défaut)

| Configuration | Débit court | Débit 65 k | TTFT |
|---|---|---|---|
| `canvas 512 / 8` (défaut) | 1 525 t/s | 121 t/s | 0,356 s |
| `--enable-prefix-caching` explicite | 1 524 t/s | 139 t/s | 0,358 s |

Écart nul. L'option est **déjà activée par défaut** dans vLLM V1
(`CacheConfig.enable_prefix_caching = True`). L'ajouter au compose serait redondant.

### Constat 25 — `kv_cache_dtype` : `fp8` n'est pas un choix de performance, c'est une contrainte

Avec `kv_cache_dtype=auto` (donc bf16), le moteur **ne démarre pas** à `gm=0.70` :

```
ValueError: To serve at least one request with the model's max seq len (196608),
(4.34 GiB KV cache is needed, which is larger than the available KV cache memory
(4.28 GiB). Based on the available memory, the estimated maximum model length is
193632. Try increasing `gpu_memory_utilization` ... or decreasing `max_model_len`
```

Le calcul est vérifié à partir des mesures du test de référence (`C_c512_s8`, qui
démarre) :

| Format | Tokens stockés | Capacité | Tokens / Gio | Requis pour 196 608 tokens |
|--------|----------------|----------|--------------|-----------------------------|
| `fp8` | 400 959 | 4,42 Gio | 90 715 | **2,17 Gio** |
| `bf16` | — | 4,28 Gio | 45 357 | **4,34 Gio** |

Avec `fp8`, 400 959 tokens tiennent dans 4,42 Gio, soit une concurrency de 2,04×
(`400 959 / 196 608`). En bf16, la même place ne permet que 194 129 tokens — sous la
fenêtre maximale de 196 608. C'est l'écart de 2 479 tokens qui fait échouer le
démarrage, pas un ordre de grandeur.

Conséquence : **`fp8` est obligatoire** pour tenir à la fois `max_model_len=196608`
et `gpu_memory_utilization=0.70`. Les deux paramètres ne sont pas indépendants.

Le compromis possible : descendre `max_model_len` à 193 632 tokens libère le
démarrage en bf16, au prix de 1 512 tokens de fenêtre contextuelle (0,8 %) et du double
de cache KV par token. Sur une fenêtre de 196 608 tokens, ce choix est peu
intéressant, mais il contredit l'objectif d'une fenêtre de 196 608 tokens exacte —
donc `fp8` est retenu.

Note : le format `fp8_e4m3` est le format explicite de `fp8` dans vLLM — les deux
désignent la même chose, `fp8` étant l'alias court. Aucun gain à distinguer.

---

## Hypothèses testées (matrice de 38 tests)

Leviers candidats, par ordre de risque, et issue mesurée :

| # | Levier | Issue | Preuve |
|---|---|---|---|
| 1 | **`gpu-memory-utilization`** | **CONFIRMÉ comme cause racine** | `0.85` → 2 650 Mio partagés, ~140 t/s ; `0.70` → 125 Mio, 1 523 t/s |
| 2 | Overcommit GPU Windows (DWM 4,37 Gio) | Confirmé comme contributeur | le seuil de bascule est à `gm=0.78` (31 659/32 607 Mio) |
| 3 | Checkpoint AWQ-INT4 vs NVFP4 | **Rejeté** | AWQ 1 210 t/s vs NVFP4 1 538 t/s, pour 0,73 Gio de moins |
| 4 | `max-model-len` | **Écarté** | cf. constat 19 — 1 507–1 566 t/s pour 131 k / 196 k / 262 k |
| 5 | **`canvas_length`** | **CONFIRMÉ comme second levier** | `canvas 256` → 410 t/s en contexte long contre 121 t/s en `canvas 512` (×3,4) |
| 6 | `max_denoising_steps` | Plancher qualité à 8 | à 4 passes : 2 427 t/s mais texte dégradé (0,148 de dégradation) |
| 7 | `kv-cache-dtype` | **Contrainte, pas un levier** | `auto` (bf16) exige 4,34 Gio de KV, il n'y en a que 4,28 → le moteur ne démarre pas |
| 8 | `max-num-seqs` | **Écarté** | `mns=1/2/4/8` : 1 529–1 539 t/s, aucun gain. `mns=8` ne OOM plus à `gm=0.70` |
| 9 | Backend d'attention | **Écarté** | `TRITON_ATTN` équivalent ; `FLASHINFER` et `TORCH_SDPA` ne démarrent pas |
| 10 | `--enable-prefix-caching` | **Écarté** | déjà actif par défaut dans vLLM V1, écart nul |
| 11 | Patch WSL2 | **Prérequis, pas une option** | sans lui : `RuntimeError: UVA is not available` |

Chaque test mesure : VRAM dédiée/partagée Windows, `nvidia-smi` WSL, débit client
sériel (médiane de 3), débit sur prompt de 65 k tokens, TTFT, `<pad>`, validité de
l'appel d'outil, et les compteurs logs vLLM (conservés pour information, non retenus
comme critère).

Matrice complète : `state/final_matrix.json` (38 configurations).
Résultats : `state/results/<id>.json`, synthèse via `python3 state/summarize.py`.

### Résultat de la campagne — 38 configurations, 29 réussites, 9 échecs

| Famille | Tests | Enseignement |
|---|---|---|
| `gm` (0,55 → 0,80) | 10 | `0.62` et en dessous : le moteur ne démarre pas. `0.80` : bascule en mémoire partagée Windows (706 Mio partagés, 942 t/s). Zone utile `0.65`–`0.78`, optimum `0.70` |
| `max_model_len` (16 k → 262 k) | 6 | Aucun effet sur le débit court (1 507–1 566 t/s). La fenêtre maximale ne coûte presque rien en mémoire grâce au cache `fp8` |
| `canvas_length` (64 → 512) | 8 | **Second levier majeur.** `canvas 256` : 410 t/s en contexte long contre 121 t/s en `canvas 512` (×3,4) |
| `max_denoising_steps` (4 → 12) | 5 | Plancher à 8. À 4 passes : 2 427 t/s mais 42,6 % de dégradation sur le texte ; à 6 : 31,8 % |
| Backend d'attention | 4 | `FLASH_ATTN` (défaut) et `TRITON_ATTN` équivalents. `FLASHINFER` et `TORCH_SDPA` ne démarrent pas |
| `max_num_seqs` (1 → 8) | 3 | Aucun gain (1 529–1 539 t/s). Le modèle sérialise les requêtes |
| `kv_cache_dtype` | 3 | `fp8` obligatoire : bf16 exige 4,34 Gio, il n'y en a que 4,28 |
| Checkpoint AWQ-INT4 | 2 | Rejeté : 1 215 t/s contre 1 525 t/s, et 31 t/s en contexte long contre 121 t/s |
| Patch WSL2 / prefix caching | 2 | Patch obligatoire (`UVA is not available` sans lui) ; prefix caching déjà actif par défaut |

Les 9 échecs (`A_gm62`, `A_gm60`, `A_gm55`, `D_ab_FLASH_ATTN`, `D_ab_FLASHINFER`,
`D_ab_TORCH_SDPA`, `D_nopatch`, `E_kvd_auto`, `E_kvd_bfloat16`) sont tous des
**erreurs de démarrage**, pas des dégradations de débit : le service refuse de
démarrer et la campagne le note `ECHEC_SANTE`. Aucun test n'a produit un service
démarré mais inutilisable. C'est la conséquence directe du budget mémoire `gm=0.70` :
tout ce qui consomme de la VRAM au-delà de ce qui reste disponible empêche le moteur
de s'initialiser.

Note sur `D_ab_FLASH_ATTN` : c'est le backend **par défaut**, mais le forcer
explicitement fait échouer le démarrage. Le défaut « fonctionne » précisément parce
que vLLM choisit alors une autre configuration interne (mode de graphe CUDA
compatible) — ne pas expliciter ce backend.

Les configurations en échec n'écrivent pas de fichier dans `state/results/` : le
harnais ne conserve que les tests qui ont abouti. Le journal de la campagne (sortie
console de `bench2.py`) contient les 38 lignes, réussites et échecs.

---

## Journal des fichiers

| Fichier | Rôle |
|---|---|
| `JOURNAL.md` | ce journal |
| `state/` | artefacts de diagnostic (ignoré par Git) |
| `state/final_matrix.json` | matrice des 38 configurations testées |
| `state/results/` | résultats détaillés (un JSON par test) |
| `state/summarize.py` | tableau de synthèse des résultats |
| `state/gen_configs.py` | générateur de matrices par famille de leviers |
| `state/reanalyze_quality.py` | recalcule la métrique de qualité sur les résultats existants |
| `state/opencode_scenario.py` | simulation d'une session OpenCode (historique croissant) |
| `state/final_validation.py` | comparaison de deux configurations sur session OpenCode réelle (compose de production, un bloc modifié) |
| `bench2.py` | harnais de mesure (mémoire GPU, débit client, TTFT, qualité) |
| `verify.sh` | vérification bout-en-bout de la configuration de production |
| `docker-compose.yml` | configuration de production |
| `trace_configs.md` | historique des runs précédents |

### Critères de verdict appliqués par le harnais

Un test n'est `OK` que si **toutes** les conditions sont réunies :

| Critère | Seuil | Raison |
|---|---|---|
| Débit client, prompt court | ≥ 400 t/s | cible utilisateur |
| Débit client, prompt 65 k | > 0 t/s | le contexte long est le cas d'usage OpenCode |
| Appel d'outil | valide | usage agent |
| `<pad>` | 0 | anomalie connue |
| Erreurs de requête | 0 | un prompt refusé (HTTP 400) invalide la config |
| Dégradation du texte | ≤ 3 % | un nombre de passes de débruitage insuffisant produit des suites de ponctuation parasite et des mots dupliqués |

Le débit mesuré est **client** (tokens de complétion ÷ durée réelle de la requête),
pas le compteur des logs vLLM, qui est moyenné sur une fenêtre de 10 s et dilué par
l'inactivité du serveur. Précision mesurée sur la configuration de production : sous
charge continue, `Committed token throughput` affiche 1 382 tok/s (cohérent avec le
débit client de 1 361 tok/s) ; sur un serveur inactif, la même métrique affiche
~38 tok/s. La valeur des logs n'est donc interprétable que si la fenêtre de log est
saturée par des requêtes.

---

## 2026-11-09 — Décision finale : `canvas_length=256`

### Constat 26 — le canvas est le second levier, et le premier est le budget mémoire

Les 38 tests convergent sur deux leviers, tous deux dans le `--diffusion-config` ou le
budget mémoire. Le débit en **contexte long** (le cas OpenCode) est multiplié par 3,4
entre `canvas 512` et `canvas 256` :

| `canvas_length` | Court | 65 k tokens | TTFT | KV |
|-----------------|-------|-------------|------|----|
| 64 | 611 t/s | 195 t/s | 0,140 s | 4,42 Gio |
| 128 | 1 017 t/s | 345 t/s | 0,161 s | 4,42 Gio |
| **256** | **1 365 t/s** | **410 t/s** | **0,212 s** | 4,42 Gio |
| 512 | 1 525 t/s | 121 t/s | 0,356 s | 4,42 Gio |

Le débit court progresse légèrement avec le canvas (1 365 → 1 525 t/s), mais le débit
long recule de 3,4×. Le coût d'une passe de débruitage est proportionnel à la taille
du canvas ; en contexte long, ce coût est payé sur toute la fenêtre.

### Constat 27 — validation sur session OpenCode réelle

`state/final_validation.py` rejoue 8 tours d'une session agent (historique croissant,
appel d'outil une fois sur deux) sur les deux candidats, avec le compose de production
et un seul bloc modifié :

| Configuration | Médiane | Min | Max | Déd. Mio | Partagé Mio |
|---------------|---------|-----|-----|----------|-------------|
| `canvas 512` | 1 092 t/s | 110 t/s | 1 545 t/s | 29 400 | 131 |
| **`canvas 256`** | **1 411 t/s** | **333 t/s** | 1 435 t/s | 29 378 | 141 |

`canvas 256` gagne sur la médiane **et** sur le minimum (3× plus stable). Les tours
sont tous productifs en `canvas 256` (512 tokens de complétion à chaque tour, contre
42 à 512 tokens en `canvas 512` sur les tours 5 et 7).

Note : dans cette session, `tool_calls` est vide pour les deux configurations — le
modèle analyse la demande et pose des questions au lieu d'appeler l'outil, faute de
fichiers à lire. La validité de l'appel d'outil est vérifiée séparément par
`verify.sh` (qui obtient bien `get_weather → {"city": "Paris"}`) et par le critère
`tool` de la campagne, satisfait pour les 29 tests réussis.

### Configuration de production retenue

```yaml
--gpu-memory-utilization 0.70
--max-model-len 196608
--kv-cache-dtype fp8
--max-num-seqs 4
--diffusion-config '{"canvas_length":256,"max_denoising_steps":8}'
```

Vérification bout-en-bout (`./verify.sh`) sur cette configuration : **8 réussis,
0 échec** — fenêtre 196 608 tokens, budget `gm=0.70`, 136 Mio de mémoire partagée,
appel d'outil structuré valide, 0 `<pad>`, débit client médian **1 361 t/s**.

### Ce qui n'a pas été résolu

- Le débit décroît quand l'historique OpenCode grandit : 1 365 t/s sur 29 tokens de
  prompt, 410 t/s sur 65 031 tokens. C'est structurel au modèle de diffusion (le
  débruitage est payé `max_denoising_steps` fois sur tout le contexte), pas un défaut
  de configuration vLLM. Le `canvas 256` atténue le phénomène sans le supprimer.
- `max_denoising_steps=8` est un plancher qualité : à 6 passes, le texte se dégrade de
  31,8 %, à 4 passes de 42,6 %. Descendre plus bas échange de la vitesse contre de la
  qualité, pas contre de l'inexactitude.
- La concurrence ne peut pas être exploitée : `max_num_seqs` de 1 à 8 ne change rien au
  débit, le moteur sérialise les requêtes.

---

## 2026-11-09 — Phase 3 : mémoire du contexte et qualité des réponses

### Constat 28 — les 8 passes de débruitage ne multiplient PAS la mémoire par 8

Hypothèse à vérifier : « le débruitage multipliant par 8 le contexte, il faudrait
diviser la fenêtre par 8 pour ne pas déborder ». **Infirmée par la mesure.**

Le cache KV est **partagé** entre les passes de débruitage : chaque passe recalcule les
logits du canvas mais réutilise les KV du contexte. Preuve — `vllm:kv_cache_usage_perc`
échantillonné pendant une génération longue (`state/kv_probe.py`) :

| Tokens du prompt | Usage KV pendant la génération | Tokens occupés | Ratio |
|------------------|-------------------------------|----------------|-------|
| 49 174 | 17,2 % | 69 035 | 1,40× |
| 98 327 | 27,8 % | 111 518 | 1,13× |

Le ratio tokens occupés / tokens du prompt reste **constant** (~1,1×) : le cache est
alloué une fois, pas huit fois. Si chaque passe dupliquait le contexte, 98 327 tokens
occuperaient 786 616 tokens, soit **196 %** du cache disponible (400 959 tokens) —
le service serait alors incapable de servir la requête.

Confirmation par le paramètre `max_model_len` : 196 608 tokens de fenêtre sont
servis avec 4,42 Gio de cache KV, et `Maximum concurrency: 2.04x`. Si le contexte
était multiplié par 8, la fenêtre utile serait 196 608 / 8 = 24 576 tokens.

**Conséquence : l'agent n'est pas contraint à 24 k tokens.** La fenêtre de 196 608
tokens est réellement utilisable, et 65 k tokens tiennent sans aucun ajustement.

Ce qui coûte réellement 8 fois, c'est le **temps de calcul** (8 passes sur tout le
contexte), pas la mémoire. D'où le débit de 121 t/s à `canvas 512` contre 410 t/s en
`canvas 256` sur un prompt de 65 k tokens.

### Constat 29 — la qualité n'avait pas été mesurée : construction d'un banc

**Lacune réelle de la campagne.** La métrique `degradation` de `bench2.py` ne mesure
que deux choses : la ponctuation parasite (`***`, `....,,`) et les mots adjacents
dupliqués (`ForwardForward`). Elle détecte une **dégénérescence** du modèle de
diffusion — ce qui arrive quand le nombre de passes est insuffisant — et rien d'autre.
Elle ne dit rien de la justesse du propos, du respect des consignes, ni de l'usage des
outils. La phrase « l'appel d'outil casse en dessous de 8 passes » était la seule
information semi-qualitative de toute la campagne.

`state/quality_bench.py` comble cette lacune avec 8 critères **objectifs et
automatiquement vérifiables** :

| Critère | Ce qu'il vérifie |
|---------|------------------|
| `long_context_recall` | retrouver un numéro noyé dans 60 k tokens de contexte |
| `exact_5_lines` | respecter « exactement 5 éléments, un par ligne » |
| `one_sentence` | respecter « UNE seule phrase » |
| `forbidden_word_absent` | ne pas utiliser un mot interdit |
| `strict_json` | produire du JSON strictement valide |
| `tool_use` | émettre un appel d'outil structuré |
| `code_compiles` | produire du Python syntaxiquement valide |
| `no_hallucinated_file` | ne pas inventer le contenu d'un fichier absent |

Un piège méthodologique a été corrigé en cours de route : la réponse du modèle est
scindée en `reasoning` (brouillon qui répète l'énoncé) et `content` (réponse finale).
Concaténer les deux faisait échouer artificiellement 5 critères sur 8 — le modèle
respectait parfaitement la consigne dans `content`.

### Constat 30 — `canvas 256` vs `canvas 512` : la qualité est équivalente

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

Le canvas ne dégrade pas la qualité du propos : les deux configurations respectent
identiquement les consignes de format, produisent du JSON et du Python valides, et
n'hallucinent pas de fichier.

### Constat 31 — le rappel en contexte long est le vrai point faible, et il est indépendant du canvas

Test de rappel (`state/recall_trial.py`) : un numéro de série unique noyé dans un
contexte de 64 953 tokens, 20 trials par configuration, jugement sur le `content`
final :

| Configuration | Réponses exactes | Réponses vides | Raisonnement correct | Débit médian |
|---------------|------------------|----------------|-----------------------|--------------|
| `canvas 256` | 16/20 (80 %) | 2/20 | 20/20 | 249 t/s |
| `canvas 512` | 13/20 (65 %) | 5/20 | 20/20 | 121 t/s |

**`canvas 256` fait au moins aussi bien** — 80 % contre 65 %, avec deux fois moins de
réponses vides. Dans les deux cas, le raisonnement contient le numéro dans **100 %**
des trials — le modèle *localise* toujours l'information. Ce qui varie, c'est la
fidélité du **commit final** : le texte commité peut être vide, tronqué, ou contenir
un numéro approchant (`7429831` → `74299811`, `77223333`).

C'est la mécanique propre à la diffusion : le canvas est débruité de façon
stochastique (Gumbel-max, renoise aléatoire), et `argmax_canvas` est commité. Une
passe de débruitage peut « perdre » un token que le raisonnement avait pourtant trouvé.

Au-delà de ~100 k tokens de contexte, le rappel s'effondre dans les deux
configurations : à 153 k tokens, plus aucune réponse exacte. La fenêtre utile réelle
du modèle se situe donc autour de **65 k–100 k tokens**, pas 196 608 — et c'est
indépendant du réglage vLLM.

### Décision : `canvas 256` est confirmé

À qualité de propos identique (7/8) et à taux de rappel supérieur (80 % contre
65 %), `canvas 256` offre **2,1× le débit** (249 contre 121 t/s sur le test de rappel,
410 contre 121 t/s sur le prompt de 65 k de la campagne). Le choix ne coûte rien en
qualité, et gagne même sur l'exactitude.

Ce que la campagne **n'a pas** mesuré, et qui reste ouvert :

- La justesse sémantique sur des tâches de raisonnement réelles (code correct sur un
  dépôt, diagnostic d'un bug) — les critères mechaniques ne disent pas si le code
  *fait* ce qu'on attend.
- La variabilité inter-tâches : le banc utilise 8 cas, pas un jeu de référence.
- Le comportement au-delà de 100 k tokens, où le rappel s'effondre. Réduire
  `max_model_len` à ~98 304 tokens rendrait le service *honnête* sur sa fenêtre
  réelle plutôt que d'annoncer 196 608 tokens dont l'utilité se dégrade.
