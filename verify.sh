#!/usr/bin/env bash
# Verification bout-en-bout de la configuration de production DiffusionGemma.
#
# Controle les trois criteres :
#   1. fenetre contextuelle de 262144 tokens
#   2. appel d'outil structure, sans anomalie <pad>
#   3. debit "Committed token throughput" > 500 tokens/s dans les logs vLLM
#
# Usage : ./verify.sh [duree_charge_secondes]
set -uo pipefail

SUSTAIN_S="${1:-40}"
BASE="http://localhost:8000"
MODEL="diffusiongemma"
CONTAINER="diffusiongemma-vllm"
PASS=0
FAIL=0

check() {
    local label="$1" ok="$2" detail="$3"
    if [[ "$ok" == "1" ]]; then
        printf '  [OK]   %-46s %s\n' "$label" "$detail"
        PASS=$((PASS + 1))
    else
        printf '  [ECHEC] %-46s %s\n' "$label" "$detail"
        FAIL=$((FAIL + 1))
    fi
}

post() {
    curl -s --max-time 300 -H 'Content-Type: application/json' \
         -d "$2" "$BASE$1"
}

echo "=== Verification DiffusionGemma / vLLM ==="

# --- 0. Sante du service ---------------------------------------------------
for _ in $(seq 1 90); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/health" 2>/dev/null)
    [[ "$code" == "200" ]] && break
    sleep 10
done
check "service pret (/health = 200)" \
      "$([[ "$code" == "200" ]] && echo 1 || echo 0)" "code=$code"

# --- 1. Fenetre contextuelle ---------------------------------------------
MAXLEN=$(docker logs "$CONTAINER" 2>&1 | grep -oE "max_model_len=[0-9]+" | tail -1 | cut -d= -f2)
[[ -z "$MAXLEN" ]] && MAXLEN=$(docker logs "$CONTAINER" 2>&1 | grep -oE "Maximum concurrency for [0-9,]+ tokens per request" | tail -1 | grep -oE "[0-9,]+" | tr -d ,)
check "fenetre contextuelle = 262144" \
      "$([[ "$MAXLEN" == "262144" ]] && echo 1 || echo 0)" "max_model_len=$MAXLEN"

# --- 2. Appel d'outil + absence de <pad> ---------------------------------
TOOL_BODY='{"model":"'"$MODEL"'","messages":[{"role":"system","content":"You are an assistant that uses tools."},{"role":"user","content":"Use the get_weather tool to get the weather in Paris."}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Recupere la meteo dune ville","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"tool_choice":"required","max_tokens":512,"skip_special_tokens":false}'

TOOL_RESP=$(post /v1/chat/completions "$TOOL_BODY")
PAD_COUNT=$(printf '%s' "$TOOL_RESP" | grep -o "<pad>" | wc -l)
TOOL_NAMES=$(printf '%s' "$TOOL_RESP" | grep -o '"name"[[:space:]]*:[[:space:]]*"get_weather"' | wc -l)

check "appel d'outil structure (get_weather)" \
      "$([[ "$TOOL_NAMES" -ge 1 ]] && echo 1 || echo 0)" "occurrences=$TOOL_NAMES"
check "aucune anomalie <pad> (reponse outil)" \
      "$([[ "$PAD_COUNT" -eq 0 ]] && echo 1 || echo 0)" "<pad> x$PAD_COUNT"

# --- 3. Reponse texte sans <pad> -----------------------------------------
TEXT_BODY='{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Explique en detail le fonctionnement d un modele de diffusion."}],"max_tokens":1024,"skip_special_tokens":false}'
TEXT_RESP=$(post /v1/chat/completions "$TEXT_BODY")
TEXT_PAD=$(printf '%s' "$TEXT_RESP" | grep -o "<pad>" | wc -l)
check "aucune anomalie <pad> (reponse texte)" \
      "$([[ "$TEXT_PAD" -eq 0 ]] && echo 1 || echo 0)" "<pad> x$TEXT_PAD"

# --- 4. Debit dans les logs vLLM sous charge continue ----------------------
echo "  ... charge continue pendant ${SUSTAIN_S}s (requetes seriees)"
python3 - "$SUSTAIN_S" <<'PYEOF'
import json, sys, time, urllib.request

duration = float(sys.argv[1])
url = "http://localhost:8000/v1/chat/completions"
payload = json.dumps({
    "model": "diffusiongemma",
    "messages": [{"role": "user",
                  "content": "Explique en detail le fonctionnement d un modele de diffusion."}],
    "max_tokens": 512,
    "skip_special_tokens": False,
}).encode()

tokens = 0
pads = 0
errors = 0
end = time.time() + duration
while time.time() < end:
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read())
        usage = body.get("usage", {}) or {}
        tokens += usage.get("completion_tokens", 0) or 0
        for ch in body.get("choices", []):
            msg = ch.get("message", {}) or {}
            pads += (msg.get("content") or "").count("<pad>")
            pads += (msg.get("reasoning_content") or "").count("<pad>")
    except Exception as exc:  # noqa: BLE001
        errors += 1
        print(f"    erreur: {exc}", file=sys.stderr)
print(f"    {tokens} tokens generes, {pads} <pad>, {errors} erreurs")
PYEOF

BEST=$(docker logs "$CONTAINER" 2>&1 \
       | grep -oE "Committed token throughput: [0-9.]+" \
       | tail -30 | grep -oE "[0-9.]+" \
       | sort -g | tail -1)
[[ -z "$BEST" ]] && BEST=0
check "debit logs vLLM > 500 tok/s" \
      "$(awk -v v="$BEST" 'BEGIN{print (v>500)?1:0}')" \
      "Committed token throughput = $BEST tok/s"

echo
echo "=== $PASS reussis, $FAIL echoues ==="
[[ "$FAIL" -eq 0 ]] && exit 0 || exit 1
