#!/usr/bin/env bash
set -e

# Crée un module sitecustomize qui sera importé par TOUS les processus Python
mkdir -p /tmp/vllm_patch
cat > /tmp/vllm_patch/sitecustomize.py << 'EOF'
import sys
print("[vllm_wsl_patch] Applying UVA / pin_memory patch...", file=sys.stderr)

try:
    import vllm.platforms as p
    p.Platform.is_pin_memory_available = classmethod(lambda cls: True)
    print("[vllm_wsl_patch] Platform.is_pin_memory_available patched", file=sys.stderr)
except Exception as e:
    print(f"[vllm_wsl_patch] platforms patch failed: {e}", file=sys.stderr)

try:
    import vllm.v1.worker.gpu.buffer_utils as b
    b.is_uva_available = lambda: True
    if hasattr(b.is_uva_available, "cache_clear"):
        b.is_uva_available.cache_clear()
    print("[vllm_wsl_patch] buffer_utils.is_uva_available patched", file=sys.stderr)
except Exception as e:
    print(f"[vllm_wsl_patch] buffer_utils patch failed: {e}", file=sys.stderr)

try:
    import vllm.utils.platform_utils as pu
    pu.is_uva_available = lambda: True
    if hasattr(pu.is_uva_available, "cache_clear"):
        pu.is_uva_available.cache_clear()
    print("[vllm_wsl_patch] platform_utils.is_uva_available patched", file=sys.stderr)
except Exception as e:
    print(f"[vllm_wsl_patch] platform_utils patch failed: {e}", file=sys.stderr)
EOF

export PYTHONPATH="/tmp/vllm_patch:${PYTHONPATH:-}"

echo "[wsl_entrypoint] UVA/pin_memory patched via sitecustomize; launching vllm serve"
exec vllm serve "$@"