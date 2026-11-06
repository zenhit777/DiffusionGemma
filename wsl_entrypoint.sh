#!/usr/bin/env bash
# WSL2 workaround: vLLM V2 runner (required for diffusion models) requires UVA,
# which WSL2 reports as unavailable. Force is_uva_available()=True and
# pin_memory available so the engine can init. Throughput will be degraded
# vs native Linux (no real pinned/UVA memory), but lets us verify the path.
python3 - <<'PY'
import vllm.platforms as p
from vllm.platforms import current_platform
p.Platform.is_pin_memory_available = classmethod(lambda cls: True)
import vllm.v1.worker.gpu.buffer_utils as b
b.is_uva_available = (lambda: True)
try:
    b.is_uva_available.cache_clear()
except Exception:
    pass
PY
echo "[wsl_entrypoint] UVA/pin_memory patched; launching vllm serve"
exec vllm serve "$@"
