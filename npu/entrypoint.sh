#!/usr/bin/env bash
set -e
source /opt/ryzen-ai/venv/bin/activate
source /opt/xilinx/xrt/setup.sh
if [[ -n "${RYZEN_AI_INSTALLATION_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:${RYZEN_AI_INSTALLATION_PATH}/onnxruntime/lib/:${LD_LIBRARY_PATH:-}"
fi
exec "$@"
