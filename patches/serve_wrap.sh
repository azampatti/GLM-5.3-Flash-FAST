#!/bin/bash
# Runs inside each rank's container: patch the container's vLLM for this build (patch_glm53.py), then exec vllm unchanged.
# The image on disk is never modified; every new container starts clean and is patched again.
set -e
echo "[glm53-lever] $(hostname) GLM53_TOPK=${GLM53_TOPK:-} GLM53_MXFP8=${GLM53_MXFP8:-}"
python3 "$(dirname "$0")/patch_glm53.py"
exec "$@"
