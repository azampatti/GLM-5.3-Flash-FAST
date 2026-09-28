#!/usr/bin/env bash
# Boot the kda4 build on both nodes (TP=2 over RoCE), from the HEAD node. Returns once it is serving and verified.
#   ./serve.sh              boot (~16 min), wait for health, verify the build, one smoke prompt
#   ./serve.sh --dry-run    print the launch command and exit
# Stop it with ./stop.sh. Settings: config.env (any of them can be set for one run: PORT=8100 ./serve.sh).
set -euo pipefail
cd "$(dirname "$0")"; . tools/common.sh
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

on_head || die "run this on the head node ($HEAD_IP)"
if [ "$DRY" = 0 ]; then
  [ -f "$SETUP_OK" ] && [ -f "$KDA_FILE" ] || die "not set up yet -- run ./setup.sh"
  [ "$(cut -d' ' -f1 "$SETUP_OK")" = "$KDA_FILE_SHA256" ] || die "setup is stale for this config -- run ./setup.sh"
  wssh "test -f '$SETUP_OK' && test -f '$KDA_FILE'" || die "the worker has no copy of this repo at $REPO -- run ./setup.sh"
fi

# RoCE v2 IPv4 GID index (it renumbers across reboots)
FIRST_HCA="${HCAS%%,*}"; GID=""
for i in $(seq 0 15); do
  t=$(cat "/sys/class/infiniband/$FIRST_HCA/ports/1/gid_attrs/types/$i" 2>/dev/null || true)
  g=$(cat "/sys/class/infiniband/$FIRST_HCA/ports/1/gids/$i" 2>/dev/null || true)
  case "$t" in *"RoCE v2"*) case "$g" in *0000:0000:0000:0000:ffff:*) GID="$i"; break;; esac ;; esac
done
[ -n "$GID" ] || die "no RoCE v2 IPv4 GID on $FIRST_HCA"

# DFlash2 speculative config, exactly as measured: k=7, sampled drafts, standard verification
SPEC_CONFIG=$(python3 - "/workspace/models/$DRAFT_SNAP_REL" "$K" <<'EOF'
import json, sys
cfg = {"method": "dflash", "model": sys.argv[1], "num_speculative_tokens": int(sys.argv[2]),
       "attention_backend": "TRITON_ATTN", "kv_cache_dtype": "auto",
       "draft_sample_method": "probabilistic", "rejection_sample_method": "standard",
       "enable_adaptive_verification": False, "disable_eagle_block_drop": False}
print(json.dumps(cfg))
EOF
)

# The build (patch_glm53.py reads these inside each container):
#   online MXFP8 for the BF16 shared experts, MLA q_b/o projections and lm_head;
#   KDA in_proj split + the offline 4-bit KDA weights (q/k/v/o NVFP4 W4A16, b/f_a/g_a MXFP8) from ./glm53-builds.
LEVERS=(-e GLM53_MXFP8=se,mla,head -e GLM53_KDA_SPLIT=1 -e GLM53_NVFP4_SERIALIZED=kda -e GLM53_MXFP8_SERIALIZED=kdabfg
        -e GLM53_TOPK= -e GLM53_DRAFT_QUANT_OK=)
mkdir -p logs; BLOG="$REPO/logs/boot-$(date +%Y%m%d-%H%M%S).log"
CMD=(bash "$LAUNCHER" "$IMAGE" "$HEAD_IP"
  --head "$HF_CACHE" --no-ray --workers "$WORKER_IP"
  --master-port "$MASTER_PORT" --container-name "$CONTAINER"
  --ipc=host --privileged --ulimit memlock=-1 --ulimit stack=67108864
  -e NCCL_SOCKET_IFNAME="$MN_IF" -e NCCL_IB_HCA="$HCAS" -e NCCL_IB_GID_INDEX="$GID"
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e GLOO_SOCKET_IFNAME="$MN_IF" -e VLLM_HOST_IP="$HEAD_IP"
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e CUTE_DSL_ARCH=sm_121a -e TORCH_USE_RTLD_GLOBAL=1
  -e TORCHINDUCTOR_COMPILE_THREADS=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  -e VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=auto -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
  "${LEVERS[@]}"
  -v "$HF_CACHE/hub:/workspace/models"
  -v "$HF_CACHE:$HF_CACHE:ro" -v "$REPO:$REPO:ro"
  -- bash "$REPO/patches/serve_wrap.sh" vllm serve "$WEIGHTS"
    --host 0.0.0.0 --port "$PORT"
    --distributed-executor-backend mp --tensor-parallel-size "$TP_SIZE"
    --dtype bfloat16 --trust-remote-code --moe-backend b12x --linear-backend b12x
    --gpu-memory-utilization "$GMU" --no-enable-flashinfer-autotune
    --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice
    --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill --async-scheduling
    --max-num-batched-tokens "$BATCH" --max-model-len "$CTX" --kv-cache-memory "$KV_CACHE_MEMORY"
    --kv-cache-dtype fp8 --block-size 256 --max-num-seqs "$SEQS" --max-cudagraph-capture-size "$CAPTURE_SIZE"
    --skip-mm-profiling --served-model-name "$SERVED_MODEL_NAME" --speculative-config "$SPEC_CONFIG")
if [ "$DRY" = 1 ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi

say "Stopping any running $CONTAINER, freeing memory on both nodes"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; wssh "docker rm -f $CONTAINER" >/dev/null 2>&1 || true
# Unified memory: the GPU and the page cache share one pool, and a fresh boot needs ~107 GiB free at engine init.
eval "$DROP" >/dev/null 2>&1 || true; wssh "$DROP" >/dev/null 2>&1 || true
for t in $(seq 0 5 180); do
  a=$(avail_gib); b=$(worker_avail_gib)
  [ "$a" -ge "$MIN_FREE_GIB" ] && [ "$b" -ge "$MIN_FREE_GIB" ] && break
  [ "$t" -ge 180 ] && die "only $a GiB (head) / $b GiB (worker) available after 3 min; need $MIN_FREE_GIB (something else running?)"
  sleep 5
done
echo "  available: head $(avail_gib) GiB, worker $(worker_avail_gib) GiB"

say "Booting (log: $BLOG), about 16 minutes"
{ echo "── GLM-5.3-Flash kda4: k=$K ctx $CTX kv-pin $KV_CACHE_MEMORY GMU $GMU seqs $SEQS batch $BATCH graphs<=$CAPTURE_SIZE port $PORT"
  echo "   weights  : $WEIGHTS"
  echo "   spec     : $SPEC_CONFIG"; } > "$BLOG"
setsid nohup "${CMD[@]}" >> "$BLOG" 2>&1 < /dev/null &
BASE="http://localhost:$PORT"
for i in $(seq 1 135); do            # 45 min
  sleep 20
  curl -sf --max-time 4 "$BASE/health" >/dev/null 2>&1 && break
  grep -aq "\[glm53-lever\] FATAL" "$BLOG" && die "a patch did not apply: $(grep -a 'glm53-lever\] FATAL' "$BLOG" | head -1)"
  if [ "$i" -ge 6 ] && ! docker inspect -f '{{.State.Running}}' "$CONTAINER" >/dev/null 2>&1; then
    grep -aiE 'Error|out of memory' "$BLOG" | tail -5; die "the container died during boot (see $BLOG)"
  fi
  [ "$i" -eq 135 ] && die "not healthy after 45 min (see $BLOG)"
done
echo "  healthy after ~$((i * 20 / 60)) min"

say "Verifying the build"
fail=0
chk(){ if grep -aqE "$2" "$BLOG"; then echo "  ok: $1"; else echo "  MISSING: $1"; fail=1; fi; }
chk "all 10 in-container patches" "patches: topk, mxfp8-linear, mxfp8-head, draft-quant, fp8-attn-guard, index-filter, kda-split, kda-split-fwd, kda-split-map, a16-marlin"
chk "index filter on the 4-bit KDA weights" "index filter active for $WEIGHTS"
chk "4-bit KDA layers (NVFP4 W4A16)" "serialized NVFP4-W4A16 #"
chk "MXFP8 KDA b/f/g layers" "serialized MXFP8 #"
chk "online MXFP8 (shared expert / MLA / lm_head)" "online MXFP8 #"
chk "draft depth $K" "\"num_speculative_tokens\": $K"
grep -aE "GPU KV cache size" "$BLOG" | tail -1 | sed 's/.*GPU KV/  KV:/'
ans=$(curl -s --max-time 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$SERVED_MODEL_NAME\",
  \"messages\":[{\"role\":\"user\",\"content\":\"What is 12+30? Answer with just the number.\"}],\"max_tokens\":512,\"temperature\":0,
  \"reasoning_effort\":\"low\"}" | python3 -c "import json,sys;print((json.load(sys.stdin)['choices'][0]['message'].get('content') or '').strip())" 2>/dev/null || true)
case "$ans" in *42*) echo "  ok: smoke prompt answered $ans" ;; *) echo "  FAILED: smoke prompt answered '${ans:0:120}'"; fail=1 ;; esac
[ "$fail" = 0 ] || die "the server is up but does not look like the validated build (see above and $BLOG)"
printf '\n\033[1mServing\033[0m %s on http://%s:%s/v1  (./stop.sh to stop)\n\n' "$SERVED_MODEL_NAME" "$HEAD_IP" "$PORT"
