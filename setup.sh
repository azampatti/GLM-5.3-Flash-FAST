#!/usr/bin/env bash
# One-time setup, run on the HEAD node. Re-runnable: every step skips what is already done and verified.
#   ./setup.sh
#
#   1. checks the upstream clone (its .env, and the launcher this repo borrows from it)
#   2. the pinned vLLM image on both nodes
#   3. the model and the DFlash2 draft at their pinned revisions, in the HF cache of both nodes
#   4. rebuilds the 4-bit KDA weights into ./glm53-builds/kda-nvfp4 (CPU, ~1 min) and checks them byte for byte
#   5. copies this repo to the worker at the same path (the containers on both nodes mount it)
set -euo pipefail
cd "$(dirname "$0")"; . tools/common.sh

# ---------------------------------------------------------------------------------------------------------------------
say "1/5 Checking the upstream clone and the two nodes"
command -v docker >/dev/null || die "docker is not installed"
on_head || die "run this on the head node ($HEAD_IP from $UPSTREAM_DIR/.env)"
[ -f "$LAUNCHER" ] || die "missing $LAUNCHER -- is $UPSTREAM_DIR a clone of the upstream repo?"
got="$(sha256sum "$LAUNCHER" | cut -d' ' -f1)"
if [ "$got" != "$LAUNCHER_SHA256" ]; then
  echo "  upstream vendor/run_cluster_dual.sh differs from the validated one (upstream commit ${UPSTREAM_COMMIT:0:7})."
  echo "  To match exactly:  git -C $UPSTREAM_DIR checkout $UPSTREAM_COMMIT"
  [ "${ALLOW_LAUNCHER_CHANGE:-0}" = 1 ] || die "launcher changed (set ALLOW_LAUNCHER_CHANGE=1 to use it anyway)"
fi
echo "  ok: upstream at $UPSTREAM_DIR ($(git -C "$UPSTREAM_DIR" rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout'))"
wssh true || die "cannot ssh to the worker ($WORKER_SSH_TARGET) without a password"
wssh "test -d '$HF_CACHE'" || die "the worker has no $HF_CACHE (both nodes need the HF cache at the same path)"
echo "  ok: head $HEAD_IP, worker $WORKER_SSH_TARGET, HF cache $HF_CACHE on both"
case "$REPO" in *" "*) die "the repo path has a space in it: $REPO" ;; esac
wssh "mkdir -p '$REPO'" || die "cannot create $REPO on the worker (this repo must live at the same path on both nodes)"

# ---------------------------------------------------------------------------------------------------------------------
say "2/5 vLLM image ($IMAGE, pinned)"
if [ "$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null)" != "$IMAGE_ID" ]; then
  echo "  pulling $IMAGE_PIN"
  docker pull "$IMAGE_PIN"
  docker tag "$IMAGE_PIN" "$IMAGE"
fi
[ "$(docker image inspect -f '{{.Id}}' "$IMAGE")" = "$IMAGE_ID" ] || die "$IMAGE is not the pinned build $IMAGE_ID"
echo "  ok: head has $IMAGE = ${IMAGE_ID:7:12}"
if [ "$(wssh "docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null" || true)" != "$IMAGE_ID" ]; then
  echo "  shipping it to the worker (docker save | docker load, ~25 GB)"
  docker save "$IMAGE" | ssh -o BatchMode=yes "$WORKER_ROCE_SSH_TARGET" "docker load" | tail -1
fi
[ "$(wssh "docker image inspect -f '{{.Id}}' $IMAGE" 2>/dev/null)" = "$IMAGE_ID" ] || die "the worker's $IMAGE is not $IMAGE_ID"
echo "  ok: worker has the same image"

# ---------------------------------------------------------------------------------------------------------------------
say "3/5 Model and draft at their pinned revisions"
hf_get(){   # hf_get <repo> <revision>: into $HF_CACHE (host 'hf' if present, else the image's)
  if [ "${OFFLINE:-0}" = 1 ]; then echo "  OFFLINE=1: not contacting the hub for $1"; return 0; fi
  echo "  $1 @ ${2:0:8}"
  if command -v hf >/dev/null 2>&1; then
    HF_HOME="$HF_CACHE" hf download "$1" --revision "$2" >/dev/null
  else
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME="$HF_CACHE" ${HF_TOKEN:+-e HF_TOKEN} \
      -v "$HF_CACHE:$HF_CACHE" --entrypoint hf "$IMAGE" download "$1" --revision "$2" >/dev/null
  fi
}
hf_get "$MODEL_REPO" "$MODEL_REVISION"
hf_get "$DRAFT_REPO" "$DRAFT_REVISION"
[ -f "$SNAP/config.json" ] && [ -f "$SNAP/model.safetensors.index.json" ] || die "model snapshot incomplete: $SNAP"
[ -f "$HF_CACHE/hub/$DRAFT_SNAP_REL/config.json" ] || die "draft snapshot incomplete: $HF_CACHE/hub/$DRAFT_SNAP_REL"
echo "  ok: both in $HF_CACHE/hub on the head"
for entry in "models--${MODEL_REPO//\//--}" "models--${DRAFT_REPO//\//--}"; do
  echo "  syncing $entry to the worker (only what is missing moves)"
  rsync -a "$HF_CACHE/hub/$entry" "$WORKER_ROCE_SSH_TARGET:$HF_CACHE/hub/"
done
wssh "test -f '$SNAP/config.json' && test -f '$HF_CACHE/hub/$DRAFT_SNAP_REL/config.json'" || die "the worker is missing the model or the draft"
echo "  ok: worker has both"

# ---------------------------------------------------------------------------------------------------------------------
say "4/5 4-bit KDA weights -> $WEIGHTS"
sha(){ sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
weights_ok(){ [ "$(sha "$KDA_FILE")" = "$KDA_FILE_SHA256" ] && [ "$(sha "$WEIGHTS/model.safetensors.index.json")" = "$KDA_INDEX_SHA256" ] \
              && [ -z "$(find -L "$WEIGHTS" -maxdepth 1 -type l)" ] && [ -e "$WEIGHTS/model-00001-of-00033.safetensors" ]; }
if weights_ok; then
  echo "  already built and verified"
else
  docker ps --format '{{.Names}}' | grep -qx "$CONTAINER" && die "$CONTAINER is running; stop it first (./stop.sh): the build needs ~40 GB of RAM"
  [ "$(avail_gib)" -ge 45 ] || die "only $(avail_gib) GiB available; the build needs ~40 GB"
  mkdir -p "$WEIGHTS"
  echo "  quantizing KDA q/k/v/o -> NVFP4 and b/f_a/g_a -> MXFP8 inside the image (CPU only, about a minute)"
  # Both paths are mounted at their host paths, so the relative links the build writes resolve on the host and in the
  # serving containers alike.
  docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp --memory 40g --entrypoint bash \
    -v "$HF_CACHE:$HF_CACHE:ro" -v "$REPO:$REPO" "$IMAGE" \
    -lc "python3 '$REPO/tools/build_kda_nvfp4.py' '$SNAP' '$WEIGHTS'" 2>&1 | grep -E "KDA 4-bit view|Error|error" || true
  weights_ok || {
    echo "  sha256 got $(sha "$KDA_FILE") (want $KDA_FILE_SHA256)"
    echo "  index  got $(sha "$WEIGHTS/model.safetensors.index.json") (want $KDA_INDEX_SHA256)"
    echo "  dangling links: $(find -L "$WEIGHTS" -maxdepth 1 -type l | wc -l)"
    die "the rebuilt weights are not byte-identical to the validated build"
  }
fi
echo "  ok: model-kda-nvfp4.safetensors ${KDA_FILE_SHA256:0:12}, index ${KDA_INDEX_SHA256:0:12}; everything else links to $SNAP"

# ---------------------------------------------------------------------------------------------------------------------
say "5/5 Copying this repo to the worker at $REPO"
rsync -a --delete --exclude .git --exclude logs "$REPO/" "$WORKER_ROCE_SSH_TARGET:$REPO/"
wssh "cd '$REPO' && [ \"\$(sha256sum glm53-builds/kda-nvfp4/model-kda-nvfp4.safetensors | cut -d' ' -f1)\" = $KDA_FILE_SHA256 ] \
      && [ -z \"\$(find -L glm53-builds/kda-nvfp4 -maxdepth 1 -type l)\" ]" || die "the worker's copy does not verify"
echo "  ok: worker copy verified (same bytes, every link resolves)"

stamp="$KDA_FILE_SHA256 $IMAGE_ID $MODEL_REVISION $DRAFT_REVISION"
echo "$stamp" > "$SETUP_OK"; wssh "echo '$stamp' > '$SETUP_OK'"
printf '\n\033[1mSetup complete.\033[0m  Start the server with:  ./serve.sh\n\n'
