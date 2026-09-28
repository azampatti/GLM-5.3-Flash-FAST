# Sourced by setup.sh, serve.sh, stop.sh and check.sh. Loads site values from the upstream .env, then this repo's config.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
say(){ printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die(){ echo "ERROR: $*" >&2; exit 1; }

# Our knobs win over anything the upstream .env sets for the same names (its K, CTX, PORT... describe ITS recipe);
# a value set in your shell for this one run wins over both.
_KNOBS="PORT SERVED_MODEL_NAME K CTX KV_CACHE_MEMORY GMU SEQS BATCH CAPTURE_SIZE MIN_FREE_GIB IMAGE"
declare -A _OVR=()
for _v in $_KNOBS; do [ -n "${!_v+x}" ] && _OVR[$_v]="${!_v}"; done
UPSTREAM_DIR="${UPSTREAM_DIR:-$HOME/GLM53F-NvidiaNVFP4-DFlash}"
[ -f "$UPSTREAM_DIR/.env" ] || die "no $UPSTREAM_DIR/.env -- clone the upstream repo there and fill its .env (see README), or set UPSTREAM_DIR"
set -a; . "$UPSTREAM_DIR/.env"; set +a
for _v in $_KNOBS; do if [ -n "${_OVR[$_v]+x}" ]; then export "$_v=${_OVR[$_v]}"; else unset "$_v"; fi; done
. "$REPO/config.env"

# site values (upstream .env)
: "${CONTAINER:=glm53f-nvidia}"; : "${MASTER_PORT:=29503}"; : "${TP_SIZE:=2}"
: "${WORKER_SSH_TARGET:=${WORKER_IP:-}}"; : "${WORKER_ROCE_SSH_TARGET:=$WORKER_SSH_TARGET}"
for _v in HEAD_IP WORKER_IP MN_IF HCAS HF_CACHE WORKER_SSH_TARGET; do
  [ -n "${!_v:-}" ] || die "$_v is not set in $UPSTREAM_DIR/.env"
done
HF_CACHE="${HF_CACHE%/}"

# derived
LAUNCHER="$UPSTREAM_DIR/vendor/run_cluster_dual.sh"
WEIGHTS="$REPO/glm53-builds/kda-nvfp4"          # the path must contain /glm53-builds/: the loader's index filter keys on it
KDA_FILE="$WEIGHTS/model-kda-nvfp4.safetensors"
SNAP="$HF_CACHE/hub/models--${MODEL_REPO//\//--}/snapshots/$MODEL_REVISION"
DRAFT_SNAP_REL="models--${DRAFT_REPO//\//--}/snapshots/$DRAFT_REVISION"   # relative to $HF_CACHE/hub (mounted as /workspace/models)
SETUP_OK="$REPO/glm53-builds/.setup-ok"
DROP='docker run --rm --privileged -v /proc:/hostproc nvidia/cuda:13.0.0-base-ubuntu24.04 sh -c "sync; echo 3 > /hostproc/sys/vm/drop_caches"'

wssh(){ ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH_TARGET" "$@"; }
avail_gib(){ awk '/^MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
worker_avail_gib(){ wssh "awk '/^MemAvailable/{printf \"%d\", \$2/1048576}' /proc/meminfo" 2>/dev/null || echo 0; }
on_head(){ ip -4 -o addr 2>/dev/null | grep -qwF "$HEAD_IP"; }
