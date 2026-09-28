#!/usr/bin/env bash
# Stop the server on both nodes.
set -euo pipefail
cd "$(dirname "$0")"; . tools/common.sh
pkill -f "run_cluster_dual.sh $IMAGE" 2>/dev/null || true
wssh "docker rm -f $CONTAINER" >/dev/null 2>&1 && echo "worker: stopped" || echo "worker: nothing to stop (or unreachable)"
docker rm -f "$CONTAINER" >/dev/null 2>&1 && echo "head:   stopped" || echo "head:   nothing to stop"
