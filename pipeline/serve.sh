#!/bin/bash
# pipeline/serve.sh start|stop configs/models/<name>.env
# Starts a vLLM OpenAI-compatible server for one model profile (MODEL SERVED_NAME PORT GPU_FRACTION MAX_LEN KV_BYTES
# EXTRA_ARGS) with a fixed KV-cache size, prefix caching, local-file images allowed under the shared root (so the frames
# in DATA must resolve to paths inside it), and all caches under that root. The GPU lock is held only while the server
# loads. The server runs in its own process group, so `stop` removes it completely.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SHARED=${SHARED_ROOT:-/data/cinesage}
action=$1; profile=$2; name=$(basename "$profile" .env)
mkdir -p "$ROOT/logs/serve"
export pidf=$ROOT/logs/serve/$name.pid logf=$ROOT/logs/serve/$name.log

if [ "$action" = stop ]; then
  if [ -f "$pidf" ]; then
    pid=$(cat "$pidf"); kill -TERM -- "-$pid" 2>/dev/null
    for _ in $(seq 60); do pgrep -g "$pid" >/dev/null || break; sleep 2; done
    pgrep -g "$pid" >/dev/null && kill -KILL -- "-$pid"
    rm -f "$pidf"
  fi
  echo "stopped $name"; exit 0
fi

set -a; source "$profile"; set +a
export PATH=$SHARED/envs/vllm/bin:$PATH HF_HOME=$SHARED/hf-cache HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 SHARED \
  XDG_CACHE_HOME=$SHARED/cache/serving-home/xdg-cache VLLM_CACHE_ROOT=$SHARED/cache/serving-home/vllm \
  TRITON_CACHE_DIR=$SHARED/cache/serving-home/triton TORCHINDUCTOR_CACHE_DIR=$SHARED/cache/serving-home/torchinductor
# unified memory: release page-cache pages first so the GPU pool can be allocated
(cd "$ROOT" && "$SHARED/envs/client/bin/python" -m pipeline.page_cache)
echo "$(date +%T) starting $name ($MODEL) on :$PORT"
flock -o -w 1800 "$SHARED/.gpu.lock" bash -c '
  nohup setsid vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" --served-model-name "$SERVED_NAME" \
    --gpu-memory-utilization "$GPU_FRACTION" --max-model-len "$MAX_LEN" --kv-cache-memory-bytes "$KV_BYTES" \
    --allowed-local-media-path "$SHARED" --enable-prefix-caching --enable-prompt-tokens-details $EXTRA_ARGS \
    > "$logf" 2>&1 < /dev/null &
  echo $! > "$pidf"
  for _ in $(seq 360); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { echo "$(date +%T) ready"; exit 0; }
    kill -0 "$(cat "$pidf")" 2>/dev/null || break; sleep 5
  done
  echo "failed to start; see $logf"; tail -20 "$logf"
  kill -TERM -- "-$(cat "$pidf")" 2>/dev/null; rm -f "$pidf"; exit 1'
