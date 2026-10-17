#!/usr/bin/env bash
# start.sh — one command: start Nemotron 3.5 Super VL on this DGX Spark.
#
#   ./start.sh              full 1M context, video off
#   VIDEO=1 ./start.sh      262,144 context, video on
#   PORT=8000 ./start.sh    different port
#
# Then: curl -s localhost:18300/v1/models
set -eu
cd "$(dirname "$0")"

MAXLEN="${MAXLEN:-1048576}"
VIDEO="${VIDEO:-0}"
# video on cannot fit 1M (encoder budget == whole context), so default it down.
if [ "$VIDEO" != "0" ] && [ "${MAXLEN:-}" = "1048576" ]; then
  echo "[start] video=$VIDEO with 1M: lowering MAXLEN to 262144 (else it will not start)"
  MAXLEN=262144
fi

MAXLEN="$MAXLEN" VIDEO="$VIDEO" UTIL="${UTIL:-0.85}" PORT="${PORT:-18300}" \
  bash config/tuned.sh

echo
echo "waiting for the server (load takes ~8-10 min)..."
for _ in $(seq 1 90); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 \
         "http://127.0.0.1:${PORT:-18300}/health" || true)
  if [ "$code" = "200" ]; then
    echo "[start] up: http://127.0.0.1:${PORT:-18300}/v1  (model: nemotron-3.5-super-vl)"
    exit 0
  fi
  sleep 10
done
echo "[start] did not come up in time. check: docker logs supervl" >&2
exit 1