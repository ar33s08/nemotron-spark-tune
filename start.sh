#!/usr/bin/env bash
# start.sh — one command: start Nemotron 3.5<｜begin▁of▁sentence｜>
#
#   ./start.sh              video ON, 524,288 context   (default; best measured)
#   VIDEO=0 ./start.sh      video OFF, 1,048,576 context
#   MAXLEN=262144 ./start.sh
#   PORT=8000 ./start.sh
#
# Then: curl -s localhost:18300/v1/models
set -eu
cd "$(dirname "$0")"

# Video ON is the default: measured FASTER than video off, and it keeps video
# understanding. Its ceiling is 524,288 because the encoder cache budget is set
# to the whole context when video is on (nano_nemotron_vl.py:361-363).
VIDEO="${VIDEO:-1}"
MAXLEN="${MAXLEN:-}"
if [ -z "$MAXLEN" ]; then
  if [ "$VIDEO" = "0" ]; then MAXLEN=1048576; else MAXLEN=524288; fi
fi
if [ "$VIDEO" != "0" ] && [ "$MAXLEN" -gt 524288 ]; then
  echo "[start] video=$VIDEO cannot fit $MAXLEN (encoder budget == full context)."
  echo "[start] using 524288, the highest video-on context that boots."
  MAXLEN=524288
fi
# util must be HIGH when video is on: the encoder cache is sized from the context
# and takes its memory before KV does. At 0.80 KV starves (needs 11.47 GiB, 4.68
# available) and the engine exits. 0.90 leaves room for both.
if [ "$VIDEO" = "0" ]; then UTIL="${UTIL:-0.85}"; else UTIL="${UTIL:-0.90}"; fi

echo "[start] context=$MAXLEN  video=$VIDEO  util=$UTIL"
MAXLEN="$MAXLEN" VIDEO="$VIDEO" UTIL="$UTIL" PORT="${PORT:-18300}" \
  bash config/tuned.sh

echo
echo "waiting for the server (load takes ~8-13 min)..."
for _ in $(seq 1 110); do
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