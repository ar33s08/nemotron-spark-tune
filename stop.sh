#!/usr/bin/env bash
# stop.sh — stop the server and wait for memory to come back.
set -eu
NAME="${NAME:-supervl}"
docker rm -f "$NAME" >/dev/null 2>&1 || true
sync
for _ in $(seq 1 60); do
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  [ "$avail" -ge 112 ] && { echo "[stop] stopped, ${avail}GB free"; exit 0; }
  sleep 5
done
echo "[stop] stopped. warning: only ${avail}GB free (want >=112)" >&2