#!/usr/bin/env bash
# sysfix.sh — make the box reliably BOOT the 70GB model.
#
# Root cause of the OOM during load:
#   * vm.min_free_kbytes = 45167 (44MB) -> nearly no reserve on a 121GB box
#   * vm.swappiness = 60 -> kernel eagerly swaps the model pages
#   * swap had 10GB of stale pages from earlier runs
#   * page cache (28GB) from the checkpoint read wasn't dropped before profiling
#
# Fix: real reserve + lazy swap + drop caches before launch. Idempotent.
set -eu

echo "== before =="
cat /proc/sys/vm/min_free_kbytes /proc/sys/vm/swappiness

# 2GB reserve so the profiler always has headroom
sysctl -w vm.min_free_kbytes=2097152
# only swap when really necessary
sysctl -w vm.swappiness=10
# let allocations succeed rather than fail-fast
sysctl -w vm.overcommit_memory=1

# release the stale swap pages
swapoff -a 2>/dev/null || true
swapon -a 2>/dev/null || true

# drop page cache (the 28GB of checkpoint reads)
sync
echo 3 > /proc/sys/vm/drop_caches

echo "== after =="
cat /proc/sys/vm/min_free_kbytes /proc/sys/vm/swappiness
awk '/MemAvailable/{print "available: " int($2/1048576) "GB"}' /proc/meminfo
free -g | sed -n '2,3p'