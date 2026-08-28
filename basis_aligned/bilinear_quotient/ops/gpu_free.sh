#!/bin/bash
# gpu_free.sh -- print the GPU's actual occupancy and exit nonzero if anything is on it.
#
# Written after I appended a queue entry while Codex had a collector resident, because my shell
# habit was `nvidia-smi --query-compute-apps ...; echo "(empty=free)"` -- an unconditional label
# that says "free" whether or not the command above it printed processes. Same defect family as
# LESSON 41's addendum: a signal that cannot report the state it names.
#
# Exits 0 only when no compute process is resident, so `ops/gpu_free.sh && echo path >> queue.txt`
# is safe to chain.
set -u
apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)
read -r used total < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits \
    2>/dev/null | tr -d ',')
if [ -z "$apps" ]; then
    echo "GPU FREE  (${used}/${total} MiB used)"
    exit 0
fi
echo "GPU BUSY  (${used}/${total} MiB used) -- resident:"
while IFS= read -r line; do
    pid=${line%%,*}
    cmd=$(ps -o cmd= -p "$pid" 2>/dev/null | head -c 90)
    echo "    ${line}   ${cmd:-<gone>}"
done <<< "$apps"
exit 1
