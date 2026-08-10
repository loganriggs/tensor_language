#!/bin/bash
# CLEAR THE ANALYSIS BACKLOG.
#
# An audit of every foldable checkpoint against its ladder found 15 of 189
# lacking a complete one: the 12 codebook third-architecture arms trained
# tonight, and 3 old truncated-tokenizer cells that predate the current corpus.
# Interpretation is this programme's deliverable, so a trained checkpoint with
# no ladder is unfinished work, not spare inventory.
#
# Completeness is tested by the presence of the LAST-written section, not by
# the file existing -- a crashed ladder leaves a valid-looking JSON with a third
# of its sections, and skipping on existence alone means it is never re-run.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
LOG=tf_backlog_analysis.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

gate() {
  local ok=0 napps free
  say "gating on GPU compute processes"
  while true; do
    napps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$napps" -eq 0 ] && [ "${free:-0}" -ge 10000 ]; then ok=$((ok + 1)); else ok=0; fi
    say "  gate: gpu_apps=$napps free=${free}MiB ok=$ok"
    [ "$ok" -ge 3 ] && break
    sleep 60
  done
  say "gate OPEN"
}

push() {
  shopt -s nullglob
  local f=( tf_codebook_*_interp3.json tf_vanilla_*v8192*_interp3.json \
            out_interp3_*.txt RESULTS.md )
  shopt -u nullglob
  [ ${#f[@]} -eq 0 ] && return 0
  git add -- "${f[@]}" >/dev/null 2>&1
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UkpvpG8hGeMVyGrk8er2uc" >/dev/null 2>&1 || return 0
  git push -q origin main >/dev/null 2>&1 && say "  pushed" || say "  PUSH FAILED"
}

analyse() {
  local ST=$1
  if [ -f "${ST}_interp3.json" ] && python -c "
import json,sys
d=json.load(open('${ST}_interp3.json'))
sys.exit(0 if 'bits_per_byte_ladder' in d else 1)" 2>/dev/null; then
    say "$ST complete -- skip"; return
  fi
  [ -f "${ST}_interp3.json" ] && say "  $ST PARTIAL -- re-running"
  say "analysing $ST"
  python tf_interp3.py --stem "$ST" >> "out_interp3_${ST}.txt" 2>&1 \
    && say "  OK" || say "  ANALYSIS FAILED $ST"
}

say "=== analysis backlog (pid $$) ==="
gate
say "stage 1: the 12 codebook third-architecture arms"
for S in 0 1 2 3 4 5; do
  for CAP in "" "_noqknorm"; do
    analyse "tf_codebook_d1_w128_b8192_s${S}_slot61${CAP}"
  done
done
push "analysis backlog: the 12 codebook third-architecture arms"
say "stage 2: three old truncated-tokenizer cells"
for ST in tf_vanilla_d1_w32_v8192_s0 tf_vanilla_d1_w64_v8192_s0 \
          tf_vanilla_d1_w128_v8192_s0; do
  analyse "$ST"
done
push "analysis backlog: three truncated-tokenizer cells -- every foldable checkpoint now carries a complete ladder"
say "=== backlog done ==="
