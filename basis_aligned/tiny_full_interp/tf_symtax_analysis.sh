#!/bin/bash
# ANALYSIS PASS for the symmetric-tax arms.
#
# tf_symtax_chain2.sh trains but does not analyse: the arms it produces have a
# .pt and a .json and NOTHING ELSE -- no fold gate, no induction probe, no
# ladder.  Caught by listing the files rather than assuming the trainer did it.
# Without this, prediction K4 (does the predicate arm's hand-installed
# induction survive removing the query/key cap?) cannot be scored, and 12 new
# checkpoints sit unanalysed while the programme's stated deliverable is
# interpretation.
#
# Runs tf_interp3.py -- the same variant-agnostic ladder every published
# family number came through, separately gated against tf_interp2 on the
# vanilla checkpoint -- on every symmetric-tax arm that lacks an _interp3.json.
#
# GATE: on actual GPU compute processes, per the stale-wrapper failure mode.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
LOG=tf_symtax_analysis.log
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
  local f=( tf_predicate_*_slot32*_interp3.json out_interp3_*.txt RESULTS.md )
  shopt -u nullglob
  [ ${#f[@]} -eq 0 ] && return 0
  git add -- "${f[@]}" >/dev/null 2>&1
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UkpvpG8hGeMVyGrk8er2uc" >/dev/null 2>&1 || return 0
  git push -q origin main >/dev/null 2>&1 && { say "  pushed"; return 0; }
  git fetch -q origin main >/dev/null 2>&1
  git merge --no-edit -q origin/main >/dev/null 2>&1 \
    && git push -q origin main >/dev/null 2>&1 \
    && say "  pushed after merge" || say "  PUSH FAILED"
}

say "=== symmetric-tax analysis pass (pid $$) ==="
gate
n=0
for D in 2 3; do
  for CAP in "" "_noqknorm"; do
    for S in 0 1 2; do
      ST="tf_predicate_d${D}_w128_b8192_s${S}_slot32${CAP}"
      [ -f "${ST}.pt" ] || { say "$ST not trained yet -- skip"; continue; }
      if [ -f "${ST}_interp3.json" ]; then say "$ST already analysed -- skip"; continue; fi
      say "analysing $ST"
      python tf_interp3.py --stem "$ST" >> "out_interp3_${ST}.txt" 2>&1 \
        && { say "  OK"; n=$((n+1)); } || say "  ANALYSIS FAILED $ST"
    done
    push "symmetric-tax analysis: predicate slot-32 depth $D, cap${CAP:-_on}, ladder + fold gate + induction"
  done
done
say "=== analysis pass done, $n newly analysed ==="
