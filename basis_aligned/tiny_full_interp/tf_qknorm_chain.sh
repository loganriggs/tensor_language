#!/bin/bash
# THE FOLDABLE FAMILY'S QUERY/KEY-NORM CONTROL.
#
# The conventional baseline's query/key-norm control came in enormous: removing
# per-head query/key RMSNorm buys the conventional model 0.171 nats of held
# cross-entropy and takes its induction score from +0.1356 to +1.2834 (model-
# seed t = 81.7).  Query/key norm was inherited from THIS FAMILY's history and
# imposed on the conventional arm, so the conventional model has now been
# un-handicapped and the foldable family has not.  Until this chain runs, the
# comparison is asymmetric and no tax number should be quoted from it.
#
# Runs the foldable arm at qk_norm=False through tf_factorial.py's (bilin,
# bilin) path -- which gate G1 shows reproduces tf_model.TinyBilin vanilla
# bit-for-bit, and which is parameter-identical either way (body 590,080).
# tf_model.py is NOT edited: it hardcodes query/key norm and is imported by
# chains that are running.
#
# Predictions registered in tf_qknorm_predictions.json BEFORE this file existed.
#
# GATED behind BOTH the baseline chain and the factorial chain.  Exact-name
# pgrep with a [c]haracter class so the gate cannot self-match.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
LOG=tf_qknorm_chain.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

gate() {
  local ok=0 busy free
  say "gating on the baseline and factorial chains + a quiet card"
  while true; do
    busy=0
    pgrep -f -x "/bin/bash ./tf_baseline_chain.sh"   > /dev/null && busy=1
    pgrep -f -x "/bin/bash ./tf_factorial_chain.sh"  > /dev/null && busy=1
    pgrep -f -- 'python tf_[b]aseline_std\.py'       > /dev/null && busy=1
    pgrep -f -- 'python tf_[f]actorial\.py'          > /dev/null && busy=1
    pgrep -f -- 'python tf_[t]rain\.py'              > /dev/null && busy=1
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
           | head -1)
    if [ "$busy" -eq 0 ] && [ "${free:-0}" -ge 10000 ]; then
      ok=$((ok + 1))
    else
      ok=0
    fi
    say "  gate: busy=$busy free=${free}MiB consecutive_ok=$ok"
    [ "$ok" -ge 3 ] && break
    sleep 120
  done
  say "gate OPEN"
}

push() {
  shopt -s nullglob
  local files=( tff_bilin_bilin_*_noqknorm*.json out_tff_*noqknorm*.txt \
                tf_qknorm_predictions.json tf_qknorm_table.md RESULTS.md )
  shopt -u nullglob
  [ ${#files[@]} -eq 0 ] && { say "  nothing to stage"; return 0; }
  git add -- "${files[@]}" > /dev/null 2>&1
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UkpvpG8hGeMVyGrk8er2uc" \
    > /dev/null 2>&1 || { say "  nothing to commit"; return 0; }
  git push -q origin main > /dev/null 2>&1 && { say "  pushed"; return 0; }
  git fetch -q origin main > /dev/null 2>&1
  git merge --no-edit -q origin/main > /dev/null 2>&1 \
    && git push -q origin main > /dev/null 2>&1 \
    && { say "  pushed after merge"; return 0; }
  say "  PUSH FAILED (next stage retries)"
}

run_seed() {
  local S=$1
  local STEM="tff_bilin_bilin_d2_w128_b8192_s${S}_noqknorm"
  if [ -f "${STEM}.pt" ]; then
    say "$STEM exists -- skip training"
  else
    say "training ${STEM}"
    python tf_factorial.py cell --attn bilin --mlp bilin --depth 2 \
      --width 128 --seed "$S" --suffix _noqknorm --no-qk-norm \
      >> "out_${STEM}.txt" 2>&1 || { say "  TRAIN FAILED ${STEM}"; return; }
    say "  trained"
  fi
  if [ -f "${STEM}_induction.json" ]; then
    say "  $STEM already probed -- skip"
  else
    python tf_factorial_probe.py --stem "$STEM" \
      >> "out_${STEM}_probe.txt" 2>&1 && say "  probe OK" \
      || say "  PROBE FAILED ${STEM}"
  fi
}

say "=== foldable query/key-norm control start (pid $$) ==="
gate
for S in 0 1 2; do
  run_seed "$S"
  push "foldable query/key-norm control: depth 2 width 128 seed $S"
done
python tf_qknorm_report.py >> tf_qknorm_report_stdout.txt 2>&1 \
  && say "report OK" || say "REPORT FAILED"
push "foldable query/key-norm control: FINAL -- scored against tf_qknorm_predictions.json"
say "=== foldable query/key-norm control done ==="
