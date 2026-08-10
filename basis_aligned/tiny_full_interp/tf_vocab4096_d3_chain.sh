#!/bin/bash
# VOCABULARY-SIZE CHECK of the dissolution trend: cap-off 2x2 at depth 3
# width 128, V=4096, seeds 0-2 (12 cells).
#
# Written directly, not derived; full arm list spelled out.
# Predictions: tf_vocab4096_d3_predictions.json, registered before this file.
#
# CONCURRENCY EXPERIMENT (Logan, 2026-08-10): the three seeds of each arm train
# CONCURRENTLY (~2.5 GB each, small kernels under-occupy the card). Wall time
# per group is logged; sequential baseline is ~5.5-6 min per training, so a
# group beating 16.5 min wins. Probes stay sequential after each group (they
# are seconds, and probe determinism is not worth co-tenancy questions).
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
LOG=tf_vocab4096_d3_chain.log
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
  local f=( tff_*_d3_w128_b4096_s[012]_noqknorm*.json tfb_std7_d3_w128_b4096_s[012]_noqknorm*.json \
            out_*d3_w128_b4096*.txt RESULTS.md )
  shopt -u nullglob
  [ ${#f[@]} -eq 0 ] && return 0
  git add -- "${f[@]}" >/dev/null 2>&1
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UkpvpG8hGeMVyGrk8er2uc" >/dev/null 2>&1 || return 0
  git push -q origin main >/dev/null 2>&1 && say "  pushed" || say "  PUSH FAILED"
}

# arm_group CMDPREFIX STEMPREFIX -- trains seeds 0-2 concurrently, then probes sequentially
arm_group() {
  local stem_prefix=$1; shift
  local t0=$SECONDS pids=() S
  for S in 0 1 2; do
    local ST="${stem_prefix/SEED/$S}"
    if [ -f "${ST}.pt" ]; then say "$ST exists -- skip"; continue; fi
    say "training $ST (concurrent)"
    "$@" --seed "$S" >> "out_${ST}.txt" 2>&1 &
    pids+=($!)
  done
  local rc=0 p
  for p in "${pids[@]:-}"; do [ -n "$p" ] && { wait "$p" || rc=1; }; done
  say "  group wall time $((SECONDS - t0)) s (sequential baseline ~1000 s)"
  [ "$rc" -ne 0 ] && say "  TRAIN FAILED in group ${stem_prefix}"
  for S in 0 1 2; do
    local ST="${stem_prefix/SEED/$S}"
    [ -f "${ST}.pt" ] || continue
    if [ ! -f "${ST}_induction.json" ]; then
      if [[ "$ST" == tfb_* ]]; then
        python tf_baseline_probe.py --stem "$ST" >> "out_${ST}_probe.txt" 2>&1 \
          && say "  probe OK $ST" || say "  PROBE FAILED $ST"
      else
        python tf_factorial_probe.py --stem "$ST" >> "out_${ST}_probe.txt" 2>&1 \
          && say "  probe OK $ST" || say "  PROBE FAILED $ST"
      fi
    fi
  done
}

say "=== vocab-4096 depth-3 trend check (pid $$) ==="
gate
arm_group "tff_bilin_bilin_d3_w128_b4096_sSEED_noqknorm" \
  python tf_factorial.py cell --attn bilin --mlp bilin --depth 3 --width 128 --vocab 4096 --suffix _noqknorm --no-qk-norm
arm_group "tff_bilin_gelu_d3_w128_b4096_sSEED_noqknorm" \
  python tf_factorial.py cell --attn bilin --mlp gelu --depth 3 --width 128 --vocab 4096 --suffix _noqknorm --no-qk-norm
arm_group "tff_softmax_bilin_d3_w128_b4096_sSEED_noqknorm" \
  python tf_factorial.py cell --attn softmax --mlp bilin --depth 3 --width 128 --vocab 4096 --suffix _noqknorm --no-qk-norm
arm_group "tfb_std7_d3_w128_b4096_sSEED_noqknorm" \
  python tf_baseline_std.py cell --exp 7 --depth 3 --width 128 --vocab 4096 --suffix _noqknorm --no-qk-norm
push "vocab-4096 depth-3 trend check: all four cap-off corners at depth 3 width 128, seeds 0-2"
say "=== vocab-4096 depth-3 check done ==="
