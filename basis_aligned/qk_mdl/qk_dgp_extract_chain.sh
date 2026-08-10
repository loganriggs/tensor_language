#!/bin/bash
# Chain wrapper for archetype extraction into a standalone machine
# (qk_dgp_extract.py; aggregate list #5).  Registered predictions:
# qk_dgp_extract_predictions.json (E1-E4; E4 gates).  Order: gate on a free
# GPU and no other qk_dgp/tf chain, smoke first, then oracle (E4, hard stop
# on failure), then the identifiable recovered assembly (E1), the overlap
# recovered assembly (E3), and the sub-additivity ablations (E2), committing
# and pushing after each stage with the numbers.  Bracket-trick pgrep
# patterns: substring pgrep self-matches killed us twice (AGENT_BRIEF).
set -u
cd /workspace/tensor_language/basis_aligned/qk_mdl || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
LOG=qk_dgp_extract_chain.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

gate() {
  local ok=0 napps free otherchains
  say "gating: need 0 GPU apps AND >=8000 MiB free AND no other qk_dgp/tf chain, 3 consecutive 60s checks"
  while true; do
    napps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    otherchains=$(pgrep -fc 'qk_dgp_chain[.]sh|tf_v2_transition_chain[.]sh|tf_v4096_seeds35_chain[.]sh' || true)
    if [ "$napps" -eq 0 ] && [ "${free:-0}" -ge 8000 ] && [ "${otherchains:-0}" -eq 0 ]; then
      ok=$((ok + 1))
    else
      ok=0
    fi
    say "  gate: gpu_apps=$napps free=${free}MiB other_chains=${otherchains:-0} ok=$ok"
    [ "$ok" -ge 3 ] && break
    sleep 60
  done
  say "gate OPEN"
}

commit_push() {
  # $1 = files to add (space-separated), $2 = message
  git add $1 "$LOG" >/dev/null 2>&1
  git commit -q -m "$2

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UkpvpG8hGeMVyGrk8er2uc" >/dev/null 2>&1 \
    && say "committed" || say "nothing to commit / commit failed"
  git push -q origin main >/dev/null 2>&1 && say "pushed" || say "PUSH FAILED (fix with fetch+rebase, never stash)"
}

stage_numbers() {
  # $1 = stage key -> one-line headline numbers from qk_dgp_extract.json
  python - "$1" <<'PY'
import json, sys
k = sys.argv[1]
try:
    r = json.load(open('qk_dgp_extract.json'))
except Exception:
    print('no JSON found'); raise SystemExit
out = []
if k == 'oracle':
    for vn, d in r.get('E4', {}).items():
        if isinstance(d, dict):
            out.append(f"{vn}: gap {d.get('gap_vs_law_paired')} {'PASS' if d.get('pass') else 'FAIL'}")
    out.append(f"gate {r.get('E4', {}).get('gate')}")
elif k in ('identifiable', 'overlap'):
    key = 'E1' if k == 'identifiable' else 'E3'
    for arm, d in r.get(key, {}).get(k, {}).items():
        if isinstance(d, dict) and 'machine_ce_blocked' in d:
            out.append(f"{arm}: machine {d['machine_ce_blocked']} vs model {d.get('model_ce')} "
                       f"(gap {d.get('machine_minus_model')})")
elif k == 'subadd':
    for nm, d in r.get('E2', {}).items():
        if isinstance(d, dict) and 'damage_ratio_shared_over_single_matched' in d:
            out.append(f"{nm}: shared dCE {d.get('zero_shared', {}).get('dce')} vs matched-single "
                       f"{d.get('zero_single_energy_matched', {}).get('dce')} ratio "
                       f"{d.get('damage_ratio_shared_over_single_matched')}")
        elif nm == 'oracle_reference' and isinstance(d, dict):
            out.append(f"oracle ref dCE {d.get('mean_dce')}")
print('; '.join(out) if out else 'stage produced no numbers (see JSON)')
PY
}

say "=== archetype-extraction chain (pid $$) ==="
gate

say "smoke run (oracle + tiny recovered assembly)"
if QK_SMOKE=1 python qk_dgp_extract.py > qk_dgp_extract_smoke.out 2>&1; then
  say "smoke OK"
else
  say "SMOKE FAILED (see qk_dgp_extract_smoke.out) — stopping chain"
  commit_push "qk_dgp_extract.py qk_dgp_extract_chain.sh qk_dgp_extract_smoke.json qk_dgp_extract_smoke.out" \
    "archetype-extraction chain: smoke FAILED, chain stopped — see qk_dgp_extract_smoke.out"
  exit 1
fi
commit_push "qk_dgp_extract.py qk_dgp_extract_chain.sh qk_dgp_extract_smoke.json qk_dgp_extract_smoke.out" \
  "archetype extraction (aggregate #5) smoke pass: oracle assembly + tiny recovered assembly ran end to end"

run_stage() {
  local ST=$1
  say "running qk_dgp_extract.py --stage $ST"
  if python qk_dgp_extract.py --stage "$ST" >> qk_dgp_extract.out 2>&1; then
    say "stage $ST finished OK"
  else
    say "stage $ST FAILED or hard-stopped (see qk_dgp_extract.out) — committing whatever landed"
    NUMS=$(stage_numbers "$ST")
    commit_push "qk_dgp_extract.json qk_dgp_extract.out" \
      "archetype extraction stage ${ST} stopped: ${NUMS} — full numbers in qk_dgp_extract.json"
    if [ "$ST" = oracle ]; then
      say "E4 gate did not pass — E1-E3 uninterpretable; chain stops per registration"
      exit 1
    fi
    return
  fi
  NUMS=$(stage_numbers "$ST")
  say "stage $ST numbers: $NUMS"
  commit_push "qk_dgp_extract.json qk_dgp_extract.out" \
    "archetype extraction stage ${ST}: ${NUMS} — full numbers in qk_dgp_extract.json"
}

for ST in oracle identifiable overlap subadd; do
  run_stage "$ST"
done
say "=== chain done ==="
