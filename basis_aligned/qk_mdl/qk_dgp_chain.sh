#!/bin/bash
# Chain wrapper for the planted-modular DGP experiment (qk_dgp_modular.py;
# aggregate list #4).  Registered predictions + 2026-08-10 amendment:
# qk_dgp_modular_predictions.json.  Order: gate on a free GPU and no tf_*
# chain, run the ANALYTIC CONTROLS for both variants first (cheap, hard-stop
# rules per variant), then the four cells (identifiable/overlap x
# semi/learned), committing and pushing after each with the numbers.
# Bracket-trick pgrep patterns: substring pgrep self-matches killed us twice
# (AGENT_BRIEF).
set -u
cd /workspace/tensor_language/basis_aligned/qk_mdl || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
LOG=qk_dgp_chain.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

gate() {
  local ok=0 napps free tfchains
  say "gating: need 0 GPU apps AND >=10000 MiB free AND no tf_* chain, 3 consecutive 60s checks"
  while true; do
    napps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    tfchains=$(pgrep -fc 'tf_v2_transition_chain[.]sh|tf_v4096_seeds35_chain[.]sh' || true)
    if [ "$napps" -eq 0 ] && [ "${free:-0}" -ge 10000 ] && [ "${tfchains:-0}" -eq 0 ]; then
      ok=$((ok + 1))
    else
      ok=0
    fi
    say "  gate: gpu_apps=$napps free=${free}MiB tf_chains=${tfchains:-0} ok=$ok"
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

cell_numbers() {
  # $1 = variant, $2 = arm -> one-line headline numbers from the variant JSON
  python - "$1" "$2" <<'PY'
import json, sys
v, a = sys.argv[1], sys.argv[2]
try:
    r = json.load(open(f'qk_dgp_modular_{v}.json'))
except Exception:
    print('no JSON found')
    raise SystemExit
out = []
g = r.get('arms', {}).get(a, {}).get('task_gate', {})
if g:
    out.append(f"held CE {g.get('held_ce')} paired gap {g.get('gap_vs_true_law_paired')} "
               f"task gate {'PASS' if g.get('pass') else 'FAIL'}")
if a == 'semi' and 'D1' in r:
    d = r['D1']
    out.append(f"D1 frac>=0.9 {d.get('frac_units_matched_cos_ge_0.9')} "
               f"(ceiling {d.get('noiseless_ceiling_frac_ge_0.9')})")
if a == 'learned' and 'D2' in r:
    out.append(f"D2 frac>=0.9 {r['D2'].get('frac_units_matched_cos_ge_0.9')}")
if a == 'semi' and 'D3' in r:
    out.append(f"D3 codable {r['D3'].get('codable_n')}/{r['D3'].get('total_coords')}")
if a == 'semi' and 'D4' in r:
    out.append(f"D4 sel-vs-pca top2 {r['D4'].get('privilege_ratio_sel_vs_pca_top2')} "
               f"sel-vs-shrink top2 {r['D4'].get('privilege_ratio_sel_vs_shrink_top2')}")
print('; '.join(out) if out else 'stage skipped (see JSON gates)')
PY
}

say "=== planted-modular DGP chain (pid $$) ==="
gate

say "analytic controls, both variants (hard-stop rules per variant)"
python qk_dgp_modular.py --controls-only >> qk_dgp_controls.out 2>&1 \
  || say "controls-only invocation FAILED (see qk_dgp_controls.out)"
ID_PASS=$(python -c "import json; print(json.load(open('qk_dgp_controls.json'))['identifiable']['pass'])" 2>/dev/null || echo False)
OV_PASS=$(python -c "import json; print(json.load(open('qk_dgp_controls.json'))['overlap']['pass'])" 2>/dev/null || echo False)
say "controls: identifiable pass=$ID_PASS overlap pass=$OV_PASS"
commit_push "qk_dgp_controls.json qk_dgp_controls.out" \
  "planted-modular DGP analytic controls: identifiable pass=$ID_PASS, overlap pass=$OV_PASS — numbers in qk_dgp_controls.json"

run_cell() {
  local VAR=$1 ARM=$2
  say "running qk_dgp_modular.py --variant $VAR --arm $ARM"
  if python qk_dgp_modular.py --variant "$VAR" --arm "$ARM" >> "qk_dgp_modular_${VAR}.out" 2>&1; then
    say "cell $VAR/$ARM finished OK"
  else
    say "cell $VAR/$ARM FAILED (see qk_dgp_modular_${VAR}.out) — committing whatever landed"
  fi
  NUMS=$(cell_numbers "$VAR" "$ARM")
  say "cell $VAR/$ARM numbers: $NUMS"
  commit_push "qk_dgp_modular_${VAR}.json qk_dgp_modular_${VAR}.out" \
    "planted-modular DGP ${VAR}/${ARM} run complete: ${NUMS} — full numbers in qk_dgp_modular_${VAR}.json"
}

for VAR in identifiable overlap; do
  PASS_VAR=$([ "$VAR" = identifiable ] && echo "$ID_PASS" || echo "$OV_PASS")
  if [ "$PASS_VAR" != "True" ]; then
    say "variant $VAR: analytic control FAILED under its rule — hard stop, skipping its cells"
    continue
  fi
  for ARM in semi learned; do
    run_cell "$VAR" "$ARM"
  done
done
say "=== chain done ==="
