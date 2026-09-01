"""UNNUMBERED / PARKED: target-free reduced-rank output-state repair of ct96.

WHY: ct96 and t120 have nearly collinear signed CE-damage vectors (cosine 0.984), but rung 280
falsified the stronger universal damage-family algebra. A CE vector depends on the true next token and
cannot itself be installed as a legal model correction. This experiment moves the question to the final
normalized residual state, where a frozen activation-to-activation map is executable without targets.

ARM: rebuild the registered ct96 compiled program. On the pre-existing FIT documents only, observe its
final normalized state x and the native teacher state h, and fit
    h_hat_r = x + b_r + (x W_r) U_r^T,
where U_r are the leading residual-state PCA directions and W_r is ridge-fit from centered compiled state.
Freeze {b,W,U}; score ranks r={0,1,4,16,64} on the disjoint 1,000-document census. No target token,
position ID, behavior label, or census row is an input to the correction. The targets are used only after
the frozen map runs, for evaluation. Price is D*(2r+1) stored values and 2*D*r multiplies/token.

REGISTERED PREDICTIONS (CE added above real; LOWER IS BETTER):
  (a) LIVE CONFIG: uncorrected held-out census damage is in [0.040, 0.070].
  (b) FLOOR BREAK: rank-16 damage <= 0.0277 (removes at least half of the +0.0553 anchor).
  (c) CERTIFICATES: rank-16 preserves at least 11/62 registered circuit certificates.
NULL: rank-64 held-out damage > 0.040, meaning final-state affine repair does not expose the common CE
mode even with 148,608 stored values. Rank 64 is an identification ceiling, not an adoption proposal.
ADOPTION: even if (a-c) hold, fresh/OOD and frozen cross-config transfer plus intervention fidelity must
hold in later rungs before this is a verified floor-breaking milestone. Self-reviewed. GPU via bqrunner.
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/tensor_language/basis_aligned/bilinear_quotient')
OUT = ROOT / 'output_state_repair_ct96_results.json'
FACTORS = ROOT / 'output_state_repair_ct96_factors.pt'

if os.environ.get('BQLIB_DRYRUN') == '1':
    needed = [
        ROOT / 'frontier_tail_traj_results.json',
        ROOT / 'circuits/BATTERY.json',
        ROOT / 'census_state_diverse.pt',
        ROOT / 'ops/cevdump_ct96.py',
    ]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        print(f'DRYRUN FAIL: missing {missing}')
        raise SystemExit(1)
    print('DRYRUN OK: rung 281 ct96 target-free output-state repair')
    raise SystemExit(0)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'ops'))
sys.path.insert(0, '/workspace/rspd')

import torch
import torch.nn.functional as F
import census_lib as CN
import cevdump_ct96 as C

D = C.D
RANKS = (0, 1, 4, 16, 64)
FIT_DOCS = 212


@torch.no_grad()
def native_states(rows):
    """Final normalized native states, in the same document/position order as evalV."""
    states = []
    C.SEL['qk_tail_on'] = False
    for i in range(0, rows.shape[0], 4):
        idx = rows[i:i + 4, :256].to(C.DEV)
        x = F.rms_norm(C.m.transformer.wte(idx), (D,))
        x0 = x
        v1 = None
        for block in C.m.transformer.h:
            x, v1 = block(x, v1, x0)
        states.append(F.rms_norm(x, (D,)).to(torch.float16).cpu())
    return torch.cat(states).reshape(-1, D).contiguous()


@torch.no_grad()
def score_rank(compiled, native, targets, base_ce, leaves, refs, rank, mx, my, w, u):
    ces = []
    residual_sse = 0.0
    native_delta_sse = 0.0
    for i in range(0, compiled.shape[0], 2048):
        x = compiled[i:i + 2048].to(C.DEV, dtype=torch.float32)
        h_native = native[i:i + x.shape[0]].to(C.DEV, dtype=torch.float32)
        correction = my
        if rank:
            correction = correction + ((x - mx) @ w[:, :rank]) @ u[:, :rank].T
        h = x + correction
        residual_sse += float((h - h_native).square().sum())
        native_delta_sse += float((x - h_native).square().sum())
        logits = 30 * torch.tanh(C.m.lm_head(h) / 30)
        ce = F.cross_entropy(logits, targets[i:i + x.shape[0]].to(C.DEV), reduction='none')
        ces.append(ce.cpu())
    d = torch.cat(ces).float() - base_ce
    valid = 0
    member_abs = {}
    for tag, member in leaves.items():
        val = float(d[member].abs().mean())
        member_abs[tag] = round(val, 6)
        valid += int(val < 0.5 * refs[tag])
    nscalar = compiled.shape[0] * D
    return d, valid, member_abs, (residual_sse / nscalar) ** 0.5, \
        (residual_sse / max(native_delta_sse, 1e-30)) ** 0.5


@torch.no_grad()
def main():
    started = time.time()
    CN.use_state('census_state_diverse.pt')
    census = CN.rows().cpu()
    base_ce = CN.base_ce().float().cpu()
    nflat = CN.nflat()
    fit = C.FW[C.CA:C.CB, :513].cpu()
    assert fit.shape[0] == FIT_DOCS and census.shape[0] == 1000
    rows = torch.cat([fit, census], dim=0)

    C.CROWS = census
    C.CBASE = base_ce
    C.NFLAT = nflat
    C.ANCH = json.load(open(ROOT / 'frontier_tail_traj_results.json'))
    C.SEL.update({
        'mode': 'norm', 'K': 4608, 'K69': 4608, 'K69MAP': {},
        'skipset': tuple(range(10, 18)), 'motif_off': (), 'clsdmg': True,
        'ext_rows': rows, 'cp_swap': 4608, 'qk_r': 96,
        'qk_rmap': {li: 128 for li in range(2, 10)}, 'qk_tail': True,
        'drop_tailE': True, 'capture_final_state': True,
    })
    print('ARM: ct96 plus frozen reduced-rank output-state repair', flush=True)
    C.main()
    compiled = C.SEL['final_state']
    compiled_ce = C.SEL['cev'].float()
    assert compiled.shape == (rows.shape[0] * 256, D)
    assert compiled_ce.numel() == rows.shape[0] * 256

    native = native_states(rows)
    nfit = FIT_DOCS * 256
    xfit = compiled[:nfit].to(C.DEV, dtype=torch.float32)
    yfit = native[:nfit].to(C.DEV, dtype=torch.float32) - xfit
    mx = xfit.mean(0)
    my = yfit.mean(0)
    xc = xfit - mx
    yc = yfit - my
    gram = xc.T @ xc
    ridge = 1e-3 * float(gram.diagonal().mean())
    cov_y = yc.T @ yc
    evals, evecs = torch.linalg.eigh(cov_y)
    u = evecs[:, -max(RANKS):].flip(1).contiguous()
    z = yc @ u
    w = torch.linalg.solve(gram + ridge * torch.eye(D, device=C.DEV), xc.T @ z)
    explained = evals.flip(0).clamp_min(0)
    explained = torch.cumsum(explained, 0) / explained.sum().clamp_min(1e-12)

    eval_compiled = compiled[nfit:]
    eval_native = native[nfit:]
    targets = census[:, 1:257].reshape(-1).long()
    assert targets.numel() == nflat

    battery = json.load(open(ROOT / 'circuits/BATTERY.json'))['by_tag']
    leaves = {}
    refs = {}
    for tag, receipt in battery.items():
        try:
            leaf = CN.leaf(tag)
        except Exception:
            continue
        member = leaf['member'].long()
        if member.numel() == 0:
            continue
        leaves[tag] = member
        refs[tag] = receipt['mean_ablation']['top'][0]['abs_dce_members']

    baseline_d = compiled_ce[nfit:] - base_ce
    baseline_damage = float(baseline_d.mean())
    del xfit, yfit, xc, yc, gram, cov_y, evals, evecs, z
    torch.cuda.empty_cache()
    arms = {}
    for rank in RANKS:
        d, valid, member_abs, state_rmse, state_fraction = score_rank(
            eval_compiled, eval_native, targets, base_ce, leaves, refs,
            rank, mx, my, w, u,
        )
        arms[str(rank)] = {
            'price_values': D * (2 * rank + 1),
            'multiplies_per_token': 2 * D * rank,
            'census_damage': round(float(d.mean()), 7),
            'certificates_valid': valid,
            'state_rmse': round(state_rmse, 7),
            'state_error_fraction': round(state_fraction, 7),
            'member_abs_dce': member_abs,
        }
        print(f"rank {rank:2d}: damage {arms[str(rank)]['census_damage']:+.6f}, "
              f"certs {valid}/62, state fraction {arms[str(rank)]['state_error_fraction']:.4f}",
              flush=True)

    pred_a = 0.040 <= baseline_damage <= 0.070
    pred_b = arms['16']['census_damage'] <= 0.0277
    pred_c = arms['16']['certificates_valid'] >= 11
    null_triggered = arms['64']['census_damage'] > 0.040
    result = {
        'convention': 'CE added above native model; lower is better',
        'fit_documents': FIT_DOCS,
        'heldout_census_documents': 1000,
        'target_free_map': True,
        'baseline_census_damage': round(baseline_damage, 7),
        'residual_pca_cumulative_explained': {
            str(r): round(float(explained[r - 1]), 7) for r in RANKS if r > 0
        },
        'arms': arms,
        'pred_a_live_config': bool(pred_a),
        'pred_b_floor_break_rank16': bool(pred_b),
        'pred_c_certificates_rank16': bool(pred_c),
        'null_triggered': bool(null_triggered),
        'decision_level': 'identification; adoption still requires fresh/OOD, transfer, and interventions',
        'runtime_s': round(time.time() - started, 1),
    }
    torch.save({
        'mx': mx.cpu().half(), 'my': my.cpu().half(),
        'w': w.cpu().half(), 'u': u.cpu().half(),
        'ranks': RANKS, 'fit_docs': FIT_DOCS,
    }, FACTORS)
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'arms'}, indent=2), flush=True)
    print(f'wrote {OUT} and {FACTORS}', flush=True)
    if not pred_a:
        raise SystemExit('INSTRUMENT FAIL: ct96 baseline outside registered band')


if __name__ == '__main__':
    main()
