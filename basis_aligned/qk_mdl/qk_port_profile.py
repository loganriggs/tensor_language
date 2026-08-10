"""STATIC-FRACTION-BY-DEPTH PROFILE (aggregate list #3).

For each layer L in {2, 3, 4, 6, 9, 13, 17}: how much of layer L's QK pattern
function is static token identity? Machinery ported wholesale from the layer-1
port test (tick 193, qk_l1_port.py) and the shrunk-table variant (tick 204,
qk_l1_lowrank_ctx.py):

  (a) token-conditional mean-residual tables at layer L's lambda-mixed attention
      input, estimated on the disjoint co-occurrence corpus (N_EST=1024 seqs,
      the layer-1 budget), with the tick-204 SHRINKAGE estimator: tau=8 toward
      the embedding prior (wte rows), which fixed rare-token noise at layer 1
      and matters more with depth. All tap layers accumulated in ONE pass.
  (b) port cost: standard 307k FineWeb audit (T=512, baseline 3.0763) with layer
      L's PATTERN INPUT computed from the static tables — both branch scores
      s1, s2 replaced via scores_from_factors (tables -> per-head unit-RMS
      factors -> rotary C/S expansion), exactly the tick-193 substitution.
      Values and everything else stay real.
  (c) destruction floor: layer L's pattern zeroed, matching tick 193 exactly —
      s1 branch scores zeroed (pattern = s1*s2 -> 0), values untouched.
  (d) static fraction = 1 - port/floor, with paired per-position SEs (and
      seq-clustered SEs) on each delta-CE.

GATE (mandatory): layer 1 rerun through this exact code path must reproduce the
published shrunk-table port cost +0.0515 (tick 204 rank0) within 0.002 and the
destruction floor +2.70 (tick 193) within 0.05. If layer 1 fails, deeper layers
do not count and the run aborts.

Positive control: an identity score_patch must audit at EXACTLY zero delta.

--smoke: tables from 16 cooc seqs (~8k tokens), audit 4 seqs (~2k predictions),
layers {1, 2} only, layer-1 gate at loose tolerance (0.05 port / 0.5 floor),
model in bf16 to stay under ~2 GB while another job holds the GPU.
"""
import argparse
import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward, rope_tables, apply_rot
from tier2_folding import scores_from_factors

ap = argparse.ArgumentParser()
ap.add_argument('--smoke', action='store_true')
ARGS = ap.parse_args()

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT_JSON = f'{QK}/qk_port_profile.json'
TAU = 8.0
BASE_PUB = 3.07630                       # tick-192 full-audit baseline
PUB_L1_PORT_SHRUNK = 0.0515              # tick 204, rank0 (shrunk tables)
PUB_L1_FLOOR = 2.70                      # tick 193, s1-branch zeroed
PUB_L1_PORT_RAW = 0.02738                # tick 193, raw-mean tables (context only)

if ARGS.smoke:
    LAYERS = [1, 2]                      # 1 = gate, 2 = plumbing proof
    # Full table budget even in smoke: with a few-thousand-token table the
    # layer-1 port cost is coverage-dominated (+0.37 at 8.8% mass coverage in a
    # 16-seq trial) and the gate can never pass; the accumulation pass only
    # runs blocks 0-2 here, so the full budget still costs ~30 s and < 2 GB.
    N_EST, EST_BATCH = 1024, 2
    N_AUDIT_SEQ, AUD_BATCH = 4, 2
    TOL_PORT, TOL_FLOOR = 0.05, 0.5
    MODEL_DTYPE = torch.bfloat16
else:
    LAYERS = [1, 2, 3, 4, 6, 9, 13, 17]  # 1 first: it is the gate
    N_EST, EST_BATCH = 1024, 4
    N_AUDIT_SEQ, AUD_BATCH = 600, 4
    TOL_PORT, TOL_FLOOR = 0.002, 0.05
    MODEL_DTYPE = torch.float32          # published numbers used the fp32 load

t0 = time.time()


def say(msg):
    print(f'[{time.time() - t0:7.1f}s] {msg}', flush=True)


m, cfg = load_elriggs('bilin18', dtype=MODEL_DTYPE)
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
FINEWEB = torch.from_numpy(
    np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))[:N_AUDIT_SEQ]
COOC = torch.from_numpy(
    np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
say(f'model loaded ({MODEL_DTYPE}); audit {len(FINEWEB)} seqs, est {N_EST} seqs, layers {LAYERS}')

# ---------- one pass over the disjoint cooc corpus, tapping every layer ----------
# Replicates reference_forward's block body exactly (lambdas mix, v1 lamb mix,
# bf16 rope tables, per-head rms then rotary); taps the lambda-mixed input of
# each layer in LAYERS. This is qk_l1_port.block1_input / qk_l2_port.block2_input
# generalized to arbitrary taps in a single pass.
MAXTAP = max(LAYERS)


@torch.no_grad()
def tapped_inputs(idx):
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    v1 = None
    B, T = idx.shape
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    taps = {}
    for li in range(MAXTAP + 1):
        blk = m.transformer.h[li]
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        if li in LAYERS:
            taps[li] = x
        if li == MAXTAP:
            break
        a = blk.attn
        hcur = F.rms_norm(x, (x.size(-1),))

        def qk(lin):
            z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,))
            return apply_rot(z, cosb, sinb)

        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k = qk(a.c_q), qk(a.c_k)
        q2, k2 = qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
        x = x + a.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    return taps


say('accumulating token-conditional mean residuals (one pass, all tap layers)...')
sum_x = {L: torch.zeros(V, D, device=DEV) for L in LAYERS}
cnt = torch.zeros(V, device=DEV)
with torch.no_grad():
    for i in range(0, N_EST, EST_BATCH):
        b = COOC[i:i + EST_BATCH].to(DEV)
        idx = b[:, :-1]
        taps = tapped_inputs(idx)
        ids = idx.reshape(-1)
        for L in LAYERS:
            sum_x[L].index_add_(0, ids, taps[L].float().reshape(-1, D))
        cnt.index_add_(0, ids, torch.ones_like(ids, dtype=torch.float))
        del taps
seen = cnt > 0
wte = m.transformer.wte.weight.detach().float()
FQP = (torch.bincount(FINEWEB.flatten(), minlength=V).float().to(DEV) + 0.5)
cov_tok = float(seen.float().mean())
cov_mass = float((FQP * seen.float()).sum() / FQP.sum())
est_positions = N_EST * (COOC.shape[1] - 1)
say(f'tables: {int(seen.sum())}/{V} types seen ({cov_tok * 100:.1f}%), '
    f'{cov_mass * 100:.2f}% of audit token mass, {est_positions} est positions')
# park sums on CPU; per-layer tables are built (and freed) one layer at a time
SUM_CPU = {L: sum_x[L].cpu() for L in LAYERS}
del sum_x
torch.cuda.empty_cache()


def build_tables(L):
    """Shrunk (tau=8 toward the wte embedding prior — tick 204/195 estimator)
    token-conditional tables -> per-head unit-RMS factor tables for layer L."""
    a = m.transformer.h[L].attn
    with torch.no_grad():
        s = SUM_CPU[L].to(DEV)
        mean_x = torch.where(seen[:, None], s / cnt[:, None].clamp_min(1), wte)
        shr = ((cnt / (cnt + TAU))[:, None] * mean_x
               + (TAU / (cnt + TAU))[:, None] * wte)
        xn = F.rms_norm(shr, (D,)).to(m.transformer.wte.weight.dtype)
        tabs = {}
        for name, lin in (('q1', a.c_q), ('k1', a.c_k),
                          ('q2', a.c_q2), ('k2', a.c_k2)):
            z = lin(xn).view(V, NH, HD).float()
            tabs[name] = F.rms_norm(z, (HD,)).contiguous()
        del s, mean_x, shr, xn
    return tabs


# ---------- audits: per-position CE so deltas get paired SEs ----------
@torch.no_grad()
def audit(patch_fn):
    """Returns per-position CE, shape (n_seq, T)."""
    rows = []
    for i in range(0, len(FINEWEB), AUD_BATCH):
        b = FINEWEB[i:i + AUD_BATCH].to(DEV)
        idx = b[:, :-1]
        logits = reference_forward(
            m, idx, 'bf16',
            score_patch=None if patch_fn is None else patch_fn(idx)).float()
        ce = F.cross_entropy(logits.reshape(-1, V), b[:, 1:].reshape(-1),
                             reduction='none')
        rows.append(ce.view(idx.shape[0], -1).cpu())
    return torch.cat(rows)


def paired_stats(ce_arm, ce_base):
    d = (ce_arm - ce_base).reshape(-1).double()
    n = d.numel()
    se_pos = float(d.std(unbiased=True) / math.sqrt(n))
    per_seq = (ce_arm - ce_base).mean(1).double()
    se_seq = float(per_seq.std(unbiased=True) / math.sqrt(per_seq.numel()))
    return float(d.mean()), se_pos, se_seq


def patch_tables(tabs, L):
    def mk(idx):
        def p(li, s1, s2):
            if li != L:
                return s1, s2
            n1 = scores_from_factors(tabs['q1'], tabs['k1'], idx, HD)
            n2 = scores_from_factors(tabs['q2'], tabs['k2'], idx, HD)
            return n1.to(s1.dtype), n2.to(s2.dtype)
        return p
    return mk


def patch_zero(L):
    """Destruction floor, tick-193 definition: zero the s1 branch scores at
    layer L (pattern = s1*s2 -> 0). Values untouched."""
    def mk(idx):
        def p(li, s1, s2):
            if li != L:
                return s1, s2
            return torch.zeros_like(s1), s2
        return p
    return mk


out = {'mode': 'smoke' if ARGS.smoke else 'full',
       'layers': {}, 'tau': TAU,
       'n_est_seqs': N_EST, 'est_positions': est_positions,
       'coverage_types': round(cov_tok, 4), 'coverage_mass': round(cov_mass, 5),
       'n_audit_seqs': len(FINEWEB),
       'published': {'base_ce': BASE_PUB,
                     'l1_port_shrunk': PUB_L1_PORT_SHRUNK,
                     'l1_port_rawmean': PUB_L1_PORT_RAW,
                     'l1_floor': PUB_L1_FLOOR}}


def dump():
    json.dump(out, open(OUT_JSON, 'w'), indent=2)


say('baseline audit...')
CE_BASE = audit(None)
base_ce = float(CE_BASE.double().mean())
out['base_ce'] = round(base_ce, 5)
say(f'baseline CE {base_ce:.5f} (published {BASE_PUB})')
dump()

# positive control: identity patch must be EXACTLY zero (first audit batch)
say('positive control: identity score_patch...')


def patch_identity(idx):
    return lambda li, s1, s2: (s1, s2)


bsub = FINEWEB[:AUD_BATCH].to(DEV)
with torch.no_grad():
    lg_a = reference_forward(m, bsub[:, :-1], 'bf16', score_patch=None).float()
    lg_b = reference_forward(m, bsub[:, :-1], 'bf16',
                             score_patch=patch_identity(bsub[:, :-1])).float()
ident_err = float((lg_a - lg_b).abs().max())
out['identity_control_max_logit_err'] = ident_err
say(f'identity control max |dlogit| = {ident_err:.2e}')
assert ident_err == 0.0, 'identity patch must be exact'
dump()


def run_layer(L):
    say(f'--- layer {L}: building shrunk tables ---')
    tabs = build_tables(L)
    say(f'layer {L}: port audit (static tables on the pattern input)...')
    ce_port = audit(patch_tables(tabs, L))
    port, port_se, port_se_seq = paired_stats(ce_port, CE_BASE)
    say(f'layer {L}: port dCE {port:+.5f} (pos-SE {port_se:.5f}, seq-SE {port_se_seq:.5f})')
    del tabs, ce_port
    torch.cuda.empty_cache()
    say(f'layer {L}: destruction floor (s1 branch zeroed)...')
    ce_fl = audit(patch_zero(L))
    floor, floor_se, floor_se_seq = paired_stats(ce_fl, CE_BASE)
    say(f'layer {L}: floor dCE {floor:+.5f} (pos-SE {floor_se:.5f}, seq-SE {floor_se_seq:.5f})')
    del ce_fl
    frac = 1.0 - port / floor if floor != 0 else float('nan')
    out['layers'][str(L)] = {
        'port_dce': round(port, 5), 'port_se_pos': round(port_se, 5),
        'port_se_seq': round(port_se_seq, 5),
        'floor_dce': round(floor, 5), 'floor_se_pos': round(floor_se, 5),
        'floor_se_seq': round(floor_se_seq, 5),
        'static_fraction': round(frac, 4),
        'table_seen_types': int(seen.sum()), 'table_est_positions': est_positions}
    say(f'layer {L}: static fraction {frac:.4f}')
    dump()
    return port, floor


# ---------- layer 1 first: the gate ----------
l1_port, l1_floor = run_layer(1)
gate_port_ok = abs(l1_port - PUB_L1_PORT_SHRUNK) <= TOL_PORT
gate_floor_ok = abs(l1_floor - PUB_L1_FLOOR) <= TOL_FLOOR
out['gate'] = {'l1_port_dce': round(l1_port, 5),
               'l1_floor_dce': round(l1_floor, 5),
               'published_port': PUB_L1_PORT_SHRUNK, 'published_floor': PUB_L1_FLOOR,
               'tol_port': TOL_PORT, 'tol_floor': TOL_FLOOR,
               'port_ok': gate_port_ok, 'floor_ok': gate_floor_ok,
               'pass': bool(gate_port_ok and gate_floor_ok)}
say(f"GATE {'PASS' if out['gate']['pass'] else 'FAIL'}: "
    f'port {l1_port:+.5f} vs {PUB_L1_PORT_SHRUNK} (tol {TOL_PORT}), '
    f'floor {l1_floor:+.5f} vs {PUB_L1_FLOOR} (tol {TOL_FLOOR})')
dump()
if not out['gate']['pass'] and not ARGS.smoke:
    say('gate FAILED in full mode — deeper layers would not count; aborting.')
    sys.exit(1)

for L in LAYERS[1:]:
    run_layer(L)

out['done'] = True
dump()
say(f'PORT PROFILE DONE -> {OUT_JSON}')
