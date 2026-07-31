"""CERTIFIED RESTRICTED-CORE PROXY -- part 2: certification controls.

Adds to qk_certified_proxy.json (same subsample, same 24 candidates, truth loaded
from part 1 -- deterministic, harness-verified against the census there):
  (a) RANDOM-BASIS control: restricted cores of the SAME ranks (K_in=288 x K_out=144)
      but random orthonormal input/output bases instead of train-gram bases. If the
      causal-prediction fidelity is basis-specific, this should collapse.
  (b) CAPACITY sweep: half cores (144x72) and double cores (576x288). Does rank
      restore the one part-1 miss (h.L9.7: true +0.017, restricted-288 -0.001)?
All core math / folding / forward VERBATIM from qk_certified_proxy.py.
"""
import json, sys, math, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'

def gpu_guard(min_free=4500, tries=45, sleep=20):
    for _ in range(tries):
        free = int(subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits']
        ).decode().split('\n')[0].strip())
        if free >= min_free:
            print(f"GPU guard: {free} MiB free -- proceeding.", flush=True); return
        print(f"GPU guard: only {free} MiB free (<{min_free}); sleeping {sleep}s ...", flush=True)
        time.sleep(sleep)
    raise RuntimeError("GPU guard timed out waiting for free memory")
gpu_guard()

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h); N_SVD = 4
DFF = cfg['expansion_factor'] * D
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
TRAIN = FINEWEB[0:256, :SEQL].to(DEV)
HELD = FINEWEB[448:600, :SEQL].to(DEV)
NHELD = HELD.shape[0]; BATCH = 6; KCAUSAL = 200

MAIN = json.load(open(f'{QK}/qk_certified_proxy.json'))
SUBIDX = np.array(MAIN['meta']['subsample_rows'])
SUB = HELD[torch.from_numpy(SUBIDX).to(DEV)]
NSUB = SUB.shape[0]
recs1 = {r['comp']: r for r in MAIN['records']}
CAND_NAMES = [r['comp'] for r in MAIN['records']]
def parse(comp):
    if comp.startswith('h.'):
        _, l, h = comp.split('.'); return ('head', int(l[1:]), int(h))
    _, l, d = comp.split('.'); return ('mlp', int(l[1:]), int(d[1:]))
CANDS = [(c,) + parse(c) for c in CAND_NAMES]
def rkey(kind, li): return li if kind == 'head' else li + 1
KEYS = sorted({rkey(k, l) for _, k, l, _ in CANDS if rkey(k, l) < NL})

# ---- TRAIN gram pass (verbatim part 1) ----
gram_in = [torch.zeros(D, D, device=DEV) for _ in range(NL)]
gram_out = [torch.zeros(D, D, device=DEV) for _ in range(NL)]
@torch.no_grad()
def fwd_gram(idx):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        gram_in[li] += torch.einsum('btd,bte->de', x, x)
        mo = blk.mlp(F.rms_norm(x, (D,)))
        gram_out[li] += torch.einsum('btd,bte->de', mo, mo)
        x = x + mo
print("TRAIN gram pass ...", flush=True)
for i in range(0, TRAIN.shape[0], BATCH): fwd_gram(TRAIN[i:i+BATCH])
mlp_dirs = torch.zeros(NL, N_SVD, D, device=DEV)
EVIN = {}; EVOUT = {}
for li in range(NL):
    _e, evecs = torch.linalg.eigh(gram_out[li])
    mlp_dirs[li] = evecs[:, -N_SVD:].T.flip(0)
    EVOUT[li] = evecs.flip(1).contiguous()
    _e, evecs = torch.linalg.eigh(gram_in[li])
    EVIN[li] = evecs.flip(1).contiguous()
del gram_in, gram_out
torch.cuda.empty_cache()

# random orthonormal bases (control), one Q per layer per side, seeded
g = torch.Generator(device=DEV).manual_seed(7)
QIN = {li: torch.linalg.qr(torch.randn(D, D, generator=g, device=DEV))[0] for li in range(NL)}
QOUT = {li: torch.linalg.qr(torch.randn(D, D, generator=g, device=DEV))[0] for li in range(NL)}

# ---- HELD PASS A: per-position means (verbatim; no activation collection needed) ----
YH_SUM = {li: torch.zeros(SEQL, NH, HD, device=DEV) for li in range(NL)}
PROJ_SUM = {li: torch.zeros(SEQL, N_SVD, device=DEV) for li in range(NL)}
X_SUM = {li: torch.zeros(SEQL, D, device=DEV) for li in range(NL)}
MO_SUM = {li: torch.zeros(SEQL, D, device=DEV) for li in range(NL)}
@torch.no_grad()
def fwd_passA(idx):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        YH_SUM[li] += yh4.sum(0)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); X_SUM[li] += x.sum(0)
        mo = blk.mlp(F.rms_norm(x, (D,))); MO_SUM[li] += mo.sum(0)
        PROJ_SUM[li] += torch.einsum('btd,nd->btn', mo, mlp_dirs[li]).sum(0)
        x = x + mo
print("HELD PASS A ...", flush=True)
for i in range(0, NHELD, BATCH): fwd_passA(HELD[i:i+BATCH])
YHMEAN = {li: YH_SUM[li] / NHELD for li in range(NL)}
PROJMEAN = {li: PROJ_SUM[li] / NHELD for li in range(NL)}
MX = {li: X_SUM[li] / NHELD for li in range(NL)}
MOMEAN = {li: MO_SUM[li] / NHELD for li in range(NL)}
del YH_SUM, PROJ_SUM, X_SUM, MO_SUM

def build_core(k_in, k_out, basis):
    core = {}
    for li in range(NL):
        blk = m.transformer.h[li]
        Pin = (EVIN[li] if basis == 'gram' else QIN[li])[:, :k_in].contiguous()
        Pout = (EVOUT[li] if basis == 'gram' else QOUT[li])[:, :k_out].contiguous()
        mx = MX[li]
        mx_par = mx @ Pin
        mx_perp = mx - mx_par @ Pin.T
        WL, WR, WD = blk.mlp.Left.weight, blk.mlp.Right.weight, blk.mlp.Down.weight
        core[li] = {'Pin': Pin, 'Pout': Pout,
                    'perp_sq': (mx_perp*mx_perp).sum(-1),
                    'uL': mx_perp @ WL.T, 'uR': mx_perp @ WR.T,
                    'AL': WL @ Pin, 'AR': WR @ Pin,
                    'Dc': Pout.T @ WD,
                    'bc': (blk.mlp.Down_bias.unsqueeze(0) - MOMEAN[li]) @ Pout}
    return core

@torch.no_grad()
def mlp_restricted(core, li, x):
    P = core[li]
    cc = x @ P['Pin']
    ssq = (P['perp_sq'].unsqueeze(0) + (cc*cc).sum(-1)) / D
    hL = P['uL'].unsqueeze(0) + cc @ P['AL'].T
    hR = P['uR'].unsqueeze(0) + cc @ P['AR'].T
    hidden = (hL * hR) / ssq.clamp_min(1e-12).unsqueeze(-1)
    moc = hidden @ P['Dc'].T + P['bc'].unsqueeze(0)
    return MOMEAN[li].unsqueeze(0) + moc @ P['Pout'].T

@torch.no_grad()
def forward_flex(idx, core=None, ablate=None, restrict_from=None):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        if ablate is not None and ablate[0] == 'head' and ablate[1] == li:
            yh4 = yh4.clone(); yh4[:, :, ablate[2]] = YHMEAN[li][:, ablate[2]].unsqueeze(0)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        if restrict_from is not None and li >= restrict_from:
            mo = mlp_restricted(core, li, x)
        else:
            mo = blk.mlp(F.rms_norm(x, (D,)))
        if ablate is not None and ablate[0] == 'mlp' and ablate[1] == li:
            kk = ablate[2]
            pr = torch.einsum('btd,d->bt', mo, mlp_dirs[li, kk])
            mo = mo - (pr - PROJMEAN[li][:, kk].unsqueeze(0)).unsqueeze(-1) * mlp_dirs[li, kk]
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return logits

def ce_of(logits, tgt):
    logp = F.log_softmax(logits[:, :SEQL-1].float(), dim=-1)
    return -logp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)

def stats(s, sq, n):
    if n <= 1: return 0.0, 0.0
    mean = s/n; var = max(sq/n - mean*mean, 0.0)*n/(n-1)
    return mean, math.sqrt(var/n)
def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0: return 0.0
    return float(np.corrcoef(a, b)[0, 1])
def spearman(a, b):
    return pearson(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))

SETTINGS = [
    ('gram_144x72', 144, 72, 'gram'),
    ('gram_576x288', 576, 288, 'gram'),
    ('rand_288x144', 288, 144, 'rand'),
]
truth = {c: recs1[c]['true_global_dCE'] for c in CAND_NAMES}
truth_z = {c: recs1[c]['true_global_z'] for c in CAND_NAMES}
main288 = {c: recs1[c]['restr_global_dCE'] for c in CAND_NAMES}

results = {}
for name, k_in, k_out, basis in SETTINGS:
    print(f"\n===== SETTING {name} =====", flush=True)
    core = build_core(k_in, k_out, basis)
    est = {}; rcost = {}
    t0 = time.time()
    accs = {c: [0.0, 0.0, 0] for c in CAND_NAMES}
    rbc = {k: [0.0, 0.0, 0] for k in KEYS}
    for i in range(0, NSUB, BATCH):
        sb = slice(i, min(i+BATCH, NSUB))
        idx = SUB[sb]; tgt = idx
        base_ce = ce_of(forward_flex(idx), tgt)
        rce = {}
        for k in KEYS:
            rce[k] = ce_of(forward_flex(idx, core=core, restrict_from=k), tgt)
            dv = rce[k] - base_ce
            rbc[k][0] += float(dv.sum()); rbc[k][1] += float((dv*dv).sum()); rbc[k][2] += int(dv.numel())
        for (comp, kind, li, ix) in CANDS:
            key = rkey(kind, li)
            if key >= NL:
                accs[comp] = None; continue                  # trivial: restricted == true
            d = ce_of(forward_flex(idx, core=core, ablate=(kind, li, ix), restrict_from=key), tgt) - rce[key]
            accs[comp][0] += float(d.sum()); accs[comp][1] += float((d*d).sum()); accs[comp][2] += int(d.numel())
    for c in CAND_NAMES:
        if accs[c] is None:
            est[c] = {'dCE': truth[c], 'SE': None, 'trivial': True}
        else:
            mn, se = stats(*accs[c])
            est[c] = {'dCE': round(mn, 6), 'SE': round(se, 6), 'trivial': False}
    for k in KEYS:
        mn, se = stats(*rbc[k])
        rcost[f'L{k}'] = round(mn, 4)
    nont = [c for c in CAND_NAMES if not est[c]['trivial']]
    pv = [est[c]['dCE'] for c in nont]; tv = [truth[c] for c in nont]
    clear = [abs(truth_z[c]) >= 2 for c in nont]
    sa_all = float(np.mean([np.sign(p) == np.sign(t) for p, t in zip(pv, tv)]))
    sa_cl = float(np.mean([np.sign(p) == np.sign(t) for p, t, cl in zip(pv, tv, clear) if cl]))
    results[name] = {
        'k_in': k_in, 'k_out': k_out, 'basis': basis,
        'spearman_vs_truth_nontrivial': round(spearman(pv, tv), 3),
        'pearson_vs_truth_nontrivial': round(pearson(pv, tv), 3),
        'sign_agreement_all_nontrivial': round(sa_all, 3),
        'sign_agreement_signclear_nontrivial': round(sa_cl, 3),
        'mean_abs_err_nontrivial': round(float(np.mean(np.abs(np.array(pv)-np.array(tv)))), 5),
        'restriction_base_cost_per_key': rcost,
        'per_candidate': {c: est[c] for c in CAND_NAMES},
        'h.L9.7': est['h.L9.7'], 'h.L0.3': est['h.L0.3'],
        'sec': round(time.time()-t0, 1),
    }
    print(f"  spearman {results[name]['spearman_vs_truth_nontrivial']}"
          f" pearson {results[name]['pearson_vs_truth_nontrivial']}"
          f" sign(all/clear) {sa_all:.3f}/{sa_cl:.3f}"
          f"  h.L9.7 {est['h.L9.7']['dCE']} (true {truth['h.L9.7']})"
          f"  h.L0.3 {est['h.L0.3']['dCE']} (true {truth['h.L0.3']})"
          f"  [{results[name]['sec']}s]", flush=True)

# reference row for the part-1 main setting
nont = [c for c in CAND_NAMES if recs1[c]['n_restricted_downstream_mlps'] > 0]
results['gram_288x144_part1'] = {
    'k_in': 288, 'k_out': 144, 'basis': 'gram',
    'spearman_vs_truth_nontrivial': round(spearman([main288[c] for c in nont], [truth[c] for c in nont]), 3),
    'note': 'main setting, computed in part 1',
}

MAIN['capacity_and_random_controls'] = {
    'note': 'restricted-core proxy re-run at other core ranks and with random orthonormal '
            'bases (same folding, same subsample); truth reused from part 1 records. '
            'Nontrivial = candidates with >=1 restricted downstream MLP (21 of 24).',
    'settings': results,
}
json.dump(MAIN, open(f'{QK}/qk_certified_proxy.json', 'w'), indent=2)
print("\nSaved controls into qk_certified_proxy.json", flush=True)
print("QK CERTIFIED PROXY 2 DONE", flush=True)
