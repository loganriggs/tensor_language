"""Can the EXPLICIT textbook induction predicate replace the model's patterns? (Logan: 'don't we
already understand induction and can fully express it?')
For every circuit head in layers 2-10, replace its pattern with the 3-parameter explicit form
    pat_hat = a * MATCH + b * TEMPLATE + c            (causal-masked)
where MATCH[i,j] = 1[token[j-1] == token[i]] (the textbook induction predicate, recomputed per
input) and TEMPLATE = the head's position-based mean pattern. a,b,c fit per head by least squares
ON NATURAL repeated text; then TESTED on natural AND on shuffled prefixes (params frozen ->
generalization). Conditions: (i) full model; (ii) full model + explicit patterns at circuit heads
L2-10; (iii) the 43-component minimal circuit with explicit patterns (rest mean-ablated) = the
'small explicit program' version. If (ii)/(iii) retain the advantage, induction is 100%
expressible as predicate + 3 numbers/head; the earlier 80% was the linear-residual lens, not our
understanding.
"""
import json, sys, ast
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
MINCOMP = json.load(open(f'{QK}/qk_understanding_props.json'))['minimality']['locally_minimal_components']
KEEP = set(ast.literal_eval(c) for c in MINCOMP)
SUBST = {(li, h) for (t, li, h) in [c for c in KEEP if c[0] == 'h'] if 2 <= li <= 10}
ALLC = [('h', li, h) for li in range(NL) for h in range(NH)] + [('m', li) for li in range(NL)]
print(f"substituting explicit predicate at {len(SUBST)} circuit heads in layers 2-10", flush=True)

P = 64; NSEQ = 48
def make_eval(shuffled):
    pref = FINEWEB[:NSEQ, 1:1+P].clone()
    if shuffled:
        g = torch.Generator().manual_seed(11)
        for r in range(NSEQ): pref[r] = pref[r][torch.randperm(P, generator=g)]
    return torch.cat([pref, pref], 1).to(DEV)
EVN, EVS = make_eval(False), make_eval(True)
FIR = torch.arange(1, P-1, device=DEV); SEC = torch.arange(P, 2*P-1, device=DEV)

def match_matrix(idx):
    """MATCH[b,i,j] = 1 if token[j-1]==token[i], j>=1, causal."""
    B, T = idx.shape
    eq = idx.unsqueeze(2) == torch.roll(idx, 1, dims=1).unsqueeze(1)   # [b,i,j] tok[i]==tok[j-1]
    eq[:, :, 0] = False
    causal = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    return (eq & causal).float()


@torch.no_grad()
def forward(EV, keep=None, MEAN=None, subst=None, PARAMS=None, TEMPL=None, fit_collect=None, collect=False):
    idx = EV[:, :-1]; B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)); means = {}
    MM = match_matrix(idx) if (subst or fit_collect is not None) else None
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)
        if fit_collect is not None:
            for h in range(NH):
                if (li, h) in fit_collect['heads']:
                    Pt = pat[:, h]; Tm = Pt.mean(0)                       # position template (T,T)
                    fit_collect['templ'][(li, h)] = Tm
                    # least squares on causal entries: features [MATCH, TEMPL, 1]
                    Xf = torch.stack([MM[mask.expand(B, T, T)], Tm.unsqueeze(0).expand(B, T, T)[mask.expand(B, T, T)],
                                      torch.ones_like(MM[mask.expand(B, T, T)])], 1)          # (n,3)
                    yf = Pt[mask.expand(B, T, T)]
                    sol = torch.linalg.lstsq(Xf, yf.unsqueeze(1)).solution.squeeze(1)
                    fit_collect['params'][(li, h)] = sol
        if subst:
            for h in range(NH):
                if (li, h) in subst:
                    aa, bb, cc = PARAMS[(li, h)]
                    pat[:, h] = (aa*MM + bb*TEMPL[(li, h)].unsqueeze(0) + cc).masked_fill(~mask, 0.0)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        if collect: means[('h', li)] = yh4.mean((0, 1))
        if keep is not None:
            for h in range(NH):
                if ('h', li, h) not in keep: yh4[:, :, h, :] = MEAN[('h', li)][h]
        x = x + a.c_proj(yh4.reshape(B, T, -1)); mo = blk.mlp(F.rms_norm(x, (D,)))
        if collect: means[('m', li)] = mo.mean((0, 1))
        if keep is not None and ('m', li) not in keep: mo = MEAN[('m', li)].expand_as(mo)
        x = x + mo
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float(), means

def adv_of(lg, EV):
    tgt = EV[:, 1:]; ce = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction='none').view(EV.shape[0], -1)
    return ce[:, FIR].mean().item() - ce[:, SEC].mean().item()

# fit a,b,c per head on NATURAL eval
fc = {'heads': SUBST, 'templ': {}, 'params': {}}
_, _ = forward(EVN, fit_collect=fc)
PARAMS = fc['params']; TEMPL = fc['templ']
amean = float(np.mean([PARAMS[k][0].item() for k in PARAMS]))
print(f"fitted; mean match-coefficient a = {amean:.4f}", flush=True)

res = {}
for name, EV in [('natural', EVN), ('shuffled', EVS)]:
    _, MEAN = forward(EV, collect=True)
    af = adv_of(forward(EV)[0], EV); an = adv_of(forward(EV, keep=set(), MEAN=MEAN)[0], EV)
    a_sub = adv_of(forward(EV, subst=SUBST, PARAMS=PARAMS, TEMPL=TEMPL)[0], EV)
    a_circ = adv_of(forward(EV, keep=KEEP, MEAN=MEAN)[0], EV)
    a_circ_sub = adv_of(forward(EV, keep=KEEP, MEAN=MEAN, subst=SUBST, PARAMS=PARAMS, TEMPL=TEMPL)[0], EV)
    r = lambda a: (a - an) / (af - an + 1e-9)
    res[name] = {'adv_full': round(af, 3), 'adv_none': round(an, 3),
                 'explicit_in_full': {'adv': round(a_sub, 3), 'ret': round(r(a_sub), 3)},
                 'circuit_raw': {'adv': round(a_circ, 3), 'ret': round(r(a_circ), 3)},
                 'circuit_explicit': {'adv': round(a_circ_sub, 3), 'ret': round(r(a_circ_sub), 3)}}
    print(f"[{name}] full {af:+.3f} | explicit-patterns-in-full {a_sub:+.3f} ({r(a_sub):.1%}) | "
          f"circuit-raw {a_circ:+.3f} ({r(a_circ):.1%}) | circuit+explicit {a_circ_sub:+.3f} ({r(a_circ_sub):.1%})", flush=True)
json.dump(res, open(f'{QK}/qk_induction_predicate.json', 'w'), indent=2)
print("QK INDUCTION PREDICATE DONE", flush=True)
