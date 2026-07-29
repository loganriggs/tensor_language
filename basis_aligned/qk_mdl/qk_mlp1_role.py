"""Where does MLP1-ablation damage bilin18 induction -- the MATCH (attention pattern) or the COPY
(value)? On a repeated-prefix eval, the correct induction target for query i (second copy, i>=P) is
key position i-P+1 (its previous token equals token[i], so copying that key's token predicts
token[i+1]). Measure the attention mass landing on the correct key, summed over heads/layers, with
MLP1 intact vs mean-ablated. If MLP1 ablation collapses the match mass in bilin18 (but not in the
softmax control swiglu18), MLP1 feeds the two-branch MATCH computation; if match mass is unchanged,
MLP1 acts on the copy/value or readout side.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
P = 64; NSEQ = 48
pref = FINEWEB[100:100+NSEQ, 1:1+P]; EV = torch.cat([pref, pref], 1).to(DEV)


def make(short):
    m, cfg = load_elriggs(short); NH, D = cfg['n_head'], cfg['n_embd']; HD = D//NH; V = cfg['vocab_size']; NL = len(m.transformer.h)
    return dict(m=m, NH=NH, HD=HD, D=D, V=V, NL=NL, two=bool(cfg.get('bilinear_attn')) and bool(cfg.get('squared_attn')), sq=bool(cfg.get('squared_attn')))


@torch.no_grad()
def forward(M, idx, ablate_mlp=frozenset(), MEAN=None, collect_mean=False):
    m = M['m']; NH, HD, D, NL, V = M['NH'], M['HD'], M['D'], M['NL'], M['V']
    B, T = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)); means = {}
    # correct induction key per query: for i in [P,2P), key = i-P+1
    qidx = torch.arange(P, 2*P-1, device=DEV); kidx = qidx - P + 1
    match_mass = 0.0; nlay = 0
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        if M['two']:
            q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
            pat = (s1*s2).masked_fill(~mask, 0.0)
            patn = pat / pat.sum(-1, keepdim=True).clamp_min(1e-9)   # normalize just to read match fraction
        elif M['sq']:
            q, k = qk(a.c_q), qk(a.c_k); s = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD
            pat = s.square().masked_fill(~mask, 0.0); patn = pat/pat.sum(-1, keepdim=True).clamp_min(1e-9)
        else:
            q, k = qk(a.c_q), qk(a.c_k); s = torch.einsum('bqhd,bkhd->bhqk', q, k)/(HD**0.5)
            s = s.masked_fill(~mask, float('-inf')); patn = F.softmax(s, -1)
        # match mass: prob at correct induction key, averaged over query positions/heads/batch
        # induction-match: does query i (2nd copy) attend MOST to the correct copy key i-P+1?
        rawpat = (pat if M['two'] else patn).clone()
        rawpat = rawpat.masked_fill(~mask, float('-inf'))   # only causal positions compete for argmax
        am = rawpat[:, :, qidx, :].argmax(-1)               # (B,NH,len(qidx)) argmax key per query
        match_mass += (am == kidx.view(1, 1, -1)).float().mean().item(); nlay += 1
        realpat = pat if M['two'] else patn   # true head pattern (unnormalized for two-branch)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', realpat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); mo = blk.mlp(F.rms_norm(x, (D,)))
        if collect_mean: means[('m', li)] = mo.mean((0, 1))
        if li in ablate_mlp: mo = MEAN[('m', li)].expand_as(mo)
        x = x + mo
    lg = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()
    tgt = EV[:, 1:]; ce = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction='none').view(B, -1)
    FIR = torch.arange(1, P-1, device=DEV); SEC = torch.arange(P, 2*P-1, device=DEV)
    adv = ce[:, FIR].mean().item() - ce[:, SEC].mean().item()
    return match_mass/nlay, adv, means


res = {}
for short in ['bilin18', 'swiglu18', 'bilinsm12']:
    M = make(short)
    _, _, MEAN = forward(M, EV[:, :-1], collect_mean=True)
    mm_intact, adv_intact, _ = forward(M, EV[:, :-1])
    mm_abl, adv_abl, _ = forward(M, EV[:, :-1], ablate_mlp={1}, MEAN=MEAN)
    res[short] = {'match_mass_intact': round(mm_intact, 5), 'match_mass_ablateMLP1': round(mm_abl, 5),
                  'match_mass_drop_frac': round((mm_intact - mm_abl)/(mm_intact+1e-9), 3),
                  'adv_intact': round(adv_intact, 4), 'adv_ablateMLP1': round(adv_abl, 4)}
    print(f"[{short}] induction-match attn mass: intact {mm_intact:.5f} -> ablate MLP1 {mm_abl:.5f} "
          f"(drop {res[short]['match_mass_drop_frac']:.0%}) | adv {adv_intact:.3f}->{adv_abl:.3f}", flush=True)
json.dump(res, open(f'{QK}/qk_mlp1_role.json', 'w'), indent=2)
print("QK MLP1 ROLE DONE", flush=True)
