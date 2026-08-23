"""Why DOUBLE-QK? The squared-attention pattern is a PRODUCT of two bilinear score matrices:
pattern = (q.k/D) * (q2.k2/D). Do the two factors SPECIALIZE — e.g. one carries content-similarity and the other
position/induction — so the product combines two routing criteria (a gate x content structure)? Capture BOTH
factors separately (verbatim-copy monkeypatch stashing scores/D and scores2/D) at mid layer L8 and per-head
correlate EACH factor with recency / content-similarity / induction(token-match), vs the product (pattern).

REGISTERED PREDICTIONS:
  (0) SANITY: factor1*factor2 (masked) reproduces the pattern of §981 (same per-head profile).
  (a) THE TWO FACTORS SPECIALIZE: on at least some heads the two factors have DIFFERENT feature profiles (e.g. one
      factor content-similarity-heavy, the other recency/induction-heavy), so the double-QK combines two distinct
      routing criteria multiplicatively -> the product is why routing can be jointly content-AND-position selective;
  (b) report per-head corr(factor1, {rec,con,ind}) and corr(factor2, {rec,con,ind}); summarize how often the two
      factors' dominant feature differs."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd'); sys.path.insert(0, '/workspace/tensor_language')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import jacclust.tt_model as TT
from einops import einsum
import torch.nn.functional as F
import types
from collections import defaultdict

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'qk_double_factor_results.json'
NEVAL = 48; SEQ = 128; L = 8
STASH = {}


def make_patched(attn_self):
    def patched(self, q, k, v, q2, k2):
        B, T, H, Dh = q.shape
        s1 = einsum(q, k, "b sq h d, b sk h d -> b h sq sk") / Dh
        s2 = einsum(q2, k2, "b sq h d, b sk h d -> b h sq sk") / Dh
        causal = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
        s1m = s1.masked_fill(causal.logical_not(), 0.0); s2m = s2.masked_fill(causal.logical_not(), 0.0)
        STASH['f1'] = s1m.detach(); STASH['f2'] = s2m.detach()
        pattern = (s1 * s2).masked_fill(causal.logical_not(), 0.0)
        z = einsum(pattern, v, "b h sq sk, b sk h d -> b h sq d")
        return z
    return patched


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); nb = blocks.shape[0]
    attn = m.transformer.h[L].attn; H = attn.n_head
    attn.squared_attention = types.MethodType(make_patched(attn), attn)
    acc = {f: {feat: defaultdict(list) for feat in ['rec', 'con', 'ind']} for f in ['f1', 'f2']}
    for bi in range(0, nb, 4):
        idx = blocks[bi:bi+4].to(DEV)[:, :-1].contiguous(); Tt = idx.shape[1]; b = idx.shape[0]
        emb = F.rms_norm(m.transformer.wte(idx), (D,)).float(); forward_logits(idx)
        f1 = STASH['f1'].float(); f2 = STASH['f2'].float()
        pos = torch.arange(Tt, device=DEV)
        rec = -(pos.view(1, Tt, 1) - pos.view(1, 1, Tt)).float().expand(b, Tt, Tt)
        en = emb/(emb.norm(dim=-1, keepdim=True)+1e-9); con = torch.bmm(en, en.transpose(1, 2))
        tok = idx; tkm1 = torch.full_like(tok, -1); tkm1[:, 1:] = tok[:, :-1]
        ind = (tkm1.view(b, 1, Tt) == tok.view(b, Tt, 1)).float()
        causal = torch.tril(torch.ones(Tt, Tt, device=DEV, dtype=torch.bool)); rowok = pos.view(Tt, 1) >= 1
        msk = causal & rowok
        feats = {'rec': rec, 'con': con, 'ind': ind}
        for fname, F_ in [('f1', f1), ('f2', f2)]:
            for h in range(H):
                fh = F_[:, h]
                for j in range(b):
                    x1 = fh[j][msk]
                    for feat, arr in feats.items():
                        x2 = arr[j][msk]
                        if x1.std() > 1e-6 and x2.std() > 1e-6:
                            acc[fname][feat][h].append(float(torch.corrcoef(torch.stack([x1, x2]))[0, 1]))
    attn.squared_attention = types.MethodType(TT.CausalBilinearSelfAttention.squared_attention, attn)
    def meanh(dd): return {int(h): round(float(np.mean(v)), 3) for h, v in sorted(dd.items())}
    out = {'layer': L, 'n_head': H, 'factor1': {f: meanh(acc['f1'][f]) for f in ['rec', 'con', 'ind']},
           'factor2': {f: meanh(acc['f2'][f]) for f in ['rec', 'con', 'ind']}}
    # per head: dominant feature (by |corr|) of each factor; count heads where they differ
    diff = 0
    for h in range(H):
        d1 = max(['rec', 'con', 'ind'], key=lambda f: abs(out['factor1'][f].get(h, 0)))
        d2 = max(['rec', 'con', 'ind'], key=lambda f: abs(out['factor2'][f].get(h, 0)))
        if d1 != d2: diff += 1
    out['n_heads_factors_differ'] = diff
    out['pred_a_factors_specialize'] = bool(diff >= 3)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print("factor1 (rec/con/ind means):", {f: round(np.mean(list(out['factor1'][f].values())), 3) for f in ['rec','con','ind']}, flush=True)
    print("factor2 (rec/con/ind means):", {f: round(np.mean(list(out['factor2'][f].values())), 3) for f in ['rec','con','ind']}, flush=True)
    print(f"heads where the two factors' dominant feature differs: {diff}/{H}", flush=True)
    print(f"(a) the two QK factors specialize: {out['pred_a_factors_specialize']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
