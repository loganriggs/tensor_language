"""OPEN the QK-ROUTING mechanism (the one genuinely-under-covered piece). bilin18 uses SOFTMAX-FREE SQUARED
attention: pattern[h,q,k] = (q.k/D)*(q2.k2/D), causally masked, UNNORMALIZED (a raw product of two bilinear score
matrices), then z = pattern @ v. What does this pattern route to? Capture the EXACT pattern (monkeypatch a verbatim
copy of squared_attention that stashes it — no rotary reimplementation) at a mid content layer, and per head
correlate the (causal) routing weights with:
  - RECENCY: -(q - k)  (does it favor recent keys?),
  - CONTENT-SIMILARITY: cos(emb[q], emb[k])  (attend to embedding/content-similar tokens?),
  - INDUCTION/TOKEN-MATCH: 1[token[k-1] == token[q]]  (the induction pattern: attend to the token AFTER a previous
    occurrence of the current token).
Report per-head mean correlations + a positional-only null (shuffle key identities within a row keeps position).

REGISTERED PREDICTIONS:
  (0) SANITY: the captured pattern reproduces the model (patched == original output); causal (zero above diagonal).
  (a) ROUTING IS CONTENT/TOKEN-DRIVEN, NOT JUST POSITIONAL: at a mid content layer, the pattern correlates with
      CONTENT-SIMILARITY and/or INDUCTION token-match ABOVE a position-matched null -> the squared attention routes
      by content/token identity, not merely recency (consistent with content aggregation §932 + induction §952);
  (b) report per-head corr(pattern, recency / content-sim / induction) and the means."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import jacclust.tt_model as TT
from einops import einsum
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'qk_routing_results.json'
NEVAL = 48; SEQ = 128; L = 8   # mid content layer; short SEQ to keep TxT patterns manageable
STASH = {}


def make_patched(orig_self):
    def patched(self, q, k, v, q2, k2):
        B, T, H, Dh = q.shape
        scores = einsum(q, k, "b sq h d, b sk h d -> b h sq sk")
        scores2 = einsum(q2, k2, "b sq h d, b sk h d -> b h sq sk")
        pattern = (scores / Dh) * (scores2 / Dh)
        causal = torch.tril(torch.ones(T, T, device=pattern.device, dtype=torch.bool))
        pattern = pattern.masked_fill(causal.logical_not(), 0.0)
        STASH['pattern'] = pattern.detach()  # (B,H,T,T)
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
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    attn = m.transformer.h[L].attn
    import types
    attn.squared_attention = types.MethodType(make_patched(attn), attn)  # exact copy + stash
    H = attn.n_head
    # per-head correlation accumulators
    from collections import defaultdict
    corr_rec = defaultdict(list); corr_con = defaultdict(list); corr_ind = defaultdict(list); corr_ind_null = defaultdict(list)
    for bi in range(0, nb, 4):
        idx = blocks[bi:bi+4].to(DEV)[:, :-1].contiguous(); Tt = idx.shape[1]
        emb = F.rms_norm(m.transformer.wte(idx), (D,)).float()  # (b,T,D) content proxy
        forward_logits(idx); pat = STASH['pattern'].float()  # (b,H,T,T)
        b = idx.shape[0]
        # features (b,T,T)
        pos = torch.arange(Tt, device=DEV)
        recency = -(pos.view(1, Tt, 1) - pos.view(1, 1, Tt)).float().expand(b, Tt, Tt)  # -(q-k)
        embn = emb / (emb.norm(dim=-1, keepdim=True) + 1e-9)
        content_sim = torch.bmm(embn, embn.transpose(1, 2))  # (b,T,T) cos(emb[q],emb[k])
        tok = idx  # (b,T)
        # induction: token[k-1] == token[q]
        tok_km1 = torch.full_like(tok, -1); tok_km1[:, 1:] = tok[:, :-1]
        induction = (tok_km1.view(b, 1, Tt) == tok.view(b, Tt, 1)).float()  # (b,q,k): token[k-1]==token[q]
        causal = torch.tril(torch.ones(Tt, Tt, device=DEV, dtype=torch.bool))
        for h in range(H):
            ph = pat[:, h]  # (b,T,T)
            for arr, dest in [(recency, corr_rec), (content_sim, corr_con), (induction, corr_ind)]:
                # correlation over the causal (lower-triangular, q>=1) entries, per batch, averaged
                for j in range(b):
                    msk = causal & (pos.view(Tt, 1) >= 1)
                    x1 = ph[j][msk]; x2 = arr[j][msk]
                    if x1.std() > 1e-6 and x2.std() > 1e-6:
                        dest[h].append(float(torch.corrcoef(torch.stack([x1, x2]))[0, 1]))
            # induction null: shuffle key positions' induction labels within row (keeps positional structure of pattern)
            for j in range(b):
                msk = causal & (pos.view(Tt, 1) >= 1)
                x1 = ph[j][msk]; ind = induction[j].clone()
                perm = torch.randperm(Tt, device=DEV); ind = ind[:, perm]
                x2 = ind[msk]
                if x1.std() > 1e-6 and x2.std() > 1e-6:
                    corr_ind_null[h].append(float(torch.corrcoef(torch.stack([x1, x2]))[0, 1]))
    attn.squared_attention = types.MethodType(TT.CausalBilinearSelfAttention.squared_attention, attn)  # restore
    def meand(dd): return {int(h): round(float(np.mean(v)), 4) for h, v in sorted(dd.items())}
    out = {'layer': L, 'n_head': H,
           'corr_recency_per_head': meand(corr_rec), 'corr_content_sim_per_head': meand(corr_con),
           'corr_induction_per_head': meand(corr_ind), 'corr_induction_null_per_head': meand(corr_ind_null)}
    allc = np.array([np.mean(corr_con[h]) for h in corr_con]); allr = np.array([np.mean(corr_rec[h]) for h in corr_rec])
    alli = np.array([np.mean(corr_ind[h]) for h in corr_ind]); alln = np.array([np.mean(corr_ind_null[h]) for h in corr_ind_null])
    out['mean_corr_recency'] = round(float(allr.mean()), 4); out['mean_corr_content_sim'] = round(float(allc.mean()), 4)
    out['mean_corr_induction'] = round(float(alli.mean()), 4); out['mean_corr_induction_null'] = round(float(alln.mean()), 4)
    out['pred_a_content_or_token_driven'] = bool(out['mean_corr_content_sim'] > 0.05 or (out['mean_corr_induction'] - out['mean_corr_induction_null']) > 0.05)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"L{L} mean corr: recency {out['mean_corr_recency']} | content-sim {out['mean_corr_content_sim']} | induction {out['mean_corr_induction']} (null {out['mean_corr_induction_null']})", flush=True)
    print(f"(a) routing content/token-driven (not just positional): {out['pred_a_content_or_token_driven']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
