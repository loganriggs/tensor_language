"""Resolve §982's confound: recency and content-similarity CO-VARY in short contexts, so §981's content-sim
correlation could be recency in disguise. Recompute the routing correlations using ONLY LONG-RANGE query-key
pairs (q - k > GAP), where recency is nearly constant and content-similarity varies independently. If content-sim
(and induction) still correlate with the routing pattern on long-range pairs, the routing is GENUINELY content/
token-selective, not just recency. Per-head, mid layer L8. Longer SEQ so there are enough long-range pairs.

REGISTERED PREDICTIONS:
  (0) SANITY: enough long-range pairs per head (n large); null (shuffled key content) ~0.
  (a) GENUINE CONTENT/TOKEN ROUTING AT RANGE: on long-range pairs (q-k>GAP), the content-similarity heads (§981
      h7/h8) still show positive content-sim correlation and the induction heads (§981 h4/h6) still show positive
      induction correlation, ABOVE a content-shuffled null -> routing is genuinely content/token-selective beyond
      recency; recency correlation is suppressed by construction;
  (b) report per-head long-range corr(pattern, content-sim / induction) + shuffled-content null."""
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
OUT = PT + 'qk_routing_longrange_results.json'
NEVAL = 48; SEQ = 200; L = 8; GAP = 30
STASH = {}


def make_patched(attn_self):
    def patched(self, q, k, v, q2, k2):
        B, T, H, Dh = q.shape
        s1 = einsum(q, k, "b sq h d, b sk h d -> b h sq sk") / Dh
        s2 = einsum(q2, k2, "b sq h d, b sk h d -> b h sq sk") / Dh
        causal = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
        pattern = (s1 * s2).masked_fill(causal.logical_not(), 0.0)
        STASH['pattern'] = pattern.detach()
        return einsum(pattern, v, "b h sq sk, b sk h d -> b h sq d")
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
    con_h = defaultdict(list); ind_h = defaultdict(list); connull_h = defaultdict(list); npairs = 0
    for bi in range(0, nb, 4):
        idx = blocks[bi:bi+4].to(DEV)[:, :-1].contiguous(); Tt = idx.shape[1]; b = idx.shape[0]
        emb = F.rms_norm(m.transformer.wte(idx), (D,)).float(); forward_logits(idx); pat = STASH['pattern'].float()
        pos = torch.arange(Tt, device=DEV)
        gap = (pos.view(Tt, 1) - pos.view(1, Tt))  # q-k
        lr = (gap > GAP)  # long-range causal pairs
        en = emb/(emb.norm(dim=-1, keepdim=True)+1e-9); con = torch.bmm(en, en.transpose(1, 2))
        tok = idx; tkm1 = torch.full_like(tok, -1); tkm1[:, 1:] = tok[:, :-1]
        ind = (tkm1.view(b, 1, Tt) == tok.view(b, Tt, 1)).float()
        # content-shuffled null: permute key positions' embeddings
        for h in range(H):
            ph = pat[:, h]
            for j in range(b):
                m2 = lr
                x1 = ph[j][m2]
                if x1.numel() < 50 or x1.std() < 1e-6: continue
                c2 = con[j][m2]; i2 = ind[j][m2]
                if c2.std() > 1e-6: con_h[h].append(float(torch.corrcoef(torch.stack([x1, c2]))[0, 1]))
                if i2.std() > 1e-6: ind_h[h].append(float(torch.corrcoef(torch.stack([x1, i2]))[0, 1]))
                # null: shuffle key identity for content
                perm = torch.randperm(Tt, device=DEV); cs = torch.bmm(en, en[:, perm].transpose(1, 2))[j][m2]
                if cs.std() > 1e-6: connull_h[h].append(float(torch.corrcoef(torch.stack([x1, cs]))[0, 1]))
        npairs += int(lr.sum())
    attn.squared_attention = types.MethodType(TT.CausalBilinearSelfAttention.squared_attention, attn)
    def meanh(dd): return {int(h): round(float(np.mean(v)), 3) for h, v in sorted(dd.items()) if v}
    out = {'layer': L, 'gap': GAP, 'content_sim_per_head': meanh(con_h), 'induction_per_head': meanh(ind_h),
           'content_null_per_head': meanh(connull_h)}
    cvals = np.array([np.mean(con_h[h]) for h in con_h if con_h[h]]); ivals = np.array([np.mean(ind_h[h]) for h in ind_h if ind_h[h]])
    nvals = np.array([np.mean(connull_h[h]) for h in connull_h if connull_h[h]])
    out['mean_content_sim_longrange'] = round(float(cvals.mean()), 4); out['mean_induction_longrange'] = round(float(ivals.mean()), 4)
    out['mean_content_null'] = round(float(nvals.mean()), 4)
    # top content head and top induction head (from §981: h8 content, h4 induction)
    out['h8_content'] = out['content_sim_per_head'].get(8); out['h4_induction'] = out['induction_per_head'].get(4)
    out['pred_a_genuine_content_token'] = bool((out['mean_content_sim_longrange'] - out['mean_content_null'] > 0.03) or out['mean_induction_longrange'] > 0.05)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"long-range (q-k>{GAP}) mean corr: content-sim {out['mean_content_sim_longrange']} (null {out['mean_content_null']}) | induction {out['mean_induction_longrange']}", flush=True)
    print(f"h8 content {out['h8_content']} | h4 induction {out['h4_induction']}", flush=True)
    print(f"(a) genuine content/token routing at range: {out['pred_a_genuine_content_token']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
