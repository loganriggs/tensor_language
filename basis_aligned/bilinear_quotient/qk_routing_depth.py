"""Does QK ROUTING differ by DEPTH? §952/§954: induction is a FRONT-attention phenomenon. Measure the INDUCTION
routing correlation (long-range, to avoid the recency confound §983) and RECENCY routing across depth (L2, L5, L8,
L11, L15) by capturing the exact squared-attention pattern per layer. Report the strongest-induction head per layer
and the mean. Ties the routing MECHANISM to the induction depth-localization (§954, induction heads early).

REGISTERED PREDICTIONS:
  (0) SANITY: pattern captured at each layer; long-range induction null ~0.
  (a) INDUCTION ROUTING IS FRONT-PEAKED: the max-head induction routing correlation (long-range) is HIGHER in the
      front/early layers (L2-L8) than late (L11-L15) -> induction routing lives early, consistent with §954;
      recency routing is present at all depths;
  (b) report per-layer max-head and mean induction routing corr (+ recency) on long-range pairs."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd'); sys.path.insert(0, '/workspace/tensor_language')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import jacclust.tt_model as TT
from einops import einsum
import torch.nn.functional as F
import types

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'qk_routing_depth_results.json'
NEVAL = 40; SEQ = 200; GAP = 30; LAYERS = [2, 5, 8, 11, 15]
STASH = {}


def make_patched():
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
def measure_layer(L, blocks):
    attn = m.transformer.h[L].attn; H = attn.n_head
    attn.squared_attention = types.MethodType(make_patched(), attn)
    from collections import defaultdict
    ind_h = defaultdict(list); rec_h = defaultdict(list); indnull_h = defaultdict(list)
    for bi in range(0, blocks.shape[0], 4):
        idx = blocks[bi:bi+4].to(DEV)[:, :-1].contiguous(); Tt = idx.shape[1]; b = idx.shape[0]
        forward_logits(idx); pat = STASH['pattern'].float()
        pos = torch.arange(Tt, device=DEV); gap = pos.view(Tt, 1) - pos.view(1, Tt); lr = (gap > GAP)
        tok = idx; tkm1 = torch.full_like(tok, -1); tkm1[:, 1:] = tok[:, :-1]
        ind = (tkm1.view(b, 1, Tt) == tok.view(b, Tt, 1)).float()
        recency = -(gap.float()).unsqueeze(0).expand(b, Tt, Tt)
        for h in range(H):
            ph = pat[:, h]
            for j in range(b):
                x1 = ph[j][lr]
                if x1.numel() < 50 or x1.std() < 1e-6: continue
                i2 = ind[j][lr]; r2 = recency[j][lr]
                if i2.std() > 1e-6: ind_h[h].append(float(torch.corrcoef(torch.stack([x1, i2]))[0, 1]))
                if r2.std() > 1e-6: rec_h[h].append(float(torch.corrcoef(torch.stack([x1, r2]))[0, 1]))
                perm = torch.randperm(Tt, device=DEV); ish = ind[j][:, perm][lr]
                if ish.std() > 1e-6: indnull_h[h].append(float(torch.corrcoef(torch.stack([x1, ish]))[0, 1]))
    attn.squared_attention = types.MethodType(TT.CausalBilinearSelfAttention.squared_attention, attn)
    indm = {h: float(np.mean(v)) for h, v in ind_h.items() if v}
    recm = {h: float(np.mean(v)) for h, v in rec_h.items() if v}
    nullm = {h: float(np.mean(v)) for h, v in indnull_h.items() if v}
    return {'max_induction': round(max(indm.values()), 3) if indm else 0.0,
            'mean_induction': round(float(np.mean(list(indm.values()))), 3) if indm else 0.0,
            'max_induction_null': round(max(nullm.values()), 3) if nullm else 0.0,
            'mean_recency': round(float(np.mean(list(recm.values()))), 3) if recm else 0.0,
            'top_induction_head': int(max(indm, key=indm.get)) if indm else -1}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous()
    out = {'gap': GAP, 'by_layer': {}}
    for L in LAYERS:
        r = measure_layer(L, blocks); out['by_layer'][str(L)] = r
        print(f"L{L:>2}: max-induction {r['max_induction']} (head {r['top_induction_head']}, null {r['max_induction_null']}) | mean-induction {r['mean_induction']} | mean-recency {r['mean_recency']}", flush=True)
    front = np.mean([out['by_layer'][str(L)]['max_induction'] for L in [2, 5, 8]])
    back = np.mean([out['by_layer'][str(L)]['max_induction'] for L in [11, 15]])
    out['front_L2_8_max_induction'] = round(float(front), 3); out['back_L11_15_max_induction'] = round(float(back), 3)
    out['pred_a_induction_front_peaked'] = bool(front > back + 0.05)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"front(L2-8) max-induction {front:.3f} vs back(L11-15) {back:.3f}", flush=True)
    print(f"(a) induction routing front-peaked: {out['pred_a_induction_front_peaked']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
