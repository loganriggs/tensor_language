"""THREAD B (per-head map): FIRST head-level map of bilin18's attention. All prior attention work (§1043-1047,
§1054, §1069) treated each layer's 9 heads as one unit. Two measurements per (layer, head): (1) PATTERN PROFILE --
share of |pattern| mass by key distance (self d=0 / prev d=1 / local 2-8 / mid 9-64 / far >64) on prose, plus an
INDUCTION score on repeated-random sequences (mass at the induction offset); (2) CAUSAL COST -- zero that head's
output slice (c_proj input columns h*128:(h+1)*128) and measure CE cost on prose. Heads classified by registered
thresholds: prev (prev-share>0.4), local (self+prev+local>0.6), induction (induction-share>0.2), pooler
(mid+far>0.5), inert (CE cost < 0.01). Patterns recomputed from module weights incl. rotary+qk-norm (weight-based).

REGISTERED PREDICTIONS:
  (0) SANITY: summed per-head CE costs per layer correlate with known layer importance (front+middle >> late 15-17).
  (a) HEADS SPECIALIZE: within most layers head roles are NOT uniform -- in >= 12/18 layers, the top-2 heads carry
      >= 50% of the layer's summed per-head CE cost (head-level sparsity);
  (b) ROLE MAP MATCHES THE BAND STORY AT FINER GRAIN: L0-2 heads mostly local/prev with at least one prev-token
      head in L0; induction-scored heads concentrate in L0 and L5 (the §877 induction pair); L3-14 dominated by
      pooler-profile heads; L15-17 heads nearly all inert;
  (c) if roles are uniform within layers (no head specialization), the layer-as-unit picture was already complete
      (report plainly)."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; NH = 9; HD = 128; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_head_map_results.json'
NSEQ = 96; SEQ = 256; NSYN = 24; LREP = 64
H = m.transformer.h
MOD = sys.modules[type(H[0].attn).__module__]
ZERO = {'layer': -1, 'head': -1}
CAPX = {}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def zero_hook(L):
    def h(mo, args):
        if ZERO['layer'] != L: return None
        y = args[0].clone(); hh = ZERO['head']
        y[..., hh*HD:(hh+1)*HD] = 0.0
        return (y,) + tuple(args[1:])
    return h


def capx_hook(L):
    def h(mo, args): CAPX[L] = args[0].detach()
    return h


@torch.no_grad()
def pattern_for(attn, x):
    """replicate forward up to the causal pattern: [B, NH, T, T]."""
    B, T, C = x.shape
    q = attn.c_q(x).view(B, T, NH, HD); k = attn.c_k(x).view(B, T, NH, HD)
    q2 = attn.c_q2(x).view(B, T, NH, HD); k2 = attn.c_k2(x).view(B, T, NH, HD)
    cos, sin = attn.rotary(q)
    q, k = F.rms_norm(q, (HD,)), F.rms_norm(k, (HD,))
    q, k = MOD.apply_rotary_emb(q, cos, sin), MOD.apply_rotary_emb(k, cos, sin)
    q2, k2 = F.rms_norm(q2, (HD,)), F.rms_norm(k2, (HD,))
    q2, k2 = MOD.apply_rotary_emb(q2, cos, sin), MOD.apply_rotary_emb(k2, cos, sin)
    s1 = torch.einsum('bqhd,bkhd->bhqk', q, k); s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)
    pat = (s1/HD)*(s2/HD)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
    return pat.masked_fill_(mask.logical_not(), 0.0)


@torch.no_grad()
def ce(blocks):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].reshape(-1)
        lp = F.log_softmax(fwd(idx).float(), -1)
        tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tgt.shape[0], device=DEV), tgt].sum()); n += tgt.shape[0]
    return tot/n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    g = torch.Generator(device=DEV).manual_seed(0)
    base_syn = torch.randint(0, 50000, (NSYN, LREP), generator=g, device=DEV)
    syn = torch.cat([base_syn, base_syn], 1)

    # ---- pattern profiles (capture attn inputs, recompute per-head patterns) ----
    hcap = [H[L].attn.register_forward_pre_hook(capx_hook(L)) for L in range(18)]
    prof = {L: torch.zeros(NH, 5, device=DEV) for L in range(18)}  # self/prev/local/mid/far
    nb = 0
    for i in range(0, 32, 8):  # 32 seqs enough for pattern stats
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); fwd(idx); nb += 1
        T = idx.shape[1]
        di = torch.arange(T, device=DEV).view(-1, 1) - torch.arange(T, device=DEV).view(1, -1)
        bucket = torch.full((T, T), -1, device=DEV, dtype=torch.long)
        bucket[di == 0] = 0; bucket[di == 1] = 1
        bucket[(di >= 2) & (di <= 8)] = 2; bucket[(di >= 9) & (di <= 64)] = 3; bucket[di > 64] = 4
        for L in range(18):
            pat = pattern_for(H[L].attn, CAPX[L]).abs()  # B,NH,T,T
            for b in range(5):
                prof[L][:, b] += pat[:, :, bucket == b].sum((0, 2))
            del pat
    for L in range(18): prof[L] = (prof[L] / prof[L].sum(1, keepdim=True).clamp_min(1e-9)).cpu()
    # induction share on synthetic
    ind = torch.zeros(18, NH, device=DEV)
    idx = syn[:, :-1].contiguous(); fwd(idx); T = idx.shape[1]
    qi = torch.arange(LREP, T, device=DEV); ki = qi - LREP + 1
    for L in range(18):
        pat = pattern_for(H[L].attn, CAPX[L]).abs()
        tot = pat[:, :, qi, :].sum((0, 2, 3)).clamp_min(1e-9)
        ind[L] = pat[:, :, qi, ki].sum((0, 2)) / tot
        del pat
    for h in hcap: h.remove()
    ind = ind.cpu()

    # ---- causal per-head CE cost ----
    hz = [H[L].attn.c_proj.register_forward_pre_hook(zero_hook(L)) for L in range(18)]
    ZERO['layer'] = -1
    ce_blocks = blocks[:64]
    base = ce(ce_blocks)
    cost = torch.zeros(18, NH)
    for L in range(18):
        for hh in range(NH):
            ZERO['layer'] = L; ZERO['head'] = hh
            cost[L, hh] = ce(ce_blocks) - base
            ZERO['layer'] = -1
        print(f"L{L} head costs: {[round(float(c),3) for c in cost[L]]}", flush=True)
    for h in hz: h.remove()

    # ---- classify ----
    heads = {}
    for L in range(18):
        row = []
        for hh in range(NH):
            p = prof[L][hh]; c = float(cost[L, hh]); i_s = float(ind[L, hh])
            roles = []
            if c < 0.01: roles.append('inert')
            if i_s > 0.2: roles.append('induction')
            if float(p[1]) > 0.4: roles.append('prev')
            if float(p[0]+p[1]+p[2]) > 0.6: roles.append('local')
            if float(p[3]+p[4]) > 0.5: roles.append('pooler')
            row.append({'cost': round(c, 4), 'induction': round(i_s, 3),
                        'profile': [round(float(v), 3) for v in p], 'roles': roles or ['mixed']})
        heads[str(L)] = row
    top2_frac = {}
    for L in range(18):
        cc = cost[L].clamp_min(0); s = float(cc.sum())
        top2_frac[str(L)] = round(float(cc.topk(2).values.sum())/max(s, 1e-6), 3) if s > 0.02 else None
    sparse_layers = sum(1 for v in top2_frac.values() if v is not None and v >= 0.5)
    out = {'base_ce': round(base, 4), 'heads': heads, 'top2_cost_frac_by_layer': top2_frac,
           'n_layers_head_sparse': sparse_layers,
           'layer_cost_sum': {str(L): round(float(cost[L].clamp_min(0).sum()), 3) for L in range(18)},
           'pred_a_heads_specialize': bool(sparse_layers >= 12),
           'runtime_s': round(time.time()-t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"head-sparse layers (top-2 >= 50% of cost): {sparse_layers}/18 | pred_a {out['pred_a_heads_specialize']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
