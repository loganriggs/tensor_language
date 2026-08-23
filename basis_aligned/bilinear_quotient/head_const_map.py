"""Generalize §1089 across ALL 162 heads: L5H7 (the biggest attention node) turned out to be replaceable by a
single CONSTANT vector at 1.5% of its zeroing cost — a bias head. How much of the WHOLE attention stack is bias
heads? For every (layer, head): capture the head's output mean (one fixed vector), then measure CE cost of
(1) ZERO (reproduces §1083's map) and (2) CONST replacement (output := its global mean vector everywhere).
DYNAMIC value of a head := zero_cost − const_cost (what the head's actual computation adds beyond its bias).
Also (3) L5H7 SPARSITY: replacement constant restricted to its top-k |coords| (k=4,8,16,32,64) — how few dims
carry the gain payload?

REGISTERED PREDICTIONS:
  (0) SANITY: zero costs reproduce §1083 (L5H7 ~0.88, L0H3 ~0.09, L1H1 ~0.06); const-replacement never costs
      much MORE than zeroing (const ~ superset of zero information).
  (a) BIAS-DOMINATED STACK: summed const-replacement cost across all heads < 35% of summed zero cost -> most of
      what attention 'does' causally is deliver fixed biases; the DYNAMIC remainder concentrates in the known
      specialists (front local/prev heads L0H3, L1H1, L2H5 and the induction head L5H5 stays dynamic-cheap on
      prose);
  (b) SPECIALISTS ARE DYNAMIC: for the front specialists (L0H3, L1H1), const-replacement recovers < 50% of their
      zero cost (their value IS the token-dependent routing), unlike L5H7's 98.5%;
  (c) L5H7 SPARSE PAYLOAD: k=16 of 128 coords of the constant retain >= 80% of the full-constant recovery
      (the payload is the few massive/gain dims, §1089)."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; NH = 9; HD = 128; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'head_const_map_results.json'
NSEQ = 64; SEQ = 256
H = m.transformer.h
CTL = {'layer': -1, 'head': -1, 'mode': None, 'vec': None}
MEANS = {}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def head_hook(L):
    def h(mo, args):
        if CTL['layer'] != L or CTL['mode'] is None: return None
        y = args[0].clone(); hh = CTL['head']; sl = slice(hh*HD, (hh+1)*HD)
        if CTL['mode'] == 'zero':
            y[..., sl] = 0.0
        else:  # const / sparse-const: vec provided
            y[..., sl] = CTL['vec'].view(1, 1, HD).to(y.dtype)
        return (y,) + tuple(args[1:])
    return h


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

    # pass 1: per-head global mean vectors
    caps = {L: torch.zeros(NH, HD, device=DEV) for L in range(18)}; cnt = 0
    hooks = []
    for L in range(18):
        def mk(L):
            def h(mo, args):
                y = args[0].detach().float()  # B,T,C
                caps[L] += y.reshape(-1, NH, HD).mean(0) * y.shape[0]*y.shape[1]
            return h
        hooks.append(H[L].attn.c_proj.register_forward_pre_hook(mk(L)))
    npos = 0
    for i in range(0, NSEQ, 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); fwd(idx); npos += idx.numel()
    for h in hooks: h.remove()
    for L in range(18): MEANS[L] = caps[L] / npos

    # pass 2: zero + const per head
    hooks = [H[L].attn.c_proj.register_forward_pre_hook(head_hook(L)) for L in range(18)]
    CTL['layer'] = -1; base = ce(blocks)
    zero = torch.zeros(18, NH); const = torch.zeros(18, NH)
    for L in range(18):
        for hh in range(NH):
            CTL['layer'] = L; CTL['head'] = hh; CTL['mode'] = 'zero'
            zero[L, hh] = ce(blocks) - base
            CTL['mode'] = 'const'; CTL['vec'] = MEANS[L][hh]
            const[L, hh] = ce(blocks) - base
            CTL['layer'] = -1; CTL['mode'] = None
        print(f"L{L} zero: {[round(float(v),3) for v in zero[L]]}", flush=True)
        print(f"L{L} const: {[round(float(v),3) for v in const[L]]}", flush=True)

    # pass 3: L5H7 sparse-constant sweep
    sparse = {}
    full_vec = MEANS[5][7]
    for k in [4, 8, 16, 32, 64]:
        v = torch.zeros_like(full_vec)
        topk = full_vec.abs().topk(k).indices
        v[topk] = full_vec[topk]
        CTL['layer'] = 5; CTL['head'] = 7; CTL['mode'] = 'const'; CTL['vec'] = v
        sparse[str(k)] = round(ce(blocks) - base, 4)
        CTL['layer'] = -1; CTL['mode'] = None
        print(f"L5H7 const top-{k}: cost {sparse[str(k)]}", flush=True)
    for h in hooks: h.remove()

    zsum = float(zero.clamp_min(0).sum()); csum = float(const.clamp_min(0).sum())
    dyn = (zero - const).clamp_min(0)
    top_dynamic = sorted([(round(float(dyn[L, hh]), 4), f'L{L}H{hh}') for L in range(18) for hh in range(NH)], reverse=True)[:12]
    out = {'base_ce': round(base, 4), 'zero_sum': round(zsum, 3), 'const_sum': round(csum, 3),
           'const_over_zero': round(csum/max(zsum, 1e-6), 3),
           'zero': {f'L{L}': [round(float(v), 4) for v in zero[L]] for L in range(18)},
           'const': {f'L{L}': [round(float(v), 4) for v in const[L]] for L in range(18)},
           'top_dynamic_heads': top_dynamic, 'l5h7_sparse_const': sparse}
    z57 = float(zero[5, 7]); c57 = float(const[5, 7])
    k16_recov = 1 - sparse['16']/max(z57, 1e-6); full_recov = 1 - c57/max(z57, 1e-6)
    def recov(L, hh): return 1 - float(const[L, hh])/max(float(zero[L, hh]), 1e-6)
    out['pred_a_bias_dominated'] = bool(csum < 0.35*zsum)
    out['pred_b_specialists_dynamic'] = bool(recov(0, 3) < 0.5 and recov(1, 1) < 0.5)
    out['pred_c_sparse_payload'] = bool(k16_recov >= 0.8*full_recov)
    out['recov_L0H3'] = round(recov(0, 3), 3); out['recov_L1H1'] = round(recov(1, 1), 3); out['recov_L5H7'] = round(full_recov, 3)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"sum zero {zsum:.3f} | sum const {csum:.3f} ({out['const_over_zero']}) | top dynamic {top_dynamic[:5]}", flush=True)
    print(f"recov: L5H7 {out['recov_L5H7']} | L0H3 {out['recov_L0H3']} | L1H1 {out['recov_L1H1']} | preds a {out['pred_a_bias_dominated']} b {out['pred_b_specialists_dynamic']} c {out['pred_c_sparse_payload']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
