"""ARCHITECTURE-LEVEL test enabled by §1091's bias map: make ALL 162 heads STATIC (each output := its global-mean
constant) -- a bilin18 with NO attention dynamics at all, only attention-shaped biases. The per-head const costs
summed to just 0.645 nats, but per-head measurement hides COLLECTIVE redundancy: the middle's content pooling is
carried collectively (§1049/§1054/§1085) so removing all dynamics at once should reveal what per-head removal
cannot. Conditions: base | ALL-CONST | ALL-ZERO (control: no attention at all, not even biases) | const-except-
FRONT (L0-2 dynamics restored) | const-except-DYN6 (the §1091 top dynamic heads L0H3,L2H5,L1H1,L6H3,L9H7,L7H8
restored) | const-except-MIDDLE (L3-14 dynamics restored). Rare/frequent CE split for the content signature.

REGISTERED PREDICTIONS:
  (0) SANITY: all-zero >> all-const (the §1089/§1091 biases matter); each restore condition <= all-const.
  (a) COLLECTIVE POOLING REVEALED: all-const cost >= 3x the summed per-head const costs (0.645) -> the middle's
      content gathering is collectively load-bearing but per-head redundant; its rare/freq cost ratio >= 2
      (what dies is the content machine);
  (b) MIDDLE > FRONT for the collective function: restoring the MIDDLE band's dynamics recovers more of the
      all-const cost than restoring the front (content pooling lives in L3-14), and the 6 named dynamic heads
      alone recover < 40% (the collective function is not in the named routers);
  (c) if all-const ~ 0.645 (per-head sum), there is NO hidden collective function -- attention dynamics are fully
      per-head-accounted and bilin18's attention is 'one bias + 6 routers' outright (report plainly)."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; NH = 9; HD = 128; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_static_results.json'
NSEQ = 96; SEQ = 256; RARE_MAX = 2
H = m.transformer.h
DYN6 = {(0, 3), (2, 5), (1, 1), (6, 3), (9, 7), (7, 8)}
CTL = {'mode': None}   # mode: None | 'all' | 'zero' | set of (L,h) kept dynamic
MEANS = {}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def hook(L):
    def h(mo, args):
        md = CTL['mode']
        if md is None: return None
        y = args[0].clone(); B, T, _ = y.shape
        for hh in range(NH):
            if md == 'zero':
                y[..., hh*HD:(hh+1)*HD] = 0.0
            elif md == 'all' or (isinstance(md, set) and (L, hh) not in md):
                y[..., hh*HD:(hh+1)*HD] = MEANS[L][hh].view(1, 1, HD).to(y.dtype)
        return (y,) + tuple(args[1:])
    return h


@torch.no_grad()
def ce_split(blocks, is_rare):
    tot = 0.0; n = 0; tr = 0.0; nr = 0; tf = 0.0; nf = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].reshape(-1)
        lp = F.log_softmax(fwd(idx).float(), -1)
        ce_tok = -lp.reshape(-1, lp.shape[-1])[torch.arange(tgt.shape[0], device=DEV), tgt]
        rm = is_rare[tgt]
        tot += float(ce_tok.sum()); n += tgt.shape[0]
        tr += float(ce_tok[rm].sum()); nr += int(rm.sum())
        tf += float(ce_tok[~rm].sum()); nf += int((~rm).sum())
    return tot/n, tr/max(nr, 1), tf/max(nf, 1)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0])
    tfreq = torch.zeros(V, device=DEV)
    ta = blocks[:, 1:].to(DEV).reshape(-1); tfreq.index_add_(0, ta, torch.ones_like(ta, dtype=torch.float))
    is_rare = tfreq <= RARE_MAX

    # pass 1: per-head means
    caps = {L: torch.zeros(NH, HD, device=DEV) for L in range(18)}
    hs = []
    for L in range(18):
        def mk(L):
            def h(mo, args):
                y = args[0].detach().float()
                caps[L] += y.reshape(-1, NH, HD).sum(0)
            return h
        hs.append(H[L].attn.c_proj.register_forward_pre_hook(mk(L)))
    npos = 0
    for i in range(0, NSEQ, 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); fwd(idx); npos += idx.numel()
    for h in hs: h.remove()
    for L in range(18): MEANS[L] = caps[L] / npos

    hs = [H[L].attn.c_proj.register_forward_pre_hook(hook(L)) for L in range(18)]
    CTL['mode'] = None
    base, base_r, base_f = ce_split(blocks, is_rare)
    conds = {
        'all_const': 'all', 'all_zero': 'zero',
        'const_except_front': {(L, h) for L in range(3) for h in range(NH)},
        'const_except_dyn6': DYN6,
        'const_except_middle': {(L, h) for L in range(3, 15) for h in range(NH)},
    }
    res = {}
    for name, md in conds.items():
        CTL['mode'] = md
        c, cr, cf = ce_split(blocks, is_rare)
        res[name] = {'cost': round(c-base, 4), 'rare_cost': round(cr-base_r, 4), 'freq_cost': round(cf-base_f, 4),
                     'rare_freq_ratio': round((cr-base_r)/max(cf-base_f, 1e-4), 2)}
        CTL['mode'] = None
        print(f"{name:>20}: cost {res[name]['cost']} | rare {res[name]['rare_cost']} | freq {res[name]['freq_cost']} | ratio {res[name]['rare_freq_ratio']}", flush=True)
    for h in hs: h.remove()

    ac = res['all_const']['cost']
    out = {'base_ce': round(base, 4), 'conditions': res, 'perhead_const_sum': 0.645,
           'collective_factor': round(ac/0.645, 2),
           'recov_front': round(1 - res['const_except_front']['cost']/max(ac, 1e-6), 3),
           'recov_dyn6': round(1 - res['const_except_dyn6']['cost']/max(ac, 1e-6), 3),
           'recov_middle': round(1 - res['const_except_middle']['cost']/max(ac, 1e-6), 3)}
    out['pred_a_collective'] = bool(ac >= 3*0.645 and res['all_const']['rare_freq_ratio'] >= 2)
    out['pred_b_middle_over_front'] = bool(out['recov_middle'] > out['recov_front'] and out['recov_dyn6'] < 0.4)
    out['pred_c_no_collective'] = bool(ac <= 1.3*0.645)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"collective factor {out['collective_factor']} | recov front {out['recov_front']} dyn6 {out['recov_dyn6']} middle {out['recov_middle']}", flush=True)
    print(f"preds: a collective {out['pred_a_collective']} | b middle>front {out['pred_b_middle_over_front']} | c no-collective {out['pred_c_no_collective']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
