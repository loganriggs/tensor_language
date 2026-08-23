"""THREAD C continuation: L4 is the FIRST TRUE CONTEXT MLP (§1084/§1088: tok-only recovers ~0.05, nothing partial
works; highest dev-share of any layer 0.47) and separately mlp4 manufactures the position-0 sink constant (§439).
At ORDINARY positions, WHAT context does L4's dev×dev term consume? Candidates: (a) the CONTENT-subspace deviation
(the pooled topic signal, U_c = top-64 of pooled L8-12 deviations) vs (b) non-content deviation (local/positional/
class residue). Substitute L4's output with the MLP applied to reduced inputs: mtok + content-projected dev |
mtok + non-content dev | mtok + random-64-projected dev (control) | mtok only | full (sanity). CE cost vs
mean-ablation; rare/frequent split of the mean-ablation cost for L4's signature.

REGISTERED PREDICTIONS:
  (0) SANITY: full-input substitution ~0 cost; mtok-only reproduces §1084's ~0.05 recovery of the ~0.10 gap.
  (a) CONTENT-FED: mtok + content-dev recovers >= 60% of L4's mean-ablation gap while random-64 dev adds < half
      of what content-dev adds over mtok-only -> L4's context input is the CONTENT signal (it is the first
      content×content multiplier, pushing §1041's deep-middle structure down to L4);
  (b) NON-CONTENT-FED: if instead mtok + non-content dev recovers more than mtok + content-dev, L4 consumes a
      different (local/class) context variable -> the transition band computes something the content basis
      misses (report which; that would make L4 a genuinely new variable's home);
  (c) L4's mean-ablation rare/freq ratio reported (content machinery ~2+, grammar ~1, §1075)."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'l4_function_results.json'
NSEQ = 96; SEQ = 256; L4 = 4; REF = [8, 10, 12]; K = 64; RARE_MAX = 2
H = m.transformer.h
SUB = {'mode': None}
ST = {}; CUR = {}


def fwd(idx):
    CUR['tok'] = idx
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def sub_hook(mo, i_, o_):
    if SUB['mode'] is None: return None
    x = (i_[0] if isinstance(i_, tuple) else i_)
    mt = ST['xbar'][CUR['tok']].to(x.dtype)
    dv = x - mt
    md = SUB['mode']
    if md == 'meanabl':
        return ST['obar'].view(1, 1, D).expand_as(o_).to(o_.dtype)
    if md == 'full': xin = x
    elif md == 'mtok': xin = mt
    elif md == 'content': xin = mt + ((dv.float() @ ST['Uc']) @ ST['Uc'].T).to(x.dtype)
    elif md == 'noncontent': xin = x - ((dv.float() @ ST['Uc']) @ ST['Uc'].T).to(x.dtype)
    elif md == 'rand64': xin = mt + ((dv.float() @ ST['Ur']) @ ST['Ur'].T).to(x.dtype)
    y = mo.Down(mo.Left(xin)*mo.Right(xin)) + mo.Down_bias
    return y.to(o_.dtype)


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

    # pass 1: L4 input mean per token + output mean + content basis (pooled L8-12 deviation)
    capL4, capO, capR = [], [], {Lr: [] for Lr in REF}
    def cap4(mo, i_, o_):
        capL4.append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
        capO.append(o_.detach().float().reshape(-1, D))
        return None
    hs = [H[L4].mlp.register_forward_hook(cap4)]
    for Lr in REF:
        def mk(Lr):
            def h(mo, i_, o_): capR[Lr].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(H[Lr].mlp.register_forward_hook(mk(Lr)))
    idsL = []
    for i in range(0, NSEQ, 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fwd(idx)
    for h in hs: h.remove()
    tok = torch.cat(idsL, 0)
    X4 = torch.cat(capL4, 0); capL4.clear()
    xb = torch.zeros(V, D, device=DEV); cn = torch.zeros(V, device=DEV)
    xb.index_add_(0, tok, X4); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
    ST['xbar'] = (xb/cn.clamp_min(1).unsqueeze(1)).half()
    ST['obar'] = torch.cat(capO, 0).mean(0); capO.clear()
    devsum = None
    for Lr in REF:
        X = torch.cat(capR[Lr], 0); capR[Lr] = []
        xbr = torch.zeros(V, D, device=DEV); xbr.index_add_(0, tok, X)
        dv = X - (xbr/cn.clamp_min(1).unsqueeze(1))[tok]
        devsum = dv if devsum is None else devsum + dv; del X
    dev = devsum/len(REF); dev = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(dev, full_matrices=False); ST['Uc'] = Vt[:K].T.contiguous()
    g = torch.Generator(device=DEV).manual_seed(0)
    ST['Ur'] = torch.linalg.qr(torch.randn(D, K, generator=g, device=DEV))[0]
    del dev, X4

    hk = H[L4].mlp.register_forward_hook(sub_hook)
    SUB['mode'] = None
    base, base_r, base_f = ce_split(blocks, is_rare)
    res = {}
    for md in ['full', 'mtok', 'content', 'noncontent', 'rand64', 'meanabl']:
        SUB['mode'] = md
        c, cr, cf = ce_split(blocks, is_rare)
        res[md] = {'cost': round(c-base, 4), 'rare': round(cr-base_r, 4), 'freq': round(cf-base_f, 4)}
        SUB['mode'] = None
        print(f"{md:>11}: cost {res[md]['cost']} | rare {res[md]['rare']} | freq {res[md]['freq']}", flush=True)
    hk.remove()
    abl = max(res['meanabl']['cost'], 1e-6)
    def recov(md): return round(1 - res[md]['cost']/abl, 3)
    out = {'base_ce': round(base, 4), 'conditions': res,
           'recov': {md: recov(md) for md in ['full', 'mtok', 'content', 'noncontent', 'rand64']},
           'rare_freq_ratio_meanabl': round(res['meanabl']['rare']/max(res['meanabl']['freq'], 1e-4), 2)}
    add_content = out['recov']['content'] - out['recov']['mtok']
    add_rand = out['recov']['rand64'] - out['recov']['mtok']
    out['pred_a_content_fed'] = bool(out['recov']['content'] >= 0.6 and add_rand < 0.5*max(add_content, 1e-6))
    out['pred_b_noncontent_fed'] = bool(out['recov']['noncontent'] > out['recov']['content'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"recov: {out['recov']} | rare/freq(meanabl) {out['rare_freq_ratio_meanabl']}", flush=True)
    print(f"pred_a content-fed {out['pred_a_content_fed']} | pred_b noncontent-fed {out['pred_b_noncontent_fed']} | wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
