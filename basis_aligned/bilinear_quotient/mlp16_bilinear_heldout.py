"""Registered upgrade from §1131: §1040 found mlp16's effective bilinear loss-rank ≈ 64 (compact); if that
certifies held-out, mlp16 crosses the 90% line with a WEIGHT-NATIVE stand-in (the MLP's own top neurons — no
fitted tables). Method: rank mlp16's 4608 neurons on half A by activation energy (||Down[:,j]|| × std of
(Left_j·x)(Right_j·x) on A); keep only the top-k neurons (k = 64/128/256, plus k=4608 sanity) — the stand-in
IS the model's own bilinear restricted to k neurons; CE substitution on half B vs mean-ablation. Selection on
A only (no leak); also a RANDOM-k neuron control at k=64.

REGISTERED PREDICTIONS:
  (0) SANITY: k=4608 recovery = 1; recovery monotone in k; random-64 << top-64.
  (a) CERTIFIED ≥90%: top-64 held-out recovery >= 0.9 -> mlp16 = a 64-neuron bilinear circuit, certified;
      benchmark table updated (readout band then has mlp16 ✓90%, mlp17 0.84);
  (b) SELECTION-INFLATED: top-64 < 0.85 -> §1040's rank-64 was in-sample selection; report the certified k
      needed for 0.9 (the honest compactness number)."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; HID = 4608; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp16_bilinear_heldout_results.json'
NSEQ = 384; SEQ = 256; L = 16
H = m.transformer.h
SUB = {'mask': None, 'mode': None}
ST = {}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def sub_hook(mo, i_, o_):
    if SUB['mode'] is None: return None
    if SUB['mode'] == 'meanabl':
        return ST['obar'].view(1, 1, D).expand_as(o_).to(o_.dtype)
    x = (i_[0] if isinstance(i_, tuple) else i_)
    hpre = mo.Left(x) * mo.Right(x)
    hpre = hpre * SUB['mask'].to(hpre.dtype)
    y = mo.Down(hpre) + mo.Down_bias
    return y.to(o_.dtype)


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
    rows = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    A = rows[:NSEQ//2]; B = rows[NSEQ//2:]
    mlp = H[L].mlp

    # rank neurons on A: energy = ||Down[:,j]|| * std of pre-activation on A
    acts_sq = torch.zeros(HID, device=DEV); acts_mu = torch.zeros(HID, device=DEV); npos = 0
    capO = []
    def cap_hook(mo, i_, o_):
        x = (i_[0] if isinstance(i_, tuple) else i_)
        hpre = (mo.Left(x)*mo.Right(x)).detach().float().reshape(-1, HID)
        nonlocal_acc(hpre)
        capO.append(o_.detach().float().reshape(-1, D))
        return None
    def nonlocal_acc(hpre):
        nonlocal npos
        acts_sq.add_((hpre**2).sum(0)); acts_mu.add_(hpre.sum(0)); npos += hpre.shape[0]
    hk = mlp.register_forward_hook(cap_hook)
    for i in range(0, A.shape[0], 8): fwd(A[i:i+8].to(DEV)[:, :-1].contiguous())
    hk.remove()
    mu = acts_mu/npos; var = acts_sq/npos - mu**2
    dn = mlp.Down.weight.float().norm(dim=0)
    energy = dn * var.clamp_min(0).sqrt()
    order = energy.argsort(descending=True)
    ST['obar'] = torch.cat(capO, 0).mean(0); capO.clear()

    hk = mlp.register_forward_hook(sub_hook)
    SUB['mode'] = None; base = ce(B)
    res = {}
    g = torch.Generator().manual_seed(0)
    conds = {'top64': order[:64], 'top128': order[:128], 'top256': order[:256],
             'rand64': torch.randperm(HID, generator=g)[:64].to(DEV), 'all': order}
    SUB['mode'] = 'meanabl'; abl = ce(B) - base; SUB['mode'] = None
    for nm, keep in conds.items():
        msk = torch.zeros(HID, device=DEV); msk[keep] = 1.0
        SUB['mask'] = msk; SUB['mode'] = 'keep'
        c = ce(B) - base
        SUB['mode'] = None
        res[nm] = {'cost': round(c, 4), 'recov': round(1 - c/max(abl, 1e-6), 3)}
        print(f"{nm:>7}: cost {c:+.4f} | recovery {res[nm]['recov']}", flush=True)
    hk.remove()

    out = {'base_ce': round(base, 4), 'meanabl': round(abl, 4), 'conditions': res}
    out['pred_a_certified_90'] = bool(res['top64']['recov'] >= 0.9)
    out['pred_b_selection_inflated'] = bool(res['top64']['recov'] < 0.85)
    k90 = next((nm for nm in ['top64', 'top128', 'top256'] if res[nm]['recov'] >= 0.9), 'none<=256')
    out['smallest_k_for_090'] = k90
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"smallest k for 0.9: {k90} | pred_a certified90 {out['pred_a_certified_90']} | pred_b inflated {out['pred_b_selection_inflated']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
