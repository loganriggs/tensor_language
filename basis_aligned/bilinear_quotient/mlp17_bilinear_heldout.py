"""Completes the readout-ceiling story (§1132 found mlp16's two-instrument ceiling at 0.81): same weight-native
top-k neuron certification for MLP17 (fitted-linear certified 0.842, §1131). Selection on half A (energy =
||Down col|| x pre-activation std), CE keep-only substitution on half B; k = 64/128/256; random-64 control.

REGISTERED PREDICTIONS:
  (0) SANITY: all-neurons = 1.0; random-64 catastrophic; monotone in k.
  (a) SAME CEILING LAW: mlp17's top-256 recovery lands within 0.05 of its linear 0.842 (two instruments
      converge again) -> the readout MLPs share the ceiling structure: ~0.81-0.84 capturable, the tail neither
      sparse nor linear (module-level §660 law, now certified across the band);
  (b) if top-64 >= 0.9, mlp17 IS neuron-compact (unlike mlp16) — crosses the 90% line weight-natively
      (report which neurons; §660 said ~4 quadratic functions get 75% — a compact core exists)."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; HID = 4608; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp17_bilinear_heldout_results.json'
NSEQ = 384; SEQ = 256; L = 17
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
    out['pred_a_same_ceiling'] = bool(abs(res['top256']['recov'] - 0.842) <= 0.05)
    out['pred_b_compact'] = bool(res['top64']['recov'] >= 0.9)
    k90 = next((nm for nm in ['top64', 'top128', 'top256'] if res[nm]['recov'] >= 0.9), 'none<=256')
    out['smallest_k_for_090'] = k90
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"smallest k for 0.9: {k90} | pred_a same-ceiling {out['pred_a_same_ceiling']} | pred_b compact {out['pred_b_compact']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
