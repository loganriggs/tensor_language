"""Reconcile §842 (mlp0's top-24 CLASS units are a self-square: Left & Right read the SAME class, sharpening) with §1077
(FULL mlp0: Left/Right weight rows ~orthogonal, forcing global self-product Right:=Left costs +2.4 nats). Hypothesis:
mlp0 is a MIX -- a self-square SUBSET (Left(x)_i and Right(x)_i highly correlated per neuron, the §842 class units) plus
a CONJUNCTION majority (low per-neuron correlation, the §1077 cost source). Test: per-neuron activation correlation
corr(Left(x)_i, Right(x)_i) for mlp0; then FORCE self-product (Right:=Left) restricted to the HIGH-corr quartile vs the
LOW-corr quartile and measure CE cost. If high-corr units are cheap to force and low-corr units expensive, §842 and §1077
are reconciled: mlp0 has both self-square and conjunction neurons.

REGISTERED PREDICTIONS:
  (0) SANITY: forcing self-product on a HIGH-corr neuron is ~free (Left~=Right there); random-quartile null in between.
  (a) RECONCILED MIX: mlp0 has a nonzero self-square subset (frac neurons with corr>0.7 clearly >0) AND forcing self-
      product on the LOW-corr quartile costs much more than on the HIGH-corr quartile -> §842 (class self-square subset)
      and §1077 (full-gate conjunction) are both correct, describing different neuron subsets;
  (b) report the corr distribution + high/low/random-quartile force-self-product CE costs (mlp0), with mlp8 corr for
      contrast."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'gate_reconcile_results.json'
NEVAL = 160; SEQ = 256
SUB = {'L': None, 'mask': None}   # mask: bool (HID,) neurons where Right:=Left is forced


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def force_hook(L):
    mlp = m.transformer.h[L].mlp
    def h(mo, i_, o_):
        if SUB['L'] != L or SUB['mask'] is None: return None
        x = (i_[0] if isinstance(i_, tuple) else i_).float()
        Lx = mlp.Left(x); Rx = mlp.Right(x)
        gate = Lx * torch.where(SUB['mask'], Lx, Rx)   # Right:=Left on masked neurons
        ny = mlp.Down(gate) + mlp.Down_bias
        return ny.to((o_[0] if isinstance(o_, tuple) else o_).dtype)
    return h


@torch.no_grad()
def neuron_corr(L, blocks):
    mlp = m.transformer.h[L].mlp; capL = []; capR = []; hs = []
    hs.append(mlp.Left.register_forward_hook(lambda mo, i_, o_: capL.append(o_.detach().float().reshape(-1, o_.shape[-1]))))
    hs.append(mlp.Right.register_forward_hook(lambda mo, i_, o_: capR.append(o_.detach().float().reshape(-1, o_.shape[-1]))))
    for i in range(0, blocks.shape[0], 8): fwd(blocks[i:i+8].to(DEV)[:, :-1].contiguous())
    for h in hs: h.remove()
    Lx = torch.cat(capL, 0); Rx = torch.cat(capR, 0)
    Lc = Lx - Lx.mean(0); Rc = Rx - Rx.mean(0)
    corr = (Lc*Rc).sum(0) / (Lc.norm(dim=0)*Rc.norm(dim=0) + 1e-8)   # (HID,)
    return corr


@torch.no_grad()
def ce(blocks):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(fwd(idx).float(), -1); tf = tgt.reshape(-1)
        tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tf.shape[0], device=DEV), tf].sum()); n += tf.shape[0]
    return tot / n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous()
    corr0 = neuron_corr(0, blocks); corr8 = neuron_corr(8, blocks)
    HID = corr0.shape[0]; q = HID // 4
    order = torch.argsort(corr0, descending=True)
    high_q = torch.zeros(HID, dtype=torch.bool, device=DEV); high_q[order[:q]] = True     # top-corr quartile
    low_q = torch.zeros(HID, dtype=torch.bool, device=DEV); low_q[order[-q:]] = True       # bottom-corr quartile
    g = torch.Generator(device=DEV).manual_seed(0); randsel = torch.randperm(HID, generator=g, device=DEV)[:q]
    rand_q = torch.zeros(HID, dtype=torch.bool, device=DEV); rand_q[randsel] = True
    hooks = [m.transformer.h[0].mlp.register_forward_hook(force_hook(0))]
    SUB['L'] = None; base = ce(blocks)
    def cost(mask): SUB['L'] = 0; SUB['mask'] = mask; c = ce(blocks); SUB['L'] = None; SUB['mask'] = None; return round(c-base, 4)
    out = {'base_ce': round(base, 4), 'HID': HID,
           'mlp0_corr': {'mean': round(float(corr0.mean()), 4), 'frac_gt_0.7': round(float((corr0 > 0.7).float().mean()), 4),
                         'frac_gt_0.5': round(float((corr0 > 0.5).float().mean()), 4), 'frac_abs_gt_0.5': round(float((corr0.abs() > 0.5).float().mean()), 4)},
           'mlp8_corr': {'mean': round(float(corr8.mean()), 4), 'frac_gt_0.7': round(float((corr8 > 0.7).float().mean()), 4)},
           'force_selfproduct_cost': {'high_corr_quartile': cost(high_q), 'low_corr_quartile': cost(low_q), 'random_quartile': cost(rand_q)}}
    for h in hooks: h.remove()
    hi = out['force_selfproduct_cost']['high_corr_quartile']; lo = out['force_selfproduct_cost']['low_corr_quartile']
    out['pred_a_reconciled_mix'] = bool(out['mlp0_corr']['frac_gt_0.7'] > 0.0 and lo > 3*max(hi, 1e-4))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"mlp0 corr: mean {out['mlp0_corr']['mean']} | frac>0.7 {out['mlp0_corr']['frac_gt_0.7']} | frac|.|>0.5 {out['mlp0_corr']['frac_abs_gt_0.5']} (mlp8 mean {out['mlp8_corr']['mean']})", flush=True)
    print(f"force self-product cost: high-corr Q {hi} | low-corr Q {lo} | random Q {out['force_selfproduct_cost']['random_quartile']}", flush=True)
    print(f"pred_a reconciled mix: {out['pred_a_reconciled_mix']} | wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
