"""BOTTOM-UP, top of the stack: the READOUT band (L16-17). These read the built residual to form the output. Measure
each readout module's loss-recovery under a LINEAR map of its own input (held-out) -- if high, the readout is
understood as a (near-)linear read of the residual. For the MLPs also compare the model's-own-neuron bilinear-rank
recovery (§1040: mlp16 rank-64 = 0.94). Completes the per-module bottom-up map.

REGISTERED PREDICTIONS:
  (0) SANITY: mean-ablate = 0; shuffled-input linear ~0.
  (a) READOUT IS A ~LINEAR READ: attn16/17 and mlp16/17 reach HIGH loss-recovery under a linear map of their input
      (target > 0.7) -> the readout is understood as a near-linear read of the residual (consistent with §941 L17
      ~85% linear);
  (b) report linear-map loss-recovery per readout module + shuffled-input null."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'bottomup_readout_results.json'
NEVAL = 200; SEQ = 256; MODS = [('attn', 16), ('mlp', 16), ('attn', 17), ('mlp', 17)]; RIDGE = 1e2
SUB = {'tag': None, 'mode': None, 'M': {}, 'gmean': {}}


def submod(kind, L): return getattr(m.transformer.h[L], kind)


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def hook_factory(tag, kind, L):
    def h(mo, i_, o_):
        if SUB['tag'] != tag or SUB['mode'] is None: return None
        x = (i_[0] if isinstance(i_, tuple) else i_).float()
        o = o_[0] if isinstance(o_, tuple) else o_; B, T, _ = o.shape
        if SUB['mode'] == 'mean':
            ny = SUB['gmean'][tag].view(1, 1, D).expand(B, T, D)
        else:
            xf = x.reshape(-1, D)
            if SUB['mode'] == 'shuf': xf = xf[torch.randperm(xf.shape[0], device=DEV)]
            x1 = torch.cat([xf, torch.ones(xf.shape[0], 1, device=DEV)], 1)
            ny = (x1 @ SUB['M'][tag]).reshape(B, T, D)
        return (ny.to(o.dtype),) + tuple(o_[1:]) if isinstance(o_, tuple) else ny.to(o.dtype)
    return h


@torch.no_grad()
def capture(blocks):
    caps = {f'{k}{L}': [] for k, L in MODS}; xin = {f'{k}{L}': [] for k, L in MODS}; hs = []
    for kind, L in MODS:
        tag = f'{kind}{L}'; mod = submod(kind, L)
        def mk(tag, mod):
            def h(mo, i_, o_):
                xin[tag].append((i_[0] if isinstance(i_, tuple) else i_).float().reshape(-1, D).cpu())
                caps[tag].append((o_[0] if isinstance(o_, tuple) else o_).float().reshape(-1, D).cpu())
            return h
        hs.append(mod.register_forward_hook(mk(tag, mod)))
    SUB['tag'] = None
    for i in range(0, blocks.shape[0], 4): forward_logits(blocks[i:i+4].to(DEV)[:, :-1].contiguous())
    for h in hs: h.remove()
    return {t: torch.cat(caps[t], 0) for t in caps}, {t: torch.cat(xin[t], 0) for t in xin}


@torch.no_grad()
def ce(blocks):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1); tf = tgt.reshape(-1)
        tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tf.shape[0], device=DEV), tf].sum()); n += tf.shape[0]
    return tot / n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    nb = rows.shape[0]; ntr = int(0.6*nb); tr = rows[:ntr, :SEQ].contiguous(); te = rows[ntr:, :SEQ].contiguous()
    Y, X = capture(tr)
    for kind, L in MODS:
        tag = f'{kind}{L}'; Xl = X[tag].to(DEV); Yl = Y[tag].to(DEV)
        X1 = torch.cat([Xl, torch.ones(Xl.shape[0], 1, device=DEV)], 1)
        SUB['M'][tag] = torch.linalg.solve(X1.T @ X1 + RIDGE*torch.eye(D+1, device=DEV), X1.T @ Yl)
        SUB['gmean'][tag] = Yl.mean(0); del Xl, Yl
    hooks = [submod(k, L).register_forward_hook(hook_factory(f'{k}{L}', k, L)) for k, L in MODS]
    SUB['tag'] = None; ce_full = ce(te); print(f"ce_full {ce_full:.4f}", flush=True)
    out = {'ce_full': round(ce_full, 4), 'modules': {}}
    for kind, L in MODS:
        tag = f'{kind}{L}'; SUB['tag'] = tag
        SUB['mode'] = 'mean'; ce_ma = ce(te); denom = max(ce_ma - ce_full, 1e-6)
        SUB['mode'] = 'lin'; ce_lin = ce(te)
        SUB['mode'] = 'shuf'; ce_sh = ce(te)
        SUB['tag'] = None
        rl = round(float((ce_ma - ce_lin)/denom), 3); rs = round(float((ce_ma - ce_sh)/denom), 3)
        out['modules'][tag] = {'meanabl_cost': round(ce_ma-ce_full,3), 'recovery_linear': rl, 'recovery_shuffled_null': rs}
        print(f"{tag}: meanabl {ce_ma-ce_full:.3f} | linear-recovery {rl} | shuffled-null {rs}", flush=True)
    for h in hooks: h.remove()
    out['pred_a_readout_linear'] = bool(all(out['modules'][f'{k}{L}']['recovery_linear'] > 0.7 for k, L in MODS))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a readout is ~linear read: {out['pred_a_readout_linear']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
