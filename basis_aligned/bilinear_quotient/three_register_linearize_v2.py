"""RIGOROUS version of §949: disentangle MAGNITUDE from NONLINEARITY-FRACTION per register, with per-register
shuffled nulls. For each register R (FRONT L0-5, MIDDLE L6-15, READOUT L16-17), with the other registers REAL,
measure held-out CE cost of:
  C_mean  = mean-ablate all of R's MLP outputs (R's total magnitude/contribution),
  C_lin   = replace R's MLPs with fitted linear maps of their inputs (residual after best linear map),
  C_shuf  = replace with SHUFFLED-input linear maps (per-register null).
Then nonlinear_fraction(R) = C_lin / C_mean (magnitude-normalized: what fraction of R's contribution survives
the best linear map = genuinely nonlinear), and the linear map is CERTIFIED genuine iff C_lin << C_shuf.
This resolves §949: is the front's large linearize-cost genuine nonlinearity or generic damage, and is the
front near-linear in FRACTION (§941) while large in ABSOLUTE nonlinearity (magnitude)?

REGISTERED PREDICTIONS:
  (0) SANITY: C_lin < C_shuf for each register (the fitted map genuinely helps); C_mean >= C_lin.
  (a) FRONT = near-linear FRACTION but large ABSOLUTE: nonlinear_fraction(front) is SMALL (< middle's) confirming
      §941's per-layer shape, WHILE absolute C_lin(front) > C_lin(middle) (§949) because the front's magnitude
      C_mean is much larger; middle has the HIGHEST nonlinear_fraction; readout low both ways;
  (b) report C_mean, C_lin, C_shuf, nonlinear_fraction per register."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'three_register_linearize_v2_results.json'
NEVAL = 200; SEQ = 256; RIDGE_MAP = 1e3
REGS = {'front': list(range(0, 6)), 'middle': list(range(6, 16)), 'readout': [16, 17]}
MODE = {'layers': set(), 'kind': 'off', 'maps': None, 'shufmaps': None, 'means': None}


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def hook(L):
    def h(mo, i_, o_):
        if L not in MODE['layers'] or MODE['kind'] == 'off': return o_
        x = i_[0] if isinstance(i_, tuple) else i_; y = o_[0] if isinstance(o_, tuple) else o_; sh = y.shape
        if MODE['kind'] == 'mean':
            yn = MODE['means'][L].view(1, 1, D).expand(sh[0], sh[1], D).clone()
        elif MODE['kind'] == 'lin':
            yn = (x.reshape(-1, D) @ MODE['maps'][L]).reshape(sh)
        else:
            yn = (x.reshape(-1, D) @ MODE['shufmaps'][L]).reshape(sh)
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return h


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return readout(x)


@torch.no_grad()
def ce_pass(blocks):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lg = forward_logits(idx).float(); tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction='sum')); n += tgt.numel()
    return tot / n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); nb = blocks.shape[0]; ntr = int(0.7*nb)
    TRAIN = np.zeros(nb, bool); TRAIN[:ntr] = True; trm = torch.tensor(np.repeat(TRAIN, SEQ-1), device=DEV)
    store_in = {L: [] for L in range(18)}; store_out = {L: [] for L in range(18)}; hs = []
    for L in range(18):
        def mk(L):
            def h(mo, i_, o_):
                store_in[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D).cpu())
                store_out[L].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D).cpu())
            return h
        hs.append(m.transformer.h[L].mlp.register_forward_hook(mk(L)))
    for i in range(0, nb, 8): forward_logits(blocks[i:i+8].to(DEV)[:, :-1].contiguous())
    for h in hs: h.remove()
    maps = {}; shufmaps = {}; means = {}
    for L in range(18):
        Xin = torch.cat(store_in[L], 0).to(DEV); O = torch.cat(store_out[L], 0).to(DEV)
        A = Xin[trm].T @ Xin[trm] + RIDGE_MAP*torch.eye(D, device=DEV)
        maps[L] = torch.linalg.solve(A, Xin[trm].T @ O[trm]); means[L] = O.mean(0)
        pr = torch.randperm(int(trm.sum()), generator=torch.Generator(device=DEV).manual_seed(0), device=DEV)
        shufmaps[L] = torch.linalg.solve(A, Xin[trm].T @ O[trm][pr]); del Xin, O
    MODE['maps'] = maps; MODE['shufmaps'] = shufmaps; MODE['means'] = means
    hooks = [m.transformer.h[L].mlp.register_forward_hook(hook(L)) for L in range(18)]
    test = blocks[~TRAIN]
    MODE['kind'] = 'off'; MODE['layers'] = set(); ce_full = ce_pass(test)
    out = {'ce_full': round(ce_full, 4), 'registers': {}}
    for name, S in REGS.items():
        MODE['layers'] = set(S)
        MODE['kind'] = 'mean'; c_mean = ce_pass(test) - ce_full
        MODE['kind'] = 'lin'; c_lin = ce_pass(test) - ce_full
        MODE['kind'] = 'shuf'; c_shuf = ce_pass(test) - ce_full
        nf = c_lin / max(c_mean, 1e-6)
        out['registers'][name] = {'C_mean': round(c_mean, 4), 'C_lin': round(c_lin, 4), 'C_shuf': round(c_shuf, 4),
                                  'nonlinear_fraction': round(float(nf), 4), 'linmap_certified': bool(c_lin < 0.6*c_shuf)}
        print(f"{name:>8}: C_mean {c_mean:+.3f} C_lin {c_lin:+.3f} C_shuf {c_shuf:+.3f} | nonlin-frac {nf:.3f} | certified {out['registers'][name]['linmap_certified']}", flush=True)
    MODE['kind'] = 'off'; MODE['layers'] = set()
    for h in hooks: h.remove()
    r = out['registers']
    out['pred_a_front_lowfrac_highabs'] = bool(r['front']['nonlinear_fraction'] < r['middle']['nonlinear_fraction'] and r['front']['C_lin'] > r['middle']['C_lin'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"(a) front low nonlinear-fraction but high absolute C_lin (magnitude): {out['pred_a_front_lowfrac_highabs']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
