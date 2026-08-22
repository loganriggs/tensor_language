"""CAPSTONE: is the MIDDLE the ONLY essential nonlinearity? Replace MLP outputs with a fitted LINEAR map of their
input (per §941), by register, and measure held-out whole-model CE. Registers: FRONT (L0-5), MIDDLE (L6-15),
READOUT (L16-17). If linearizing front and readout costs little but linearizing the middle costs a lot, then the
model is: linear front + linear readout + a genuinely nonlinear middle — the middle multiplication is the sole
essential nonlinearity. Maps fit on 70% of rows, CE on held-out 30%.

REGISTERED PREDICTIONS:
  (0) SANITY: linearizing 0 MLPs == full CE; a shuffled-input linear map (null) is far worse.
  (a) MIDDLE IS THE ONLY ESSENTIAL NONLINEARITY: linearizing FRONT MLPs and linearizing READOUT MLPs each cost
      LITTLE; linearizing the MIDDLE MLPs costs a LOT; linearizing ALL ~= linearizing the middle alone (front and
      readout add little extra cost) -> the middle multiplication is the sole essential nonlinearity;
  (b) report held-out CE cost for linearize-front / -middle / -readout / -front+readout / -all + shuffled null."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'three_register_linearize_results.json'
NEVAL = 200; SEQ = 256; RIDGE_MAP = 1e3
FRONT = list(range(0, 6)); MIDDLE = list(range(6, 16)); READOUT = [16, 17]
LIN = {'layers': set(), 'maps': None, 'shuffle': False, 'shufmaps': None}


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def lin_hook(L):
    def h(mo, i_, o_):
        if L not in LIN['layers']: return o_
        x = i_[0] if isinstance(i_, tuple) else i_; sh = (o_[0] if isinstance(o_, tuple) else o_).shape
        M = LIN['shufmaps'][L] if LIN['shuffle'] else LIN['maps'][L]
        yn = (x.reshape(-1, D) @ M).reshape(sh)
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
    # capture each MLP input+output
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
    maps = {}; shufmaps = {}
    for L in range(18):
        Xin = torch.cat(store_in[L], 0).to(DEV); O = torch.cat(store_out[L], 0).to(DEV)
        A = Xin[trm].T @ Xin[trm] + RIDGE_MAP*torch.eye(D, device=DEV)
        maps[L] = torch.linalg.solve(A, Xin[trm].T @ O[trm])
        pr = torch.randperm(int(trm.sum()), generator=torch.Generator(device=DEV).manual_seed(0), device=DEV)
        shufmaps[L] = torch.linalg.solve(A, Xin[trm].T @ O[trm][pr])
        del Xin, O
    LIN['maps'] = maps; LIN['shufmaps'] = shufmaps
    hooks = [m.transformer.h[L].mlp.register_forward_hook(lin_hook(L)) for L in range(18)]
    test = blocks[~TRAIN]
    LIN['layers'] = set(); ce_full = ce_pass(test)
    out = {'ce_full': round(ce_full, 4), 'cost': {}}
    sets = {'front': FRONT, 'middle': MIDDLE, 'readout': READOUT, 'front+readout': FRONT+READOUT, 'all': list(range(18))}
    for name, S in sets.items():
        LIN['layers'] = set(S); LIN['shuffle'] = False; ce = ce_pass(test)
        out['cost'][name] = round(ce - ce_full, 4)
        print(f"linearize {name:>14}: Δce {ce-ce_full:+.4f}", flush=True)
    LIN['layers'] = set(range(18)); LIN['shuffle'] = True; ce_sh = ce_pass(test); LIN['shuffle'] = False
    out['shuffled_null_all'] = round(ce_sh - ce_full, 4)
    for h in hooks: h.remove()
    c = out['cost']
    out['pred_a_middle_only_nonlinearity'] = bool(c['front'] < 0.3 and c['readout'] < 0.3 and c['middle'] > 1.0 and abs(c['all'] - c['middle']) < 0.5*c['middle'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"shuffled-null (all) Δce {out['shuffled_null_all']:+.4f}", flush=True)
    print(f"(a) middle is the only essential nonlinearity: {out['pred_a_middle_only_nonlinearity']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
