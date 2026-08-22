"""WHY is the MIDDLE the reconstruction frontier (§940)? Hypothesis: the middle MLPs' output is dominated by the
BILINEAR MULTIPLICATIVE interaction (Down[(Left.x) (elementwise*) (Right.x)]), which no LINEAR read of the MLP's
input can capture; the front MLPs are more nearly a linear function of their input. Test causally per layer: fit a
ridge LINEAR MAP from each MLP's INPUT to its OUTPUT (train), replace the MLP output with that linear
reconstruction (held-out), and measure the loss recovery = (CE_meanablate - CE_linmap)/(CE_meanablate - CE_full).
The part a linear map CANNOT recover is the multiplicative nonlinearity.

REGISTERED PREDICTIONS:
  (0) SANITY: mean-ablating each MLP raises CE above full; the linear map recovers more than mean-ablate (>0).
  (a) MIDDLE IS MULTIPLICATIVE: the linear-recoverable fraction of MLP output is LOWER for middle MLPs (L6-11)
      than for the front MLP (L0/L1) -> the middle's irreducibility is the bilinear multiplicative interaction;
      front MLPs are more linearly-readable functions of their input;
  (b) report linear-recoverable fraction per layer + the multiplicative remainder (1 - fraction); null = a linear
      map from a SHUFFLED input (breaks input->output correspondence) recovers ~0."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'middle_bilinearity_results.json'
NEVAL = 200; SEQ = 256; RIDGE_MAP = 1e3
LAYERS = [0, 1, 4, 8, 11, 15, 17]
REPL = {'mode': 'off', 'L': -1, 'recon': None, 'gmean': None, 'row': 0}
ROW = [0]


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def mlp_hook(L):
    def h(mo, i_, o_):
        if REPL['mode'] == 'off' or REPL['L'] != L: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; B, T, _ = y.shape
        if REPL['mode'] == 'mean':
            yn = REPL['gmean'].expand(B, T, D).clone()
        else:
            yn = REPL['recon'][REPL['row']:REPL['row']+B*T].reshape(B, T, D).to(DEV)
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return h


@torch.no_grad()
def ce_pass(blocks):
    tot = 0.0; n = 0; REPL['row'] = ROW[0]
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lg = forward_logits(idx).float(); tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction='sum'))
        n += tgt.numel(); REPL['row'] += idx.shape[0]*(SEQ-1)
    return tot / n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); nb = blocks.shape[0]; ntr = int(0.7*nb)
    TRAIN = np.zeros(nb, bool); TRAIN[:ntr] = True; trm = np.repeat(TRAIN, SEQ-1); trm_t = torch.tensor(trm, device=DEV)
    # capture each MLP's input and output
    capin = {L: [] for L in LAYERS}; capout = {L: [] for L in LAYERS}; hs = []
    for L in LAYERS:
        def mk(L):
            def h(mo, i_, o_):
                xin = i_[0] if isinstance(i_, tuple) else i_
                capin[L].append(xin.detach().float().reshape(-1, D))
                capout[L].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
            return h
        hs.append(m.transformer.h[L].mlp.register_forward_hook(mk(L)))
    for i in range(0, nb, 8): forward_logits(blocks[i:i+8].to(DEV)[:, :-1].contiguous())
    for h in hs: h.remove()
    recons = {}; gmeans = {}; recons_shuf = {}
    for L in LAYERS:
        Xin = torch.cat(capin[L], 0); O = torch.cat(capout[L], 0); capin[L] = None; capout[L] = None
        A = Xin[trm_t].T @ Xin[trm_t] + RIDGE_MAP*torch.eye(D, device=DEV); M = torch.linalg.solve(A, Xin[trm_t].T @ O[trm_t])
        recons[L] = (Xin @ M).cpu(); gmeans[L] = O.mean(0)
        # shuffled-input null map (fit on shuffled correspondence)
        pr = torch.randperm(int(trm_t.sum()), generator=torch.Generator(device=DEV).manual_seed(0), device=DEV)
        Msh = torch.linalg.solve(A, Xin[trm_t].T @ O[trm_t][pr]); recons_shuf[L] = (Xin @ Msh).cpu()
        del Xin, O
    hooks = [m.transformer.h[L].mlp.register_forward_hook(mlp_hook(L)) for L in LAYERS]
    test = blocks[~TRAIN]; ROW[0] = ntr*(SEQ-1)
    REPL['mode'] = 'off'; REPL['L'] = -1; ce_full = ce_pass(test)
    out = {'ce_full': round(ce_full, 3), 'layers': {}}
    for L in LAYERS:
        REPL['L'] = L; REPL['gmean'] = gmeans[L]
        REPL['mode'] = 'mean'; ce_m = ce_pass(test)
        REPL['mode'] = 'set'; REPL['recon'] = recons[L]; ce_lin = ce_pass(test)
        REPL['recon'] = recons_shuf[L]; ce_sh = ce_pass(test)
        denom = max(ce_m - ce_full, 1e-6)
        frac = float((ce_m - ce_lin)/denom); fsh = float((ce_m - ce_sh)/denom)
        out['layers'][str(L)] = {'meanabl_cost': round(ce_m-ce_full, 3), 'linear_recoverable_frac': round(frac, 3),
                                 'multiplicative_remainder': round(1-frac, 3), 'shuffled_frac': round(fsh, 3)}
        print(f"L{L:>2} mlp: meanabl-cost {ce_m-ce_full:.3f} | linear-recoverable {frac:.3f} | multiplicative {1-frac:.3f} (shuf {fsh:.3f})", flush=True)
    for h in hooks: h.remove()
    front = np.mean([out['layers'][str(L)]['linear_recoverable_frac'] for L in [0, 1]])
    mid = np.mean([out['layers'][str(L)]['linear_recoverable_frac'] for L in [8, 11] if str(L) in out['layers']])
    out['front_L0_1_linfrac'] = round(float(front), 3); out['middle_L8_11_linfrac'] = round(float(mid), 3)
    out['pred_a_middle_more_multiplicative'] = bool(mid < front)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"front(L0-1) lin-frac {front:.3f} vs middle(L8-11) {mid:.3f}", flush=True)
    print(f"(a) middle more multiplicative (less linearly readable): {out['pred_a_middle_more_multiplicative']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
