"""THIRD INSTRUMENT FAMILY for the readout ceiling (open-problems plan; §1131-1133: fitted LINEAR and the
model's OWN top-k neurons both converge at 0.81-0.84 for mlp16/17 — the tail is neither sparse nor linear).
Untried: a FITTED low-rank quadratic — learn y ≈ Linear(x) + Σ_{r=1..R} (u_r·x)(v_r·x) w_r (rank-R bilinear with
free factors, R=64), trained on half A, CE-substituted on half B. If the tail is quadratic-but-not-in-the-model's-
own-neuron-basis, this finds it; if the ceiling holds across a third family, it hardens into a law.

REGISTERED PREDICTIONS:
  (0) SANITY: train R² beats linear's (0.91/0.95) — the quadratic term has capacity; held-out CE recovery >=
      the linear stand-in's (0.813/0.842) minus 0.02 (no overfit collapse).
  (a) CEILING BROKEN: fitted-bilinear held-out recovery >= 0.90 on either MLP -> the readout crosses the 90%
      line with instrument family #3; the tail was quadratic in a rotated basis;
  (b) CEILING LAW HARDENS: both land within 0.04 of their linear numbers -> three unrelated families converge;
      the readout tail is beyond quadratic (report as the module's final characterization);
  (c) intermediate: report where the gain saturates."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'readout_bilinear_fit_results.json'
NSEQ = 384; SEQ = 256; LAYERS = [16, 17]; R = 64; STEPS = 4000
H = m.transformer.h
SUB = {'layer': -1, 'mode': None}
ST = {'net': {}, 'obar': {}}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


class BilinFit(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.L = torch.nn.Parameter(torch.randn(D+1, D, generator=g)*0.01)
        self.U = torch.nn.Parameter(torch.randn(D, R, generator=g)*(D**-0.5))
        self.V = torch.nn.Parameter(torch.randn(D, R, generator=g)*(D**-0.5))
        self.W = torch.nn.Parameter(torch.randn(R, D, generator=g)*0.01)
    def forward(self, x):
        xf = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], 1)
        lin = xf @ self.L
        q = (x @ self.U) * (x @ self.V)
        return lin + q @ self.W


def sub_hook(L):
    def h(mo, i_, o_):
        if SUB['layer'] != L or SUB['mode'] is None: return None
        if SUB['mode'] == 'meanabl':
            return ST['obar'][L].view(1, 1, D).expand_as(o_).to(o_.dtype)
        x = (i_[0] if isinstance(i_, tuple) else i_).float()
        B, T, _ = x.shape
        y = ST['net'][L](x.reshape(-1, D)).view(B, T, D)
        return y.to(o_.dtype)
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
    rows = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    A = rows[:NSEQ//2]; B = rows[NSEQ//2:]

    capI = {L: [] for L in LAYERS}; capO = {L: [] for L in LAYERS}; hs = []
    for L in LAYERS:
        def mk(L):
            def h(mo, i_, o_):
                capI[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
                capO[L].append(o_.detach().float().reshape(-1, D))
            return h
        hs.append(H[L].mlp.register_forward_hook(mk(L)))
    for i in range(0, A.shape[0], 8): fwd(A[i:i+8].to(DEV)[:, :-1].contiguous())
    for h in hs: h.remove()

    train_r2 = {}
    for L in LAYERS:
        X = torch.cat(capI[L], 0); O = torch.cat(capO[L], 0); capI[L] = []; capO[L] = []
        ST['obar'][L] = O.mean(0)
        net = BilinFit(seed=L).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        with torch.enable_grad():
            for step in range(STEPS):
                ii = torch.randint(0, X.shape[0], (4096,), device=DEV)
                loss = ((net(X[ii]) - O[ii])**2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pred = torch.cat([net(X[i:i+8192]) for i in range(0, 40000, 8192)], 0)
            r2 = 1 - float(((pred - O[:pred.shape[0]])**2).sum()/((O[:pred.shape[0]] - O[:pred.shape[0]].mean(0))**2).sum())
        train_r2[L] = round(r2, 4)
        ST['net'][L] = net.eval()
        print(f"L{L}: fitted-bilinear train R2 {r2:.4f} (linear was {0.911 if L==16 else 0.947})", flush=True)
        del X, O

    hs = [H[L].mlp.register_forward_hook(sub_hook(L)) for L in LAYERS]
    SUB['layer'] = -1; base = ce(B)
    res = {}
    for L in LAYERS:
        row = {}
        for mode in ['bilin', 'meanabl']:
            SUB['layer'] = L; SUB['mode'] = mode if mode == 'meanabl' else 'bilin'
            row[mode] = round(ce(B) - base, 4)
            SUB['layer'] = -1; SUB['mode'] = None
        row['recov'] = round(1 - row['bilin']/max(row['meanabl'], 1e-6), 3)
        res[str(L)] = row
        print(f"L{L}: held-out fitted-bilinear recovery {row['recov']} (linear was {0.813 if L==16 else 0.842})", flush=True)
    for h in hs: h.remove()

    linref = {'16': 0.813, '17': 0.842}
    out = {'base_ce': round(base, 4), 'train_r2': {str(L): train_r2[L] for L in LAYERS}, 'per_layer': res,
           'gain_over_linear': {L2: round(res[L2]['recov'] - linref[L2], 3) for L2 in linref}}
    out['pred_a_ceiling_broken'] = bool(max(res[L2]['recov'] for L2 in res) >= 0.90)
    out['pred_b_ceiling_hardens'] = bool(all(abs(res[L2]['recov'] - linref[L2]) <= 0.04 for L2 in linref))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"gains over linear: {out['gain_over_linear']} | pred_a broken {out['pred_a_ceiling_broken']} | pred_b hardens {out['pred_b_ceiling_hardens']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
