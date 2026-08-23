"""FIRST SCOPING of the construction-simulation frontier (§1118/§1121: the whole-model gap = simulating the
content CONSTRUCTION, not reading the result). Question: is the construction SIMPLE IN COORDINATE SPACE?
The content object is high-rank in D-space, but its per-layer evolution might be low-complexity dynamics on the
64 coords. Train a small per-step map f_L: c_L(pos), tokenfeat(pos), pool_L(pos) -> c_{L+1}(pos), where c_L =
the layer-L residual-stream content coords (shared U_c basis), tokenfeat = the token's embedding projected to
32 dims, pool_L = causal mean of c_L over previous positions (the cheap context signal each layer's attention
adds). f = 2-layer MLP (width 256). Evaluate MULTI-STEP ROLLOUT: start from the TRUE c_5, roll f through L5->14,
measure per-position cosine/R² of predicted c_14 vs true, held-out rows. Baselines: PERSISTENCE (c stays),
LINEAR per-step map (same inputs), and a token-only map (no c_L input — is the dynamics even stateful?).

REGISTERED PREDICTIONS:
  (0) SANITY: one-step R² high for all learned maps (>0.8; adjacent-layer coords are §1049-similar);
      persistence one-step also high (drift is slow) — the TEST is multi-step rollout where errors compound.
  (a) SIMPLE DYNAMICS: the small nonlinear map rolls out L5->L14 at cosine >= 0.6 with true c_14, beating
      persistence by >= 0.15 -> the construction is low-complexity in coord space; the scratch is simple
      dynamics and a coordinate-space simulator is a viable route to closing the benchmark gap;
  (b) CONTEXT-LIMITED: if the rollout collapses toward persistence, each step needs context the pooled-mean
      input can't carry (real attention gathering per layer) — construction is irreducibly interactive and
      the frontier needs attention-in-the-loop simulation (report plainly);
  (c) STATELESS CHECK: if the token-only map matches the stateful one, the "dynamics" is just per-position
      feature computation, not evolution (would simplify everything — report plainly)."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'coord_dynamics_results.json'
NSEQ = 192; SEQ = 256; K = 64; BAND = list(range(5, 15)); EMB32 = 32; STEPS = 4000
H = m.transformer.h


class StepMLP(torch.nn.Module):
    def __init__(self, din, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w1 = torch.nn.Parameter(torch.randn(din, 256, generator=g)*(din**-0.5))
        self.b1 = torch.nn.Parameter(torch.zeros(256))
        self.w2 = torch.nn.Parameter(torch.randn(256, K, generator=g)*(256**-0.5))
        self.b2 = torch.nn.Parameter(torch.zeros(K))
    def forward(self, x): return torch.relu(x @ self.w1 + self.b1) @ self.w2 + self.b2


def fit_linear(X, Y):
    A = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], 1)
    Wl = torch.linalg.lstsq(A, Y).solution
    return lambda Z: torch.cat([Z, torch.ones(Z.shape[0], 1, device=Z.device)], 1) @ Wl


def train_mlp(X, Y, seed=0):
    net = StepMLP(X.shape[1], seed).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    with torch.enable_grad():
        for step in range(STEPS):
            ii = torch.randint(0, X.shape[0], (4096,), device=DEV)
            loss = ((net(X[ii]) - Y[ii])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return net


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0])
    nb = NSEQ; ntr = int(0.7*nb)
    T = SEQ - 1

    # capture residual stream after every band block (+L4 as the L5 input? we start at c_5 measured AFTER block 5)
    caps = {L: [] for L in BAND}
    def fcap(idx):
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for Li, blk in enumerate(H):
            x, v1 = blk(x, v1, x0)
            if Li in BAND: caps[Li].append(x.detach().float().reshape(-1, D))
    idsL = []
    for i in range(0, nb, 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fcap(idx)
    tok = torch.cat(idsL, 0)
    cn = torch.zeros(V, device=DEV); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))

    # shared basis: pooled deviations across the band's mid refs (8,10,12) as usual
    devsum = None; xbars = {}
    for L in BAND:
        X = torch.cat(caps[L], 0); caps[L] = X   # keep raw
        xb = torch.zeros(V, D, device=DEV); xb.index_add_(0, tok, X)
        xbars[L] = xb/cn.clamp_min(1).unsqueeze(1)
        if L in (8, 10, 12):
            dv = X - xbars[L][tok]
            devsum = dv if devsum is None else devsum + dv
    dev = devsum/3; dev = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(dev, full_matrices=False)
    Uc = Vt[:K].T.contiguous(); del dev, devsum

    # coords per layer (deviation-based, per-layer xbar)
    C = {L: ((caps[L] - xbars[L][tok]) @ Uc).view(nb, T, K) for L in BAND}
    for L in BAND: caps[L] = None
    # token feature: embedding projected to 32 dims
    _, _, Vte = torch.linalg.svd(m.transformer.wte.weight.float()[torch.unique(tok)], full_matrices=False)
    Pe = Vte[:EMB32].T.contiguous()
    TF = (m.transformer.wte(tok).float() @ Pe).view(nb, T, EMB32)
    # causal pooled mean of c_L
    def cpool(Cl):
        cs = Cl.cumsum(1)
        cnt = torch.arange(1, T+1, device=DEV).view(1, T, 1).float()
        return cs/cnt

    trm = torch.zeros(nb, dtype=torch.bool); trm[:ntr] = True
    def flat(x, mask): return x[mask].reshape(-1, x.shape[-1])

    # train per-step maps
    maps_mlp = {}; maps_lin = {}; maps_tokonly = {}
    for j in range(len(BAND)-1):
        L, Ln = BAND[j], BAND[j+1]
        Xin = torch.cat([C[L], TF, cpool(C[L])], -1)
        Xtr = flat(Xin, trm); Ytr = flat(C[Ln], trm)
        maps_mlp[L] = train_mlp(Xtr, Ytr, seed=j)
        maps_lin[L] = fit_linear(Xtr, Ytr)
        Xtok = flat(torch.cat([TF, cpool(C[L])*0], -1), trm)   # token-only (no state, no pool)
        maps_tokonly[L] = train_mlp(torch.cat([Xtr[:, K:K+EMB32], Xtr[:, K+EMB32:]*0], 1) if False else Xtok, Ytr, seed=100+j)
        print(f"step L{L}->L{Ln} trained", flush=True)

    # evaluation: one-step + rollout on held-out rows
    ev = ~trm
    def cosR2(pred, true):
        cos = float(F.cosine_similarity(pred, true, dim=-1).mean())
        r2 = 1 - float(((pred-true)**2).sum()/((true - true.mean(0))**2).sum())
        return round(cos, 3), round(r2, 3)
    res = {}
    # one-step at L8->L9
    L = 8; Ln = 9
    Xin = flat(torch.cat([C[L], TF, cpool(C[L])], -1), ev)
    true1 = flat(C[Ln], ev)
    res['one_step_L8'] = {'mlp': cosR2(maps_mlp[L](Xin), true1),
                          'linear': cosR2(maps_lin[L](Xin), true1),
                          'persistence': cosR2(flat(C[L], ev), true1)}
    # rollout L5 -> L14
    cur = C[BAND[0]][ev]                                   # start from TRUE c_5, held-out rows
    cur_tok = TF[ev]
    for j in range(len(BAND)-1):
        L = BAND[j]
        Xin = torch.cat([cur, cur_tok, cpool(cur)], -1).reshape(-1, K+EMB32+K)
        cur = maps_mlp[L](Xin).view(cur.shape)
    trueE = C[BAND[-1]][ev]
    ro_mlp = cosR2(cur.reshape(-1, K), trueE.reshape(-1, K))
    # linear rollout
    cur = C[BAND[0]][ev]
    for j in range(len(BAND)-1):
        L = BAND[j]
        Xin = torch.cat([cur, cur_tok, cpool(cur)], -1).reshape(-1, K+EMB32+K)
        cur = maps_lin[L](Xin).view(cur.shape)
    ro_lin = cosR2(cur.reshape(-1, K), trueE.reshape(-1, K))
    ro_pers = cosR2(C[BAND[0]][ev].reshape(-1, K), trueE.reshape(-1, K))
    # token-only "rollout" (stateless)
    cur = C[BAND[0]][ev]*0
    for j in range(len(BAND)-1):
        L = BAND[j]
        Xin = torch.cat([cur_tok, (cur*0)], -1).reshape(-1, EMB32+K)
        cur = maps_tokonly[L](Xin).view(C[BAND[0]][ev].shape)
    ro_tok = cosR2(cur.reshape(-1, K), trueE.reshape(-1, K))
    res['rollout_L5_to_L14'] = {'mlp': ro_mlp, 'linear': ro_lin, 'persistence': ro_pers, 'token_only': ro_tok}
    print(f"one-step L8: {res['one_step_L8']}", flush=True)
    print(f"rollout: mlp {ro_mlp} | linear {ro_lin} | persistence {ro_pers} | token-only {ro_tok}", flush=True)

    out = {'results': res}
    out['pred_a_simple_dynamics'] = bool(ro_mlp[0] >= 0.6 and ro_mlp[0] - ro_pers[0] >= 0.15)
    out['pred_b_context_limited'] = bool(ro_mlp[0] - ro_pers[0] < 0.05)
    out['pred_c_stateless'] = bool(abs(ro_tok[0] - ro_mlp[0]) < 0.05)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a simple-dynamics {out['pred_a_simple_dynamics']} | pred_b context-limited {out['pred_b_context_limited']} | pred_c stateless {out['pred_c_stateless']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
