"""ORACLE-INJECTION upgrade of §1122 (registered there): the linear coord dynamics captures ~53% of the
band-end content; the remainder should be what attention injects fresh each layer. Add, as a per-step input,
the TRUE attention-output coords a_L = U_c^T(attn_out_L) at each position (an ORACLE diagnostic — at rollout
time these come from the real run; the point is to locate the missing information, not to build a stand-in).
Maps: linear only (per §1122, nonlinearity adds nothing). Rollout L5->L14 with inputs [c_L, tokenfeat, pool,
a_L^true] vs §1122's no-injection rollout and a SHUFFLED-injection null (a_L from a random other position).

REGISTERED PREDICTIONS:
  (0) SANITY: no-injection linear rollout reproduces §1122 (~0.635 cos); shuffled injection <= no-injection.
  (a) ATTENTION INJECTIONS COMPLETE THE PICTURE: with true a_L per step, rollout cosine >= 0.85 -> construction
      = LINEAR coord dynamics + attention injections, fully characterized; the remaining benchmark frontier
      is simulating the attention stream alone (a named, bounded object);
  (b) if injection lifts < 0.1, the missing half is NOT the attention outputs (it lives in MLP nonlinearity or
      off-coord D-space paths — report plainly)."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'coord_dynamics_attn_results.json'
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
    caps = {L: [] for L in BAND}; capsA = {L: [] for L in BAND}
    hsA = []
    for L in BAND:
        def mkA(L):
            def h(mo, i_, o_):
                y = o_[0] if isinstance(o_, tuple) else o_
                capsA[L].append(y.detach().float().reshape(-1, D))
            return h
        hsA.append(H[L].attn.register_forward_hook(mkA(L)))
    def fcap(idx):
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for Li, blk in enumerate(H):
            x, v1 = blk(x, v1, x0)
            if Li in BAND: caps[Li].append(x.detach().float().reshape(-1, D))
    idsL = []
    for i in range(0, nb, 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fcap(idx)
    for h in hsA: h.remove()
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
    A = {L: ((torch.cat(capsA[L], 0)) @ Uc).view(nb, T, K) for L in BAND}
    for L in BAND: caps[L] = None; capsA[L] = None
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

    # train per-step LINEAR maps: base (no injection) + inj (with true attention coords)
    maps_base = {}; maps_inj = {}
    for j in range(len(BAND)-1):
        L, Ln = BAND[j], BAND[j+1]
        Xb = flat(torch.cat([C[L], TF, cpool(C[L])], -1), trm)
        Xi = flat(torch.cat([C[L], TF, cpool(C[L]), A[Ln]], -1), trm)   # injection = NEXT block's attn coords (what happens during the step)
        Ytr = flat(C[Ln], trm)
        maps_base[L] = fit_linear(Xb, Ytr)
        maps_inj[L] = fit_linear(Xi, Ytr)
        print(f"step L{L}->L{Ln} fit", flush=True)

    # evaluation: one-step + rollout on held-out rows
    ev = ~trm
    def cosR2(pred, true):
        cos = float(F.cosine_similarity(pred, true, dim=-1).mean())
        r2 = 1 - float(((pred-true)**2).sum()/((true - true.mean(0))**2).sum())
        return round(cos, 3), round(r2, 3)
    res = {}
    cur_tok = TF[ev]; trueE = C[BAND[-1]][ev]
    def rollout(use_inj, shuffle_inj=False):
        cur = C[BAND[0]][ev]
        g2 = torch.Generator(device=DEV).manual_seed(7)
        for j in range(len(BAND)-1):
            L, Ln = BAND[j], BAND[j+1]
            if use_inj:
                a = A[Ln][ev]
                if shuffle_inj:
                    a = a.reshape(-1, K)[torch.randperm(a.numel()//K, generator=g2, device=DEV)].view(a.shape)
                Xin = torch.cat([cur, cur_tok, cpool(cur), a], -1).reshape(-1, K+EMB32+K+K)
                cur = maps_inj[L](Xin).view(cur.shape)
            else:
                Xin = torch.cat([cur, cur_tok, cpool(cur)], -1).reshape(-1, K+EMB32+K)
                cur = maps_base[L](Xin).view(cur.shape)
        return cosR2(cur.reshape(-1, K), trueE.reshape(-1, K))
    ro_base = rollout(False)
    ro_inj = rollout(True)
    ro_shuf = rollout(True, shuffle_inj=True)
    ro_pers = cosR2(C[BAND[0]][ev].reshape(-1, K), trueE.reshape(-1, K))
    res['rollout'] = {'base_linear': ro_base, 'with_true_attn': ro_inj, 'shuffled_attn': ro_shuf, 'persistence': ro_pers}
    print(f"rollout: base {ro_base} | +true-attn {ro_inj} | +shuffled-attn {ro_shuf} | persistence {ro_pers}", flush=True)

    out = {'results': res}
    out['pred_a_injections_complete'] = bool(ro_inj[0] >= 0.85)
    out['pred_b_not_attention'] = bool(ro_inj[0] - ro_base[0] < 0.1)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a injections-complete {out['pred_a_injections_complete']} | pred_b not-attention {out['pred_b_not_attention']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
