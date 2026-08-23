"""Registered in §1123: the coord rollout caps at R² 0.53 (0.68 with true per-step attention), consistent with
per-step compounding (0.92⁹≈0.47). DISCRIMINATOR: fit DIRECT linear maps across spans — no chaining, no
compounding: c_start(+tok+pool) -> c_end for spans L5→L9, L9→L14, L5→L14; each also ± the TRUE band-summed
attention coords (sum of a_L over the span — the total context injected during the span, oracle input).
If a direct L5→L14 map matches per-step-level fidelity, the §1123 compounding reading is confirmed and
simulation should use direct/skip maps; if direct ≈ chained, the end-band content depends on information NOT
present at L5 (it arrives mid-band via attention) — then the +attn-sum condition prices exactly how much.

REGISTERED PREDICTIONS:
  (0) SANITY: chained baseline reproduces §1122 (0.53); direct >= chained always (strictly less compounding);
      span R² decreases with span length.
  (a) COMPOUNDING CONFIRMED: direct L5→L14 (no oracle) reaches R² >= 0.75 — the information is at L5, rollouts
      lose it to compounding, and the simulation route is skip/direct maps;
  (b) INFO-ARRIVES-MID-BAND: direct (no oracle) ≈ chained (< 0.6) AND direct + attn-sum >= 0.85 — the linear
      closure is c_5 PLUS the span's total attention input; the construction is then FULLY characterized as
      linear(state, total-attention-injection) and the frontier reduces to simulating attention only;
  (c) if even direct + attn-sum < 0.75, a genuinely nonlinear/off-coordinate term remains (report plainly)."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'coord_direct_results.json'
NSEQ = 192; SEQ = 256; K = 64; BAND = list(range(5, 15)); EMB32 = 32
H = m.transformer.h
SPANS = [(5, 9), (9, 14), (5, 14)]


def fit_linear(X, Y):
    A = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], 1)
    Wl = torch.linalg.lstsq(A, Y).solution
    return lambda Z: torch.cat([Z, torch.ones(Z.shape[0], 1, device=Z.device)], 1) @ Wl


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0]); nb = NSEQ; ntr = int(0.7*nb); T = SEQ - 1

    caps = {L: [] for L in BAND}; capsA = {L: [] for L in BAND}
    hsA = []
    for L in BAND:
        def mkA(L):
            def h(mo, i_, o_):
                y = o_[0] if isinstance(o_, tuple) else o_
                capsA[L].append(y.detach().float().reshape(-1, D))
            return h
        hsA.append(H[L].attn.register_forward_hook(mkA(L)))
    idsL = []
    for i in range(0, nb, 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1))
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for Li, blk in enumerate(H):
            x, v1 = blk(x, v1, x0)
            if Li in BAND: caps[Li].append(x.detach().float().reshape(-1, D))
    for h in hsA: h.remove()
    tok = torch.cat(idsL, 0)
    cn = torch.zeros(V, device=DEV); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))

    devsum = None; xbars = {}
    for L in BAND:
        X = torch.cat(caps[L], 0); caps[L] = X
        xb = torch.zeros(V, D, device=DEV); xb.index_add_(0, tok, X)
        xbars[L] = xb/cn.clamp_min(1).unsqueeze(1)
        if L in (8, 10, 12):
            dv = X - xbars[L][tok]
            devsum = dv if devsum is None else devsum + dv
    dev = devsum/3; dev = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(dev, full_matrices=False)
    Uc = Vt[:K].T.contiguous(); del dev, devsum

    C = {L: ((caps[L] - xbars[L][tok]) @ Uc).view(nb, T, K) for L in BAND}
    A = {L: ((torch.cat(capsA[L], 0)) @ Uc).view(nb, T, K) for L in BAND}
    for L in BAND: caps[L] = None; capsA[L] = None
    _, _, Vte = torch.linalg.svd(m.transformer.wte.weight.float()[torch.unique(tok)], full_matrices=False)
    TF = (m.transformer.wte(tok).float() @ Vte[:EMB32].T.contiguous()).view(nb, T, EMB32)
    def cpool(Cl):
        cs = Cl.cumsum(1)
        return cs/torch.arange(1, T+1, device=DEV).view(1, T, 1).float()

    trm = torch.zeros(nb, dtype=torch.bool); trm[:ntr] = True
    ev = ~trm
    def flat(x, mask): return x[mask].reshape(-1, x.shape[-1])
    def cosR2(pred, true):
        return (round(float(F.cosine_similarity(pred, true, dim=-1).mean()), 3),
                round(1 - float(((pred-true)**2).sum()/((true - true.mean(0))**2).sum()), 3))

    res = {}
    for (Ls, Le) in SPANS:
        attn_sum = sum(A[L2] for L2 in range(Ls+1, Le+1))       # total attention injected during the span
        Xb = torch.cat([C[Ls], TF, cpool(C[Ls])], -1)
        Xa = torch.cat([C[Ls], TF, cpool(C[Ls]), attn_sum], -1)
        Y = C[Le]
        fb = fit_linear(flat(Xb, trm), flat(Y, trm))
        fa = fit_linear(flat(Xa, trm), flat(Y, trm))
        rb = cosR2(fb(flat(Xb, ev)), flat(Y, ev))
        ra = cosR2(fa(flat(Xa, ev)), flat(Y, ev))
        pers = cosR2(flat(C[Ls], ev), flat(Y, ev))
        res[f'L{Ls}_to_L{Le}'] = {'direct': rb, 'direct_plus_attnsum': ra, 'persistence': pers}
        print(f"L{Ls}->L{Le}: direct {rb} | +attn-sum {ra} | persistence {pers}", flush=True)

    d514 = res['L5_to_L14']
    out = {'spans': res, 'chained_baseline_1122': [0.635, 0.529]}
    out['pred_a_compounding'] = bool(d514['direct'][1] >= 0.75)
    out['pred_b_info_midband'] = bool(d514['direct'][1] < 0.6 and d514['direct_plus_attnsum'][1] >= 0.85)
    out['pred_c_nonlinear_remainder'] = bool(d514['direct_plus_attnsum'][1] < 0.75)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a compounding {out['pred_a_compounding']} | pred_b info-midband {out['pred_b_info_midband']} | pred_c nonlinear-remainder {out['pred_c_nonlinear_remainder']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
