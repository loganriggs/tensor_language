# exclaim_probe_gate2: TWO-SITE PROBE UPGRADE (§1364). v1's gate shed 93% of the
# elsewhere bill but kept only 41% of the target increment — recall-capped by AUC 0.618
# on the L3-entry route stream. Same §105-safe design, richer features: capture at L3
# AND L8 entries on the same route pass, concatenated (2304-dim ridge). The register
# signal §1350 located through L3-5 should be more legible after the band.
#
# Registered predictions:
#   pred_a keeps >= 55% of the band's target increment over route.
#   pred_b still sheds >= 60% of the band's elsewhere increment.
#   pred_c AUC >= 0.70. (A second AUC miss bounds the exclamatory register's linear
#          legibility in route-grade streams — a scoped negative worth having.)
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'exclaim_probe_gate2_results.json'
NMEAN = 24; NFITP = 960; NR = 1920; KAHEAD = 16
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
PAIR = {(17, 2), (17, 3)}
BAND05 = (0, 1, 2, 3, 4, 5)


@torch.no_grad()
def fwd_arm(idx, arm, vmeans, ymeans, gatemask, capture_l3=None):
    """arm full|ymean|route|route_solo|band05|gated. route grain; pair live in solo/
    band05/gated; band05 layers live everywhere (band05) or inside gatemask (gated).
    capture_l3: list to append the L3-entry hybrid stream (detached) — same code path
    for fit and apply (§105)."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(H):
        if L in (3, 8) and capture_l3 is not None:
            capture_l3.append(x.detach().float())
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
        q = are(F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,)), cos, sin)
        k = are(F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,)), cos, sin)
        q2 = are(F.rms_norm(at.c_q2(xin).view(B, T, 9, 128), (128,)), cos, sin)
        k2 = are(F.rms_norm(at.c_k2(xin).view(B, T, 9, 128), (128,)), cos, sin)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
            * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
        tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        pat = pat.masked_fill(~tril, 0.0)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        if arm == 'full':
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
        elif arm == 'ymean':
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
            for h in range(9):
                y[:, :, h] = ymeans[L][h].to(y.dtype)
        else:
            keep_pair = arm in ('route_solo', 'band05', 'gated')
            vr = v.clone()
            for h in range(9):
                if not (keep_pair and (L, h) in PAIR):
                    vr[:, :, h] = vmeans[L][h].to(vr.dtype)
            vvr = (1 - at.lamb) * vr + at.lamb * v1.view_as(vr)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vvr.dtype), vvr)
            if L in BAND05 and arm in ('band05', 'gated'):
                vv_live = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
                y_live = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv_live.dtype), vv_live)
                if arm == 'band05':
                    y = y_live
                else:
                    gm = gatemask.view(B, T, 1, 1)
                    y = torch.where(gm, y_live, y)
        yo = at.c_proj(y.reshape(B, T, D))
        x = xm + yo
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    ex_t = set()
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        if '!' in d:
            ex_t.add(tok)
    ex_ids = torch.tensor(sorted(ex_t))

    ROWS = cl.fineweb_rows(NMEAN + NFITP + NR)[:, :T + 1].contiguous()
    MEANR = ROWS[:NMEAN]
    FITR = ROWS[NMEAN:NMEAN + NFITP]
    EVR = ROWS[NMEAN + NFITP:]
    cl.assert_disjoint(FITR, EVR, label='probe_fit_vs_eval')

    # per-head means
    vs = [[[] for _ in range(9)] for _ in range(18)]
    ys = [[[] for _ in range(9)] for _ in range(18)]
    for i in range(0, NMEAN, 4):
        idx = MEANR[i:i + 4, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        B = idx.shape[0]
        for L, blk in enumerate(H):
            at = blk.attn
            xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
            xin = F.rms_norm(xm, (D,))
            cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
            q = are(F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,)), cos, sin)
            k = are(F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,)), cos, sin)
            q2 = are(F.rms_norm(at.c_q2(xin).view(B, T, 9, 128), (128,)), cos, sin)
            k2 = are(F.rms_norm(at.c_k2(xin).view(B, T, 9, 128), (128,)), cos, sin)
            pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
                * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
            tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
            pat = pat.masked_fill(~tril, 0.0)
            v = at.c_v(xin).view(B, T, 9, 128)
            if v1 is None:
                v1 = v
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
            for h in range(9):
                vs[L][h].append(v[:, :, h].float().mean((0, 1)).cpu())
                ys[L][h].append(y[:, :, h].float().mean((0, 1)).cpu())
            x = xm + at.c_proj(y.reshape(B, T, D))
            x = x + blk.mlp(F.rms_norm(x, (D,)))
    vmeans = [torch.stack([torch.stack(vs[L][h]).mean(0) for h in range(9)]).to(DEV)
              for L in range(18)]
    ymeans = [torch.stack([torch.stack(ys[L][h]).mean(0) for h in range(9)]).to(DEV)
              for L in range(18)]

    def labels_for(rows_tok):
        """(B,T) bool: an '!'-token occurs in the next KAHEAD target positions."""
        istgt = torch.isin(rows_tok[:, 1:], ex_ids)
        lab = torch.zeros_like(istgt)
        for w in range(KAHEAD):
            sh = torch.zeros_like(istgt)
            if w == 0:
                sh = istgt
            else:
                sh[:, :-w] = istgt[:, w:]
            lab |= sh
        return lab

    # ---- fit probe on ROUTE-ONLY stream at L3 entry (same code path as apply)
    Xs = []; Ys = []
    for i in range(0, NFITP, 8):
        idx = FITR[i:i + 8, :-1].to(DEV).contiguous()
        cap = []
        fwd_arm(idx, 'route', vmeans, ymeans, None, capture_l3=cap)
        xs = torch.cat([F.rms_norm(cap[0], (D,)), F.rms_norm(cap[1], (D,))], -1)
        xs = xs.reshape(-1, 2 * D)
        Xs.append(xs[::4].cpu())          # subsample for memory
        Ys.append(labels_for(FITR[i:i + 8]).reshape(-1)[::4].cpu())
    X = torch.cat(Xs); Yl = torch.cat(Ys).float()
    n_hold = X.shape[0] // 5
    Xtr, Xho = X[n_hold:], X[:n_hold]
    Ytr, Yho = Yl[n_hold:], Yl[:n_hold]
    lam = 10.0
    A = Xtr.T @ Xtr + lam * torch.eye(2 * D)
    w = torch.linalg.solve(A, Xtr.T @ (Ytr * 2 - 1))
    sc_ho = Xho @ w
    pos = sc_ho[Yho > 0.5]; neg = sc_ho[Yho < 0.5]
    # AUC by rank statistic
    allsc = torch.cat([pos, neg])
    ranks = allsc.argsort().argsort().float()
    auc = float((ranks[:len(pos)].mean() - (len(pos) - 1) / 2) / max(len(neg), 1))
    sc_all = X @ w
    thr = torch.quantile(sc_all, 0.90)
    posrate = float((sc_all > thr).float().mean())
    print(f"probe: fit n {X.shape[0]} | pos rate {float(Yl.mean()):.3f} | "
          f"holdout AUC {auc:.3f} | gate thr q90 (rate {posrate:.2f})", flush=True)
    w = w.to(DEV); thr = float(thr)

    # ---- eval
    tgt_all = EVR[:, 1:]
    TARGET = torch.isin(tgt_all, ex_ids)
    TARGET[:, :64] = False
    ELSE = ~TARGET; ELSE[:, :64] = False
    print(f"targets {int(TARGET.sum())}", flush=True)

    def ce_cond(arm):
        st = se = 0.0; nt = ne = 0; gate_frac = 0.0; nb = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            gm = None
            if arm == 'gated':
                cap = []
                fwd_arm(idx, 'route', vmeans, ymeans, None, capture_l3=cap)
                feats = torch.cat([F.rms_norm(cap[0], (D,)), F.rms_norm(cap[1], (D,))], -1)
                sc = (feats @ w)
                gm = sc > thr
                gate_frac += float(gm.float().mean()); nb += 1
            lo = fwd_arm(idx, arm, vmeans, ymeans, gm).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mt = TARGET[i:i + 8].to(DEV); me = ELSE[i:i + 8].to(DEV)
            st += float(ce[mt].sum()); nt += int(mt.sum())
            se += float(ce[me].sum()); ne += int(me.sum())
        return st / max(nt, 1), se / max(ne, 1), (gate_frac / max(nb, 1))

    res = {}
    for arm in ('full', 'ymean', 'route', 'route_solo', 'band05', 'gated'):
        tce, ece, gf = ce_cond(arm)
        res[arm] = {'target': round(tce, 4), 'else': round(ece, 4)}
        extra = f" | gate frac {gf:.2f}" if arm == 'gated' else ""
        print(f"{arm}: target {tce:.4f} | else {ece:.4f}{extra}", flush=True)

    gt = res['ymean']['target'] - res['full']['target']
    ge = res['ymean']['else'] - res['full']['else']
    rec = {a: {'target': round((res['ymean']['target'] - res[a]['target']) / max(gt, 1e-6), 4),
               'else': round((res['ymean']['else'] - res[a]['else']) / max(ge, 1e-6), 4)}
           for a in res if a != 'ymean'}
    t_inc_band = rec['band05']['target'] - rec['route']['target']
    e_inc_band = rec['band05']['else'] - rec['route']['else']
    t_inc_gate = rec['gated']['target'] - rec['route']['target']
    e_inc_gate = rec['gated']['else'] - rec['route']['else']
    pa = t_inc_gate >= 0.55 * max(t_inc_band, 1e-6)
    pb = e_inc_gate <= 0.40 * max(e_inc_band, 1e-6)
    pc = auc >= 0.70
    out = {'n_rows': NR, 'probe_auc': round(auc, 4), 'gate_threshold_q': 0.90,
           'ce': res, 'recovery': rec,
           'increments': {'band_t': round(t_inc_band, 4), 'band_e': round(e_inc_band, 4),
                          'gate_t': round(t_inc_gate, 4), 'gate_e': round(e_inc_gate, 4)},
           'pred_a_keeps_capability': bool(pa), 'pred_b_sheds_bill': bool(pb),
           'pred_c_mode_legible': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nincrements over route: band t+{t_inc_band:.3f}/e+{e_inc_band:.3f} | "
          f"gated t+{t_inc_gate:.3f}/e+{e_inc_gate:.3f}")
    print(f"pred_a keeps {pa} | pred_b sheds {pb} | pred_c legible {pc} (AUC {auc:.3f})")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
