# comparative_refine_heads: LOCALIZE THE REFINE STAGE (§1344). kit + live L10-12 reaches
# 0.776 with a 1.8x target-specific increment; this run finds which of the 27 heads carry
# it. Stage 1: drop-one from kit+b1012 (960 rows, RANKING ONLY). Stage 2: nested kits
# kit + top-k refine heads (k=2,4,8) scored at 1920 rows.
#
# Registered predictions:
#   pred_a THE REFINEMENT LOCALIZES: <= 4 heads carry >= 70% of the band increment
#          (top-4 kit's increment >= 0.70 * b1012's +0.117).
#   pred_b 10.5 MOONLIGHTS HERE: the question head is in the top-4 by drop cost — the
#          §1310 question->than off-diagonal (0.171, replicated) predicts exactly this
#          cross-circuit guest appearance.
#   pred_c SLIMMING PRESERVES SPECIFICITY: the top-4 kit's elsewhere increment over the
#          §1333 kit <= 0.03.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'comparative_refine_heads_results.json'
NMEAN = 24; NR = 1920
B1012 = [(L, h) for L in (10, 11, 12) for h in range(9)]
CURBAND = {'layers': (10, 11, 12), 'heads': set(B1012)}
NR1 = 960
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
A02 = {(L, h) for L in (0, 1, 2) for h in range(9)}
KEEP81 = {(8, 1)}


@torch.no_grad()
def fwd_arm(idx, arm, vmeans, ymeans, keymask, qrymask, clean_rms=None):
    """arm full|ymean|route|kit|kit_gf. kit = route + a02 gated (key|qry windows) + 8.1.
    kit_gf: same, but the FINAL rms_norm divides by the clean run's per-token rms
    (clean_rms, (B,T)) instead of its own. Returns (logits, final_prenorm_rms)."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    gate = (keymask | qrymask)
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
        if arm == 'full':
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
        elif arm == 'ymean':
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
            for h in range(9):
                y[:, :, h] = ymeans[L][h].to(y.dtype)
        else:
            vr = v.clone()
            for h in range(9):
                if not (arm != 'route' and (L, h) in KEEP81):
                    vr[:, :, h] = vmeans[L][h].to(vr.dtype)
            vvr = (1 - at.lamb) * vr + at.lamb * v1.view_as(vr)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vvr.dtype), vvr)
            if arm in ('kit', 'kitband'):
                if L in (0, 1, 2):
                    vv_live = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
                    y_live = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv_live.dtype), vv_live)
                    gm = gate.view(B, T, 1, 1)
                    y = torch.where(gm, y_live, y)
                if arm == 'kitband' and L in CURBAND['layers']:
                    vv_live = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
                    y_live2 = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv_live.dtype), vv_live)
                    for h in range(9):
                        if (L, h) in CURBAND['heads']:
                            y[:, :, h] = y_live2[:, :, h]
        yo = at.c_proj(y.reshape(B, T, D))
        x = xm + yo
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    rms = x.float().pow(2).mean(-1).sqrt()
    if arm == 'kit_gf':
        xn = x / clean_rms.clamp_min(1e-6).unsqueeze(-1).to(x.dtype)
    else:
        xn = F.rms_norm(x, (D,))
    return 30.0 * torch.tanh(m.lm_head(xn) / 30.0), rms


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    COMP = ['bigger', 'smaller', 'better', 'worse', 'larger', 'greater', 'higher',
            'lower', 'faster', 'slower', 'older', 'younger', 'stronger', 'weaker',
            'easier', 'harder', 'longer', 'shorter', 'cheaper', 'richer', 'more', 'less',
            'fewer', 'rather']
    than = set(); comp = set()
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        if d.strip().lower() == 'than':
            than.add(tok)
        if d.strip().lower() in COMP:
            comp.add(tok)
    than_t = torch.tensor(sorted(than)); comp_t = torch.tensor(sorted(comp))

    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]

    # per-head means (same estimator as the whole template thread)
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

    # masks
    tgt_all = EVR[:, 1:]; toks = EVR[:, :-1]
    is_comp = torch.isin(toks, comp_t)
    ctx = torch.zeros_like(is_comp)
    for w in range(2, 21):
        sh = torch.zeros_like(is_comp)
        sh[:, w:] = is_comp[:, :-w]
        ctx |= sh
    TARGET = torch.isin(tgt_all, than_t) & ctx
    TARGET[:, :64] = False
    ELSE = ~TARGET; ELSE[:, :64] = False
    ntar = int(TARGET.sum())
    print(f"targets {ntar}", flush=True)

    def ce_pair(arm, nr):
        st = se = 0.0; nt = ne = 0
        for i in range(0, nr, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            km = torch.isin(idx.cpu(), comp_t).to(DEV)
            qm_ = ctx[i:i + 8].to(DEV)
            lo, _ = fwd_arm(idx, arm, vmeans, ymeans, km, qm_)
            ce = F.cross_entropy(lo.float().reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mt = TARGET[i:i + 8].to(DEV); me = ELSE[i:i + 8].to(DEV)
            st += float(ce[mt].sum()); nt += int(mt.sum())
            se += float(ce[me].sum()); ne += int(me.sum())
        return st / max(nt, 1), se / max(ne, 1)

    # stage 1: drop-one ranking
    CURBAND['heads'] = set(B1012)
    base_t, _ = ce_pair('kitband', NR1)
    print(f"kit+b1012 (NR1) target {base_t:.4f}", flush=True)
    drop = {}
    for hd in B1012:
        CURBAND['heads'] = set(B1012) - {hd}
        tce, _ = ce_pair('kitband', NR1)
        drop[hd] = tce - base_t
        print(f"drop L{hd[0]}.{hd[1]}: +{drop[hd]:.4f}", flush=True)
        json.dump({'partial': True,
                   'drop': {f'{a}.{b}': round(v, 4) for (a, b), v in drop.items()}},
                  open(OUT, 'w'), indent=1)
    ranked = sorted(B1012, key=lambda hd: -drop[hd])

    # stage 2
    res = {}
    for arm in ('full', 'ymean', 'route', 'kit'):
        t, e = ce_pair(arm, NR)
        res[arm] = {'target': round(t, 4), 'else': round(e, 4)}
        print(f"{arm}: target {t:.4f} | else {e:.4f}", flush=True)
    CURBAND['heads'] = set(B1012)
    t, e = ce_pair('kitband', NR)
    res['kit_b1012'] = {'target': round(t, 4), 'else': round(e, 4)}
    print(f"kit_b1012: target {t:.4f} | else {e:.4f}", flush=True)
    for kk in (8, 4, 2):
        CURBAND['heads'] = set(ranked[:kk])
        t, e = ce_pair('kitband', NR)
        res[f'kit_top{kk}'] = {'target': round(t, 4), 'else': round(e, 4)}
        print(f"kit_top{kk}: target {t:.4f} | else {e:.4f}", flush=True)

    gt = res['ymean']['target'] - res['full']['target']
    ge = res['ymean']['else'] - res['full']['else']
    def rt(a): return (res['ymean']['target'] - res[a]['target']) / max(gt, 1e-6)
    def re(a): return (res['ymean']['else'] - res[a]['else']) / max(ge, 1e-6)
    kit_t, kit_e = rt('kit'), re('kit')
    band_inc = rt('kit_b1012') - kit_t
    top4_inc = rt('kit_top4') - kit_t
    top4 = [f'{a}.{b}' for a, b in ranked[:4]]
    pa = top4_inc >= 0.70 * max(band_inc, 1e-6)
    pb = '10.5' in top4
    pc = (re('kit_top4') - kit_e) <= 0.03
    out = {'n_targets': ntar, 'n_rows': NR, 'ce': res,
           'recovery': {a: {'target': round(rt(a), 4), 'else': round(re(a), 4)}
                        for a in res if a != 'ymean'},
           'drop_costs': {f'{a}.{b}': round(v, 4) for (a, b), v in drop.items()},
           'ranked_top8': [f'{a}.{b}' for a, b in ranked[:8]],
           'band_increment': round(band_inc, 4), 'top4_increment': round(top4_inc, 4),
           'pred_a_localizes': bool(pa), 'pred_b_105_moonlights': bool(pb),
           'pred_c_specific': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\ntop4 {top4} | band inc {band_inc:.4f} top4 inc {top4_inc:.4f}")
    print(f"pred_a localizes {pa} | pred_b 10.5 {pb} | pred_c specific {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
