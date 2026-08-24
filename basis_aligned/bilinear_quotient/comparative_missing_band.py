# comparative_missing_band: WHERE DOES THE MISSING 0.34 LIVE? §1343 named it target-
# specific dynamic CONTENT (rank collapse; entropy channels refuted). This scan adds ONE
# fully-live attention band on top of the §1333 kit per arm and asks which band closes
# the gap: L3-5 / L6-9 / L10-12 / L14-17 (L13 excluded — 13.8's circuit stays out of
# frame). MLPs live throughout (attention-only convention, §1343 explicit).
#
# Registered predictions:
#   pred_a SOME BAND CARRIES IT: best band adds >= 0.10 target recovery over the kit.
#   pred_b THE MID POOL IS THE CARRIER: the winner is L3-5 or L6-9 (the content-pooling
#          band, §1099/§1093 priors).
#   pred_c TARGET-SPECIFIC (risky, registered against the §1329 generic-band pattern):
#          the winner's target increment >= 1.5x its elsewhere increment. FALSE would
#          mean the missing content is generic pooling, itself informative.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'comparative_missing_band_results.json'
NMEAN = 24; NR = 1920
BANDS = {'b35': (3, 4, 5), 'b69': (6, 7, 8, 9), 'b1012': (10, 11, 12), 'b1417': (14, 15, 16, 17)}
CURBAND = {'layers': ()}
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
                    y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv_live.dtype), vv_live)
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

    sums = {}
    cnts = {'t': 0, 'e': 0}
    ARMS = ['full', 'ymean', 'route', 'kit'] + [f'kit+{b}' for b in BANDS]
    for a in ARMS:
        sums[a] = {'t': 0.0, 'e': 0.0}
    for i in range(0, NR, 8):
        bb = EVR[i:i + 8].to(DEV)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
        km = torch.isin(idx.cpu(), comp_t).to(DEV)
        qm_ = ctx[i:i + 8].to(DEV)
        mt = TARGET[i:i + 8].to(DEV); me = ELSE[i:i + 8].to(DEV)
        outs = {}
        for a in ('full', 'ymean', 'route', 'kit'):
            outs[a], _ = fwd_arm(idx, a, vmeans, ymeans, km, qm_)
        for bname, blayers in BANDS.items():
            CURBAND['layers'] = blayers
            outs[f'kit+{bname}'], _ = fwd_arm(idx, 'kitband', vmeans, ymeans, km, qm_)
        for a, lo in outs.items():
            ce = F.cross_entropy(lo.float().reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            sums[a]['t'] += float(ce[mt].sum()); sums[a]['e'] += float(ce[me].sum())
        cnts['t'] += int(mt.sum()); cnts['e'] += int(me.sum())
    ce_res = {a: {'target': round(sums[a]['t'] / cnts['t'], 4),
                  'else': round(sums[a]['e'] / cnts['e'], 4)} for a in sums}
    for a in ARMS:
        print(f"{a}: target {ce_res[a]['target']} | else {ce_res[a]['else']}", flush=True)

    gt = ce_res['ymean']['target'] - ce_res['full']['target']
    ge = ce_res['ymean']['else'] - ce_res['full']['else']
    def rt(a): return (ce_res['ymean']['target'] - ce_res[a]['target']) / max(gt, 1e-6)
    def re(a): return (ce_res['ymean']['else'] - ce_res[a]['else']) / max(ge, 1e-6)
    kit_t, kit_e = rt('kit'), re('kit')
    inc = {b: {'target': round(rt(f'kit+{b}') - kit_t, 4),
               'else': round(re(f'kit+{b}') - kit_e, 4)} for b in BANDS}
    winner = max(inc, key=lambda b: inc[b]['target'])
    pa = inc[winner]['target'] >= 0.10
    pb = winner in ('b35', 'b69')
    pc = inc[winner]['target'] >= 1.5 * max(inc[winner]['else'], 1e-4)
    out = {'n_targets': ntar, 'n_rows': NR, 'ce': ce_res,
           'recovery': {a: {'target': round(rt(a), 4), 'else': round(re(a), 4)}
                        for a in ARMS if a != 'ymean'},
           'band_increments': inc, 'winner': winner,
           'pred_a_band_carries': bool(pa), 'pred_b_mid_pool': bool(pb),
           'pred_c_target_specific': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nincrements over kit: " +
          " ".join(f"{b} t{inc[b]['target']:+.3f}/e{inc[b]['else']:+.3f}" for b in BANDS))
    print(f"winner {winner} | pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
