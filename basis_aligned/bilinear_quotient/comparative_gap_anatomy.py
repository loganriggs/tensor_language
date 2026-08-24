# comparative_gap_anatomy: WHAT IS THE MISSING 0.34 OF THE "THAN" KIT? (user push). The
# closed comparative kit (§1333: route + 2 gates + 8.1, ATTENTION-ONLY extraction — all
# 18 MLPs live throughout, stated here because it was only implicit before) recovers
# 0.659; its CE at targets is 3.65 vs full's 1.25. The user's conjecture: part of the
# gap is not missing CONTENT but a missing ENTROPY-SETTER — mean-ablating 129 heads
# perturbs the stream scale, and the final rms-norm + logit soft-cap turn scale into
# distribution sharpness, so the kit may rank "than" correctly but flattened.
#
# The decomposition, three instruments:
#   RANK      at each target, the rank of the true "than" token under full vs kit
#             logits. Content loss moves rank; pure confidence loss does not.
#   GAIN-FREEZE  the §116-117 regime this template never ran: re-run the kit with the
#             FINAL rms-norm's per-token scale clamped to the clean run's (lockstep dual
#             forward — kit stream direction, clean stream scale). This injects the
#             correct per-token entropy scale with zero content. Its recovery gain is
#             the per-token entropy-setter's share of the gap.
#   TEMPERATURE  the cheap global version: one scalar s on the kit's logits, fit on a
#             2000-position ELSEWHERE sample (never on targets; sampled elsewhere CE
#             disclosed as an estimate), scored at targets.
#
# Registered predictions:
#   pred_a CONTENT IS ALREADY THERE: among targets where the full model puts "than" in
#          its top-5, the kit keeps it top-5 at >= 70%.
#   pred_b (USER'S ENTROPY-SETTER): gain-freezing recovers >= 0.25 of the remaining gap
#          — kit_gf target recovery >= 0.659 + 0.25*(1-0.659) ~ 0.744.
#   pred_c THE GLOBAL VERSION IS WEAKER BUT REAL: temperature alone recovers >= 0.10 of
#          the remaining gap.
# If pred_a holds and pred_b fails, the gap is distributed content (the crowd's dynamic
# service) and the entropy-setter is not the story; if both hold, the next kit adds the
# scale channel explicitly (a per-position scalar — cheap description) and the exact-
# extraction frontier moves to the MLP side.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'comparative_gap_anatomy_results.json'
NMEAN = 24; NR = 1920; NSAMP = 2000
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
            if L in (0, 1, 2) and arm in ('kit', 'kit_gf'):
                vv_live = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
                y_live = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv_live.dtype), vv_live)
                gm = gate.view(B, T, 1, 1)
                y = torch.where(gm, y_live, y)
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

    # elsewhere sample for temperature fitting (never targets)
    g = torch.Generator().manual_seed(19)
    escore = torch.rand(ELSE.shape, generator=g); escore[~ELSE] = -1.0
    eflat = escore.flatten()
    samp_idx = eflat.topk(NSAMP).indices

    sums = {a: {'t': 0.0, 'e': 0.0} for a in ('full', 'ymean', 'route', 'kit', 'kit_gf')}
    cnts = {'t': 0, 'e': 0}
    tgt_logits_kit = []; tgt_ids = []; tgt_rank_full = []; tgt_rank_kit = []
    samp_logits_kit = []; samp_ids = []
    for i in range(0, NR, 8):
        bb = EVR[i:i + 8].to(DEV)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
        km = torch.isin(idx.cpu(), comp_t).to(DEV)
        qm_ = ctx[i:i + 8].to(DEV)
        mt = TARGET[i:i + 8].to(DEV); me = ELSE[i:i + 8].to(DEV)
        lo_full, rms_clean = fwd_arm(idx, 'full', vmeans, ymeans, km, qm_)
        arms_out = {'full': lo_full}
        for a in ('ymean', 'route', 'kit'):
            arms_out[a], _ = fwd_arm(idx, a, vmeans, ymeans, km, qm_)
        arms_out['kit_gf'], _ = fwd_arm(idx, 'kit_gf', vmeans, ymeans, km, qm_,
                                        clean_rms=rms_clean)
        for a, lo in arms_out.items():
            ce = F.cross_entropy(lo.float().reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            sums[a]['t'] += float(ce[mt].sum()); sums[a]['e'] += float(ce[me].sum())
        cnts['t'] += int(mt.sum()); cnts['e'] += int(me.sum())
        # rank + logit capture at targets
        if mt.any():
            lf = arms_out['full'].float()[mt]; lk = arms_out['kit'].float()[mt]
            ids = tg[mt]
            tgt_ids.append(ids.cpu())
            tgt_rank_full.append((lf > lf.gather(1, ids.unsqueeze(1))).sum(1).cpu())
            tgt_rank_kit.append((lk > lk.gather(1, ids.unsqueeze(1))).sum(1).cpu())
            tgt_logits_kit.append(lk.half().cpu())
        # elsewhere sample capture (flat indices within this batch's block)
        base_flat = i * (T)
        blk_pos = samp_idx[(samp_idx >= base_flat) & (samp_idx < base_flat + 8 * T)] - base_flat
        if blk_pos.numel():
            lk_all = arms_out['kit'].float().reshape(-1, arms_out['kit'].shape[-1])
            samp_logits_kit.append(lk_all[blk_pos.to(DEV)].half().cpu())
            samp_ids.append(tg.reshape(-1)[blk_pos.to(DEV)].cpu())
    ce_res = {a: {'target': round(sums[a]['t'] / cnts['t'], 4),
                  'else': round(sums[a]['e'] / cnts['e'], 4)} for a in sums}
    for a in ce_res:
        print(f"{a}: target {ce_res[a]['target']} | else {ce_res[a]['else']}", flush=True)

    # temperature fit on the elsewhere sample
    SL = torch.cat(samp_logits_kit).float(); SI = torch.cat(samp_ids)
    best = (None, 1e9)
    for s in [round(0.6 + 0.05 * i, 2) for i in range(29)]:
        cesamp = float(F.cross_entropy(SL * s, SI))
        if cesamp < best[1]:
            best = (s, cesamp)
    s_star = best[0]
    TL = torch.cat(tgt_logits_kit).float(); TI = torch.cat(tgt_ids)
    ce_temp_t = float(F.cross_entropy(TL * s_star, TI))
    ce_res['kit_temp'] = {'target': round(ce_temp_t, 4),
                          'else_sampled': round(best[1], 4)}
    print(f"temperature s*={s_star} | target CE {ce_temp_t:.4f} "
          f"(sampled-else fit {best[1]:.4f})", flush=True)

    # ranks
    RF = torch.cat(tgt_rank_full); RK = torch.cat(tgt_rank_kit)
    full_top5 = RF < 5; kit_top5 = RK < 5
    keep5 = float((kit_top5 & full_top5).sum()) / max(int(full_top5.sum()), 1)
    rank_stats = {'full_top1_frac': round(float((RF == 0).float().mean()), 4),
                  'kit_top1_frac': round(float((RK == 0).float().mean()), 4),
                  'full_top5_frac': round(float(full_top5.float().mean()), 4),
                  'kit_top5_frac': round(float(kit_top5.float().mean()), 4),
                  'kit_keeps_full_top5': round(keep5, 4),
                  'median_rank_full': int(RF.median()), 'median_rank_kit': int(RK.median())}
    print(rank_stats, flush=True)

    gt = ce_res['ymean']['target'] - ce_res['full']['target']
    def rec(a):
        return (ce_res['ymean']['target'] - ce_res[a]['target']) / max(gt, 1e-6)
    rkit, rgf, rtemp = rec('kit'), rec('kit_gf'), rec('kit_temp')
    remaining = 1.0 - rkit
    pa = keep5 >= 0.70
    pb = rgf >= rkit + 0.25 * remaining
    pc = rtemp >= rkit + 0.10 * remaining
    out = {'n_targets': ntar, 'n_rows': NR, 'ce': ce_res,
           'recovery': {'kit': round(rkit, 4), 'kit_gf': round(rgf, 4),
                        'kit_temp': round(rtemp, 4), 'route': round(rec('route'), 4)},
           'temperature': s_star, 'rank': rank_stats,
           'mlp_note': 'ATTENTION-ONLY extraction: all 18 MLPs live in every arm',
           'pred_a_content_there': bool(pa), 'pred_b_user_entropy_setter': bool(pb),
           'pred_c_temperature_real': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nrecovery: kit {rkit:.4f} | +gainfreeze {rgf:.4f} | +temp {rtemp:.4f}")
    print(f"pred_a content {pa} ({keep5:.2f}) | pred_b entropy-setter {pb} | pred_c temp {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
