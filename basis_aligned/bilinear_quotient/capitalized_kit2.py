# capitalized_kit2: WHERE DOES THE KIT'S GAP LIVE? (§1400: kit .644, committee marginal
# only .029, gate covers .485.) Same template; scoring SPLIT by GATE (gated vs ungated
# targets), plus an ALL-GATE arm (commons live everywhere — does gate SCOPE bind?).
# Assumptions as §1400 (CAPGATE, committee ungated, NR=1920).
#
# Registered predictions:
#   pred_a the gate binds: kit recovery on GATED targets >= ungated + 0.15.
#   pred_b gate scope is costing: all-gate kit gains >= 0.05 recovery (all targets)
#          over the gated kit.
#   pred_c the committee earns its keep in context: committee marginal on GATED
#          targets >= 0.05.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'capitalized_kit2_results.json'
NMEAN = 24; NR = 1920
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
COMMITTEE = {(15, 3), (16, 0), (16, 4), (16, 5), (17, 0), (17, 1), (17, 2)}

_c = json.load(open(PT + 'closer_band_slim_results.json'))['ranked_top16']
_q = json.load(open(PT + 'question_kit_slim_results.json'))['ranked_heads'][:16]
COMMONS = {tuple(int(x) for x in h.split('.')) for h in set(_c) | set(_q)}


@torch.no_grad()
def fwd_arm(idx, arm, vmeans, ymeans, gatemask, use_committee=True):
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
        if arm == 'full':
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
        elif arm == 'ymean':
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
            for h in range(9):
                y[:, :, h] = ymeans[L][h].to(y.dtype)
        else:  # route / kit
            vr = v.clone()
            for h in range(9):
                if not (arm == 'kit' and use_committee and (L, h) in COMMITTEE):
                    vr[:, :, h] = vmeans[L][h].to(vr.dtype)
            vvr = (1 - at.lamb) * vr + at.lamb * v1.view_as(vr)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vvr.dtype), vvr)
            if arm == 'kit' and any((L, h) in COMMONS for h in range(9)):
                vv_live = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
                y_live = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv_live.dtype), vv_live)
                gm = gatemask.view(B, T, 1)
                for h in range(9):
                    if (L, h) in COMMONS:
                        y[:, :, h] = torch.where(gm, y_live[:, :, h], y[:, :, h])
        x = xm + at.c_proj(y.reshape(B, T, D))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    cap = set(); gate_tok = set()
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        if len(d) >= 2 and d[0] == ' ' and d[1].isupper() and d[1:].isalpha():
            cap.add(tok)
        if any(c in d for c in '.!?\n:"') or (len(d) >= 1 and d.lstrip(' ')[:1].isupper()):
            gate_tok.add(tok)
    cap_ids = torch.tensor(sorted(cap)); gate_ids = torch.tensor(sorted(gate_tok))
    print(f"cap vocab {len(cap)} gate vocab {len(gate_tok)} commons {len(COMMONS)}",
          flush=True)

    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]

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

    toks = EVR[:, :-1]; tgt = EVR[:, 1:]
    TARGET = torch.isin(tgt, cap_ids)
    TARGET[:, :64] = False
    GATE = torch.isin(toks, gate_ids)
    coverage = float((TARGET & GATE).sum()) / max(float(TARGET.sum()), 1.0)
    ELSE = ~TARGET; ELSE[:, :64] = False
    TG = TARGET & GATE; TU = TARGET & ~GATE
    print(f"targets {int(TARGET.sum())} (gated {int(TG.sum())} / ungated {int(TU.sum())})"
          f" | coverage {coverage:.3f}", flush=True)

    def ce_run(arm, use_committee=True, allgate=False):
        acc = {'t': [0.0, 0], 'tg': [0.0, 0], 'tu': [0.0, 0], 'e': [0.0, 0]}
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg2 = bb[:, 1:].contiguous()
            gm = (torch.ones_like(GATE[i:i + 8]) if allgate else GATE[i:i + 8]).to(DEV)
            lo = fwd_arm(idx, arm, vmeans, ymeans, gm, use_committee).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg2.reshape(-1),
                                 reduction='none').view(tg2.shape)
            for key, M in (('t', TARGET), ('tg', TG), ('tu', TU), ('e', ELSE)):
                mm = M[i:i + 8].to(DEV)
                acc[key][0] += float(ce[mm].sum()); acc[key][1] += int(mm.sum())
        return {k: acc[k][0] / max(acc[k][1], 1) for k in acc}

    res = {}
    for name, arm, uc, ag in (('full', 'full', True, False), ('ymean', 'ymean', True, False),
                              ('kit', 'kit', True, False),
                              ('kit_nocommittee', 'kit', False, False),
                              ('kit_allgate', 'kit', True, True)):
        r = ce_run(arm, uc, ag)
        res[name] = {k: round(v, 4) for k, v in r.items()}
        print(f"{name}: " + " ".join(f"{k} {r[k]:.4f}" for k in r), flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    def rec(a, key):
        gap = res['ymean'][key] - res['full'][key]
        return (res['ymean'][key] - res[a][key]) / max(gap, 1e-6)
    kg, ku = rec('kit', 'tg'), rec('kit', 'tu')
    ka, kt = rec('kit_allgate', 't'), rec('kit', 't')
    cg = rec('kit', 'tg') - rec('kit_nocommittee', 'tg')
    pa = kg >= ku + 0.15
    pb = (ka - kt) >= 0.05
    pc = cg >= 0.05
    out = {'n_targets': int(TARGET.sum()), 'coverage': round(coverage, 4), 'res': res,
           'recovery': {'kit_gated': round(kg, 4), 'kit_ungated': round(ku, 4),
                        'kit_all': round(kt, 4), 'kit_allgate_all': round(ka, 4),
                        'committee_marginal_gated': round(cg, 4)},
           'pred_a_gate_binds': bool(pa), 'pred_b_scope_costs': bool(pb),
           'pred_c_committee_in_context': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"gated {kg:.3f} ungated {ku:.3f} | allgate {ka:.3f} vs kit {kt:.3f} | "
          f"committee-marginal-gated {cg:.3f}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
