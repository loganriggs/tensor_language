# capitalized_committee_grain: PATTERN OR PAYLOAD? (§1398: committee-of-7 = clean
# surgical unit, 71% of band.) Split the committee's contribution by grain: QK-mean
# (replace each committee head's attention pattern with its 24-row mean pattern, values
# live) vs OV-mean (y-slice mean, = the §1397/98 ablation). Per-layer sub-arms test
# uniformity. Manual-forward implementation (crosskit §105 code path); patterns are
# position-dependent [T,T] means. Assumption registered: mean pattern from NMEAN=24 rows,
# same masks as §1397/98, NR=960.
#
# Registered predictions:
#   pred_a PAYLOAD-DOMINANT: committee OV damage >= 2x committee QK damage on targets
#          (late annotators read register state; their patterns should be generic).
#   pred_b BOTH grains surgical: else <= 10% of target damage for each unit arm.
#   pred_c UNIFORM: every layer sub-arm agrees OV > QK on targets (3 of 3).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'capitalized_committee_grain_results.json'
NMEAN = 24; NR = 960
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
COMMITTEE = {(15, 3), (16, 0), (16, 4), (16, 5), (17, 0), (17, 1), (17, 2)}
LAYERS = (15, 16, 17)


@torch.no_grad()
def fwd_arm(idx, qk_set, ov_set, meanpat, ymean, capture=None):
    """qk_set: heads whose pattern is replaced by meanpat; ov_set: heads whose y output
    is replaced by ymean. capture: dict L->list to record mean pattern/y during ref pass."""
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
        if qk_set:
            for h in range(9):
                if (L, h) in qk_set:
                    pat[:, h] = meanpat[(L, h)].to(pat.dtype)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
        if capture is not None and L in LAYERS:
            capture[L]['pat'].append(pat.float().mean(0).cpu())
            capture[L]['y'].append(y.float().mean((0, 1)).cpu())
        if ov_set:
            for h in range(9):
                if (L, h) in ov_set:
                    y[:, :, h] = ymean[(L, h)].to(y.dtype)
        x = xm + at.c_proj(y.reshape(B, T, D))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    cap = set()
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        if len(d) >= 2 and d[0] == ' ' and d[1].isupper() and d[1:].isalpha():
            cap.add(tok)
    cap_ids = torch.tensor(sorted(cap))

    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]

    capture = {L: {'pat': [], 'y': []} for L in LAYERS}
    for i in range(0, NMEAN, 4):
        fwd_arm(MEANR[i:i + 4, :-1].to(DEV).contiguous(), None, None, None, None,
                capture=capture)
    meanpat = {}; ymean = {}
    for L in LAYERS:
        mp = torch.stack(capture[L]['pat']).mean(0)          # [9,T,T]
        ym = torch.stack(capture[L]['y']).mean(0)            # [9,128]
        for h in range(9):
            if (L, h) in COMMITTEE:
                meanpat[(L, h)] = mp[h].to(DEV)
                ymean[(L, h)] = ym[h].to(DEV)
    print("refs cached", flush=True)

    tgt_all = EVR[:, 1:]
    TARGET = torch.isin(tgt_all, cap_ids)
    TARGET[:, :64] = False
    JIT = torch.zeros_like(TARGET)
    JIT[:, 2:] = TARGET[:, :-2]
    JIT &= ~TARGET
    g = torch.Generator().manual_seed(97)
    sc = torch.rand(TARGET.shape, generator=g)
    sc[TARGET | JIT] = -1.0; sc[:, :64] = -1.0
    k = int(TARGET.sum())
    flat = sc.flatten()
    RAND = torch.zeros_like(flat, dtype=torch.bool)
    RAND[flat.topk(min(k, int((flat > 0).sum()))).indices] = True
    RAND = RAND.view(TARGET.shape)
    ELSE = ~TARGET & ~JIT & ~RAND; ELSE[:, :64] = False
    print(f"targets {k}", flush=True)

    def ce_all(qk_set, ov_set):
        sums = dict(t=0.0, j=0.0, r=0.0, e=0.0); ns = dict(t=0, j=0, r=0, e=0)
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd_arm(idx, qk_set, ov_set, meanpat, ymean).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            for key, M in (('t', TARGET), ('j', JIT), ('r', RAND), ('e', ELSE)):
                mm = M[i:i + 8].to(DEV)
                sums[key] += float(ce[mm].sum()); ns[key] += int(mm.sum())
        return {kk: sums[kk] / max(ns[kk], 1) for kk in sums}

    base = ce_all(None, None)
    arms = {}
    per_layer = {L: {p for p in COMMITTEE if p[0] == L} for L in LAYERS}
    runs = [('ov_all', None, COMMITTEE), ('qk_all', COMMITTEE, None)]
    for L in LAYERS:
        runs.append((f'ov_L{L}', None, per_layer[L]))
        runs.append((f'qk_L{L}', per_layer[L], None))
    for name, qs, os_ in runs:
        r = ce_all(qs, os_)
        arms[name] = {kk: round(r[kk] - base[kk], 4) for kk in r}
        print(f"{name}: tgt {arms[name]['t']:+.4f} jit {arms[name]['j']:+.4f} "
              f"rand {arms[name]['r']:+.4f} else {arms[name]['e']:+.4f}", flush=True)
        json.dump({'partial': True, 'arms': arms}, open(OUT, 'w'), indent=1)

    pa = arms['ov_all']['t'] >= 2.0 * max(arms['qk_all']['t'], 1e-4)
    pb = all(abs(arms[a]['e']) <= 0.10 * max(arms[a]['t'], 1e-4)
             for a in ('ov_all', 'qk_all'))
    pc = all(arms[f'ov_L{L}']['t'] > arms[f'qk_L{L}']['t'] for L in LAYERS)
    out = {'n_targets': k, 'n_rows': NR, 'base': {kk: round(v, 4) for kk, v in base.items()},
           'arms': arms,
           'ov_over_qk': round(arms['ov_all']['t'] / max(arms['qk_all']['t'], 1e-4), 3),
           'pred_a_payload_dominant': bool(pa), 'pred_b_both_surgical': bool(pb),
           'pred_c_uniform': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"ov/qk ratio {out['ov_over_qk']} | pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
