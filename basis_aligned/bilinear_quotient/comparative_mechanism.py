# comparative_mechanism: HOW does 8.1 know "than" is licensed? (§1304 next step; mirrors
# question_mechanism, the 10.5 instrument.) (1) OFFSET/READ: at "than" targets, does
# 8.1's |pattern| mass sit on the licensing COMPARATIVE's position? (2) WEIGHTS-ONLY
# criterion: raw rms(wte) codes of comparative tokens as keys vs 512 ordinary in-corpus
# tokens under 8.1's q/k pipelines.
#
# Registered predictions:
#   pred_a COMPARATIVE-FETCHER: mean |pattern| on the exact comparative key >= 3x a random
#          earlier key at the same targets.
#   pred_b CRITERION IS EMBEDDING-NATIVE (the 8.7-lexicon bet, OPPOSITE of the state
#          heads): weights-only comparative-key ratio >= 2.0.
#   pred_c CONTROL HEAD FLAT: inert 8.0 shows comparative-fetch ratio <= 1.5.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'comparative_mechanism_results.json'
NR = 960; QPOS = 200; KPOS = 72; LQ = 8
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h


@torch.no_grad()
def pattern_at(idx, LSTOP):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(m.transformer.h):
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
        if L == LSTOP:
            return pat.abs()
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return None


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
    qm_t = torch.tensor(sorted(than)); wh_t = torch.tensor(sorted(comp))

    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    toks = ROWS[:, :-1]; tgt_all = ROWS[:, 1:]
    is_comp = torch.isin(toks, wh_t)
    B2, T2 = toks.shape
    OPENER = torch.full_like(toks, -1)
    opener_pos = torch.full((B2,), -1, dtype=torch.long)
    for p in range(T2):
        opener_pos = torch.where(is_comp[:, p], torch.full_like(opener_pos, p), opener_pos)
        stale = (opener_pos >= 0) & (p - opener_pos > 20)
        opener_pos = torch.where(stale, torch.full_like(opener_pos, -1), opener_pos)
        OPENER[:, p] = opener_pos
    dist = torch.arange(T2).view(1, -1) - OPENER
    TARGET = torch.isin(tgt_all, qm_t) & (OPENER >= 8) & (dist >= 2) & (dist <= 20)
    TARGET[:, :64] = False
    ntar = int(TARGET.sum())
    print(f"targets {ntar}", flush=True)

    g = torch.Generator().manual_seed(6)
    res_open = {1: {'op': [], 'rnd': []}, 0: {'op': [], 'rnd': []}}
    res_share = {1: {'tar': [], 'els': []}, 0: {'tar': [], 'els': []}}
    for i in range(0, NR, 4):
        idx = toks[i:i + 4].to(DEV).contiguous()
        pat = pattern_at(idx, LQ)                              # (B,9,T,T) abs
        iswh_b = torch.isin(idx, wh_t.to(DEV))
        tm = TARGET[i:i + 4]
        for h in (1, 0):
            p = pat[:, h]
            mass = (p * iswh_b.unsqueeze(1).float()).sum(-1)
            tot = p.sum(-1).clamp_min(1e-9)
            share = mass / tot
            em = ~tm; em2 = em.clone(); em2[:, :64] = False
            res_share[h]['tar'].append(share[tm.to(DEV)].cpu())
            res_share[h]['els'].append(share[em2.to(DEV)].cpu())
            for b, q in tm.nonzero():
                op = int(OPENER[i + b, q])
                rnd = int(torch.randint(8, int(q), (1,), generator=g))
                if rnd == op:
                    rnd = max(8, op - 5)
                res_open[h]['op'].append(float(p[b, q, op]))
                res_open[h]['rnd'].append(float(p[b, q, rnd]))
    stats = {}
    for h in (1, 0):
        mo = sum(res_open[h]['op']) / max(len(res_open[h]['op']), 1)
        mr = sum(res_open[h]['rnd']) / max(len(res_open[h]['rnd']), 1)
        st = float(torch.cat(res_share[h]['tar']).mean()) if res_share[h]['tar'] else 0.0
        se = float(torch.cat(res_share[h]['els']).mean()) if res_share[h]['els'] else 1e-9
        stats[h] = {'opener_absmean': round(mo, 5), 'randkey_absmean': round(mr, 5),
                    'opener_ratio': round(mo / max(mr, 1e-9), 2),
                    'wh_share_tar': round(st, 4), 'wh_share_els': round(se, 4),
                    'wh_share_ratio': round(st / max(se, 1e-9), 2)}
        print(f"8.{h}: opener {mo:.5f} vs rand {mr:.5f} (ratio {stats[h]['opener_ratio']}) | "
              f"wh-share tar {st:.4f} els {se:.4f}", flush=True)

    # weights-only: WH keys vs ordinary in-corpus keys under 10.5 / 10.0
    rows_all = toks[:8].reshape(-1)
    uniq = torch.unique(rows_all)
    ordinary = uniq[~torch.isin(uniq, wh_t)]
    ord_sel = ordinary[torch.randperm(len(ordinary), generator=g)[:512]].to(DEV)
    wh_sel = wh_t.to(DEV)
    at = H[LQ].attn
    res_w = {}
    for h in (1, 0):
        x_q = F.rms_norm(m.transformer.wte(ord_sel), (D,))
        x_kw = F.rms_norm(m.transformer.wte(wh_sel), (D,))
        x_ko = F.rms_norm(m.transformer.wte(ord_sel), (D,))
        dummy = torch.zeros(1, QPOS + 1, 9, 128, device=DEV)
        cos_t, sin_t = at.rotary(dummy)
        def pipe(lin, x, pos):
            z = F.rms_norm(lin(x).view(-1, 9, 128), (128,)).view(1, -1, 9, 128)
            return are(z, cos_t[:, pos:pos + 1], sin_t[:, pos:pos + 1])[0, :, h]
        q1 = pipe(at.c_q, x_q, QPOS); q2 = pipe(at.c_q2, x_q, QPOS)
        kw1 = pipe(at.c_k, x_kw, KPOS); kw2 = pipe(at.c_k2, x_kw, KPOS)
        ko1 = pipe(at.c_k, x_ko, KPOS); ko2 = pipe(at.c_k2, x_ko, KPOS)
        sw = (torch.einsum('qd,kd->qk', q1.float(), kw1.float()) / 128) * \
             (torch.einsum('qd,kd->qk', q2.float(), kw2.float()) / 128)
        so = (torch.einsum('qd,kd->qk', q1.float(), ko1.float()) / 128) * \
             (torch.einsum('qd,kd->qk', q2.float(), ko2.float()) / 128)
        res_w[h] = {'wh_absmean': round(float(sw.abs().mean()), 5),
                    'ord_absmean': round(float(so.abs().mean()), 5),
                    'ratio': round(float(sw.abs().mean() / so.abs().mean().clamp_min(1e-9)), 2)}
    print(f"weights scores {res_w}", flush=True)

    out = {'n_targets': ntar,
           'pattern': {str(k): v for k, v in stats.items()},
           'weights_scores': {str(k): v for k, v in res_w.items()},
           'pred_a_comp_fetcher': bool(stats[1]['opener_ratio'] >= 3),
           'pred_b_embedding_native': bool(res_w[1]['ratio'] >= 2.0),
           'pred_c_control_flat': bool(stats[0]['opener_ratio'] <= 1.5),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a fetch {out['pred_a_comp_fetcher']} | pred_b native {out['pred_b_embedding_native']} | pred_c ctrl {out['pred_c_control_flat']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
