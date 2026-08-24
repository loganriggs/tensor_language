# question_mechanism: HOW does 10.5 know the sentence is a question? (§1282/§1284 next
# step; mirrors quote_mechanism.py, the 13.8 instrument.) Two measurements:
#  (1) OFFSET/READ profile: at "?"-prediction targets, does 10.5's |pattern| mass sit on
#      the question's WH-OPENER position (13.8-style opener fetch)? Metrics: (a) mean
#      |pattern| on the exact opener key vs a random earlier key, per target; (b) share of
#      total mass on WH-token keys, target vs elsewhere positions.
#  (2) WEIGHTS-ONLY criterion (§1238 instrument): raw rms(wte) codes of WH tokens as keys
#      vs 512 ordinary in-corpus tokens under 10.5's q/k pipelines.
#
# Registered predictions:
#   pred_a OPENER-READER: mean |pattern| on the exact opener key >= 3x a random earlier
#          key at the same targets.
#   pred_b CRITERION IS STREAM-COMPUTED, NOT EMBEDDING-NATIVE (the 13.8 sibling pattern):
#          weights-only WH-key ratio <= 2.0 — raw WH codes do NOT stand out structurally.
#   pred_c CONTROL HEAD FLAT: inert 10.0 (share -0.006 in question_heads) shows opener
#          ratio <= 1.5.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'question_mechanism_results.json'
NR = 192; QPOS = 200; KPOS = 72; LQ = 10
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
    qm = set(); sent_end = set(); wh = set()
    WH = ['Who', 'What', 'When', 'Where', 'Why', 'How', 'Which', 'Is', 'Are', 'Do', 'Does',
          'Did', 'Can', 'Could', 'Will', 'Would', 'Should']
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        if '?' in d:
            qm.add(tok)
        if any(c in d for c in '.!?'):
            sent_end.add(tok)
        if d.strip() in WH:
            wh.add(tok)
    qm_t = torch.tensor(sorted(qm)); se_t = torch.tensor(sorted(sent_end)); wh_t = torch.tensor(sorted(wh))

    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    toks = ROWS[:, :-1]; tgt_all = ROWS[:, 1:]
    is_end = torch.isin(toks, se_t); is_wh = torch.isin(toks, wh_t)
    B2, T2 = toks.shape
    state = torch.zeros(B2, dtype=torch.bool)
    QSTATE = torch.zeros_like(toks, dtype=torch.bool)
    OPENER = torch.full_like(toks, -1)
    recent_end = torch.full((B2,), 99, dtype=torch.long)
    opener_pos = torch.full((B2,), -1, dtype=torch.long)
    for p in range(T2):
        op = is_wh[:, p] & (recent_end <= 2)
        opener_pos = torch.where(is_end[:, p], torch.full_like(opener_pos, -1),
                                 torch.where(op & ~state, torch.full_like(opener_pos, p), opener_pos))
        state = torch.where(is_end[:, p], torch.zeros_like(state), state | op)
        QSTATE[:, p] = state
        OPENER[:, p] = opener_pos
        recent_end = torch.where(is_end[:, p], torch.zeros_like(recent_end), recent_end + 1)
    TARGET = torch.isin(tgt_all, qm_t) & QSTATE & (OPENER >= 8)
    TARGET[:, :64] = False
    ntar = int(TARGET.sum())
    print(f"targets {ntar}", flush=True)

    g = torch.Generator().manual_seed(6)
    res_open = {5: {'op': [], 'rnd': []}, 0: {'op': [], 'rnd': []}}
    res_share = {5: {'tar': [], 'els': []}, 0: {'tar': [], 'els': []}}
    for i in range(0, NR, 4):
        idx = toks[i:i + 4].to(DEV).contiguous()
        pat = pattern_at(idx, LQ)                              # (B,9,T,T) abs
        iswh_b = torch.isin(idx, wh_t.to(DEV))
        tm = TARGET[i:i + 4]
        for h in (5, 0):
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
    for h in (5, 0):
        mo = sum(res_open[h]['op']) / max(len(res_open[h]['op']), 1)
        mr = sum(res_open[h]['rnd']) / max(len(res_open[h]['rnd']), 1)
        st = float(torch.cat(res_share[h]['tar']).mean()) if res_share[h]['tar'] else 0.0
        se = float(torch.cat(res_share[h]['els']).mean()) if res_share[h]['els'] else 1e-9
        stats[h] = {'opener_absmean': round(mo, 5), 'randkey_absmean': round(mr, 5),
                    'opener_ratio': round(mo / max(mr, 1e-9), 2),
                    'wh_share_tar': round(st, 4), 'wh_share_els': round(se, 4),
                    'wh_share_ratio': round(st / max(se, 1e-9), 2)}
        print(f"10.{h}: opener {mo:.5f} vs rand {mr:.5f} (ratio {stats[h]['opener_ratio']}) | "
              f"wh-share tar {st:.4f} els {se:.4f}", flush=True)

    # weights-only: WH keys vs ordinary in-corpus keys under 10.5 / 10.0
    rows_all = toks[:8].reshape(-1)
    uniq = torch.unique(rows_all)
    ordinary = uniq[~torch.isin(uniq, wh_t)]
    ord_sel = ordinary[torch.randperm(len(ordinary), generator=g)[:512]].to(DEV)
    wh_sel = wh_t.to(DEV)
    at = H[LQ].attn
    res_w = {}
    for h in (5, 0):
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
           'pred_a_opener_reader': bool(stats[5]['opener_ratio'] >= 3),
           'pred_b_stream_computed': bool(res_w[5]['ratio'] <= 2.0),
           'pred_c_control_flat': bool(stats[0]['opener_ratio'] <= 1.5),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a opener {out['pred_a_opener_reader']} | pred_b stream {out['pred_b_stream_computed']} | pred_c ctrl {out['pred_c_control_flat']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
