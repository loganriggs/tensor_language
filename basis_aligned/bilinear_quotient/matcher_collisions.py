# matcher_collisions: GOAL-3 EXPERIMENT (user 2026-08-24: "generalization predictions —
# know from the weights which specific data points it does badly on"). The matchers'
# criterion is a weights-readable bilinear form on raw codes (§1238; match = strongly
# NEGATIVE s1*s2, AUC 0.00 "inverted"). So the weights should also tell us where the
# criterion FAILS: non-identical token pairs (t, u) whose codes COLLIDE under both
# matchers' forms. Prediction: plant "u y" in context, present t later, and the model
# falsely fetches y — an adversarial input read directly off the weights.
#
# Mining: 2048 frequency-drawn corpus tokens; m(t,u) = min(-S25, -S38) (both matchers must
# see a match). TRUE collisions = different stripped-lowercase text (case/space variants
# reported separately). Live test: random-token rows, plant [u, y] at 80/81, t at 140;
# read logp(y) at 140. Classes (64 pairs each): COLL (high m, different text), CTRL
# (matched random low-m pairs), IDENT (t == u, positive anchor).
#
# Registered predictions:
#   pred_a COLLISIONS EXIST IN WEIGHTS: >= 64 true-collision pairs with m >= 0.5x the
#          median identical-pair m.
#   pred_b FALSE FETCH IS REAL: mean logp(y) gain COLL - CTRL >= 1.0 nat (gate: IDENT -
#          CTRL >= 3 nats, else the planting instrument itself failed).
#   pred_c DOSE-RESPONSE: rank correlation between m and per-pair gain across COLL+CTRL
#          >= 0.4.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'matcher_collisions_results.json'
NTOK = 2048; QPOS = 200; KPOS = 72; NP = 64
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
MATCHERS = ((2, 5), (3, 8))


@torch.no_grad()
def score_matrix(L, h, toks):
    at = H[L].attn
    x = F.rms_norm(m.transformer.wte(toks), (D,))
    N = x.shape[0]
    dummy = torch.zeros(1, QPOS + 1, 9, 128, device=DEV)
    cos_t, sin_t = at.rotary(dummy)

    def pipe(lin, pos):
        z = F.rms_norm(lin(x).view(N, 9, 128), (128,)).view(1, N, 9, 128)
        return are(z, cos_t[:, pos:pos + 1], sin_t[:, pos:pos + 1])[0, :, h]
    q1 = pipe(at.c_q, QPOS); k1 = pipe(at.c_k, KPOS)
    q2 = pipe(at.c_q2, QPOS); k2 = pipe(at.c_k2, KPOS)
    return (torch.einsum('qd,kd->qk', q1.float(), k1.float()) / 128.0) * \
           (torch.einsum('qd,kd->qk', q2.float(), k2.float()) / 128.0)


@torch.no_grad()
def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    ROWSN = cl.fineweb_rows(48)[:, :T].reshape(-1)
    g = torch.Generator().manual_seed(31)
    perm = torch.randperm(ROWSN.shape[0], generator=g)
    seen = set(); toks = []
    for i in perm.tolist():
        t = int(ROWSN[i])
        if t not in seen:
            seen.add(t); toks.append(t)
        if len(toks) >= NTOK:
            break
    toks_t = torch.tensor(toks, device=DEV)
    S25 = score_matrix(2, 5, toks_t)
    S38 = score_matrix(3, 8, toks_t)
    M = torch.minimum(-S25, -S38)                            # match-likeness, both matchers
    diag = M.diag()
    med_ident = float(diag.median())
    print(f"median identical m {med_ident:.4f} | offdiag m mean {float((M - torch.diag(diag)).mean()):.4f}", flush=True)

    texts = [enc.decode([t]).strip().lower() for t in toks]
    Mo = M.clone(); Mo.fill_diagonal_(-1e9)
    flat = Mo.flatten().argsort(descending=True)
    coll = []; variants = 0
    for f in flat.tolist():
        i, j = f // NTOK, f % NTOK
        if Mo[i, j] < 0.5 * med_ident:
            break
        if texts[i] == texts[j]:
            variants += 1
            continue
        coll.append((i, j, float(Mo[i, j])))
        if len(coll) >= NP * 3:
            break
    print(f"true collisions >= 0.5*med: {len(coll)} (case/space variants skipped: {variants})", flush=True)
    n_coll_bar = sum(1 for _, _, v in coll if v >= 0.5 * med_ident)
    pa = n_coll_bar >= NP and len(coll) >= NP
    coll = coll[:NP]

    # controls: random different-text pairs with low m
    ctrl = []
    while len(ctrl) < NP:
        i = int(torch.randint(0, NTOK, (1,), generator=g)); j = int(torch.randint(0, NTOK, (1,), generator=g))
        if i == j or texts[i] == texts[j]:
            continue
        if float(M[i, j]) < 0.1 * med_ident:
            ctrl.append((i, j, float(M[i, j])))
    ident = [(i, i, float(M[i, i])) for i in torch.randperm(NTOK, generator=g)[:NP].tolist()]

    # live planting
    vocab_pool = ROWSN.unique()
    classes = {'coll': coll, 'ctrl': ctrl, 'ident': ident}
    gains = {}
    per_pair = {'m': [], 'lp': [], 'cls': []}
    for cname, pairs in classes.items():
        rows = vocab_pool[torch.randint(0, len(vocab_pool), (NP, T + 1), generator=g)]
        ys = vocab_pool[torch.randint(0, len(vocab_pool), (NP,), generator=g)]
        for b, (i, j, mv) in enumerate(pairs):
            rows[b, 80] = toks[j]      # u (source key token)
            rows[b, 81] = ys[b]        # y (its successor)
            rows[b, 140] = toks[i]     # t (query token)
        lps = []
        for b0 in range(0, NP, 8):
            idx = rows[b0:b0 + 8, :-1].to(DEV).contiguous()
            lo = fwd(idx).float().log_softmax(-1)
            for bb in range(idx.shape[0]):
                lps.append(float(lo[bb, 140, ys[b0 + bb]]))
        gains[cname] = sum(lps) / len(lps)
        if cname in ('coll', 'ctrl'):
            per_pair['m'] += [p[2] for p in pairs]
            per_pair['lp'] += lps
            per_pair['cls'] += [cname] * NP
        print(f"{cname}: mean logp(y)@t {gains[cname]:.4f}", flush=True)

    anchor = gains['ident'] - gains['ctrl']
    gain = gains['coll'] - gains['ctrl']
    mm = torch.tensor(per_pair['m']); lp = torch.tensor(per_pair['lp'])
    rm = mm.argsort().argsort().float(); rl = lp.argsort().argsort().float()
    rho = float(((rm - rm.mean()) * (rl - rl.mean())).sum() /
                (rm.std() * rl.std() * (len(rm) - 1) + 1e-9))
    pb = anchor >= 3.0 and gain >= 1.0
    pc = rho >= 0.4
    out = {'n_tokens': NTOK, 'median_ident_m': round(med_ident, 4),
           'n_true_collisions': len(coll), 'variants_skipped': variants,
           'mean_logp': {k: round(v, 4) for k, v in gains.items()},
           'ident_anchor_gain': round(anchor, 4), 'collision_gain': round(gain, 4),
           'rank_corr_m_vs_logp': round(rho, 4),
           'top_collisions': [[texts[i], texts[j], round(v, 3)] for i, j, v in coll[:15]],
           'pred_a_collisions_exist': bool(pa), 'pred_b_false_fetch': bool(pb),
           'pred_c_dose_response': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"anchor {anchor:.3f} | collision gain {gain:.3f} | rho {rho:.3f}")
    print(f"top collisions: {out['top_collisions'][:8]}")
    print(f"pred_a exist {pa} | pred_b fetch {pb} | pred_c dose {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
