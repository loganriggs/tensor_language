# question_annotator3: §1287 follow-up on the two reversals. (1) WHICH mid MLP destroys
# the annotation after a02 is ablated at openers? a02+mX singles for X in 3..8; if one MLP
# is the scrubber, that joint cell restores most of a02's 0.397-nat damage. (2) WHERE is
# the front-attention redundancy threshold? Pairs a0+a1, a0+a2, a1+a2 (singles were ~0,
# all three = 0.397).
#
# Registered predictions:
#   pred_a PAIRS STILL CHEAP: every front pair's opener dCE <= 30% of a02's — the
#          annotation survives any single missing layer (collective, threshold at 3).
#   pred_b ONE SCRUBBER: some a02+mX restores >= 60% of a02's damage (dCE <= 0.4x a02);
#          the specific bet mlp5 (§1247's named off-manifold scrubber) recorded as m5_wins.
#   pred_c CONTROLS (large-effect bands only, dCE >= 0.05 — §1287 denominator lesson):
#          random-side <= 20% of opener-side.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'question_annotator3_results.json'
NR = 192
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
A02 = [('attn', 0), ('attn', 1), ('attn', 2)]
BANDS = {'a01': [('attn', 0), ('attn', 1)], 'a0a2': [('attn', 0), ('attn', 2)],
         'a12': [('attn', 1), ('attn', 2)], 'a02': list(A02)}
for X in (3, 4, 5, 6, 7, 8):
    BANDS[f'a02m{X}'] = A02 + [('mlp', X)]


@torch.no_grad()
def forward_abl(idx, band, posmask, want_pat10):
    comps = [] if band is None else BANDS[band]
    aset = {L for k, L in comps if k == 'attn'}
    mset = {L for k, L in comps if k == 'mlp'}
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    pat10 = None
    pm = posmask.unsqueeze(-1).to(x.dtype) if posmask is not None else None
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
        if L == 10 and want_pat10:
            pat10 = pat.abs().detach()
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        yo = at.c_proj(y)
        if L in aset:
            yo = yo * (1 - pm)
        x = xm + yo
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if L in mset:
            mo = mo * (1 - pm)
        x = x + mo
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0), pat10


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
    OPENMASK = torch.zeros_like(toks, dtype=torch.bool)
    recent_end = torch.full((B2,), 99, dtype=torch.long)
    for p in range(T2):
        op = is_wh[:, p] & (recent_end <= 2)
        OPENMASK[:, p] = op
        state = torch.where(is_end[:, p], torch.zeros_like(state), state | op)
        QSTATE[:, p] = state
        recent_end = torch.where(is_end[:, p], torch.zeros_like(recent_end), recent_end + 1)
    TGT = torch.isin(tgt_all, qm_t) & QSTATE
    TGT[:, :64] = False
    nopen = int(OPENMASK.sum())
    g = torch.Generator().manual_seed(9)
    RANDPOS = torch.zeros_like(OPENMASK)
    flat = torch.randperm(OPENMASK.numel(), generator=g)[:nopen]
    RANDPOS.view(-1)[flat] = True
    RANDPOS &= ~OPENMASK
    print(f"opener positions {nopen} | ? targets {int(TGT.sum())}", flush=True)

    wh_dev = wh_t.to(DEV)

    def run(band, use_rand):
        ce_t = 0.0; n_t = 0; sh_t = 0.0; nb = 0
        for i in range(0, NR, 4):
            bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            pmask = (RANDPOS if use_rand else OPENMASK)[i:i + 4].to(DEV).float()
            lo, p10 = forward_abl(idx, band, pmask if band is not None else None, True)
            lo = lo.float()
            lse = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1), reduction='none').view(tg.shape)
            tm = TGT[i:i + 4].to(DEV)
            ce_t += float(lse[tm].sum()); n_t += int(tm.sum())
            iswh_k = torch.isin(idx, wh_dev)
            p = p10[:, 5]
            share = (p * iswh_k.unsqueeze(1).float()).sum(-1) / p.sum(-1).clamp_min(1e-9)
            if int(tm.sum()) > 0:
                sh_t += float(share[tm].sum()); nb += int(tm.sum())
        return ce_t / max(n_t, 1), sh_t / max(nb, 1)

    base_ce, base_sh = run(None, False)
    print(f"base: ? CE {base_ce:.4f} | 10.5 wh-share {base_sh:.4f}", flush=True)
    res = {}
    for band in BANDS:
        ce_o, sh_o = run(band, False)
        ce_r, sh_r = run(band, True)
        res[band] = {'open': {'dce': round(ce_o - base_ce, 4),
                              'share_drop': round(1 - sh_o / max(base_sh, 1e-6), 3)},
                     'rand': {'dce': round(ce_r - base_ce, 4),
                              'share_drop': round(1 - sh_r / max(base_sh, 1e-6), 3)}}
        print(f"{band}: open dCE {res[band]['open']['dce']} share-drop {res[band]['open']['share_drop']} | "
              f"rand dCE {res[band]['rand']['dce']} share-drop {res[band]['rand']['share_drop']}", flush=True)

    a02_dce = res['a02']['open']['dce']
    pairs = {b: res[b]['open']['dce'] for b in ('a01', 'a0a2', 'a12')}
    mx = {b: res[b]['open']['dce'] for b in res if b.startswith('a02m')}
    best_mx = min(mx, key=mx.get)
    pa = all(v <= 0.3 * max(a02_dce, 1e-6) for v in pairs.values())
    pb = mx[best_mx] <= 0.4 * max(a02_dce, 1e-6)
    big = [b for b in BANDS if res[b]['open']['dce'] >= 0.05]
    pc = all(abs(res[b]['rand']['dce']) <= 0.2 * res[b]['open']['dce'] for b in big) if big else False
    out = {'n_rows': NR, 'base': {'q_ce': round(base_ce, 4), 'wh_share': round(base_sh, 4)},
           'bands': res, 'a02_anchor': a02_dce, 'best_restorer': best_mx,
           'm5_wins': bool(best_mx == 'a02m5'),
           'pred_a_pairs_cheap': bool(pa), 'pred_b_one_scrubber': bool(pb),
           'pred_c_controls_flat': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"best restorer {best_mx} (m5_wins {out['m5_wins']}) | a02 anchor {a02_dce}")
    print(f"pred_a pairs-cheap {pa} | pred_b one-scrubber {pb} | pred_c ctrl {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
