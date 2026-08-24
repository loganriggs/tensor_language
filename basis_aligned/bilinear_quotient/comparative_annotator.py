# comparative_annotator: WHO writes the comparative-class mark that 8.1 fetches (§1305:
# stream-computed, weights ratio 1.01)? Band-ablate component outputs AT COMPARATIVE
# POSITIONS ONLY (vs count-matched random positions); measure (a) CE at strict "than"
# targets and (b) 8.1's comparative-key pattern share.
#
# Registered predictions:
#   pred_a A CE-CRITICAL BAND EXISTS: some band's comparative-side dCE >= 0.3 nats AND
#          >= 5x its random control.
#   pred_b IT IS a02 (front attention writes class marks — the §1286/§1288 pattern).
#   pred_c CONTENT/KEY DISSOCIATION REPEATS: the share-drop winner is a mid-MLP band,
#          NOT the CE winner (registered direction from §1286).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'comparative_annotator_results.json'
NR = 960
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
BANDS = {'a02': ('attn', [0, 1, 2]), 'm02': ('mlp', [0, 1, 2]),
         'a38': ('attn', [3, 4, 5, 6, 7, 8]), 'm38': ('mlp', [3, 4, 5, 6, 7, 8]),
         'late9': ('both', [9])}


@torch.no_grad()
def forward_abl(idx, band, posmask, want_pat10):
    kind, layers = (None, []) if band is None else BANDS[band]
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
        if L == 8 and want_pat10:
            pat10 = pat.abs().detach()
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        yo = at.c_proj(y)
        if band is not None and L in layers and kind in ('attn', 'both'):
            yo = yo * (1 - pm)
        x = xm + yo
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if band is not None and L in layers and kind in ('mlp', 'both'):
            mo = mo * (1 - pm)
        x = x + mo
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0), pat10


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
    TGT = torch.isin(tgt_all, qm_t) & (OPENER >= 8) & (dist >= 2) & (dist <= 20)
    TGT[:, :64] = False
    OPENMASK = is_comp
    nopen = int(OPENMASK.sum())
    g = torch.Generator().manual_seed(7)
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
            p = p10[:, 1]
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
        res[band] = {'open': {'dce': round(ce_o - base_ce, 4), 'share': round(sh_o, 4),
                              'share_drop': round(1 - sh_o / max(base_sh, 1e-6), 3)},
                     'rand': {'dce': round(ce_r - base_ce, 4),
                              'share_drop': round(1 - sh_r / max(base_sh, 1e-6), 3)}}
        print(f"{band}: open dCE {res[band]['open']['dce']} share-drop {res[band]['open']['share_drop']} | "
              f"rand dCE {res[band]['rand']['dce']} share-drop {res[band]['rand']['share_drop']}", flush=True)

    win_ce = max(res, key=lambda b: res[b]['open']['dce'])
    win_sh = max(res, key=lambda b: res[b]['open']['share_drop'])
    wc = res[win_ce]
    pa = wc['open']['dce'] >= 0.3 and wc['open']['dce'] >= 5 * max(wc['rand']['dce'], 1e-4)
    pb = win_ce == 'a02'
    pc = win_sh.startswith('m') and win_sh != win_ce
    out = {'n_rows': NR, 'base': {'q_ce': round(base_ce, 4), 'wh_share': round(base_sh, 4)},
           'bands': res, 'winner_ce': win_ce, 'winner_share': win_sh,
           'pred_a_ce_band': bool(pa), 'pred_b_a02': bool(pb),
           'pred_c_dissociation': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"winner_ce {win_ce} | winner_share {win_sh}")
    print(f"pred_a ce-band {pa} | pred_b a02 {pb} | pred_c dissoc {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
