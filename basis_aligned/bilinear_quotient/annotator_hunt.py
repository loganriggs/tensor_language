# annotator_hunt: WHO writes the opener annotation? (§1272 next step.) 13.8 fetches a
# computed "unclosed delimiter here" signal from opener positions. Ablate each BAND's
# writes AT OPENER POSITIONS ONLY (outputs zeroed there; everything else untouched) and
# measure both (i) close-target CE and (ii) 13.8's delimiter-key pattern-mass share at
# close targets. Bands: attn0-2 / mlp0-2 / attn3-8 / mlp3-8 / attn9-12+mlp9-12.
# Control: the same band ablation at COUNT-MATCHED random non-opener positions.
#
# Registered predictions:
#   pred_a AN ANNOTATOR BAND EXISTS: some band's opener-side ablation cuts 13.8's
#          delimiter-key share by >= 50% AND raises close-target CE by >= 0.15.
#   pred_b POSITION-SPECIFIC: that band's random-position ablation produces <= 20% of both
#          effects.
#   pred_c COHERENCE: the band ranking by share-collapse matches the ranking by close-CE
#          damage on the top choice (the fetch signal and the behaviour damage name the
#          same writer).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'annotator_hunt_results.json'
NR = 96
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
BANDS = {'a02': ('attn', [0, 1, 2]), 'm02': ('mlp', [0, 1, 2]),
         'a38': ('attn', [3, 4, 5, 6, 7, 8]), 'm38': ('mlp', [3, 4, 5, 6, 7, 8]),
         'late912': ('both', [9, 10, 11, 12])}


@torch.no_grad()
def forward_abl(idx, band, posmask, want_pat13):
    """Zero band components' outputs at posmask positions. Returns (logits, pat13|None)."""
    kind, layers = (None, []) if band is None else BANDS[band]
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    pat13 = None
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
        if L == 13 and want_pat13:
            pat13 = pat.abs().detach()
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
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0), pat13


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    dl = set()
    for t in range(50257):
        try:
            d = enc.decode([t])
        except Exception:
            continue
        if any(c in d for c in ['"', '(', '[', '{', ')', ']', '}']):
            dl.add(t)
    dl_ids = torch.tensor(sorted(dl))
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    tok_isdl = torch.isin(ROWS[:, :-1], dl_ids)
    tgt_isdl = torch.isin(ROWS[:, 1:], dl_ids)
    ctx = torch.zeros_like(tgt_isdl)
    for w in range(1, 65):
        sh = torch.zeros_like(tok_isdl)
        sh[:, w:] = tok_isdl[:, :-w]
        ctx |= sh
    TGT = tgt_isdl & ctx; TGT[:, :64] = False
    nopen = int(tok_isdl.sum())
    g = torch.Generator().manual_seed(5)
    RANDPOS = torch.zeros_like(tok_isdl)
    flat = torch.randperm(tok_isdl.numel(), generator=g)[:nopen]
    RANDPOS.view(-1)[flat] = True
    RANDPOS &= ~tok_isdl
    print(f"opener positions {nopen} | close targets {int(TGT.sum())}", flush=True)

    dl_ids_dev = dl_ids.to(DEV)

    def run(band, use_rand):
        ce_t = 0.0; n_t = 0; sh_t = 0.0; nb = 0
        for i in range(0, 48, 4):
            bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            pmask = (RANDPOS if use_rand else tok_isdl)[i:i + 4].to(DEV).float()
            lo, p13 = forward_abl(idx, band, pmask if band is not None else None, True)
            lo = lo.float()
            lse = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1), reduction='none').view(tg.shape)
            tm = TGT[i:i + 4].to(DEV)
            ce_t += float(lse[tm].sum()); n_t += int(tm.sum())
            isdl_k = torch.isin(idx, dl_ids_dev)
            p = p13[:, 8]
            share = (p * isdl_k.unsqueeze(1).float()).sum(-1) / p.sum(-1).clamp_min(1e-9)
            sh_t += float(share[tm].mean()); nb += 1
        return ce_t / max(n_t, 1), sh_t / nb

    base_ce, base_sh = run(None, False)
    print(f"base: close CE {base_ce:.4f} | 13.8 dl-share {base_sh:.4f}", flush=True)
    res = {}
    for band in BANDS:
        ce_o, sh_o = run(band, False)
        ce_r, sh_r = run(band, True)
        res[band] = {'open': {'dce': round(ce_o - base_ce, 4), 'share': round(sh_o, 4),
                              'share_drop': round(1 - sh_o / max(base_sh, 1e-6), 3)},
                     'rand': {'dce': round(ce_r - base_ce, 4),
                              'share_drop': round(1 - sh_r / max(base_sh, 1e-6), 3)}}
        print(f"{band}: open dCE {res[band]['open']['dce']} share-drop {res[band]['open']['share_drop']} | rand dCE {res[band]['rand']['dce']} share-drop {res[band]['rand']['share_drop']}", flush=True)

    win_sh = max(res, key=lambda b: res[b]['open']['share_drop'])
    win_ce = max(res, key=lambda b: res[b]['open']['dce'])
    w = res[win_sh]
    pa = w['open']['share_drop'] >= 0.5 and w['open']['dce'] >= 0.15
    pb = (w['rand']['share_drop'] <= 0.2 * max(w['open']['share_drop'], 1e-6) and
          w['rand']['dce'] <= 0.2 * max(w['open']['dce'], 1e-6))
    pc = win_sh == win_ce
    out = {'n_rows': 48, 'base': {'close_ce': round(base_ce, 4), 'dl_share': round(base_sh, 4)},
           'bands': res, 'winner_share': win_sh, 'winner_ce': win_ce,
           'pred_a_annotator': bool(pa), 'pred_b_position_specific': bool(pb),
           'pred_c_coherent': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"winner(share) {win_sh} | winner(CE) {win_ce}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
