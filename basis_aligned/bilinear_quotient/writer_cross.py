# writer_cross: §1289 cross-circuit question — are 1.1/1.8, the two redundant sufficient
# writers of the WH-opener annotation, ALSO the writers of the DELIMITER-opener annotation
# that 13.8 consumes (§1272-74)? Same leave-one-alive instrument, new opener class:
# positions holding quote/bracket tokens; targets = close-delimiter predictions with an
# opener in the prior 64 tokens. Conditions: clean, keep-none, keep 1.1, keep 1.8,
# keep both, keep 1.0 (control head).
#
# Registered predictions:
#   pred_a ANNOTATION NEEDED HERE TOO: keep-none costs >= 0.15 nats at close targets.
#   pred_b UNIVERSAL WRITERS: 1.1 or 1.8 alone restores >= 60% of the keep-none damage.
#   pred_c CONTROL: kept 1.0 restores <= 20%.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'writer_cross_results.json'
NR = 192
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
FRONT = [0, 1, 2]


@torch.no_grad()
def forward_keep(idx, keep, posmask):
    """Zero y-slices of all FRONT-layer heads at posmask positions, except `keep`=(L,h).
    keep=None -> keep-none; posmask=None -> clean forward."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    pm = None if posmask is None else posmask.to(x.dtype).unsqueeze(-1)
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
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        if pm is not None and L in FRONT:
            keepvec = torch.zeros(D, device=DEV, dtype=y.dtype)
            for (LL, hh) in (keep or ()):
                if LL == L:
                    keepvec[hh * 128:(hh + 1) * 128] = 1.0
            y = y * (1 - pm) + y * pm * keepvec
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    dl = set()
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        if any(c in d for c in ['"', '(', '[', '{', ')', ']', '}']):
            dl.add(tok)
    dl_t = torch.tensor(sorted(dl))

    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    toks = ROWS[:, :-1]; tgt_all = ROWS[:, 1:]
    tok_isdl = torch.isin(toks, dl_t)
    isq = torch.isin(tgt_all, dl_t)
    ctx = torch.zeros_like(tok_isdl)
    for w in range(1, 65):
        sh = torch.zeros_like(tok_isdl)
        sh[:, w:] = tok_isdl[:, :-w]
        ctx |= sh
    TGT = isq & ctx
    TGT[:, :64] = False
    OPENMASK = tok_isdl
    print(f"opener positions {int(OPENMASK.sum())} | close targets {int(TGT.sum())}", flush=True)

    def run(keep, use_mask=True):
        ce_t = 0.0; n_t = 0
        for i in range(0, NR, 4):
            bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            pmask = OPENMASK[i:i + 4].to(DEV) if use_mask else None
            lo = forward_keep(idx, keep, pmask).float()
            lse = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1), reduction='none').view(tg.shape)
            tm = TGT[i:i + 4].to(DEV)
            ce_t += float(lse[tm].sum()); n_t += int(tm.sum())
        return ce_t / max(n_t, 1)

    base = run(None, use_mask=False)
    anchor = run(None) - base
    print(f"base {base:.4f} | keep-none dCE {anchor:.4f}", flush=True)
    res = {}
    for name, keep in (('k11', {(1, 1)}), ('k18', {(1, 8)}),
                       ('k11_18', {(1, 1), (1, 8)}), ('k10ctrl', {(1, 0)})):
        d = run(keep) - base
        res[name] = {'dce': round(d, 4), 'restore': round(1 - d / max(anchor, 1e-6), 3)}
        print(f"{name}: dCE {d:.4f} restore {res[name]['restore']:.3f}", flush=True)
    pa = anchor >= 0.15
    pb = max(res['k11']['restore'], res['k18']['restore']) >= 0.6
    pc = res['k10ctrl']['restore'] <= 0.2
    out = {'n_rows': NR, 'base': round(base, 4), 'keep_none_dce': round(anchor, 4),
           'conds': res,
           'pred_a_annotation_needed': bool(pa), 'pred_b_universal_writers': bool(pb),
           'pred_c_control': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a needed {pa} | pred_b universal {pb} | pred_c ctrl {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
