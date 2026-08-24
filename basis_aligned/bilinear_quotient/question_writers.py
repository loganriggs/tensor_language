# question_writers: §1288 next step — WHICH HEADS of attn0/attn1 write the WH-opener
# annotation? Redundancy makes leave-one-out blind (singles read ~0), so the design is
# LEAVE-ONE-ALIVE: ablate ALL 27 front-head contributions (L0-2, pre-c_proj y-slices,
# zeroed) at opener positions, except one kept head; a head that alone restores the
# behaviour is a sufficient writer. Built-in null: the keep-none condition must reproduce
# the a02 anchor (0.397) — if it doesn't, the y-slice instrument disagrees with the
# output-zeroing instrument and nothing else in the run is interpretable.
#
# Registered predictions:
#   pred_a A SUFFICIENT WRITER EXISTS: some kept head restores >= 60% of the keep-none
#          damage (dCE <= 0.4x).
#   pred_b IT LIVES IN LAYER 1 (attn1 alone was fully sufficient at layer grain, §1288).
#   pred_c REDUNDANT AT HEAD GRAIN TOO: >= 2 heads each restore >= 40%.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'question_writers_results.json'
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
            if keep is not None and keep[0] == L:
                hh = keep[1]
                keepvec[hh * 128:(hh + 1) * 128] = 1.0
            y = y * (1 - pm) + y * pm * keepvec
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


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
    print(f"opener positions {int(OPENMASK.sum())} | ? targets {int(TGT.sum())}", flush=True)

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
    print(f"base {base:.4f} | keep-none dCE {anchor:.4f} (a02 output-zero anchor was 0.3966)", flush=True)
    res = {}
    for L in FRONT:
        for hh in range(9):
            d = run((L, hh)) - base
            restore = 1 - d / max(anchor, 1e-6)
            res[f'{L}.{hh}'] = {'dce': round(d, 4), 'restore': round(restore, 3)}
            print(f"keep {L}.{hh}: dCE {d:.4f} restore {restore:.3f}", flush=True)
    ranked = sorted(res.items(), key=lambda kv: -kv[1]['restore'])
    best, bw = ranked[0]
    pa = bw['restore'] >= 0.6
    pb = best.startswith('1.')
    pc = sum(1 for _, v in res.items() if v['restore'] >= 0.4) >= 2
    out = {'n_rows': NR, 'base': round(base, 4), 'keep_none_dce': round(anchor, 4),
           'anchor_consistent': bool(abs(anchor - 0.3966) <= 0.12),
           'heads': res, 'top5': ranked[:5], 'best': best,
           'pred_a_sufficient_writer': bool(pa), 'pred_b_layer1': bool(pb),
           'pred_c_multihead_redundant': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"top5 {ranked[:5]}")
    print(f"pred_a sufficient {pa} | pred_b layer1 {pb} | pred_c redundant {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
