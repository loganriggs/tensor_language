# head_partition3: CUT 3 of the head-partition program — data-dependent, respecting the
# §1292 lesson that identity content is FULL-RANK. Per source position and head, remove
# from the mixed value vv its rank-1 projection onto THAT position's own v1 direction:
# "delete this token's identity from what this head delivers", wherever in the mix it
# sits, no global subspace assumed. Null: an equally-sized removal onto a position-
# SHUFFLED v1 direction (wrong token's identity, same geometry).
#
# Registered predictions:
#   pred_a IDENTITY REMOVAL IS THE PART (main4): per-position code removal reproduces
#          >= 60% of main4's whole-head induction damage (0.395) with elsewhere <= 40%
#          of whole-head elsewhere (0.0605).
#   pred_b SHUFFLE NULL CLEAN: the shuffled-direction version <= 25% of the aligned
#          version's induction damage (main4).
#   pred_c TAIL CARRIES IDENTITY TOO: tail aligned-removal induction damage >= 2x its
#          elsewhere damage and >= 0.3 nats.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'head_partition3_results.json'
NR = 384; W = 128
H = m.transformer.h
MAIN4 = {(2, 5), (3, 8), (8, 3), (8, 4)}


def selmask(layer, name):
    s = torch.zeros(9, dtype=torch.bool)
    if name is None or layer == 0:
        return s
    if name == 'main4':
        for (L, h) in MAIN4:
            if L == layer:
                s[h] = True
    elif name == 'tail':
        s[:] = True
        for (L, h) in MAIN4:
            if L == layer:
                s[h] = False
    elif name == 'all':
        s[:] = True
    return s


@torch.no_grad()
def fwd(idx, which=None, mode=None, perm=None):
    """mode 'aligned': remove vv's projection onto this position's v1 direction;
    mode 'shuffled': onto v1 of perm-shuffled positions (null). Selected heads only."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(H):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
        are = sys.modules[type(at).__module__].apply_rotary_emb
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
        v1v = v1.view_as(v)
        vv = (1 - at.lamb) * v + at.lamb * v1v
        sel = selmask(L, which)
        if sel.any() and mode is not None:
            u = v1v if mode == 'aligned' else v1v[:, perm]
            u = u.float() / u.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
            coef = (vv.float() * u).sum(-1, keepdim=True)
            comp = (coef * u).to(vv.dtype)
            sm = sel.to(DEV).view(1, 1, 9, 1)
            vv = torch.where(sm, vv - comp, vv)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    toks = ROWS[:, :-1]; tgt = ROWS[:, 1:]
    TGT = torch.zeros_like(toks, dtype=torch.bool)
    for b0 in range(0, NR, 64):
        tb = toks[b0:b0 + 64]; gb = tgt[b0:b0 + 64]
        eq = (tb.unsqueeze(1) == tb.unsqueeze(2)) & (gb.unsqueeze(1) == gb.unsqueeze(2))
        q_i = torch.arange(T).view(1, T, 1); p_i = torch.arange(T).view(1, 1, T)
        band = (q_i < p_i) & (q_i >= p_i - W)
        TGT[b0:b0 + 64] = (eq & band).any(1)
    TGT[:, :16] = False
    ELSE = ~TGT; ELSE[:, :16] = False
    print(f"induction targets {int(TGT.sum())}", flush=True)
    g = torch.Generator().manual_seed(13)
    perm = torch.randperm(T, generator=g).to(DEV)

    def ce_sets(which, mode):
        tots = {'t': 0.0, 'e': 0.0}; ns = {'t': 0, 'e': 0}
        for i in range(0, NR, 8):
            bb = ROWS[i:i + 8].to(DEV); idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx, which, mode, perm).float()
            lse = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1), reduction='none').view(tg.shape)
            for name, mask in (('t', TGT), ('e', ELSE)):
                mm = mask[i:i + 8].to(DEV)
                tots[name] += float(lse[mm].sum()); ns[name] += int(mm.sum())
        return {k: tots[k] / max(ns[k], 1) for k in tots}

    base = ce_sets(None, None)
    print(f"base {base}", flush=True)
    res = {}
    for name, which, mode in (('main4_id', 'main4', 'aligned'), ('main4_shuf', 'main4', 'shuffled'),
                              ('tail_id', 'tail', 'aligned'), ('tail_shuf', 'tail', 'shuffled'),
                              ('all_id', 'all', 'aligned'), ('all_shuf', 'all', 'shuffled')):
        r = ce_sets(which, mode)
        res[name] = {'d_ind': round(r['t'] - base['t'], 4), 'd_else': round(r['e'] - base['e'], 4)}
        print(f"{name}: ind {res[name]['d_ind']} else {res[name]['d_else']}", flush=True)

    pa = (res['main4_id']['d_ind'] >= 0.6 * 0.395 and
          res['main4_id']['d_else'] <= 0.4 * 0.0605)
    pb = res['main4_shuf']['d_ind'] <= 0.25 * max(res['main4_id']['d_ind'], 1e-4)
    pc = (res['tail_id']['d_ind'] >= 0.3 and
          res['tail_id']['d_ind'] >= 2 * max(res['tail_id']['d_else'], 1e-4))
    out = {'n_rows': NR, 'n_targets': int(TGT.sum()),
           'base': {k: round(v, 4) for k, v in base.items()}, 'conds': res,
           'anchors_1290': {'main4_whole': [0.395, 0.0605]},
           'pred_a_identity_is_part': bool(pa), 'pred_b_shuffle_null': bool(pb),
           'pred_c_tail_identity': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a identity {pa} | pred_b null {pb} | pred_c tail {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
