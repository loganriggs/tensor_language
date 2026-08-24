# bucket_constants2: rung 2 of the user-proposed ladder — buckets conditioned on the
# CURRENT TOKEN's frequency class instead of position. §1232: position bits are worthless
# to the crowd (k16 = 3% recovery). §240's slice-conditioned constants (85% of circuits)
# conditioned on token type. Here: per-head constants indexed by the query position's
# token frequency bin (4 bins by corpus frequency rank: top-128, 129-1024, 1025-8192,
# rest — extending the §1151 function/content split).
#
# Conditions: base; zero; k1 (anchor, = §1232); freq4; freq4rand (same bin sizes, token->bin
# assignment shuffled — carries no frequency information).
#
# Registered predictions:
#   pred_a TOKEN TYPE BEATS POSITION: CE(freq4) < §1232's position-k16 (7.0409) by >= 0.05.
#   pred_b BUT STILL NOT DELETION: CE(freq4) > CE(zero) (the crowd's dynamic value is not
#          2-bit-conditionable either; registering the negative keeps the ladder honest).
#   pred_c SHUFFLED BINS ARE k1: |CE(freq4rand) - CE(k1)| <= 0.08.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'bucket_constants2_results.json'
NTRAIN = 24; NEVAL = 24; QSTART = 128
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
KS = [1, 2, 4, 16]


FREQBIN = None   # (V,) token -> bin, set in main


def token_buckets(idx):
    return FREQBIN[idx]     # (B,T)


@torch.no_grad()
def head_outputs(idx):
    """Per-layer pre-c_proj head outputs y (B,T,9,128) under the TRUE model."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    outs = []
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
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
        outs.append(y.detach())
        x = xm + at.c_proj(y.reshape(B, T, D))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return outs


@torch.no_grad()
def forward_bucketed(idx, means, bidx):
    """All heads' y replaced by bucket means (means: list of (k,9,128); bidx: (T,) bucket index).
    means=None with bidx=None -> zero condition."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x
    B = idx.shape[0]
    for L, blk in enumerate(m.transformer.h):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        if means is None:
            yrep = torch.zeros(B, T, 9, 128, device=DEV)
        else:
            yrep = means[L][bidx] if bidx.dim() == 2 else means[L][bidx].unsqueeze(0).expand(B, T, 9, 128)
        x = xm + at.c_proj(yrep.reshape(B, T, D).to(at.c_proj.weight.dtype))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NTRAIN + NEVAL)[:, :T + 1].contiguous()
    train, ev = ROWS[:NTRAIN], ROWS[NTRAIN:]
    qp = torch.arange(QSTART, T, device=DEV)

    # frequency bins from train rows
    global FREQBIN
    V = int(m.lm_head.weight.shape[0])
    cnts = torch.bincount(train[:, :T].reshape(-1), minlength=V)
    order = torch.argsort(cnts, descending=True)
    FREQBIN = torch.zeros(V, dtype=torch.long)
    FREQBIN[order[128:1024]] = 1; FREQBIN[order[1024:8192]] = 2; FREQBIN[order[8192:]] = 3
    g5 = torch.Generator().manual_seed(5)
    RANDBIN = FREQBIN[torch.randperm(V, generator=g5)].clone()
    FREQBIN = FREQBIN.to(DEV); RANDBIN = RANDBIN.to(DEV)

    # accumulate per-(layer,head,bin) mean outputs and global means on TRAIN
    K = 4
    sums = [torch.zeros(K, 9, 128, dtype=torch.float64) for _ in range(18)]
    cnt = [torch.zeros(K, dtype=torch.float64) for _ in range(18)]
    sums_r = [torch.zeros(K, 9, 128, dtype=torch.float64) for _ in range(18)]
    cnt_r = [torch.zeros(K, dtype=torch.float64) for _ in range(18)]
    gsum = [torch.zeros(9, 128, dtype=torch.float64) for _ in range(18)]
    gn = 0
    for i in range(0, NTRAIN, 4):
        idx = train[i:i + 4, :-1].to(DEV).contiguous()
        outs = head_outputs(idx)
        bidx = FREQBIN[idx]; ridx = RANDBIN[idx]
        for L in range(18):
            y = outs[L].double().cpu()          # (B,T,9,128)
            bb = bidx.cpu(); rr = ridx.cpu()
            for j in range(K):
                sel = (bb == j)
                if sel.any():
                    sums[L][j] += y[sel].sum(0); cnt[L][j] += int(sel.sum())
                selr = (rr == j)
                if selr.any():
                    sums_r[L][j] += y[selr].sum(0); cnt_r[L][j] += int(selr.sum())
            gsum[L] += y.mean((0, 1)); 
        gn += 1
    means_f = [ (sums[L] / cnt[L].clamp_min(1).view(K, 1, 1)).float().to(DEV) for L in range(18) ]
    means_r = [ (sums_r[L] / cnt_r[L].clamp_min(1).view(K, 1, 1)).float().to(DEV) for L in range(18) ]
    means_1 = [ (gsum[L] / gn).float().to(DEV).view(1, 9, 128) for L in range(18) ]

    def ce_cond(means, mode):
        tot = 0.0; n = 0
        for i in range(0, NEVAL, 4):
            bb2 = ev[i:i + 4].to(DEV); idx = bb2[:, :-1].contiguous(); tgt = bb2[:, 1:].contiguous()
            if mode == 'zero':
                bidx = None; mm = None
            elif mode == 'k1':
                bidx = torch.zeros_like(idx); mm = means
            elif mode == 'freq':
                bidx = FREQBIN[idx]; mm = means
            elif mode == 'rand':
                bidx = RANDBIN[idx]; mm = means
            lo = forward_bucketed(idx, mm, bidx).float()
            tot += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                         tgt[:, qp].reshape(-1), reduction='sum'))
            n += idx.shape[0] * len(qp)
        return tot / n

    tot = 0.0; n = 0
    for i in range(0, NEVAL, 4):
        bb2 = ev[i:i + 4].to(DEV); idx = bb2[:, :-1].contiguous(); tgt = bb2[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lt = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0).float()
        tot += float(F.cross_entropy(lt[:, qp].reshape(-1, lt.shape[-1]), tgt[:, qp].reshape(-1), reduction='sum'))
        n += idx.shape[0] * len(qp)
    base = tot / n

    CE = {'base': round(base, 4), 'zero': round(ce_cond(None, 'zero'), 4),
          'k1': round(ce_cond(means_1, 'k1'), 4),
          'freq4': round(ce_cond(means_f, 'freq'), 4),
          'freq4rand': round(ce_cond(means_r, 'rand'), 4)}
    out = {'n_train': NTRAIN, 'n_eval': NEVAL, 'ce': CE,
           'pos_k16_ref': 7.0409,
           'pred_a_token_beats_pos': bool(CE['freq4'] <= 7.0409 - 0.05),
           'pred_b_still_not_deletion': bool(CE['freq4'] > CE['zero']),
           'pred_c_shuffled_is_k1': bool(abs(CE['freq4rand'] - CE['k1']) <= 0.08),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE}")
    print(f"pred_a token>pos {out['pred_a_token_beats_pos']} | pred_b not-deletion {out['pred_b_still_not_deletion']} | pred_c shuffled {out['pred_c_shuffled_is_k1']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
