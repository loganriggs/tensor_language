# dir2_identity: three new instruments for the unnamed dominant far-channel direction
# (§1268: position-flat, half-content, single-removal 0.078).
#  (1) VOCAB READOUT: lm_head(dir2) — is it a frequency-axis direction? Registered bar on
#      the frequency tilt of its top-vs-bottom logit tokens.
#  (2) FUNCTIONAL UNIVERSALITY: correlate dir2's per-position delta coordinate with the
#      top-8 far-delta directions' coordinates in swiglu18 ON THE SAME ROWS — a functional
#      match transcends basis differences.
#  (3) WRITER: which mid-band layer's far component writes dir2 (per-layer projection norms).
#
# Registered predictions:
#   pred_a FREQUENCY AXIS: mean log-frequency of dir2's top-100 positive-logit tokens
#          differs from its bottom-100's by >= 2 (nats of log-count) in either direction.
#   pred_b UNIVERSAL: max profile correlation with a swiglu far-direction >= 0.5.
#   pred_c ONE WRITER: some single layer's far component carries >= 50% of the summed
#          per-layer dir2 projection energy at L5-9.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'dir2_identity_results.json'
NR = 24; QSTART = 128; WIN = 64; K = 64
NSEQ = 96; SEQ = 256; REF = [8, 10, 12]
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
MASK_W = None
FULL = None


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'dir2_identity_results.json'
NR = 24; QSTART = 128; WIN = 64; K = 64
NSEQ = 96; SEQ = 256; REF = [8, 10, 12]
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
MASK_W = None
FULL = None


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'dir2_identity_results.json'
NR = 24; QSTART = 128; WIN = 64; K = 64
NSEQ = 96; SEQ = 256; REF = [8, 10, 12]
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
MASK_W = None
FULL = None


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def content_basis(blocks):
    """standard idiom: top-K PCA of pooled L8-12 mlp-input deviation."""
    cap = {L: [] for L in REF}; hs = []
    for L in REF:
        def mk(L):
            def h(mo, i_, o_): cap[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(H[L].mlp.register_forward_hook(mk(L)))
    idsL = []
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fwd(idx)
    for h in hs: h.remove()
    tok = torch.cat(idsL, 0); V = int(m.lm_head.weight.shape[0]); devsum = None
    for L in REF:
        X = torch.cat(cap[L], 0); xb = torch.zeros(V, D, device=DEV); cn = torch.zeros(V, device=DEV)
        xb.index_add_(0, tok, X); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
        dv = X - (xb/cn.clamp_min(1).unsqueeze(1))[tok]
        devsum = dv if devsum is None else devsum + dv; cap[L] = []; del X
    dev = devsum/len(REF); dev = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(dev, full_matrices=False)
    return Vt[:K].T.contiguous()




@torch.no_grad()
def far_bag(idx, far=True):
    """Recency-weighted bag of v1 codes. far=True: k <= t-WIN; far=False: t-WIN < k <= t."""
    B = idx.shape[0]
    x = F.rms_norm(m.transformer.wte(idx), (D,))
    v1 = m.transformer.h[0].attn.c_v(x)                       # (B,T,D)
    ar = torch.arange(T, device=DEV)
    dist = (ar[:, None] - ar[None, :]).float()
    wmat = torch.exp(-dist / 64.0)
    valid = (dist >= WIN) if far else ((dist > 0) & (dist < WIN))
    wmat = (wmat * valid.float() * torch.tril(torch.ones(T, T, device=DEV))).to(v1.dtype)
    bag = torch.einsum('qk,bkd->bqd', wmat, v1)
    return bag / wmat.sum(-1).clamp_min(1e-6).unsqueeze(0).unsqueeze(-1)




@torch.no_grad()
def stream_at8(idx, masked):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(m.transformer.h):
        if L == 8:
            return x.detach().float()
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
        pat = pat.masked_fill(~(MASK_W if masked else FULL), 0.0)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return None





@torch.no_grad()
def forward_far_rm(idx, V16, mode):
    """mode: 'none' (base) / 'full' (remove far component) / 'proj' (remove V16-proj of far)."""
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
        pat = pat.masked_fill(~FULL, 0.0)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        yo = at.c_proj(y)
        if mode != 'none' and 5 <= L <= 9:
            pat_far = pat.masked_fill(MASK_W, 0.0)            # far = beyond WIN, pos-0 excluded
            yf = at.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat_far.to(vv.dtype), vv).reshape(B, T, D))
            if mode == 'full':
                yo = yo - yf
            else:
                yo = yo - (yf.float() @ V16 @ V16.T).to(yo.dtype)
        x = xm + yo
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def ce_far(rows, V16, mode):
    qp = torch.arange(QSTART, T, device=DEV)
    tot = 0.0; n = 0
    for i in range(0, rows.shape[0], 4):
        bb = rows[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lo = forward_far_rm(idx, V16, mode).float()
        tot += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                     tgt[:, qp].reshape(-1), reduction='sum'))
        n += idx.shape[0] * len(qp)
    return tot / n





@torch.no_grad()
def sw_stream_at8(mdl, idx, masked, MASKW_SW, FULL_SW):
    x = F.rms_norm(mdl.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    aresw = sys.modules[type(mdl.transformer.h[0].attn).__module__].apply_rotary_emb
    for L, blk in enumerate(mdl.transformer.h):
        if L == 8:
            return x.detach().float()
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        q = F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,))
        k = F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,))
        cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
        q = aresw(q, cos, sin); k = aresw(k, cos, sin)
        scores = torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / (128 ** 0.5)
        scores = scores.masked_fill(~(MASKW_SW if masked else FULL_SW), float('-inf'))
        pat = F.softmax(scores, dim=-1)
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
    global MASK_W, FULL
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ar = torch.arange(T, device=DEV)
    vis = ((ar[:, None] - ar[None, :]) < WIN) | (ar[None, :] == 0)
    FULL = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    MASK_W = FULL & vis
    FIT = cl.fineweb_rows(12)[:, :T + 1].contiguous()

    DL = []
    for i in range(0, 12, 4):
        idx = FIT[i:i + 4, :-1].to(DEV).contiguous()
        xc = stream_at8(idx, False); xm = stream_at8(idx, True)
        DL.append((xc - xm)[:, QSTART:].reshape(-1, D).cpu())
    DLc = torch.cat(DL)
    Dc = DLc - DLc.mean(0)
    _, S, Vt = torch.pca_lowrank(Dc, q=32)
    dir2 = Vt[:, 2].to(DEV).float()

    # (1) vocab readout
    logits = (m.lm_head.weight.float() @ dir2)
    V = logits.shape[0]
    rows_all = cl.fineweb_rows(48)[:, :T].reshape(-1)
    cnts = torch.bincount(rows_all, minlength=V).float().to(DEV)
    top = torch.topk(logits, 100).indices; bot = torch.topk(-logits, 100).indices
    lf_top = float(torch.log1p(cnts[top]).mean()); lf_bot = float(torch.log1p(cnts[bot]).mean())
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    toks_top = [enc.decode([t]) for t in top[:15].tolist() if t < 50257]
    toks_bot = [enc.decode([t]) for t in bot[:15].tolist() if t < 50257]
    print(f"log-freq top {lf_top:.2f} vs bot {lf_bot:.2f}", flush=True)
    print(f"top tokens: {toks_top}", flush=True)
    print(f"bot tokens: {toks_bot}", flush=True)

    # (2) swiglu functional match
    from tier2_model import load_elriggs
    mdl, cfg = load_elriggs('swiglu18'); mdl = mdl.to(DEV).eval()
    DLs = []
    for i in range(0, 12, 4):
        idx = FIT[i:i + 4, :-1].to(DEV).contiguous()
        xc = sw_stream_at8(mdl, idx, False, MASK_W, FULL)
        xm = sw_stream_at8(mdl, idx, True, MASK_W, FULL)
        DLs.append((xc - xm)[:, QSTART:].reshape(-1, D).cpu())
    DLs = torch.cat(DLs)
    Ds = DLs - DLs.mean(0)
    _, _, Vs = torch.pca_lowrank(Ds, q=8)
    prof_b = (Dc @ Vt[:, 2].unsqueeze(1)).squeeze(1)          # bilin dir2 profile
    best_r, best_j = 0.0, -1
    for j in range(8):
        prof_s = (Ds @ Vs[:, j].unsqueeze(1)).squeeze(1)
        bz = (prof_b - prof_b.mean()) / prof_b.std().clamp_min(1e-6)
        sz = (prof_s - prof_s.mean()) / prof_s.std().clamp_min(1e-6)
        r = abs(float((bz * sz).mean()))
        if r > best_r:
            best_r, best_j = r, j
    print(f"best swiglu profile match: dir {best_j} r={best_r:.3f}", flush=True)

    # (3) writer: per-layer far-component projection energy onto dir2
    energies = {}
    for Lb in (5, 6, 7, 8, 9):
        tote = 0.0
        for i in range(0, 12, 4):
            idx = FIT[i:i + 4, :-1].to(DEV).contiguous()
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            B = idx.shape[0]
            for L, blk in enumerate(m.transformer.h):
                at = blk.attn
                xm2 = blk.lambdas[0] * x + blk.lambdas[1] * x0
                xin = F.rms_norm(xm2, (D,))
                cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
                q = are(F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,)), cos, sin)
                k = are(F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,)), cos, sin)
                q2 = are(F.rms_norm(at.c_q2(xin).view(B, T, 9, 128), (128,)), cos, sin)
                k2 = are(F.rms_norm(at.c_k2(xin).view(B, T, 9, 128), (128,)), cos, sin)
                pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
                    * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
                pat = pat.masked_fill(~FULL, 0.0)
                v = at.c_v(xin).view(B, T, 9, 128)
                if v1 is None:
                    v1 = v
                vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
                y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
                if L == Lb:
                    pat_far = pat.masked_fill(MASK_W, 0.0)
                    yf = at.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat_far.to(vv.dtype), vv).reshape(B, T, D))
                    tote += float(((yf.float() @ dir2) ** 2)[:, QSTART:].sum())
                    break
                x = xm2 + at.c_proj(y)
                x = x + blk.mlp(F.rms_norm(x, (D,)))
        energies[Lb] = tote
    tot = sum(energies.values())
    shares = {str(L): round(v / tot, 3) for L, v in energies.items()}
    print(f"writer shares {shares}", flush=True)

    top_share = max(shares.values())
    out = {'logfreq': {'top100': round(lf_top, 3), 'bot100': round(lf_bot, 3)},
           'top_tokens': toks_top, 'bot_tokens': toks_bot,
           'swiglu_match': {'dir': best_j, 'r': round(best_r, 4)},
           'writer_shares': shares,
           'pred_a_freq_axis': bool(abs(lf_top - lf_bot) >= 2),
           'pred_b_universal': bool(best_r >= 0.5),
           'pred_c_one_writer': bool(top_share >= 0.5),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a freq {out['pred_a_freq_axis']} | pred_b univ {out['pred_b_universal']} | pred_c writer {out['pred_c_one_writer']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
