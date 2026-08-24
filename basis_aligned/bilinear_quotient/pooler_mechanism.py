# pooler_mechanism: WHY does the middle pool's content-similarity bias GROW with depth?
# (§1085: r = -0.02 (L6) -> 0.18 (L12), causal 1.8-4x, mechanism unknown — the last open
# mechanism item on the middle-pooling dossier.)
#
# SUBSTRATE-AVAILABILITY hypothesis: the QK circuits are not "more content-seeking" deeper —
# they see more content because the stream's content component grows with depth. Test by
# manipulating ONLY the q/k input inside the pattern computation (model forward untouched,
# features computed from the ORIGINAL input): (i) remove the content basis from x before
# q/k, (ii) AMPLIFY it 3x at the shallow layer, (iii) random-basis removal null.
#
# Registered predictions:
#   pred_a SUBSTRATE-MEDIATED: content-basis removal drops the content_sim correlation by
#          >= 70% (relative) at L10 and L12.
#   pred_b DEPTH-GROWTH = AVAILABILITY: 3x content amplification at L6 raises L6's
#          content_sim r by >= 2x (toward the deep layers' values).
#   pred_c NULL: random-64-basis removal changes each layer's r by <= 30% relative.
import json, time, sys, types, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; NH = 9; HD = 128; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'pooler_criterion_results.json'
NSEQ = 96; SEQ = 256; MID = [6, 8, 10, 12]; REF = [8, 10, 12]; K = 64
NMASK = 16; DMIN = 5
H = m.transformer.h
MOD = sys.modules[type(H[0].attn).__module__]
CAPX = {}; MASK = {'layer': -1, 'mask': None}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def capx_hook(L):
    def h(mo, args): CAPX[L] = args[0].detach()
    return h



@torch.no_grad()
def pattern_for(attn, x):
    B, T, C = x.shape
    q = attn.c_q(x).view(B, T, NH, HD); k = attn.c_k(x).view(B, T, NH, HD)
    q2 = attn.c_q2(x).view(B, T, NH, HD); k2 = attn.c_k2(x).view(B, T, NH, HD)
    cos, sin = attn.rotary(q)
    q, k = F.rms_norm(q, (HD,)), F.rms_norm(k, (HD,))
    q, k = MOD.apply_rotary_emb(q, cos, sin), MOD.apply_rotary_emb(k, cos, sin)
    q2, k2 = F.rms_norm(q2, (HD,)), F.rms_norm(k2, (HD,))
    q2, k2 = MOD.apply_rotary_emb(q2, cos, sin), MOD.apply_rotary_emb(k2, cos, sin)
    s1 = torch.einsum('bqhd,bkhd->bhqk', q, k); s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)
    pat = (s1/HD)*(s2/HD)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
    return pat.masked_fill_(mask.logical_not(), 0.0)


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
def corr_under(blocks, U_c, tfreq, transform):
    """content_sim correlation per layer with q/k input transformed by `transform(x)`."""
    hcap = [H[L].attn.register_forward_pre_hook(capx_hook(L)) for L in MID]
    feats = {L: [] for L in MID}; pats = {L: [] for L in MID}
    g = torch.Generator(device=DEV).manual_seed(0)
    for i in range(0, 32, 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); fwd(idx); T = idx.shape[1]
        for L in MID:
            x = CAPX[L].float()
            xq = transform(x) if transform is not None else x
            pat = pattern_for(H[L].attn, xq.to(CAPX[L].dtype)).abs().mean(1)
            c = x @ U_c
            for b in range(x.shape[0]):
                qi = torch.randint(DMIN+1, T, (1500,), generator=g, device=DEV)
                kj = (torch.rand(1500, generator=g, device=DEV) * (qi - DMIN).float()).long()
                csim = F.cosine_similarity(c[b, qi], c[b, kj], dim=-1)
                feats[L].append(csim.cpu()); pats[L].append(pat[b, qi, kj].cpu())
    for h in hcap: h.remove()
    out = {}
    for L in MID:
        f = torch.cat(feats[L], 0); y = torch.cat(pats[L], 0)
        fz = (f - f.mean()) / f.std().clamp_min(1e-6); yz = (y - y.mean()) / y.std().clamp_min(1e-6)
        out[L] = round(float((fz * yz).mean()), 4)
    return out


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    U_c = content_basis(blocks)
    tfreq = torch.zeros(int(m.lm_head.weight.shape[0]), device=DEV)

    g = torch.Generator(device=DEV).manual_seed(7)
    R = torch.randn(D, K, device=DEV, generator=g)
    R, _ = torch.linalg.qr(R)

    def rm_c(x): return x - (x @ U_c) @ U_c.T
    def amp_c(x): return x + 2.0 * ((x @ U_c) @ U_c.T)
    def rm_r(x): return x - (x @ R) @ R.T

    r_base = corr_under(blocks, U_c, tfreq, None)
    print(f"base r: {r_base}", flush=True)
    r_rmc = corr_under(blocks, U_c, tfreq, rm_c)
    print(f"rm_content r: {r_rmc}", flush=True)
    r_amp = corr_under(blocks, U_c, tfreq, amp_c)
    print(f"amp_content r: {r_amp}", flush=True)
    r_rmr = corr_under(blocks, U_c, tfreq, rm_r)
    print(f"rm_random r: {r_rmr}", flush=True)

    def rel_drop(L): return 1 - abs(r_rmc[L]) / max(abs(r_base[L]), 1e-6)
    pa = rel_drop(10) >= 0.7 and rel_drop(12) >= 0.7
    pb = abs(r_amp[6]) >= 2 * max(abs(r_base[6]), 1e-6)
    pc = all(abs(abs(r_rmr[L]) - abs(r_base[L])) <= 0.3 * max(abs(r_base[L]), 1e-6) for L in (8, 10, 12))
    out = {'n_seq': NSEQ, 'r': {'base': {str(k): v for k, v in r_base.items()},
                                 'rm_content': {str(k): v for k, v in r_rmc.items()},
                                 'amp_content': {str(k): v for k, v in r_amp.items()},
                                 'rm_random': {str(k): v for k, v in r_rmr.items()}},
           'pred_a_substrate': bool(pa), 'pred_b_availability': bool(pb), 'pred_c_null': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(PT + 'pooler_mechanism_results.json', 'w'), indent=1)
    print(f"pred_a substrate {pa} | pred_b avail {pb} | pred_c null {pc}")
    print(f"wrote pooler_mechanism_results.json ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
