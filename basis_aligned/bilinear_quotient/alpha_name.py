"""NAMING step for §1140 (mass driver decodable at R² 0.93). Locate WHAT the probe reads, redteam-first:
restricted-input probes on held-out data — (a) NORM-ONLY (1 feature: ||x||; is alpha a mundane magnitude
artifact of the unnormalized pattern?), (b) content coords only (U_c, 64), (c) grammar coords only (L0-1 dev
basis, 64), (d) massive-dim values only (top-8 dims), (e) full residual (reference 0.93). Plus snippets at the
top/bottom-32 predicted-alpha positions for a human-readable name.

REGISTERED PREDICTIONS:
  (0) SANITY: full-residual MLP probe reproduces ~0.93; every restricted probe <= full.
  (a) NAMED: one restricted family reaches >= 0.6 of the full probe's R² -> alpha is carried by a NAMED
      subspace (norm / content / grammar / gain dims) — the §1109 unknown finally gets its name;
  (b) NORM-ARTIFACT: if norm-only >= 0.5, alpha is largely residual magnitude (mundane but a real name —
      report which tokens carry big residuals via the snippets);
  (c) DIFFUSE: all restricted probes < 0.3 while full = 0.93 -> the driver is a broadband residual pattern
      with no known-subspace carrier (decodable but unnamed at current vocabulary; report plainly)."""
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; NH = 9; HD = 128; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'alpha_name_results.json'
NSEQ = 96; SEQ = 256; REF = [8, 10, 12]; K = 64
BANDS = {'gatherer_L3_5': [3, 4, 5]}
H = m.transformer.h
MOD = sys.modules[type(H[0].attn).__module__]
CAPX = {}
enc = tiktoken.get_encoding('gpt2')


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
    cap = {Lr: [] for Lr in REF}; hs = []
    for Lr in REF:
        def mk(Lr):
            def h(mo, i_, o_): cap[Lr].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(H[Lr].mlp.register_forward_hook(mk(Lr)))
    idsL = []
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fwd(idx)
    for h in hs: h.remove()
    tok = torch.cat(idsL, 0); V = int(m.lm_head.weight.shape[0]); devsum = None
    for Lr in REF:
        X = torch.cat(cap[Lr], 0); cap[Lr] = []
        xb = torch.zeros(V, D, device=DEV); cn = torch.zeros(V, device=DEV)
        xb.index_add_(0, tok, X); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
        dv = X - (xb/cn.clamp_min(1).unsqueeze(1))[tok]
        devsum = dv if devsum is None else devsum + dv; del X
    dev = devsum/len(REF); dev = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(dev, full_matrices=False)
    return Vt[:K].T.contiguous()


def is_function_word(tid):
    raw = enc.decode([tid]); s = raw.strip().lower()
    return s in {'the', 'a', 'an', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'to', 'and', 'or', 'but', 'is',
                 'are', 'was', 'were', 'be', 'it', 'he', 'she', 'they', 'that', 'this', 'as', 'not', 'from'}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0])
    Uc = content_basis(blocks)
    tfreq = torch.zeros(V, device=DEV)
    ta = blocks.to(DEV).reshape(-1); tfreq.index_add_(0, ta, torch.ones_like(ta, dtype=torch.float))
    func_mask = torch.zeros(V, dtype=torch.bool)
    for t in range(min(V, 50257)):
        try:
            if is_function_word(t): func_mask[t] = True
        except Exception:
            pass
    func_mask = func_mask.to(DEV)

    ALL_L = sorted({L for ls in BANDS.values() for L in ls})
    hcap = [H[L].attn.register_forward_pre_hook(capx_hook(L)) for L in ALL_L]
    T = SEQ - 1
    di = torch.arange(T, device=DEV).view(-1, 1) - torch.arange(T, device=DEV).view(1, -1)

    # kernels from first 32 seqs
    ksum = {L: torch.zeros(NH, T, device=DEV) for L in ALL_L}; kcnt = torch.zeros(T, device=DEV)
    for i in range(0, 32, 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); fwd(idx)
        for L in ALL_L:
            pat = pattern_for(H[L].attn, CAPX[L])
            for dd in range(T):
                mask = (di == dd); nel = int(mask.sum())
                if nel == 0: continue
                ksum[L][:, dd] += pat[:, :, mask].sum((0, 2))
                if L == ALL_L[0]: kcnt[dd] += nel*pat.shape[0]
            del pat
    kern = {L: ksum[L]/kcnt.clamp_min(1) for L in ALL_L}   # NH,T mean pattern by distance

    # alpha + features on next 64 seqs
    feats = {L: [] for L in ALL_L}; alphas = {L: [] for L in ALL_L}
    for i in range(32, 96, 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); fwd(idx)
        seen_before = torch.zeros_like(idx, dtype=torch.float)
        for b in range(idx.shape[0]):
            _, inv, cnts = torch.unique(idx[b], return_inverse=True, return_counts=True)
            # seen-before flag: occurrence index > 0
            first = torch.zeros_like(idx[b], dtype=torch.bool)
            seen = {}
            arr = idx[b].tolist()
            fl = [0.0]*len(arr)
            s2 = set()
            for j2, t2 in enumerate(arr):
                fl[j2] = 1.0 if t2 in s2 else 0.0
                s2.add(t2)
            seen_before[b] = torch.tensor(fl, device=DEV)
        for L in ALL_L:
            x = CAPX[L].float()
            pat = pattern_for(H[L].attn, CAPX[L]).abs().sum(-1)
            kmass = torch.cumsum(kern[L].abs(), dim=1)
            alpha = pat / kmass.unsqueeze(0).clamp_min(1e-6)
            feats[L].append(x.reshape(-1, D).cpu())                  # FULL residual as probe input
            alphas[L].append(alpha.mean(1).reshape(-1).cpu())
            del pat
    for h in hcap: h.remove()

    Fm = torch.cat([torch.cat(feats[L], 0) for L in BANDS['gatherer_L3_5']], 0).to(DEV)
    y = torch.cat([torch.cat(alphas[L], 0) for L in BANDS['gatherer_L3_5']], 0).to(DEV)
    ok = torch.isfinite(y); Fm, y = Fm[ok], y[ok]
    yz = (y - y.mean())/y.std().clamp_min(1e-6)
    N = Fm.shape[0]; ntr = int(0.7*N)
    perm = torch.randperm(N, generator=torch.Generator(device=DEV).manual_seed(0), device=DEV)
    tr, te = perm[:ntr], perm[ntr:]
    Xz = (Fm - Fm[tr].mean(0))/Fm[tr].std(0).clamp_min(1e-6)
    # ridge linear probe
    lam = 10.0
    A2 = Xz[tr].T @ Xz[tr] + lam*torch.eye(D, device=DEV)
    w = torch.linalg.solve(A2, Xz[tr].T @ yz[tr])
    r2_lin = 1 - float(((Xz[te]@w - yz[te])**2).mean()/ (yz[te]**2).mean())
    # small MLP probe
    net = torch.nn.Sequential(torch.nn.Linear(D, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    with torch.enable_grad():
        for step in range(3000):
            ii = tr[torch.randint(0, ntr, (4096,), device=DEV)]
            loss = ((net(Xz[ii]).squeeze(-1) - yz[ii])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = torch.cat([net(Xz[te][i:i+8192]).squeeze(-1) for i in range(0, te.shape[0], 8192)], 0)
    r2_mlp = 1 - float(((pred - yz[te])**2).mean()/(yz[te]**2).mean())
    # restricted-input probes
    def mlp_probe(Xin):
        Xn = (Xin - Xin[tr].mean(0))/Xin[tr].std(0).clamp_min(1e-6)
        net2 = torch.nn.Sequential(torch.nn.Linear(Xin.shape[1], 128), torch.nn.ReLU(), torch.nn.Linear(128, 1)).to(DEV)
        opt2 = torch.optim.Adam(net2.parameters(), lr=1e-3)
        with torch.enable_grad():
            for step in range(2500):
                ii = tr[torch.randint(0, ntr, (4096,), device=DEV)]
                loss2 = ((net2(Xn[ii]).squeeze(-1) - yz[ii])**2).mean()
                opt2.zero_grad(); loss2.backward(); opt2.step()
        with torch.no_grad():
            pr = torch.cat([net2(Xn[te][i:i+8192]).squeeze(-1) for i in range(0, te.shape[0], 8192)], 0)
        return 1 - float(((pr - yz[te])**2).mean()/(yz[te]**2).mean())
    # grammar basis from L0-1 mlp-input deviations
    capG = {0: [], 1: []}; hsG = []
    for Lg in (0, 1):
        def mkg(Lg):
            def h(mo, i_, o_): capG[Lg].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hsG.append(H[Lg].mlp.register_forward_hook(mkg(Lg)))
    idsG = []
    for i in range(0, 64, 8):
        idxg = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsG.append(idxg.reshape(-1)); fwd(idxg)
    for h in hsG: h.remove()
    tokg = torch.cat(idsG, 0)
    cng = torch.zeros(V, device=DEV); cng.index_add_(0, tokg, torch.ones_like(tokg, dtype=torch.float))
    devg = None
    for Lg in (0, 1):
        Xg = torch.cat(capG[Lg], 0); capG[Lg] = []
        xbg = torch.zeros(V, D, device=DEV); xbg.index_add_(0, tokg, Xg)
        dvg = Xg - (xbg/cng.clamp_min(1).unsqueeze(1))[tokg]
        devg = dvg if devg is None else devg + dvg; del Xg
    devg = devg/2; devg = devg - devg.mean(0)
    _, _, Vtg = torch.linalg.svd(devg, full_matrices=False); Ug = Vtg[:64].T.contiguous(); del devg
    massdims = Fm.abs().mean(0).topk(8).indices
    restricted = {
        'norm_only': Fm.norm(dim=1, keepdim=True),
        'content_64': Fm @ Uc,
        'grammar_64': Fm @ Ug,
        'massive_8': Fm[:, massdims],
    }
    r2_res = {}
    for nm, Xin in restricted.items():
        r2_res[nm] = round(mlp_probe(Xin), 3)
        print(f"restricted probe {nm}: R2 {r2_res[nm]}", flush=True)
    # snippets at extreme predicted alpha (full-model MLP predictions on held-out)
    import tiktoken as _tk
    _enc = _tk.get_encoding('gpt2')
    T2 = SEQ - 1
    with torch.no_grad():
        pr_full = torch.cat([net(Xz[i:i+8192]).squeeze(-1) for i in range(0, N, 8192)], 0)
    # position index mapping: the samples came from blocks[32:96] over 3 layers concatenated — snippet only within first layer block
    npos_layer = (96-32)*8//8  # careful: use first third
    first = pr_full[:64*T2*1] if pr_full.shape[0] >= 64*T2 else pr_full
    k2 = min(32, first.shape[0])
    hi = first.topk(k2).indices.tolist(); lo = (-first).topk(k2).indices.tolist()
    def snip(fi):
        s2, p2 = divmod(int(fi), T2); s2 += 32
        lo2 = max(0, p2-10)
        try: return _enc.decode(blocks[s2, lo2:p2+1].tolist()).replace('\n', ' ')
        except Exception: return '<err>'
    snips = {'high_alpha': [snip(i) for i in hi[:12]], 'low_alpha': [snip(i) for i in lo[:12]]}
    for nm2 in ('high_alpha', 'low_alpha'):
        print(nm2, snips[nm2][:6], flush=True)

    out = {'probe_r2_linear': round(r2_lin, 3), 'probe_r2_mlp': round(r2_mlp, 3),
           'restricted': r2_res, 'snippets': snips}
    best = max(r2_res, key=lambda k3: r2_res[k3])
    out['best_restricted'] = [best, r2_res[best]]
    out['pred_a_named'] = bool(r2_res[best] >= 0.6*r2_mlp)
    out['pred_b_norm_artifact'] = bool(r2_res['norm_only'] >= 0.5)
    out['pred_c_diffuse'] = bool(max(r2_res.values()) < 0.3)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"full {r2_mlp:.3f} | restricted {r2_res} | best {out['best_restricted']}", flush=True)
    print(f"pred_a named {out['pred_a_named']} | pred_b norm {out['pred_b_norm_artifact']} | pred_c diffuse {out['pred_c_diffuse']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
