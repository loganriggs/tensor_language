"""NEW THREAD (name the content variables — mechanism-over-metrics mandate): §1055/§1064 named the content
manifold's top PCA AXES (topic/register contrasts, genuinely semantic). Axes are global; FEATURES are sparse.
Train an overcomplete top-k sparse dictionary ON THE CONTENT COORDINATES (pooled L8-12 deviation, U_c top-64
coords; 256 atoms, k=8) and test the three things that killed or validated dictionaries before:
  (1) STABILITY (the §763 killer for weight-atoms, recur 0.40): 3 seeds, greedy atom matching by cosine;
  (2) NAMEABILITY: per top atom, do its top-activating contexts cohere (share distinctive tokens) vs a
      shuffled-position null? (automated proxy; snippets dumped for human reading);
  (3) CAUSAL INDIVIDUATION: removing atom i's contribution at its own active positions should hurt those
      positions >= 2x removing a DIFFERENT atom there and >= 2x a norm-matched random direction in the span.

REGISTERED PREDICTIONS:
  (0) SANITY: reconstruction R^2 > 0.9 at k=8 (64-dim coords, 4x overcomplete); atom usage spread (top atom
      < 20% of total usage).
  (a) STABLE + NAMEABLE: cross-seed mean best-match cosine >= 0.6 (beats weight-atoms' 0.40) AND context
      coherence > 2x shuffled null for >= 10 of the top-16 atoms -> the content manifold supports a stable,
      nameable sparse feature basis finer than its PCA axes;
  (b) CAUSALLY INDIVIDUATED: own-atom removal >= 2x other-atom and >= 2x matched-random on own positions for
      >= 6/8 tested atoms;
  (c) if cross-seed cosine < 0.5, the §763 instability REPLICATES on activations — the manifold's honest
      units remain the PCA axes (report plainly)."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_dictionary_results.json'
NSEQ = 192; SEQ = 256; REF = [8, 10, 12]; K = 64
NATOM = 256; TOPK = 8; STEPS = 3000
H = m.transformer.h
enc = tiktoken.get_encoding('gpt2')
ABL = {'on': False, 'dvec': None, 'mask': None}
CUR = {}


def fwd(idx):
    CUR['idx'] = idx
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def abl_hook(L):
    def h(mo, i_, o_):
        if not ABL['on']: return None
        x = (i_[0] if isinstance(i_, tuple) else i_)
        msk = ABL['mask']                                # B,T bool for this minibatch
        if msk is None or not msk.any(): return None
        v = ABL['dvec'].to(x.dtype)                      # D unit vector
        xm = x.clone()
        seg = xm[msk]
        xm[msk] = seg - (seg @ v).unsqueeze(-1) * v * ABL['scale']
        y = mo.Down(mo.Left(xm)*mo.Right(xm)) + mo.Down_bias
        return y.to(o_.dtype)
    return h


class TopKSAE(torch.nn.Module):
    def __init__(self, d, n, k, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.E = torch.nn.Parameter(torch.randn(n, d, generator=g)*0.1)
        self.Dm = torch.nn.Parameter(torch.randn(d, n, generator=g)*0.1)
        self.k = k
    def forward(self, x):
        a = x @ self.E.T
        top = a.topk(self.k, -1)
        code = torch.zeros_like(a).scatter_(-1, top.indices, top.values)
        return code @ self.Dm.T, code


def train_sae(Cc, seed):
    sae = TopKSAE(K, NATOM, TOPK, seed).to(DEV)
    opt = torch.optim.Adam(sae.parameters(), lr=3e-3)
    N = Cc.shape[0]
    with torch.enable_grad():
        for step in range(STEPS):
            idx = torch.randint(0, N, (4096,), device=DEV)
            x = Cc[idx]
            xh, code = sae(x)
            loss = ((xh - x)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        xh, code = sae(Cc[:20000])
        r2 = 1 - float(((xh - Cc[:20000])**2).sum()/ (Cc[:20000]**2).sum())
    return sae, r2


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0])

    # capture content coords
    cap = {L: [] for L in REF}; hs = []
    for L in REF:
        def mk(L):
            def h(mo, i_, o_): cap[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(H[L].mlp.register_forward_hook(mk(L)))
    idsL = []
    for i in range(0, NSEQ, 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fwd(idx)
    for h in hs: h.remove()
    tok = torch.cat(idsL, 0); devsum = None
    cn = torch.zeros(V, device=DEV); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
    for L in REF:
        X = torch.cat(cap[L], 0); cap[L] = []
        xb = torch.zeros(V, D, device=DEV); xb.index_add_(0, tok, X)
        dv = X - (xb/cn.clamp_min(1).unsqueeze(1))[tok]
        devsum = dv if devsum is None else devsum + dv; del X
    dev = devsum/len(REF); dev = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(dev, full_matrices=False)
    Uc = Vt[:K].T.contiguous()
    Cc = (dev @ Uc).contiguous(); del dev, devsum

    # (1) train 3 seeds; stability
    saes = []; r2s = []
    for seed in range(3):
        sae, r2 = train_sae(Cc, seed); saes.append(sae); r2s.append(round(r2, 4))
        print(f"seed {seed}: recon R2 {r2:.4f}", flush=True)
    def match_cos(a, b):
        Da = F.normalize(a.Dm.T, dim=-1); Db = F.normalize(b.Dm.T, dim=-1)
        sim = (Da @ Db.T).abs()
        return float(sim.max(1).values.mean())
    stab = [match_cos(saes[0], saes[1]), match_cos(saes[0], saes[2]), match_cos(saes[1], saes[2])]
    stab_mean = sum(stab)/3
    print(f"cross-seed atom match cosine: {[round(s,3) for s in stab]} (mean {stab_mean:.3f}; weight-atoms were 0.40 §763)", flush=True)

    sae = saes[0]
    _, code = sae(Cc)
    usage = (code != 0).float().mean(0)
    top_usage_frac = float(usage.max()/usage.sum())
    top_atoms = usage.argsort(descending=True)[:16].tolist()

    # (2) nameability: top-24 activating positions per atom; coherence of trailing tokens
    T = SEQ - 1
    def context_tokens(flat_pos, w=12):
        b, t = flat_pos // T, flat_pos % T
        lo = max(0, t - w)
        return set(blocks[b, lo:t+1].tolist())
    g = torch.Generator().manual_seed(0)
    coher = {}; snippets = {}
    for a in top_atoms:
        acts = code[:, a]
        top_pos = acts.argsort(descending=True)[:24].cpu()
        ctxs = [context_tokens(int(p)) for p in top_pos]
        # distinctive tokens: appear in >= 1/3 of contexts, excluding globally common (freq > N/50)
        from collections import Counter
        cnt = Counter(t2 for c in ctxs for t2 in c)
        common = {t2 for t2, c2 in cnt.items() if c2 >= 8 and cn[t2] < tok.shape[0]/50}
        share = sum(1 for c in ctxs if c & common)/len(ctxs) if common else 0.0
        # null: random positions
        rnd = torch.randint(0, code.shape[0], (24,), generator=g)
        ctxr = [context_tokens(int(p)) for p in rnd]
        cntr = Counter(t2 for c in ctxr for t2 in c)
        commonr = {t2 for t2, c2 in cntr.items() if c2 >= 8 and cn[t2] < tok.shape[0]/50}
        share_null = sum(1 for c in ctxr if c & commonr)/len(ctxr) if commonr else 0.0
        coher[a] = (round(share, 3), round(share_null, 3))
        snippets[a] = [enc.decode(blocks[int(p)//T, max(0, int(p) % T - 10):int(p) % T + 1].tolist()) for p in top_pos[:5]]
    n_coherent = sum(1 for a in top_atoms if coher[a][0] > 2*max(coher[a][1], 0.02))
    print(f"nameability: coherent atoms {n_coherent}/16 | example coher {dict(list(coher.items())[:4])}", flush=True)

    # (3) causal individuation on top-8 atoms
    hks = [H[L].mlp.register_forward_hook(abl_hook(L)) for L in REF]
    ABL['scale'] = 1.0
    def ce_on_mask(mask_flat, dvec):
        """CE on masked positions with dvec removed at masked positions (or baseline if dvec None)"""
        ABL['dvec'] = dvec
        tot = 0.0; n = 0; ptr = 0
        for i in range(0, NSEQ, 8):
            bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].reshape(-1)
            nb = idx.numel()
            mb = mask_flat[ptr:ptr+nb].view(idx.shape).to(DEV); ptr += nb
            ABL['on'] = dvec is not None; ABL['mask'] = mb
            lp = F.log_softmax(fwd(idx).float(), -1)
            ABL['on'] = False
            ce_tok = -lp.reshape(-1, lp.shape[-1])[torch.arange(tgt.shape[0], device=DEV), tgt]
            mflat = mb.reshape(-1)
            tot += float(ce_tok[mflat].sum()); n += int(mflat.sum())
        return tot/max(n, 1)
    causal = {}
    g2 = torch.Generator(device=DEV).manual_seed(1)
    ok_count = 0
    for j, a in enumerate(top_atoms[:8]):
        acts = code[:, a]
        thr = acts[acts > 0].quantile(0.75) if int((acts > 0).sum()) > 100 else 0.0
        mask = (acts > max(float(thr), 1e-6)).cpu()
        dvec_own = F.normalize(Uc @ sae.Dm[:, a].detach(), dim=0)
        other = top_atoms[(j+1) % 8]
        dvec_oth = F.normalize(Uc @ sae.Dm[:, other].detach(), dim=0)
        dvec_rnd = F.normalize(Uc @ torch.randn(K, generator=g2, device=DEV), dim=0)
        base = ce_on_mask(mask, None)
        own = ce_on_mask(mask, dvec_own) - base
        oth = ce_on_mask(mask, dvec_oth) - base
        rnd = ce_on_mask(mask, dvec_rnd) - base
        ok = own >= 2*max(oth, 1e-4) and own >= 2*max(rnd, 1e-4)
        ok_count += int(ok)
        causal[a] = {'n_pos': int(mask.sum()), 'own': round(own, 4), 'other': round(oth, 4), 'random': round(rnd, 4), 'ok': bool(ok)}
        print(f"atom {a}: own +{own:.4f} | other +{oth:.4f} | random +{rnd:.4f} | individuated {ok}", flush=True)
    for h in hks: h.remove()

    out = {'recon_r2': r2s, 'stability_cos': [round(s, 3) for s in stab], 'stability_mean': round(stab_mean, 3),
           'top_usage_frac': round(top_usage_frac, 3), 'coherence': {str(a): coher[a] for a in top_atoms},
           'n_coherent': n_coherent, 'causal': {str(a): causal[a] for a in causal}, 'n_individuated': ok_count,
           'snippets': {str(a): snippets[a] for a in top_atoms[:8]}}
    out['pred_a_stable_nameable'] = bool(stab_mean >= 0.6 and n_coherent >= 10)
    out['pred_b_individuated'] = bool(ok_count >= 6)
    out['pred_c_instability_replicates'] = bool(stab_mean < 0.5)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"stability {stab_mean:.3f} | coherent {n_coherent}/16 | individuated {ok_count}/8", flush=True)
    print(f"pred_a stable+nameable {out['pred_a_stable_nameable']} | pred_b individuated {out['pred_b_individuated']} | pred_c unstable {out['pred_c_instability_replicates']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
