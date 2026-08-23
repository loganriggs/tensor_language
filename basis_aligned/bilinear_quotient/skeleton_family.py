"""Next question after the §1113-1118 skeleton arc: is the ~8-feature content API UNIVERSAL across the family?
§1061/§1063 showed the content INFO is shared (CCA 0.95-0.97, cross-width too). Sharper: do the individual
SKELETON ATOMS match across independently-trained models? For bilin18, swiglu18 (D=1152) and bilin12 (D=768):
capture pooled content deviations on the SAME text, train each model's own k=8/256 SAE on its own coords, then
match bilin18's top-16 usage atoms to each other model's atoms by ACTIVATION CORRELATION over shared positions
(dimension-free). Null: shuffled-position correlations.

REGISTERED PREDICTIONS:
  (0) SANITY: each model's SAE R² ~0.7 (the §1114 density is family-general); shuffled null corr ~0.
  (a) UNIVERSAL API: >= 6 of bilin18's top-8 skeleton atoms find a partner with activation corr >= 0.5 in BOTH
      other models -> the named features themselves (not just the information) are convergent across
      architecture and width — the strongest universality statement of the program;
  (b) SHARED INFO, PRIVATE BASIS: if CCA-level sharing holds but atom matches stay < 0.5, each model coordinates
      the shared content in its own feature basis (report plainly — then the API is model-local and only the
      manifold is universal)."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
sys.path.insert(0, '/workspace/rspd')
from tier2_model import load_elriggs
import census_lib as cl

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'skeleton_family_results.json'
DEV = 'cuda'; NSEQ = 192; SEQ = 256; K = 64; NATOM = 256; TOPK = 8; STEPS = 3000
MODELS = {'swiglu18': [8, 10, 12], 'bilin12': [5, 6, 7]}
CAP = {}


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


def train_code(Cc, seed=0):
    sae = TopKSAE(Cc.shape[1], NATOM, TOPK, seed).to(DEV)
    opt = torch.optim.Adam(sae.parameters(), lr=3e-3)
    with torch.enable_grad():
        for step in range(STEPS):
            idx = torch.randint(0, Cc.shape[0], (4096,), device=DEV)
            x = Cc[idx]
            xh, _ = sae(x)
            loss = ((xh - x)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        xh, code_s = sae(Cc[:20000])
        r2 = 1 - float(((xh - Cc[:20000])**2).sum()/(Cc[:20000]**2).sum())
        _, code = sae(Cc)
    return code, r2


@torch.no_grad()
def coords_family(name, ref_layers, blocks):
    mdl, cfg = load_elriggs(name, device=DEV, dtype=torch.float32); mdl.eval()
    D = mdl.transformer.wte.weight.shape[1]; V = mdl.transformer.wte.weight.shape[0]
    for L in ref_layers: CAP[(name, L)] = []
    hs = []
    for L in ref_layers:
        mlp = mdl.transformer.h[L].mlp
        def mk(key):
            def h(mo, i_, o_): CAP[key].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(mlp.register_forward_hook(mk((name, L))))
    idsL = []
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1))
        mdl(idx, idx)
    for h in hs: h.remove()
    flat = torch.cat(idsL, 0)
    devsum = None
    for L in ref_layers:
        X = torch.cat(CAP[(name, L)], 0); CAP[(name, L)] = []
        xbar = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
        xbar.index_add_(0, flat, X); cnts.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
        dv = X - (xbar/cnts.clamp_min(1).unsqueeze(1))[flat]
        devsum = dv if devsum is None else devsum + dv; del X
    dev = devsum/len(ref_layers); dev = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(dev, full_matrices=False)
    Cc = (dev @ Vt[:K].T.contiguous()).contiguous()
    del mdl, dev, devsum
    torch.cuda.empty_cache()
    return Cc


@torch.no_grad()
def coords_bilin18(blocks):
    from bilin18_joint_removal import m
    D = 1152; REF = [8, 10, 12]
    H = m.transformer.h
    cap = {L: [] for L in REF}; hs = []
    for L in REF:
        def mk(L):
            def h(mo, i_, o_): cap[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(H[L].mlp.register_forward_hook(mk(L)))
    idsL = []
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1))
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in H: x, v1 = blk(x, v1, x0)
    for h in hs: h.remove()
    flat = torch.cat(idsL, 0); V = int(m.lm_head.weight.shape[0]); devsum = None
    for L in REF:
        X = torch.cat(cap[L], 0); cap[L] = []
        xbar = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
        xbar.index_add_(0, flat, X); cnts.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
        dv = X - (xbar/cnts.clamp_min(1).unsqueeze(1))[flat]
        devsum = dv if devsum is None else devsum + dv; del X
    dev = devsum/len(REF); dev = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(dev, full_matrices=False)
    return (dev @ Vt[:K].T.contiguous()).contiguous()


def match_atoms(codeA, codeB, top_atoms, g):
    """for each atom in top_atoms of A: best activation corr with any B atom + shuffled null"""
    res = {}
    zB = (codeB - codeB.mean(0)) / codeB.std(0).clamp_min(1e-6)
    perm = torch.randperm(codeA.shape[0], generator=g, device=codeA.device)
    for a in top_atoms:
        va = codeA[:, a]
        za = (va - va.mean()) / va.std().clamp_min(1e-6)
        corr = (zB * za.unsqueeze(1)).mean(0)
        best = float(corr.abs().max()); best_j = int(corr.abs().argmax())
        zs = za[perm]
        corr_s = float((zB * zs.unsqueeze(1)).mean(0).abs().max())
        res[int(a)] = {'best_corr': round(best, 3), 'partner': best_j, 'shuffled_null': round(corr_s, 3)}
    return res


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    C18 = coords_bilin18(blocks)
    code18, r18 = train_code(C18)
    usage = (code18 != 0).float().mean(0)
    top8 = usage.argsort(descending=True)[:8].tolist()
    print(f"bilin18 SAE R2 {r18:.4f} | top atoms {top8}", flush=True)
    g = torch.Generator(device=DEV).manual_seed(0)
    out = {'bilin18_r2': round(r18, 4), 'models': {}}
    for name, ref in MODELS.items():
        Cf = coords_family(name, ref, blocks)
        codeF, rF = train_code(Cf)
        mt = match_atoms(code18, codeF, top8, g)
        n_match = sum(1 for a in top8 if mt[int(a)]['best_corr'] >= 0.5)
        out['models'][name] = {'r2': round(rF, 4), 'matches': mt, 'n_matched_ge_05': n_match}
        print(f"{name}: R2 {rF:.4f} | matches >=0.5: {n_match}/8 | detail {mt}", flush=True)
        del Cf, codeF
        torch.cuda.empty_cache()
    ns = [out['models'][nm]['n_matched_ge_05'] for nm in MODELS]
    out['pred_a_universal_api'] = bool(min(ns) >= 6)
    out['pred_b_private_basis'] = bool(max(ns) <= 3)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a universal-api {out['pred_a_universal_api']} | pred_b private-basis {out['pred_b_private_basis']} | wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
