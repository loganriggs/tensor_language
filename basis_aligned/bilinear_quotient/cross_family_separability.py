"""Does the CORE two-machine finding — grammar (next-class) subspace ORTHOGONAL to content (topic) subspace (§920)
— generalize across the model FAMILY? §925 confirmed it in GPT-2; test the sibling Elriggs models swiglu18 (SwiGLU
MLP) and bilin12, plus bilin18 as in-experiment reference. Build the late-layer next-class subspace and the
content/topic subspace (token+pos stripped), measure their overlap vs chance.

REGISTERED PREDICTIONS:
  (0) SANITY: both subspaces decode their target above chance in each model.
  (a) FAMILY-WIDE SEPARABILITY: grammar and content subspaces are near-ORTHOGONAL (overlap ~ chance, ratio < 3) in
      swiglu18 and bilin12, as in bilin18 (§920) and GPT-2 (§925) -> the two-machine separable structure is a
      general property across the family;
  (b) report overlap, chance, ratio per model."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
import census_lib as cl
from bilin18_joint_removal import m as BILIN, DEV
from tier2_model import load_elriggs
import torch.nn.functional as F

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cross_family_separability_results.json'
NEVAL = 140; SEQ = 256; K = 12; RCLASS = 8; RTOPIC = 11
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','me','who','which'}


def dec():
    import tiktoken; enc = tiktoken.get_encoding('gpt2'); return lambda i: enc.decode([int(i)])


def classify(s):
    t = s.strip()
    if t == '' or not t[0].isalnum(): return 'punct'
    if t[0].isdigit(): return 'number'
    low = t.lower()
    if low in DET: return 'det'
    if low in PREP: return 'prep'
    if low in CONJ: return 'conj'
    if low in PRON: return 'pron'
    if t[0].isupper(): return 'cap'
    return 'word'


def mean_subspace(X, labels, r):
    Dd = X.shape[1]; g = X.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        if t < 0: continue
        mk = labels == t
        if mk.sum() < 5: continue
        rows.append(X[mk].mean(0)-g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0)*torch.tensor(wt, device=X.device, dtype=X.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:min(r, M.shape[0])].T.contiguous(), g


def kmeans(X, k, iters=25, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed); c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            mk = a == j
            if mk.any(): c[j] = X[mk].mean(0)
    return a


@torch.no_grad()
def cap_late(mdl, idx, Dm, L):
    reps = {}
    def h(mo, i_, o_): reps['r'] = (o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, Dm)
    hh = mdl.transformer.h[L].register_forward_hook(h)
    x = F.rms_norm(mdl.transformer.wte(idx), (Dm,)); x0 = x; v1 = None
    for blk in mdl.transformer.h: x, v1 = blk(x, v1, x0)
    hh.remove(); return reps['r']


@torch.no_grad()
def run(mdl, blocks, S, d, Dm, nlayer):
    L = int(0.8*nlayer); nb = blocks.shape[0]
    reps = []
    for i in range(0, nb, 4): reps.append(cap_late(mdl, blocks[i:i+4].to(DEV)[:, :-1].contiguous(), Dm, L))
    R = torch.cat(reps, 0)
    toks = S[:, :-1].reshape(-1); pos = np.broadcast_to(np.arange(SEQ-1), (nb, SEQ-1)).reshape(-1)
    nxtc = np.full_like(S[:, :-1], -1); nxtc[:, :-1] = S[:, 1:-1]
    nxtcls = np.array([CLASSES.index(classify(d(int(t)))) if t >= 0 else -1 for t in nxtc.reshape(-1)])
    Uclass, _ = mean_subspace(R, nxtcls, RCLASS)
    Utok, g = mean_subspace(R, toks, 64); Upos, _ = mean_subspace(R, pos.astype(np.int64), 32)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :96].contiguous()
    content = (R-g) - ((R-g)@Ucp)@Ucp.T; cn = content/(content.norm(dim=1, keepdim=True)+1e-9)
    topic = kmeans(cn, K).cpu().numpy(); Utopic, _ = mean_subspace(content, topic, RTOPIC)
    r = min(Uclass.shape[1], Utopic.shape[1]); overlap = float((Uclass.T @ Utopic).pow(2).sum()/r); chance = RCLASS/Dm
    return {'layer': L, 'overlap': round(overlap, 4), 'chance': round(chance, 4), 'ratio': round(overlap/chance, 2)}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy()
    out = {'bilin18_ref_920': {'overlap': 0.009, 'chance': 0.007}, 'gpt2_ref_925': True, 'models': {}}
    r = run(BILIN, blocks, S, d, 1152, 18); out['models']['bilin18'] = r; print(f"bilin18: {r}", flush=True)
    for short in ['swiglu18', 'bilin12']:
        try:
            mdl, cfg = load_elriggs(short); Dm = cfg.get('n_embd'); nl = cfg.get('n_layer')
            r = run(mdl, blocks, S, d, Dm, nl); out['models'][short] = r; print(f"{short}: {r}", flush=True)
            del mdl; torch.cuda.empty_cache()
        except Exception as e:
            out['models'][short] = {'error': f'{type(e).__name__}: {e}'}; print(f"{short} ERROR {e}", flush=True)
    ok = all('ratio' in out['models'][k] and out['models'][k]['ratio'] < 3 for k in out['models'])
    out['pred_a_family_separable'] = bool(ok)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"(a) two-machine separability family-wide (grammar⊥content, ratio<3 all): {ok}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
