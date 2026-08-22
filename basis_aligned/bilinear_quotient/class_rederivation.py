"""WHY is class only LATE-addressable (§961)? Because the middle RE-DERIVES class from the token, washing out
early edits. Demonstrate directly: mean-ablate the class subspace at an EARLY layer (L2), then decode the
next-token CLASS from a LATER layer (L15). If L15 class-decode is largely RECOVERED despite the L2 ablation, the
class was RE-DERIVED between L2 and L15 (not carried) — the mechanism behind §961's early-patch washout and §916's
"class is context-derived, not maintained".

REGISTERED PREDICTIONS:
  (0) SANITY: clean L15 next-class decode is high (>> base rate); ablating the class subspace AT L15 (measured at
      L15) drops it sharply (the subspace does carry class there).
  (a) CLASS IS RE-DERIVED: ablating the class subspace at L2 leaves L15 next-class decode LARGELY INTACT (recovers
      most of the clean L15 accuracy) -> class removed early is rebuilt by the middle layers; contrast a
      within-layer L15 ablation which does drop it -> demonstrates re-derivation, explaining §961/§916;
  (b) report L15 next-class decode: clean, after-L2-ablation, after-L15-ablation, base rate."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'class_rederivation_results.json'
NEVAL = 200; SEQ = 256; RCLASS = 8; RIDGE = 1e2; L_EARLY = 2; L_LATE = 15
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','me','who','which'}
ABL = {'L': -1, 'U': None}


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


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def abl_hook(L):
    def h(mo, i_, o_):
        if ABL['L'] != L: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; sh = y.shape; U = ABL['U']; v = y.reshape(-1, D)
        v2 = v - (v @ U) @ U.T
        return (v2.reshape(sh),) + tuple(o_[1:]) if isinstance(o_, tuple) else v2.reshape(sh)
    return h


def forward_capL(idx, capL):
    cap = {}
    def ch(mo, i_, o_): cap['r'] = (o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D)
    hh = m.transformer.h[capL].register_forward_hook(ch)
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    hh.remove(); return cap['r']


def mean_subspace(X, labels, r):
    g = X.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        if t < 0: continue
        mk = labels == t
        if mk.sum() < 5: continue
        rows.append(X[mk].mean(0)-g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0)*torch.tensor(wt, device=X.device, dtype=X.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:min(r, M.shape[0])].T.contiguous(), g


def acc(F_, y, ncls, tr, te):
    Y = torch.zeros(len(tr), ncls, device=DEV); Y[torch.arange(len(tr)), torch.tensor(y[tr], device=DEV)] = 1.0
    A = F_[tr].T @ F_[tr] + RIDGE*torch.eye(F_.shape[1], device=DEV); W = torch.linalg.solve(A, F_[tr].T @ Y)
    return float((F_[te] @ W).argmax(1).cpu().numpy().__eq__(y[te]).mean())


@torch.no_grad()
def cap_L15(idx):  # capture L15 with current ABL settings
    return forward_capL(idx, L_LATE)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    nxtc = np.full_like(S[:, :-1], -1); nxtc[:, :-1] = S[:, 1:-1]
    nxtcls = np.array([CLASSES.index(classify(d(int(t)))) if t >= 0 else -1 for t in nxtc.reshape(-1)])
    mask = nxtcls >= 0; y = np.where(mask, nxtcls, 0)
    hooks = [m.transformer.h[L_EARLY].register_forward_hook(abl_hook(L_EARLY)),
             m.transformer.h[L_LATE].register_forward_hook(abl_hook(L_LATE))]
    # build class subspaces at L2 and L15 from clean run
    ABL['L'] = -1
    R2 = []; R15 = []
    for i in range(0, nb, 4):
        idx = blocks[i:i+4].to(DEV)[:, :-1].contiguous()
        R2.append(forward_capL(idx, L_EARLY)); R15.append(forward_capL(idx, L_LATE))
    R2 = torch.cat(R2, 0); R15 = torch.cat(R15, 0)
    U2, _ = mean_subspace(R2, nxtcls, RCLASS); U15, _ = mean_subspace(R15, nxtcls, RCLASS)
    n = R15.shape[0]; rng = np.random.RandomState(0); perm = rng.permutation(n); ntr = int(0.7*n); tr, te = perm[:ntr], perm[ntr:]
    base = float(np.bincount(y[mask], minlength=RCLASS).max()/mask.sum())
    # probe fit on clean L15
    clean = acc(R15, y, RCLASS, tr, te)
    # (b) ablate class subspace at L2, decode next-class at L15
    ABL['L'] = L_EARLY; ABL['U'] = U2; R15_a2 = []
    for i in range(0, nb, 4): R15_a2.append(cap_L15(blocks[i:i+4].to(DEV)[:, :-1].contiguous()))
    R15_a2 = torch.cat(R15_a2, 0); after_L2 = acc(R15_a2, y, RCLASS, tr, te)
    # (c) ablate class subspace at L15 itself (within-layer), decode at L15
    ABL['L'] = L_LATE; ABL['U'] = U15; R15_a15 = []
    for i in range(0, nb, 4): R15_a15.append(cap_L15(blocks[i:i+4].to(DEV)[:, :-1].contiguous()))
    R15_a15 = torch.cat(R15_a15, 0); after_L15 = acc(R15_a15, y, RCLASS, tr, te)
    ABL['L'] = -1
    for h in hooks: h.remove()
    out = {'base_rate': round(base, 4), 'clean_L15_nextclass': round(clean, 4),
           'after_L2_ablation': round(after_L2, 4), 'after_L15_ablation': round(after_L15, 4)}
    out['recovered_frac_after_L2'] = round((after_L2 - base)/(clean - base + 1e-9), 3)
    out['pred_a_rederived'] = bool(after_L2 > base + 0.6*(clean - base) and after_L15 < after_L2 - 0.05)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"L15 next-class decode: clean {clean:.3f} | after-L2-ablation {after_L2:.3f} | after-L15-ablation {after_L15:.3f} | base {base:.3f}", flush=True)
    print(f"recovered fraction after L2 ablation: {out['recovered_frac_after_L2']}", flush=True)
    print(f"(a) class RE-DERIVED between L2 and L15 (L2-ablation recovers, L15-ablation drops): {out['pred_a_rederived']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
