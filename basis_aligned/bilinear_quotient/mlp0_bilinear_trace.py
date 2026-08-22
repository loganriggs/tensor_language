"""LAYER 0, MECHANISM: how does mlp0's BILINEAR form compute the grammatical-class output? (start
of the strict bottom-up program; weight-level, keep-only-free). mlp0 = Down[(Left·x)⊙(Right·x)] +
bias, hidden dim 4608. We know the OUTPUT encodes grammatical class (§825, named+causal). Now trace
the WEIGHTS: which hidden units write to the class output, what their two readouts (Left_k, Right_k)
detect, and whether the class computation is genuinely BILINEAR (a product/AND of two token readouts)
or effectively LINEAR (one factor ~constant).

Method (exact weights + captured real input X to mlp0):
 1. class output subspace U_class = top grammatical-class directions of mlp0's OUTPUT (token-conditional
    means, as §825).
 2. per hidden unit k, class-writing magnitude = ||U_class^T · Down[:,k]||; rank units.
 3. for the top class-writing units: activation a_k = (X·Left_k)(X·Right_k) on real tokens; which
    grammatical CLASS maximizes mean a_k (class-selective?); bilinearity = do BOTH factors vary across
    tokens (std of each factor) or is one ~constant; top tokens by a_k / by each factor.

REGISTERED PREDICTIONS:
  (0) SANITY: reconstructing out = Down[(Left·X)⊙(Right·X)]+bias matches the captured output (err<1e-3);
  (a) CLASS-SELECTIVE UNITS: the top class-writing units are grammatical-class-selective (one class
      dominates their activation), and their top tokens are that class's members;
  (b) BILINEAR vs LINEAR: report whether the class-writing units use BOTH readouts (both factors vary
      substantially → genuine product/AND) or are effectively linear (one factor low-variance);
  (c) name what layer-0's class computation does, in weight terms, to hand to layer 1."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp0_bilinear_trace_results.json'
NEVAL = 200; MINCOUNT = 8; NCLASS_DIR = 12; NUNIT = 24; NTOK = 10
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our', 'my', 'your', 'some', 'any', 'no', 'every', 'each'}
PREP = {'of', 'in', 'to', 'for', 'on', 'at', 'by', 'with', 'from', 'as', 'into', 'about', 'over', 'after', 'before', 'between', 'through', 'under', 'against'}
CONJ = {'and', 'or', 'but', 'nor', 'so', 'yet', 'because', 'although', 'while', 'if', 'than'}
PRON = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'him', 'her', 'them', 'us', 'me', 'who', 'which'}


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


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def capture(rows):
    mlp = m.transformer.h[0].mlp
    Xs = []; Os = []; toks = []
    def pre(mo, args): Xs.append(args[0].detach().float().reshape(-1, D))
    def post(mo, i_, o_): Os.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hp = mlp.register_forward_pre_hook(pre); ho = mlp.register_forward_hook(post)
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx)
        toks.append(idx.cpu().numpy().reshape(-1))
    hp.remove(); ho.remove()
    return torch.cat(Xs, 0), torch.cat(Os, 0), np.concatenate(toks)


def class_subspace(O, toks, r):
    g = O.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(toks):
        mk = toks == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL); d = dec()
    mlp = m.transformer.h[0].mlp
    Lw = mlp.Left.weight.detach().float(); Rw = mlp.Right.weight.detach().float()
    Dw = mlp.Down.weight.detach().float(); db = mlp.Down_bias.detach().float()   # (4608,1152),(4608,1152),(1152,4608),(1152)
    X, O, toks = capture(rows)
    clslab = np.array([CLASSES.index(classify(d(int(t)))) for t in toks])
    # sanity: reconstruct
    la = X @ Lw.T; ra = X @ Rw.T; hid = la * ra; recon = hid @ Dw.T + db
    err = float((recon - O).norm() / O.norm())
    U = class_subspace(O, toks, NCLASS_DIR)                      # (D, NCLASS_DIR)
    Wc = U.T @ Dw                                               # (NCLASS_DIR, 4608): each unit's write to class dirs
    unit_mag = Wc.norm(dim=0)                                   # (4608,)
    top = torch.topk(unit_mag, NUNIT).indices.tolist()
    # factor stats across tokens
    la_std = la.std(0); ra_std = ra.std(0)
    units = []
    nc = len(CLASSES); nsel = 0; nbilin = 0
    for k in top:
        a = (la[:, k] * ra[:, k]).cpu().numpy()                 # unit activation per token
        # class-selectivity: mean |a| per class
        cls_mean = np.array([np.abs(a[clslab == c]).mean() if (clslab == c).any() else 0 for c in range(nc)])
        topcls = int(cls_mean.argmax()); sel = float(cls_mean[topcls] / (cls_mean.mean() + 1e-9))
        # bilinearity: both factors must vary; measure the smaller factor's coefficient of variation
        lf = la[:, k]; rf = ra[:, k]
        cvL = float(lf.std() / (lf.abs().mean() + 1e-9)); cvR = float(rf.std() / (rf.abs().mean() + 1e-9))
        bilin = min(cvL, cvR) > 0.3
        if sel > 1.8: nsel += 1
        if bilin: nbilin += 1
        toptok = [repr(d(int(toks[j]))) for j in np.argsort(-np.abs(a))[:NTOK]]
        units.append({'unit': int(k), 'class_write_mag': round(float(unit_mag[k]), 3),
                      'top_class': CLASSES[topcls], 'class_selectivity': round(sel, 2),
                      'cvL': round(cvL, 2), 'cvR': round(cvR, 2), 'bilinear': bool(bilin),
                      'top_tokens': toptok})
    out = {'recon_err': round(err, 5), 'n_top_units': NUNIT,
           'n_class_selective': nsel, 'n_bilinear': nbilin,
           'frac_class_selective': round(nsel/NUNIT, 2), 'frac_bilinear': round(nbilin/NUNIT, 2),
           'units': units, 'runtime_s': round(time.time()-t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"recon err {err:.2e} | top-{NUNIT} class-writing units: {nsel} class-selective, {nbilin} bilinear", flush=True)
    for u in units[:12]:
        print(f"  unit {u['unit']}: writes class (mag {u['class_write_mag']}), fires for {u['top_class']} (sel {u['class_selectivity']}), bilinear={u['bilinear']} (cvL {u['cvL']}/cvR {u['cvR']}) | top {u['top_tokens'][:6]}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
