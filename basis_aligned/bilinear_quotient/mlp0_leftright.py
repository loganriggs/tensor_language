"""LAYER 0, deeper: is mlp0's class-detector bilinearity SHARPENING or CONJUNCTION? (§841 follow-up).
§841: mlp0's class-writing hidden units are class-selective AND genuinely bilinear (product of two
readouts (Left·x)(Right·x)). Now ask what the product DOES: do Left and Right read the SAME grammatical
class (self-product → SHARPENING, a soft square that sharpens class detection, cf §782) or DIFFERENT
things (Left=class, Right=gate → CONJUNCTION)? And does the product sharpen selectivity beyond either
linear readout alone?

Method (weights + real input X): for the top class-writing units, characterize the Left readout
(la=X·Left_k) and Right readout (ra=X·Right_k) SEPARATELY — which grammatical class maximizes each —
and compare to the product. sharpening = Left_class==Right_class; sharpening-factor = selectivity(la*ra)
/ max(selectivity(la), selectivity(ra)).

REGISTERED PREDICTIONS:
  (0) SANITY: reproduces §841 (units class-selective);
  (a) SHARPENING: for most class-writing units Left_class == Right_class (same class) and the product's
      class-selectivity exceeds either readout alone (sharpening-factor > 1) -> mlp0 sharpens class by
      a soft self-product;
  (b) CONJUNCTION: if Left_class != Right_class for many units, the unit ANDs two different features;
  report per-unit Left_class/Right_class/product_class + sharpening factor, and the split."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp0_leftright_results.json'
NEVAL = 200; MINCOUNT = 8; NCLASS_DIR = 12; NUNIT = 24
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
    mlp = m.transformer.h[0].mlp; Xs = []; Os = []; toks = []
    def pre(mo, args): Xs.append(args[0].detach().float().reshape(-1, D))
    def post(mo, i_, o_): Os.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hp = mlp.register_forward_pre_hook(pre); ho = mlp.register_forward_hook(post)
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx); toks.append(idx.cpu().numpy().reshape(-1))
    hp.remove(); ho.remove(); return torch.cat(Xs, 0), torch.cat(Os, 0), np.concatenate(toks)


def class_subspace(O, toks, r):
    g = O.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(toks):
        mk = toks == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


def selectivity(act, clslab, nc):
    cm = np.array([np.abs(act[clslab == c]).mean() if (clslab == c).any() else 0 for c in range(nc)])
    return int(cm.argmax()), float(cm.max()/(cm.mean()+1e-9))


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL); d = dec()
    mlp = m.transformer.h[0].mlp
    Lw = mlp.Left.weight.detach().float(); Rw = mlp.Right.weight.detach().float(); Dw = mlp.Down.weight.detach().float()
    X, O, toks = capture(rows)
    clslab = np.array([CLASSES.index(classify(d(int(t)))) for t in toks]); nc = len(CLASSES)
    U = class_subspace(O, toks, NCLASS_DIR); unit_mag = (U.T @ Dw).norm(dim=0)
    top = torch.topk(unit_mag, NUNIT).indices.tolist()
    la_all = X @ Lw.T; ra_all = X @ Rw.T
    nsharp = 0; nsame = 0; sfacs = []; units = []
    for k in top:
        la = la_all[:, k].cpu().numpy(); ra = ra_all[:, k].cpu().numpy(); prod = la*ra
        lc, ls = selectivity(la, clslab, nc); rc, rs = selectivity(ra, clslab, nc); pc, ps = selectivity(prod, clslab, nc)
        same = (lc == rc); sfac = ps/(max(ls, rs)+1e-9)
        if same: nsame += 1
        if sfac > 1.0: nsharp += 1
        sfacs.append(sfac)
        units.append({'unit': int(k), 'left_class': CLASSES[lc], 'right_class': CLASSES[rc], 'product_class': CLASSES[pc],
                      'left_sel': round(ls, 2), 'right_sel': round(rs, 2), 'product_sel': round(ps, 2),
                      'same_class': bool(same), 'sharpening_factor': round(sfac, 2)})
    out = {'n_units': NUNIT, 'n_same_class_LR': nsame, 'frac_same_class': round(nsame/NUNIT, 2),
           'n_sharpened': nsharp, 'mean_sharpening_factor': round(float(np.mean(sfacs)), 2),
           'verdict': 'sharpening (self-product)' if nsame/NUNIT >= 0.6 else 'conjunction (mixed)',
           'units': units, 'runtime_s': round(time.time()-t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"same-class Left/Right: {nsame}/{NUNIT} ({out['frac_same_class']}) | mean sharpening factor {out['mean_sharpening_factor']} | {out['verdict']}", flush=True)
    for u in units[:12]:
        print(f"  u{u['unit']}: L={u['left_class']}({u['left_sel']}) R={u['right_class']}({u['right_sel']}) -> prod={u['product_class']}({u['product_sel']}) same={u['same_class']} sharp x{u['sharpening_factor']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
