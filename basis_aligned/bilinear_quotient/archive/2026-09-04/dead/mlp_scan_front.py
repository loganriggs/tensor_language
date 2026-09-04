"""LAYER 1, MECHANISM: what does mlp1 compute — recompute/sharpen the current-token class, or a NEW feature using the previous token (now available from attn0)? Adapts the mlp0 trace to layer 1 and adds a CURRENT-class vs PREVIOUS-class selectivity test per class-writing unit. If units are selective for the PREVIOUS token's class (not just current), mlp1 computes context (prev-class); if current-class, it recomputes/sharpens. LAYER 0 orig doc:
LAYER 0, MECHANISM: how does mlp0's BILINEAR form compute the grammatical-class output? (start
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
OUT = PT + 'mlp_scan_front_results.json'
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
def capture(rows, LAYER):
    mlp = m.transformer.h[LAYER].mlp
    Xs = []; Os = []; seqs = []
    def pre(mo, args): Xs.append(args[0].detach().float().reshape(-1, D))
    def post(mo, i_, o_): Os.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hp = mlp.register_forward_pre_hook(pre); ho = mlp.register_forward_hook(post)
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx)
        seqs.append(idx.cpu().numpy())
    hp.remove(); ho.remove()
    return torch.cat(Xs, 0), torch.cat(Os, 0), np.concatenate(seqs, 0)


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
    out = {'layers': {}}
    for LAYER in [2,3,4,5]:
        mlp = m.transformer.h[LAYER].mlp
        Lw = mlp.Left.weight.detach().float(); Rw = mlp.Right.weight.detach().float(); Dw = mlp.Down.weight.detach().float()
        X, O, seqs = capture(rows, LAYER)
        toks = seqs.reshape(-1)
        prev = np.full_like(seqs, -1); prev[:,1:] = seqs[:,:-1]; prev = prev.reshape(-1)
        clslab = np.array([CLASSES.index(classify(d(int(t)))) for t in toks])
        prevcls = np.array([CLASSES.index(classify(d(int(t)))) if t>=0 else -1 for t in prev])
        nc = len(CLASSES)
        U = class_subspace(O, toks, NCLASS_DIR); unit_mag = (U.T @ Dw).norm(dim=0)
        top = torch.topk(unit_mag, NUNIT).indices.tolist()
        la_all = X @ Lw.T; ra_all = X @ Rw.T
        nsel=0; nbil=0; nprev=0; topclasses={}
        for k in top:
            a=(la_all[:,k]*ra_all[:,k]).cpu().numpy()
            cm=np.array([np.abs(a[clslab==c]).mean() if (clslab==c).any() else 0 for c in range(nc)])
            tc=int(cm.argmax()); sel=float(cm[tc]/(cm.mean()+1e-9))
            pm=np.array([np.abs(a[prevcls==c]).mean() if (prevcls==c).any() else 0 for c in range(nc)])
            ptc=int(pm.argmax()); psel=float(pm[ptc]/(pm.mean()+1e-9))
            cvL=float(la_all[:,k].std()/(la_all[:,k].abs().mean()+1e-9)); cvR=float(ra_all[:,k].std()/(ra_all[:,k].abs().mean()+1e-9))
            if sel>1.8: nsel+=1
            if min(cvL,cvR)>0.3: nbil+=1
            if psel>1.4*sel: nprev+=1
            topclasses[CLASSES[tc]]=topclasses.get(CLASSES[tc],0)+1
        out['layers'][f'mlp{LAYER}']={'n_class_selective':nsel,'n_bilinear':nbil,'n_prev_driven':nprev,'class_histogram':topclasses}
        print(f"mlp{LAYER}: class-selective {nsel}/{NUNIT}, bilinear {nbil}/{NUNIT}, prev-driven {nprev}/{NUNIT} | classes {topclasses}",flush=True)
    out['runtime_s']=round(time.time()-t0,1)
    json.dump(out,open(OUT,'w'),indent=1); print('wrote',OUT)

if __name__ == '__main__':
    main()
