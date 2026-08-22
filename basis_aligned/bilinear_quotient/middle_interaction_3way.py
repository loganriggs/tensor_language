"""RESOLVE §943's caveat: does the middle multiply TOKEN x TOKEN, TOKEN x CONTENT, or CONTENT x CONTENT? §943
split x into grammar(=token+pos+class) vs content and found the middle's products are 'grammar'-dominated, but
grammar conflated TOKEN identity (rank 64) with grammatical CLASS. Do a clean 3-way ORTHOGONAL split of the MLP
input x = x_tok + x_cls + x_con (token-mean subspace / next-class subspace orthogonalized vs token / content
rest), and measure the 6 symmetric bilinear product groups (tt, tc, tcon, cc[class], c_cls_con, con_con) by
CAUSAL ablation (output - Down[group]) at a front layer (L1) and middle layers (L8, L11). Ablation is the robust
metric (§943's variance-share had cross-cancellation). This NAMES the middle multiplication precisely.

REGISTERED PREDICTIONS:
  (0) SANITY: the 6 groups + bias reconstruct the MLP output exactly (residual ~0); front L1 dominated by a
      token-involving group.
  (a) MIDDLE IS TOKEN-CONDITIONAL: at the middle layers the token-involving groups (token x token and
      token x content) have the largest ablation cost; pure CONTENT x CONTENT is a real but smaller contributor;
      class-involving groups are small -> the middle multiplies mostly token-conditional features, with
      content x content a minority (refining §943 with token/class disentangled);
  (b) report per-layer ablation Δce for all 6 groups + the exact-reconstruction residual."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'middle_interaction_3way_results.json'
NEVAL = 160; SEQ = 256; RTOK = 64; RCLASS = 8
LAYERS = [1, 8, 11]
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','me','who','which'}
ABL = {'L': -1, 'group': None, 'sub': None}  # sub[L] = (Utok, Ucls, g)


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


def mean_subspace(X, labels, r):
    g = X.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        if t < 0: continue
        mk = labels == t
        if mk.sum() < 5: continue
        rows.append(X[mk].mean(0)-g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0)*torch.tensor(wt, device=X.device, dtype=X.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:min(r, M.shape[0])].T.contiguous(), g


def three_parts(x, Utok, Ucls, g):
    xt = ((x-g) @ Utok) @ Utok.T          # token part (mean-centered projection)
    xa = ((x-g) @ Ucls) @ Ucls.T          # class part (Ucls already orthogonalized vs Utok)
    xc = (x-g) - xt - xa                   # content part
    return xt + g, xa, xc                  # put mean into token part (arbitrary but consistent)


def group_terms(mlp, x, Utok, Ucls, g):
    xt, xa, xc = three_parts(x, Utok, Ucls, g)
    Lt, La, Lc = mlp.Left(xt), mlp.Left(xa), mlp.Left(xc)
    Rt, Ra, Rc = mlp.Right(xt), mlp.Right(xa), mlp.Right(xc)
    dn = lambda h: F.linear(h, mlp.Down.weight)
    return {
        'tok_tok':   dn(Lt*Rt),
        'tok_cls':   dn(Lt*Ra + La*Rt),
        'tok_con':   dn(Lt*Rc + Lc*Rt),
        'cls_cls':   dn(La*Ra),
        'cls_con':   dn(La*Rc + Lc*Ra),
        'con_con':   dn(Lc*Rc),
    }


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return readout(x)


def abl_hook(L, mlp):
    def h(mo, i_, o_):
        if ABL['L'] != L: return o_
        x = i_[0] if isinstance(i_, tuple) else i_; y = o_[0] if isinstance(o_, tuple) else o_
        Utok, Ucls, g = ABL['sub'][L]; terms = group_terms(mlp, x, Utok, Ucls, g)
        yn = y - terms[ABL['group']]
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return h


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    toks = S[:, :-1].reshape(-1)
    nxtc = np.full_like(S[:, :-1], -1); nxtc[:, :-1] = S[:, 1:-1]
    nxtcls = np.array([CLASSES.index(classify(d(int(t)))) if t >= 0 else -1 for t in nxtc.reshape(-1)])
    # capture MLP inputs
    store = {L: [] for L in LAYERS}
    def mk(L):
        def h(mo, i_, o_): store[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D).cpu())
        return h
    hs = [m.transformer.h[L].mlp.register_forward_hook(mk(L)) for L in LAYERS]
    for i in range(0, nb, 4): forward_logits(blocks[i:i+4].to(DEV)[:, :-1].contiguous())
    for h in hs: h.remove()
    sub = {}
    for L in LAYERS:
        Xin = torch.cat(store[L], 0).to(DEV); Utok, g = mean_subspace(Xin, toks, RTOK)
        Ucls0, _ = mean_subspace(Xin, nxtcls, RCLASS)
        # orthogonalize class subspace against token subspace
        Ucls = Ucls0 - Utok @ (Utok.T @ Ucls0); Ucls = torch.linalg.qr(Ucls)[0][:, :RCLASS].contiguous()
        sub[L] = (Utok, Ucls, g); del Xin
    ABL['sub'] = sub
    def ce_pass():
        tot = 0.0; n = 0
        for i in range(0, nb, 4):
            bb = blocks[i:i+4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
            lg = forward_logits(idx).float(); tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction='sum')); n += tgt.numel()
        return tot/n
    # exact-reconstruction sanity at L11
    L = 11; Xchk = torch.cat(store[L], 0)[:2048].to(DEV); Utok, Ucls, g = sub[L]; mlp = m.transformer.h[L].mlp
    terms = group_terms(mlp, Xchk, Utok, Ucls, g)
    bias = mlp.Down.bias if mlp.Down.bias is not None else torch.zeros(D, device=DEV)
    recon = sum(terms.values()) + bias
    with torch.no_grad(): true_out = mlp(Xchk)
    recon_resid = float((recon - true_out).pow(2).sum()/ (true_out-true_out.mean(0)).pow(2).sum())
    hks = [m.transformer.h[L].mlp.register_forward_hook(abl_hook(L, m.transformer.h[L].mlp)) for L in LAYERS]
    ABL['L'] = -1; ce_full = ce_pass()
    out = {'ce_full': round(ce_full, 4), 'recon_residual_L11': round(recon_resid, 5), 'ablation_cost': {}}
    GROUPS = ['tok_tok', 'tok_cls', 'tok_con', 'cls_cls', 'cls_con', 'con_con']
    for L in LAYERS:
        out['ablation_cost'][str(L)] = {}
        for grp in GROUPS:
            ABL['L'] = L; ABL['group'] = grp; ce = ce_pass(); ABL['L'] = -1
            out['ablation_cost'][str(L)][grp] = round(ce - ce_full, 4)
        print(f"L{L:>2} ablation Δce: {out['ablation_cost'][str(L)]}", flush=True)
    for h in hks: h.remove()
    mid = out['ablation_cost'].get('11', {})
    tokinv = mid.get('tok_tok', 0) + mid.get('tok_con', 0) + mid.get('tok_cls', 0)
    out['middle_L11_token_involving_sum'] = round(tokinv, 4); out['middle_L11_con_con'] = round(mid.get('con_con', 0), 4)
    out['pred_a_middle_token_conditional'] = bool(tokinv > mid.get('con_con', 0))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"recon-residual L11 {recon_resid:.5f} | L11 token-involving sum {tokinv:.4f} vs con_con {mid.get('con_con',0):.4f}", flush=True)
    print(f"(a) middle is token-conditional (token-involving > content x content): {out['pred_a_middle_token_conditional']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
