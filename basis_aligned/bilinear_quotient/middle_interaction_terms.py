"""WHAT does the middle bilinear MLP MULTIPLY? The frontier is the middle's multiplicative interaction (§941/§942).
A bilinear MLP output (minus bias) factors EXACTLY: with x = x_g (grammar: token+pos+class subspace) + x_c
(content: the rest),
  Down[(Left.x)(*)(Right.x)] - bias = Down[Lg*Rg] + Down[Lg*Rc] + Down[Lc*Rg] + Down[Lc*Rc]
  = gg (grammar x grammar) + gc + cg (grammar x content) + cc (content x content).
Measure each term's share of the output (variance) and its causal loss-cost (ablate = output - term) at a FRONT
layer (L1, contrast) and MIDDLE layers (L8, L11, L15). This NAMES the middle multiplication.

REGISTERED PREDICTIONS:
  (0) SANITY: the four terms + bias reconstruct the true MLP output exactly (residual ~0).
  (a) MIDDLE MULTIPLIES CONTENT x CONTENT: at the middle layers the content x content (cc) term dominates the
      output variance AND has the largest causal loss-cost when ablated -> the middle's job is CONTENT MIXING;
      the grammar-involving terms (gg, gc, cg) are secondary. At the FRONT (L1, near-linear §941) the product is
      grammar-dominated (gg + grammar-cross), not cc;
  (b) report per-layer variance share and ablation loss-cost for gg / gc / cg / cc."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'middle_interaction_terms_results.json'
NEVAL = 160; SEQ = 256; RTOK = 64; RPOS = 32; RCLASS = 8
VAR_LAYERS = [1, 8, 11, 15]; ABL_LAYERS = [1, 11]
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','me','who','which'}
ABL = {'L': -1, 'term': None, 'Us': None}


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


def mlp_terms(mlp, x, U, g):
    """Return dict of the 4 Down[term] tensors (grammar subspace U, mean g)."""
    xc_g = ((x-g) @ U) @ U.T + g  # grammar part (struct subspace + mean)
    xg = xc_g; xcc = x - xg
    Lg = mlp.Left(xg); Rg = mlp.Right(xg); Lc = mlp.Left(xcc); Rc = mlp.Right(xcc)
    dn = lambda h: F.linear(h, mlp.Down.weight)  # Down without bias
    return {'gg': dn(Lg*Rg), 'gc': dn(Lg*Rc), 'cg': dn(Lc*Rg), 'cc': dn(Lc*Rc)}


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return readout(x)


def abl_hook(L, mlp):
    def h(mo, i_, o_):
        if ABL['L'] != L: return o_
        x = i_[0] if isinstance(i_, tuple) else i_; y = o_[0] if isinstance(o_, tuple) else o_
        U, g = ABL['Us'][L]; terms = mlp_terms(mlp, x, U, g)
        yn = y - terms[ABL['term']]
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return h


@torch.no_grad()
def capL_input(L):
    cap = {}
    def h(mo, i_, o_):
        cap['in'] = (i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D)
        cap['out'] = (o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D)
    return h


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    toks = S[:, :-1].reshape(-1); pos = np.broadcast_to(np.arange(SEQ-1), (nb, SEQ-1)).reshape(-1)
    nxtc = np.full_like(S[:, :-1], -1); nxtc[:, :-1] = S[:, 1:-1]
    nxtcls = np.array([CLASSES.index(classify(d(int(t)))) if t >= 0 else -1 for t in nxtc.reshape(-1)])
    allL = sorted(set(VAR_LAYERS) | set(ABL_LAYERS))
    # capture each MLP's input and output (store per-batch on CPU)
    Us = {}
    store_in = {L: [] for L in allL}; store_out = {L: [] for L in allL}
    def mk(L):
        def h(mo, i_, o_):
            store_in[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D).cpu())
            store_out[L].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D).cpu())
        return h
    hs = [m.transformer.h[L].mlp.register_forward_hook(mk(L)) for L in allL]
    for i in range(0, nb, 4): forward_logits(blocks[i:i+4].to(DEV)[:, :-1].contiguous())
    for h in hs: h.remove()
    out = {'variance_share': {}, 'reconstruction_residual': {}}
    for L in allL:
        Xin = torch.cat(store_in[L], 0).to(DEV); Utok, g = mean_subspace(Xin, toks, RTOK)
        Upos, _ = mean_subspace(Xin, pos.astype(np.int64), RPOS); Ucls, _ = mean_subspace(Xin, nxtcls, RCLASS)
        U = torch.linalg.svd(torch.cat([Utok, Upos, Ucls], 1), full_matrices=False)[0][:, :RTOK+RPOS+RCLASS].contiguous()
        Us[L] = (U, g)
        del Xin
    # variance share per term (batched to save memory)
    for L in VAR_LAYERS:
        U, g = Us[L]; mlp = m.transformer.h[L].mlp
        sq = {'gg': 0.0, 'gc': 0.0, 'cg': 0.0, 'cc': 0.0}; tot = 0.0; recerr = 0.0; outsum = 0.0
        Xin = torch.cat(store_in[L], 0); Oout = torch.cat(store_out[L], 0)
        bias = mlp.Down.bias if mlp.Down.bias is not None else torch.zeros(D, device=DEV)
        for s in range(0, Xin.shape[0], 4096):
            x = Xin[s:s+4096].to(DEV); O = Oout[s:s+4096].to(DEV)
            terms = mlp_terms(mlp, x, U, g)
            recon = terms['gg']+terms['gc']+terms['cg']+terms['cc']+bias
            recerr += float((recon-O).pow(2).sum()); outsum += float((O-O.mean(0)).pow(2).sum())
            for k in sq: sq[k] += float(terms[k].pow(2).sum())
            tot += float((terms['gg']+terms['gc']+terms['cg']+terms['cc']).pow(2).sum())
        share = {k: round(sq[k]/max(tot, 1e-9), 3) for k in sq}
        out['variance_share'][str(L)] = share; out['reconstruction_residual'][str(L)] = round(recerr/max(outsum, 1e-9), 5)
        print(f"L{L:>2} variance share: {share} | recon-residual {out['reconstruction_residual'][str(L)]:.5f}", flush=True)
        del Xin, Oout
    # causal ablation of each term
    ROWN = nb*(SEQ-1)
    def ce_pass():
        tot = 0.0; n = 0
        for i in range(0, nb, 4):
            bb = blocks[i:i+4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
            lg = forward_logits(idx).float(); tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction='sum')); n += tgt.numel()
        return tot/n
    ABL['Us'] = Us
    hks = [m.transformer.h[L].mlp.register_forward_hook(abl_hook(L, m.transformer.h[L].mlp)) for L in ABL_LAYERS]
    ABL['L'] = -1; ce_full = ce_pass(); out['ce_full'] = round(ce_full, 4); out['ablation_cost'] = {}
    for L in ABL_LAYERS:
        out['ablation_cost'][str(L)] = {}
        for term in ['gg', 'gc', 'cg', 'cc']:
            ABL['L'] = L; ABL['term'] = term; ce = ce_pass(); ABL['L'] = -1
            out['ablation_cost'][str(L)][term] = round(ce - ce_full, 4)
        print(f"L{L:>2} ablation Δce: {out['ablation_cost'][str(L)]}", flush=True)
    for h in hks: h.remove()
    mid_share = out['variance_share'].get('11', {}); mid_abl = out['ablation_cost'].get('11', {})
    out['pred_a_middle_cc_dominates'] = bool(mid_share and mid_share['cc'] == max(mid_share.values()) and
                                             mid_abl and mid_abl['cc'] == max(mid_abl.values()))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"(a) middle multiplies content x content (cc dominates variance+ablation at L11): {out['pred_a_middle_cc_dominates']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
