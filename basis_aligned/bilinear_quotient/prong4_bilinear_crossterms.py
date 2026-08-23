"""PRONG 4 (compositional / tensor-network): §1040 showed the deep-middle MLP is full-rank as a FLAT bilinear map of
the raw 1152-dim input. But the input is a SUM of a LOCAL (current-token) part and a CONTEXT part (accumulated /
pooled from earlier layers). A bilinear form over a sum decomposes into cross-terms:
  (Wl x)(Wr x) = (Wl(x_loc+x_ctx))(Wr(x_loc+x_ctx)) = LOC×LOC + (LOC×CTX + CTX×LOC) + CTX×CTX
where x_loc = the per-token conditional mean of the input (the part that is a function of the current token) and
x_ctx = x - x_loc (the context deviation). This expresses the middle's bilinear computation in interpretable terms:
does it multiply token×token (local), token×context, or context×context? If the CONTENT/loss lives in the
context-involving terms, the middle is a STRUCTURED bilinear form (token gated/multiplied by pooled context), which is
the compositional understanding a flat rank measure misses.

For deep-middle layers, measure (a) each cross-term's share of the OUTPUT variance, and (b) LOSS-recovery when the MLP
output is replaced by cumulative term subsets (loc; loc+cross; full) + bias.

REGISTERED PREDICTIONS:
  (0) SANITY: LOC+CROSS+CTX reconstructs the output exactly (it's an identity decomposition); loss-recovery of the
      full sum = 1.
  (a) CONTENT LIVES IN THE CONTEXT TERMS: for the deep-middle (L8/L11), the LOC×LOC (token×token) term alone recovers
      LITTLE of the layer's loss, while adding the CONTEXT terms (token×context, context×context) recovers most ->
      the middle multiplies the token BY the pooled context; its bilinear form is compositionally structured, not an
      unstructured full-rank map;
  (b) report each cross-term's output-variance share and the cumulative loss-recovery (loc / loc+cross / full)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'prong4_bilinear_crossterms_results.json'
NCAL = 64; NEVAL = 160; SEQ = 256; LAYERS = [8, 11]
XBAR = {}; SUB = {'L': None, 'mode': None, 'tokids': None}


def forward_logits(idx):
    SUB['tokids'] = idx
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def crossterms(mlp, x, xloc):
    # returns dict of Down(term) for LOC×LOC, CROSS, CTX (bias added separately)
    Wl = mlp.Left.weight.float(); Wr = mlp.Right.weight.float(); Dn = mlp.Down.weight.float()
    xdev = x - xloc
    am = x @ Wl.T; bm = x @ Wr.T          # full a,b (for reference / full)
    aml = xloc @ Wl.T; bml = xloc @ Wr.T
    adl = xdev @ Wl.T; bdl = xdev @ Wr.T
    loc = (aml * bml) @ Dn.T
    cross = (aml * bdl + adl * bml) @ Dn.T
    ctx = (adl * bdl) @ Dn.T
    return loc, cross, ctx


def sub_hook_factory(L):
    mlp = m.transformer.h[L].mlp
    bias = (mlp.Down.bias.float() if mlp.Down.bias is not None else torch.zeros(D, device=DEV))
    def h(mo, i_, o_):
        if SUB['L'] != L or SUB['mode'] is None: return None
        x = (i_[0] if isinstance(i_, tuple) else i_).float().reshape(-1, D)
        xloc = XBAR[L][SUB['tokids'].reshape(-1)]
        loc, cross, ctx = crossterms(mlp, x, xloc)
        if SUB['mode'] == 'loc': y = loc + bias
        elif SUB['mode'] == 'loc_cross': y = loc + cross + bias
        else: y = loc + cross + ctx + bias   # full (identity check)
        o = o_[0] if isinstance(o_, tuple) else o_
        return y.reshape(o.shape).to(o.dtype)
    return h


@torch.no_grad()
def ce(blocks):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1); tf = tgt.reshape(-1)
        tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tf.shape[0], device=DEV), tf].sum()); n += tf.shape[0]
    return tot / n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NCAL + NEVAL)
    calib = rows[:NCAL, :SEQ].contiguous(); blocks = rows[NCAL:NCAL+NEVAL, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0])
    # per-token conditional mean of each layer's INPUT on calib (x_loc)
    caps = {L: [] for L in LAYERS}; toks = []; hs = []
    for L in LAYERS:
        mlp = m.transformer.h[L].mlp
        def mk(L):
            def h(mo, i_, o_): caps[L].append((i_[0] if isinstance(i_, tuple) else i_).float().reshape(-1, D).cpu())
            return h
        hs.append(mlp.register_forward_hook(mk(L)))
    SUB['L'] = None
    for i in range(0, calib.shape[0], 8):
        idx = calib[i:i+8].to(DEV)[:, :-1].contiguous(); toks.append(idx.reshape(-1).cpu()); forward_logits(idx)
    for h in hs: h.remove()
    tok = torch.cat(toks, 0).to(DEV)
    for L in LAYERS:
        X = torch.cat(caps[L], 0).to(DEV); xbar = X.mean(0, keepdim=True).repeat(V, 1)
        sums = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
        sums.index_add_(0, tok, X); cnts.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
        seen = cnts >= 5; xbar[seen] = sums[seen]/cnts[seen].unsqueeze(1); XBAR[L] = xbar; del X
    # output-variance share of each cross-term (on eval, captured input)
    out = {'layers': {}}
    hooks = [m.transformer.h[L].mlp.register_forward_hook(sub_hook_factory(L)) for L in LAYERS]
    SUB['L'] = None; ce_full = ce(blocks); print(f"ce_full {ce_full:.4f}", flush=True)
    out['ce_full'] = round(ce_full, 4)
    # variance shares: recompute terms on a sample of eval inputs
    vhs = {L: [] for L in LAYERS}; vtok = []; hs2 = []
    for L in LAYERS:
        mlp = m.transformer.h[L].mlp
        def mk(L):
            def h(mo, i_, o_): vhs[L].append((i_[0] if isinstance(i_, tuple) else i_).float().reshape(-1, D).detach())
            return h
        hs2.append(mlp.register_forward_hook(mk(L)))
    for i in range(0, 24, 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); vtok.append(idx.reshape(-1));
        SUB['L']=None; forward_logits(idx)
    for h in hs2: h.remove()
    vt = torch.cat(vtok, 0)
    for L in LAYERS:
        mlp = m.transformer.h[L].mlp; X = torch.cat(vhs[L], 0); xloc = XBAR[L][vt]
        loc, cross, ctx = crossterms(mlp, X, xloc)
        tv = lambda t: float(t.var(0).sum())
        vs = {'loc': tv(loc), 'cross': tv(cross), 'ctx': tv(ctx)}; tot = sum(vs.values())+1e-9
        out['layers'].setdefault(str(L), {})['var_share'] = {k: round(v/tot, 3) for k, v in vs.items()}
        del X
    # loss-recovery of cumulative subsets
    for L in LAYERS:
        SUB['L'] = L; SUB['mode'] = 'full'; ce_id = ce(blocks)   # identity check
        # mean-ablate this layer as the recovery floor
        SUB['mode'] = 'loc'; ce_loc = ce(blocks)
        SUB['mode'] = 'loc_cross'; ce_lc = ce(blocks)
        SUB['L'] = None
        # floor: replace with just bias (~mean-ablate) -- use loc with xdev=0? approximate floor via a huge; instead use ce of loc as low end
        denom = max(ce_loc - ce_full, 1e-6)  # loc-only cost as the span proxy is wrong; report raw costs
        out['layers'][str(L)]['ce_identity_full'] = round(ce_id, 4)
        out['layers'][str(L)]['cost_loc_only'] = round(ce_loc - ce_full, 4)
        out['layers'][str(L)]['cost_loc_cross'] = round(ce_lc - ce_full, 4)
        # fraction of the loc-only damage removed by adding context terms
        out['layers'][str(L)]['context_terms_fix_frac'] = round(float((ce_loc - ce_lc)/max(ce_loc - ce_full, 1e-6)), 3)
        print(f"L{L}: var_share {out['layers'][str(L)]['var_share']} | cost loc-only {ce_loc-ce_full:.3f} loc+cross {ce_lc-ce_full:.3f} (identity {ce_id-ce_full:+.4f})", flush=True)
    for h in hooks: h.remove()
    ctxshare = float(np.mean([out['layers'][str(L)]['var_share']['cross']+out['layers'][str(L)]['var_share']['ctx'] for L in LAYERS]))
    out['mean_context_var_share'] = round(ctxshare, 3)
    out['pred_a_content_in_context_terms'] = bool(ctxshare > 0.5)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"mean context-term var share {out['mean_context_var_share']}", flush=True)
    print(f"pred_a content in context terms: {out['pred_a_content_in_context_terms']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
