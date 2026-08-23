"""THREAD C (transition band L3-5): WHAT do the content-birth MLPs compute? §1052 located content onset in L3-5;
§1048 showed the pooled-content bag does NOT rescue these MLPs; their computation is the deepest unexplained gap.
Hypothesis: the transition band computes TOKEN x CONTEXT cross-terms (binding the current token to pooled context),
between the front's tok x tok and the deep-middle's context x context (§1041). The bilinear form makes this EXACT
algebra (weight-based, no fitting): split MLP input x = mtok + dv (per-token mean + deviation); then
Down[(Lx)*(Rx)] = Down[Lm*Rm] + Down[Lm*Rd + Ld*Rm] + Down[Ld*Rd] = TOK + CROSS + DEV terms exactly.
Measure per layer (front 0,1 | transition 3,4,5 | deep 8,10,12 | readout 16): (1) output-VARIANCE share of each
term; (2) CE cost of substituting the MLP output with partial sums (TOK only / TOK+CROSS / DEV only; FULL = exact
reconstruction sanity).

REGISTERED PREDICTIONS:
  (0) SANITY: FULL reconstruction CE cost ~ 0 (bilinearity exact); front L0 TOK-share is the largest of any band
      (front MLPs are token functions, §1045).
  (a) TRANSITION = BINDING: in L3-5 the CROSS term (tok x dev) holds a LARGER variance share than in either the
      front (L0-1) or the deep-middle (L8-12), and TOK+CROSS recovers most of the transition MLPs' CE gap
      (>= 70% of the full-ablation cost) while TOK alone does not (< 40%) -> the band multiplies the current
      token against the context stream (binding), which is why neither token-only (§1045) nor bag (§1048)
      stand-ins could explain it;
  (b) GRADIENT ACROSS DEPTH: TOK share falls and DEV share rises monotonically front -> deep (band means)
      -- the token->context handoff is the depth story of the whole MLP stack;
  (c) if CROSS is small everywhere, the transition is dev x dev already at L3 (report plainly; content is
      context-self-multiplication from birth)."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'transition_terms_results.json'
NSEQ = 96; SEQ = 256
LAYERS = [0, 1, 3, 4, 5, 8, 10, 12, 16]
H = m.transformer.h
SUB = {'mode': None, 'layer': -1}   # mode in {tok, tokcross, dev, full, meanabl}
XBAR = {}; CUR = {}


def fwd(idx):
    CUR['tok'] = idx
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def terms(mlp, x, mtok):
    dv = x - mtok
    Lm = mlp.Left(mtok); Rm = mlp.Right(mtok); Ld = mlp.Left(dv); Rd = mlp.Right(dv)
    t_tok = mlp.Down(Lm*Rm)
    t_cross = mlp.Down(Lm*Rd + Ld*Rm)
    t_dev = mlp.Down(Ld*Rd)
    return t_tok, t_cross, t_dev


def sub_hook(L):
    def h(mo, i_, o_):
        if SUB['layer'] != L or SUB['mode'] is None: return None
        x = (i_[0] if isinstance(i_, tuple) else i_)
        mtok = XBAR[L][CUR['tok']].to(x.dtype)
        if SUB['mode'] == 'meanabl':
            return (XBAR_OUT[L].view(1, 1, D).expand_as(o_)).to(o_.dtype)
        t_tok, t_cross, t_dev = terms(mo, x, mtok)
        b = mo.Down_bias
        if SUB['mode'] == 'tok': y = t_tok + b
        elif SUB['mode'] == 'tokcross': y = t_tok + t_cross + b
        elif SUB['mode'] == 'dev': y = t_dev + b
        else: y = t_tok + t_cross + t_dev + b
        return y.to(o_.dtype)
    return h


@torch.no_grad()
def ce(blocks):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].reshape(-1)
        lp = F.log_softmax(fwd(idx).float(), -1)
        tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tgt.shape[0], device=DEV), tgt].sum()); n += tgt.shape[0]
    return tot/n


@torch.no_grad()
def main():
    global XBAR_OUT
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0])

    # pass 1: per-token input means + output means per layer
    capI = {L: [] for L in LAYERS}; capO = {L: [] for L in LAYERS}; hs = []
    for L in LAYERS:
        def mk(L):
            def h(mo, i_, o_):
                capI[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
                capO[L].append(o_.detach().float().reshape(-1, D))
            return h
        hs.append(H[L].mlp.register_forward_hook(mk(L)))
    idsL = []
    for i in range(0, NSEQ, 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fwd(idx)
    for h in hs: h.remove()
    tok = torch.cat(idsL, 0)
    XBAR_OUT = {}
    var_shares = {}
    for L in LAYERS:
        X = torch.cat(capI[L], 0); capI[L] = []
        xb = torch.zeros(V, D, device=DEV); cn = torch.zeros(V, device=DEV)
        xb.index_add_(0, tok, X); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
        XBAR[L] = (xb / cn.clamp_min(1).unsqueeze(1)).half()
        O = torch.cat(capO[L], 0); capO[L] = []
        XBAR_OUT[L] = O.mean(0)
        # variance shares in chunks
        mlp = H[L].mlp; sums = torch.zeros(3, device=DEV); tot_var = float(((O - O.mean(0))**2).sum())
        for i in range(0, X.shape[0], 4096):
            xx = X[i:i+4096]; mt = XBAR[L][tok[i:i+4096]].float()
            tt, tc, td = terms(mlp, xx, mt)
            for j, t in enumerate((tt, tc, td)): sums[j] += (t.float()**2).sum()
        s = sums / sums.sum()
        var_shares[str(L)] = {'tok': round(float(s[0]), 4), 'cross': round(float(s[1]), 4), 'dev': round(float(s[2]), 4)}
        print(f"L{L} variance shares: tok {var_shares[str(L)]['tok']} | cross {var_shares[str(L)]['cross']} | dev {var_shares[str(L)]['dev']}", flush=True)
        del X, O

    # pass 2: CE substitutions
    hs = [H[L].mlp.register_forward_hook(sub_hook(L)) for L in LAYERS]
    SUB['layer'] = -1; base = ce(blocks)
    ce_res = {}
    for L in LAYERS:
        row = {}
        for mode in ['full', 'tok', 'tokcross', 'dev', 'meanabl']:
            SUB['layer'] = L; SUB['mode'] = mode
            row[mode] = round(ce(blocks) - base, 4)
            SUB['layer'] = -1; SUB['mode'] = None
        abl = max(row['meanabl'], 1e-6)
        row['tok_recov'] = round(1 - row['tok']/abl, 3); row['tokcross_recov'] = round(1 - row['tokcross']/abl, 3)
        ce_res[str(L)] = row
        print(f"L{L} CE cost: full {row['full']} | tok {row['tok']} (recov {row['tok_recov']}) | tok+cross {row['tokcross']} (recov {row['tokcross_recov']}) | dev {row['dev']} | mean-abl {row['meanabl']}", flush=True)
    for h in hs: h.remove()

    def band_mean(ls, key): return round(sum(var_shares[str(L)][key] for L in ls)/len(ls), 4)
    trans_cross = band_mean([3, 4, 5], 'cross')
    out = {'base_ce': round(base, 4), 'var_shares': var_shares, 'ce': ce_res,
           'band_cross': {'front01': band_mean([0, 1], 'cross'), 'trans345': trans_cross,
                          'deep81012': band_mean([8, 10, 12], 'cross')},
           'band_tok': {'front01': band_mean([0, 1], 'tok'), 'trans345': band_mean([3, 4, 5], 'tok'),
                        'deep81012': band_mean([8, 10, 12], 'tok')},
           'band_dev': {'front01': band_mean([0, 1], 'dev'), 'trans345': band_mean([3, 4, 5], 'dev'),
                        'deep81012': band_mean([8, 10, 12], 'dev')}}
    tc_rec = [ce_res[str(L)]['tokcross_recov'] for L in (3, 4, 5)]
    t_rec = [ce_res[str(L)]['tok_recov'] for L in (3, 4, 5)]
    out['pred_a_binding'] = bool(trans_cross > out['band_cross']['front01'] and trans_cross > out['band_cross']['deep81012']
                                 and min(tc_rec) >= 0.7 and max(t_rec) < 0.4)
    out['pred_b_gradient'] = bool(out['band_tok']['front01'] > out['band_tok']['trans345'] > out['band_tok']['deep81012']
                                  and out['band_dev']['front01'] < out['band_dev']['trans345'] < out['band_dev']['deep81012'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"band cross-shares front {out['band_cross']['front01']} | trans {trans_cross} | deep {out['band_cross']['deep81012']}", flush=True)
    print(f"pred_a binding {out['pred_a_binding']} | pred_b gradient {out['pred_b_gradient']} | wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
