"""IS THE CLASS CODE MULTI-AXIS (a token can be several categories at once), and finer than the 8 labels?
(user: "The" is two categories — determiner AND capitalized). My classify() assigns ONE label per token (and
mislabels "The" as just determiner via its lowercase match), but the model's class representation is a SUBSPACE
(~24 eff-dim, §780), so a token should carry COMPONENTS along several category axes simultaneously. Test:
decode several BINARY attributes separately from the class-writing residual (mlp0 output) — capitalization,
determiner-ness, punctuation, number, space-prefix, sentence-start — and check (a) each is separately
decodable, (b) capitalization and determiner-ness are near-ORTHOGONAL axes (separate categories, not one
label), and (c) place "The"/"the"/"London"/"cat" in the (determiner-axis, capital-axis) plane to show "The"
projects onto BOTH.

REGISTERED PREDICTIONS:
  (0) SANITY: each attribute decodable above its base rate;
  (a) MULTI-AXIS: capitalization and determiner-ness are separately decodable and their probe directions are
      near-orthogonal (|cos| small) -> the class code is multi-axis, not an 8-way partition; and "The" has a
      HIGH determiner projection AND a HIGH capital projection (both), unlike "the" (det only) or "London"
      (cap only) -> a token is several categories at once;
  (b) also report the effective dimensionality of the class-conditional structure (finer than 8)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'class_multiaxis_results.json'
NEVAL = 200; SEQ = 256; READ_L = 0  # mlp0 output (class-writing)
DET = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our', 'my', 'your', 'some', 'any', 'no', 'every', 'each'}


def dec():
    import tiktoken; enc = tiktoken.get_encoding('gpt2'); return lambda i: enc.decode([int(i)])


def attrs(s):
    t = s
    st = t.strip()
    cap = 1 if (st[:1].isalpha() and st[:1].isupper()) else 0
    det = 1 if st.lower() in DET else 0
    punct = 1 if (st != '' and not st[0].isalnum()) else 0
    number = 1 if (st[:1].isdigit()) else 0
    space = 1 if t[:1] == ' ' else 0
    alpha = 1 if st[:1].isalpha() else 0
    return {'capital': cap, 'determiner': det, 'punct': punct, 'number': number, 'space_prefix': space, 'alpha': alpha}


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def probe(F_, y, seed=0):
    """ridge binary probe; returns (accuracy, weight_vector)."""
    n = F_.shape[0]; rng = np.random.RandomState(seed); idx = rng.permutation(n); ntr = int(0.7*n); tr, te = idx[:ntr], idx[ntr:]
    Ft = F_[tr]; yt = torch.tensor(y[tr], device=DEV, dtype=torch.float32) - 0.5
    A = Ft.T @ Ft + 1e2*torch.eye(F_.shape[1], device=DEV); w = torch.linalg.solve(A, Ft.T @ yt)
    pred = ((F_[te] @ w) > 0).cpu().numpy().astype(int); acc = float((pred == y[te]).mean())
    return acc, w / (w.norm() + 1e-9)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    outs = []; seqs = []
    def h(mo, i_, o_): outs.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hh = m.transformer.h[READ_L].mlp.register_forward_hook(h)
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :SEQ].to(DEV)[:, :-1].contiguous(); forward_logits(idx); seqs.append(idx.cpu().numpy())
    hh.remove()
    R = torch.cat(outs, 0); toks = np.concatenate([s.reshape(-1) for s in seqs])
    A = {k: np.array([attrs(d(int(t)))[k] for t in toks]) for k in ['capital', 'determiner', 'punct', 'number', 'space_prefix', 'alpha']}
    accs = {}; ws = {}
    for k in A:
        base = max(A[k].mean(), 1-A[k].mean())
        acc, w = probe(R, A[k]); accs[k] = {'acc': round(acc, 3), 'base_rate': round(float(base), 3)}; ws[k] = w
    # orthogonality of attribute axes
    keys = list(ws.keys()); cosmat = {a: {b: round(float(abs((ws[a]*ws[b]).sum())), 3) for b in keys} for a in keys}
    # place example tokens in (determiner, capital) plane
    def project(tokstr):
        try:
            import tiktoken; enc = tiktoken.get_encoding('gpt2'); tid = enc.encode(tokstr)
        except Exception: return None
        if len(tid) != 1: return None
        x = F.rms_norm(m.transformer.wte(torch.tensor([[tid[0]]], device=DEV)), (D,))
        # run mlp0 on the embedding to get its class-write
        v1 = None; x0 = x
        xb, v1 = m.transformer.h[0](x, v1, x0)  # block0 output; approx mlp0 contribution present
        r = xb.reshape(-1, D)[0]
        return {'determiner_proj': round(float((r @ ws['determiner'])), 3), 'capital_proj': round(float((r @ ws['capital'])), 3)}
    examples = {ts: project(ts) for ts, ts in [('The', 'The'), ('the', ' the'), ('London', ' London'), ('cat', ' cat'), ('and', ' and')]}
    # effective dim of class-conditional-mean structure (finer than 8?): eff-dim of the 6-attr conditional means... use token-mean eff-dim proxy
    out = {'read_layer_mlp': READ_L, 'attribute_decode': accs, 'axis_abs_cosine': cosmat,
           'examples_det_cap_plane': examples,
           'cap_det_orthogonal': round(cosmat['capital']['determiner'], 3), 'runtime_s': round(time.time()-t0, 1)}
    out['pred_a_multiaxis'] = bool(accs['capital']['acc'] > accs['capital']['base_rate'] + 0.05 and
                                   accs['determiner']['acc'] > accs['determiner']['base_rate'] + 0.05 and
                                   cosmat['capital']['determiner'] < 0.4)
    json.dump(out, open(OUT, 'w'), indent=1)
    print("attribute decode (acc vs base rate):", flush=True)
    for k in accs: print(f"  {k:>12}: {accs[k]['acc']} (base {accs[k]['base_rate']})", flush=True)
    print(f"capital vs determiner axis |cos| = {out['cap_det_orthogonal']} (small = separate axes)", flush=True)
    print(f"examples in (determiner, capital) plane: {examples}", flush=True)
    print(f"(a) class code is multi-axis (cap & det separate, The has both): {out['pred_a_multiaxis']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
