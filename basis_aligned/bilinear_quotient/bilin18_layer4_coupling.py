"""Layer 4's forward coupling: who consumes what it writes?

§27 found layer 4 has Shapley value −0.668 in the middle-deletion game: removing its
quadratic part REPAIRS damage once other middle layers are gone, though alone it costs
+0.110. The mechanical reading is that layer 4's output is an intermediate product —
useful only to downstream consumers, harmful when left unprocessed on the bus. That
implies a locatable edge: some specific set of later layers must be the consumers.

TEST 1 — locate the consumer interventionally. Layer 4's marginal deletion cost inside
different coalitions S: m(S) = v(S ∪ {4}) − v(S), operator C throughout. If the
consumers are layers J, then m(S) should flip from positive (helpful computation) to
negative (unprocessed toxin) exactly when S ⊇ J. Sweep S over suffixes {5}, {5,6},
{5..7}, {5..9}, {5..15}, the upstream pair {2,3}, and {2,3}∪{5..15}. The flip point
names the consumers.

TEST 2 — predict the edge from weights, Phase-D style. The input-mode Gram of layer
j's bilinear tensor in the Λ metric,

    G2(j) = Left_j^T [ (Down_j^T Down_j) ∘ (Right_j S_j Right_j^T) ] Left_j
            + the Right-side twin,

gives the input directions layer j's quadratic is most sensitive to, from weights plus
the input second moment. If layer 4 writes for layers J, then layer 4's top output
directions should sit in G2(j)'s top eigenspace for j ∈ J far above (a) random spans
and (b) the same alignment computed for other writer layers. Closed form, seconds; the
interventional flip in Test 1 is its verification.
"""

import json
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, eval_ce
from bilin18_joint_removal import orth

DEV = 'cuda'
OUT = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
       'bilin18_layer4_coupling_results.json')


def main():
    t0 = time.time()
    model, cfg = load_elriggs('bilin18', device=DEV)
    tokens = torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                        'bilin18_eval_tokens.pt')
    base = eval_ce(model, tokens, batch=4)
    MID = tuple(range(2, 16))
    print(f'base CE {base:.4f}')

    # operator C machinery: intact-model mean writes + input/output collection
    store_o = {li: [] for li in MID}
    store_i = {li: [] for li in range(2, 10)}
    hooks = []

    def mk(li):
        def h(mod, inp, o):
            store_o[li].append(o.detach().reshape(-1, o.shape[-1]).float())
            if li in store_i:
                store_i[li].append(inp[0].detach().reshape(-1, inp[0].shape[-1])
                                   .float())
        return h

    for li in MID:
        hooks.append(model.transformer.h[li].mlp.register_forward_hook(mk(li)))
    for i in range(0, 32, 4):
        b = tokens[i:i + 4].to(DEV)
        model(b[:, :-1].contiguous(), b[:, 1:].contiguous())
    for h in hooks:
        h.remove()
    means = {li: torch.cat(store_o[li]).mean(0) for li in MID}
    Y4 = torch.cat(store_o[4])
    Sin = {li: (lambda X: X.T @ X / X.shape[0])(torch.cat(store_i[li]))
           for li in store_i}
    orig = {li: model.transformer.h[li].mlp.forward for li in MID}

    def mk_const(mu):
        def f(x):
            return mu.to(x.dtype).expand(x.shape[:-1] + mu.shape)
        return f

    def value(ls):
        for li in ls:
            model.transformer.h[li].mlp.forward = mk_const(means[li])
        try:
            return eval_ce(model, tokens, batch=4) - base
        finally:
            for li in ls:
                model.transformer.h[li].mlp.forward = orig[li]

    # ===== TEST 1 =====
    print('\n== T1: layer 4 marginal inside coalitions (flip locates the consumer) ==')
    coalitions = [[], [5], [5, 6], [5, 6, 7], list(range(5, 10)),
                  list(range(5, 16)), [2, 3], [2, 3] + list(range(5, 16))]
    out = {'t1': []}
    for S in coalitions:
        vS = value(S) if S else 0.0
        vS4 = value(S + [4])
        marg = vS4 - vS
        tag = '{' + ','.join(map(str, S)) + '}' if S else '{}'
        out['t1'].append({'S': S, 'marginal': marg})
        print(f'  m(4 | {tag:18s}) = {marg:+.4f}', flush=True)

    # ===== TEST 2 =====
    print('\n== T2: weight-side edge prediction (input-mode Gram, Λ metric) ==')
    _, _, Vh4 = torch.linalg.svd(Y4 - Y4.mean(0), full_matrices=False)
    W4 = orth(Vh4[:8].T)                       # layer-4 top output directions

    def g2_top(li):
        mlp = model.transformer.h[li].mlp
        L = mlp.Left.weight.detach().float()
        R = mlp.Right.weight.detach().float()
        Dw = mlp.Down.weight.detach().float()
        S = Sin[li]
        DD = Dw.T @ Dw
        K = DD * (R @ S @ R.T)
        G = L.T @ K @ L
        K2 = DD * (L @ S @ L.T)
        G = G + R.T @ K2 @ R
        ev, U = torch.linalg.eigh(G)
        return orth(U[:, ev.argsort(descending=True)[:8]])

    g = torch.Generator(device=DEV).manual_seed(0)
    Qr = orth(torch.randn(1152, 8, device=DEV, generator=g))
    out['t2'] = {}
    print(f"  {'reader j':>9} {'energy of L4-out in G2(j) top-8':>32} "
          f"{'random-span':>12}")
    for j in (5, 6, 7, 8, 9):
        Uj = g2_top(j)
        e = float((Uj.T @ W4).pow(2).sum()) / 8
        er = float((Uj.T @ Qr).pow(2).sum()) / 8
        out['t2'][j] = {'energy': e, 'random': er, 'ratio': e / max(er, 1e-9)}
        print(f"  {j:>9} {e:>32.3f} {er:>12.3f}   ({e/max(er,1e-9):.0f}x)",
              flush=True)

    out['runtime_s'] = time.time() - t0
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
