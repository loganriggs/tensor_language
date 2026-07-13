"""Is the mid-stack graph map USED by GPT-2's task behavior?

At layer LSTAR, delete the organized subspace (top-k PCs of the node-mean reps,
which carry the graph harmonics) from the residual stream at all positions past
step 200, then measure legal rate / neighbor mass at steps 350-400.
Controls: random k-dim subspaces (same k), and the next-k PCs (matched provenance).

Usage: python gpt2_usetest.py [layer]   (default 8)
"""

import sys
import torch

from llm_reps import WINDOW, build_graph, sample_walks, single_token_words
from gpt2_circuit import windowed_node_mean, org_of

N_WALKS = 96
T = 400
START = 200      # projection applied from this step onward


@torch.no_grad()
def main(lstar=8):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32,
                                                 device_map="cuda").eval()
    gen = torch.Generator().manual_seed(0)
    pool = single_token_words(tok)
    nbrs = build_graph("grid45")
    n = 20
    A = torch.zeros(n, n)
    for v, x in enumerate(nbrs):
        A[v, x] = 1.0
    perm = torch.randperm(len(pool), generator=gen)[:n]
    node_ids = torch.tensor([pool[i][1] for i in perm])
    walks = sample_walks(nbrs, N_WALKS, gen)
    ids = torch.cat([torch.full((N_WALKS, 1), tok.bos_token_id, dtype=torch.long),
                     node_ids[walks]], 1)
    is_nbr = A.bool().cuda()

    # clean node-means at lstar -> map subspace
    sums = 0
    for b0 in range(0, N_WALKS, 24):
        out = model(ids[b0:b0 + 24].cuda(), output_hidden_states=True)
        sums = sums + windowed_node_mean(out.hidden_states[lstar][:, 1:], walks[b0:b0 + 24], n)
    H = (sums / (N_WALKS // 24))
    Hc = (H - H.mean(0))
    U, S, Vt = torch.linalg.svd(Hc, full_matrices=False)
    print(f"layer {lstar}: node-mean org {org_of(H, A):+.3f}; "
          f"top-4 PC var share {(S[:4]**2).sum()/(S**2).sum():.2f}")
    # verify the map subspace is the organized one
    for k in (2, 4):
        Hk = Hc @ Vt[:k].T @ Vt[:k]
        print(f"  org of top-{k}-PC content only: {org_of(Hk + H.mean(0), A):+.3f}")

    def run(P=None):
        """P: (k x 768) orthonormal rows to project OUT at layer lstar, steps >= START."""
        hooks = []
        if P is not None:
            Pg = P.cuda()
            def fn(mod, inp):
                h = inp[0].clone()
                seg = h[:, START + 1:]
                h[:, START + 1:] = seg - (seg @ Pg.T) @ Pg
                return (h,) + inp[1:]
            hooks.append(model.transformer.h[lstar].register_forward_pre_hook(fn))
        legal = [0, 0]
        mass = 0.0
        sums = 0
        for b0 in range(0, N_WALKS, 24):
            bw = walks[b0:b0 + 24]
            out = model(ids[b0:b0 + 24].cuda(), output_hidden_states=True)
            sums = sums + windowed_node_mean(out.hidden_states[11][:, 1:], bw, n)
            logits = out.logits.float()
            lse = torch.logsumexp(logits, -1)
            nl = logits[..., node_ids.cuda()]
            probs = (nl - lse.unsqueeze(-1)).exp()
            cur = bw[:, T - WINDOW:T - 1].cuda()
            pred = nl[:, T - WINDOW + 1:T].argmax(-1)
            legal[0] += is_nbr[cur, pred].sum().item()
            legal[1] += cur.numel()
            mass += (probs[:, T - WINDOW + 1:T] * is_nbr[cur].float()).sum().item()
        for h in hooks:
            h.remove()
        return (legal[0] / legal[1], mass / legal[1],
                org_of(sums / (N_WALKS // 24), A))

    conds = {"baseline": None,
             "map top-2 PCs": Vt[:2],
             "map top-4 PCs": Vt[:4],
             "PCs 5-8 (same provenance)": Vt[4:8]}
    g = torch.Generator().manual_seed(7)
    R = torch.linalg.qr(torch.randn(768, 4, generator=g))[0].T
    conds["random 4-dim"] = R
    print(f"\nprojection at layer {lstar} from step {START}; behavior at steps 350-400:")
    for name, P in conds.items():
        lg, ms, org11 = run(P)
        print(f"  {name:26s} legal {lg:.3f}   nbr-mass {ms:.3f}   map@11 {org11:+.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
