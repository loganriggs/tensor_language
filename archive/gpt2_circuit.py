"""Which GPT-2 components build the positive graph map?

Exact decomposition (pre-LN): resid_L = embed+pos + sum_l attn_l + sum_l mlp_l.
1. Component attribution of the Gram-adjacency covariance at the best-organized layer.
2. Per-head organization of attention components.
3. Causal ablations (mean-ablate component -> org + legal rate).

Usage: python gpt2_circuit.py  -> prints tables, saves runs_llm/gpt2-circuit.json
"""

import json
from pathlib import Path

import torch

from llm_reps import WINDOW, build_graph, sample_walks, single_token_words

N_WALKS = 96
T_CTX = 400
LSTAR = 11          # measure the map where it peaks (resid after block 10 = hidden_states[11])
N_LAYERS = 12


def windowed_node_mean(x, walks, n):
    """x: B x T x D component stream (token positions, no BOS offset applied here)."""
    lo = T_CTX - WINDOW
    wn = walks[:, lo:T_CTX].reshape(-1)
    flat = x[:, lo:T_CTX].reshape(-1, x.size(-1)).float()
    sums = torch.zeros(n, x.size(-1), device=x.device)
    sums.index_add_(0, wn.to(x.device), flat)
    cnt = torch.bincount(wn, minlength=n).float().to(x.device).clamp(min=1)
    return sums / cnt.unsqueeze(-1)


def org_of(H, A):
    n = A.size(0)
    Hc = H - H.mean(0)
    off = ~torch.eye(n, dtype=torch.bool)
    return torch.corrcoef(torch.stack([(Hc @ Hc.T)[off].cpu(), A[off]]))[0, 1].item()


def cov_share(comps, A):
    """Attribution of cov(Gram, A) over component pairs. comps: dict name -> n x D."""
    n = A.size(0)
    off = ~torch.eye(n, dtype=torch.bool)
    a = A[off] - A[off].mean()
    names = list(comps)
    centered = {k: (v - v.mean(0)).cpu() for k, v in comps.items()}
    total = {}
    for i, ki in enumerate(names):
        for kj in names[i:]:
            g = (centered[ki] @ centered[kj].T)
            g = (g + g.T)[off] if ki != kj else g[off]
            total[(ki, kj)] = (g * a).mean().item()
    return total


@torch.no_grad()
def main():
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

    # ---- capture component outputs with hooks ----
    capt = {}
    hooks = []
    for l in range(N_LAYERS):
        def mk(name):
            def fn(mod, inp, out):
                capt[name] = out[0] if isinstance(out, tuple) else out
            return fn
        hooks.append(model.transformer.h[l].attn.register_forward_hook(mk(f"attn{l}")))
        hooks.append(model.transformer.h[l].mlp.register_forward_hook(mk(f"mlp{l}")))

    comps_sum = {}
    embed_sum = None
    is_nbr = A.bool().cuda()
    legal = [0, 0]
    for b0 in range(0, N_WALKS, 24):
        batch = ids[b0:b0 + 24].cuda()
        bw = walks[b0:b0 + 24]
        out = model(batch, output_hidden_states=True)
        # component windowed means (strip BOS: positions 1..T -> steps 0..T-1)
        for name, x in list(capt.items()):
            m = windowed_node_mean(x[:, 1:], bw, n)
            comps_sum[name] = comps_sum.get(name, 0) + m
        e = windowed_node_mean(out.hidden_states[0][:, 1:], bw, n)
        embed_sum = e if embed_sum is None else embed_sum + e
        nl = out.logits.float()[..., node_ids.cuda()]
        cur = bw[:, T_CTX - WINDOW:T_CTX - 1].cuda()
        pred = nl[:, T_CTX - WINDOW + 1:T_CTX].argmax(-1)
        legal[0] += is_nbr[cur, pred].sum().item()
        legal[1] += cur.numel()
    for h in hooks:
        h.remove()
    n_batches = N_WALKS // 24
    comps = {k: v / n_batches for k, v in comps_sum.items()}
    comps["embed"] = embed_sum / n_batches
    print(f"baseline legal (window): {legal[0]/legal[1]:.3f}")

    # ---- org of cumulative resid per layer (sanity) + per component ----
    upto = {k: v for k, v in comps.items()
            if k == "embed" or int(k[4:] if k.startswith("attn") else k[3:]) < LSTAR}
    resid = sum(upto.values())
    print(f"resid@L{LSTAR} org: {org_of(resid, A):+.3f}   (components: {len(upto)})")
    self_orgs = {k: org_of(v, A) for k, v in upto.items()}
    print("\ncomponent self-organization (own Gram vs A):")
    for k, v in sorted(self_orgs.items(), key=lambda x: -abs(x[1])):
        print(f"  {k:8s} {v:+.3f}")

    # ---- covariance attribution over pairs ----
    shares = cov_share(upto, A)
    tot = sum(shares.values())
    print(f"\ntop +/- pair contributions to cov(Gram,A) (total {tot:.3g}):")
    ranked = sorted(shares.items(), key=lambda x: -abs(x[1]))[:14]
    for (ki, kj), v in ranked:
        print(f"  {ki:8s} x {kj:8s} {v/abs(tot):+7.2f} (rel)")

    # ---- per-head organization for the top attn layers ----
    head_org = {}
    W = model.transformer.h
    capt2 = {}
    hooks = []
    top_attn = [k for k in self_orgs if k.startswith("attn")]
    top_attn = sorted(top_attn, key=lambda k: -self_orgs[k])[:4]
    for name in top_attn:
        l = int(name[4:])
        def mk2(l):
            def fn(mod, inp, out):
                capt2[f"z{l}"] = inp[0]      # c_proj input: concat of head outputs
            return fn
        hooks.append(W[l].attn.c_proj.register_forward_hook(mk2(l)))
    head_sums = {}
    for b0 in range(0, N_WALKS, 24):
        batch = ids[b0:b0 + 24].cuda()
        bw = walks[b0:b0 + 24]
        model(batch)
        for name in top_attn:
            l = int(name[4:])
            z = capt2[f"z{l}"][:, 1:]
            for h in range(12):
                seg = z[..., h * 64:(h + 1) * 64] @ W[l].attn.c_proj.weight[h * 64:(h + 1) * 64]
                m = windowed_node_mean(seg, bw, n)
                head_sums[(l, h)] = head_sums.get((l, h), 0) + m
    for hk in hooks:
        hk.remove()
    print("\nper-head self-organization (top organized attn layers):")
    for (l, h), v in sorted(head_sums.items(), key=lambda x: -org_of(x[1] / n_batches, A))[:10]:
        head_org[f"{l}.{h}"] = org_of(v / n_batches, A)
        print(f"  head {l}.{h}: {head_org[f'{l}.{h}']:+.3f}")

    Path("runs_llm/gpt2-circuit.json").write_text(json.dumps({
        "legal": legal[0] / legal[1], "resid_org": org_of(resid, A),
        "self_orgs": self_orgs, "head_org": head_org,
        "shares_rel": {f"{a}x{b}": v / abs(tot) for (a, b), v in ranked},
    }))
    print("\nsaved runs_llm/gpt2-circuit.json")


if __name__ == "__main__":
    main()
