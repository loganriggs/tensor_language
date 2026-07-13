"""Test: organized head outputs = LOCAL attention heads (walk-window aggregation
= graph message passing). Correlate, across all 144 GPT-2 heads:
  x = locality (attention mass on offsets 1..3, averaged over late positions)
  y = self-organization of the head's windowed-mean output.
Then causal ablations: kill top-local heads vs random heads vs late-attn layers,
measure map at L11 + legal rate.

Usage: python gpt2_localheads.py -> prints, saves runs_llm/gpt2-localheads.json
"""

import json
from pathlib import Path

import torch

from llm_reps import WINDOW, build_graph, sample_walks, single_token_words
from gpt2_circuit import windowed_node_mean, org_of

N_WALKS = 96
T = 400


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32,
                                                 device_map="cuda").eval()
    model.config._attn_implementation = "eager"
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

    # ---- pass 1: head locality (attn weights) + head output means ----
    capt = {}
    hooks = []
    for l in range(12):
        def mkz(l):
            def fn(mod, inp, out):
                capt[f"z{l}"] = inp[0]
            return fn
        hooks.append(model.transformer.h[l].attn.c_proj.register_forward_hook(mkz(l)))

    head_sums = {}
    locality = torch.zeros(12, 12)
    for b0 in range(0, N_WALKS, 12):
        batch = ids[b0:b0 + 12].cuda()
        bw = walks[b0:b0 + 12]
        out = model(batch, output_attentions=True)
        for l in range(12):
            att = out.attentions[l]          # B x H x T x T
            q = torch.arange(att.size(-2), device=att.device)
            near = torch.zeros_like(att[..., 0])
            for d in (1, 2, 3):
                idx = (q - d).clamp(min=0)
                near += att.gather(-1, idx.view(1, 1, -1, 1).expand(att.size(0), 12, -1, 1)).squeeze(-1) * (q >= d)
            locality[l] += near[:, :, T - WINDOW:].mean((0, 2)).cpu()
            z = capt[f"z{l}"][:, 1:]
            for h in range(12):
                seg = z[..., h * 64:(h + 1) * 64] @ model.transformer.h[l].attn.c_proj.weight[h * 64:(h + 1) * 64]
                m = windowed_node_mean(seg, bw, n)
                head_sums[(l, h)] = head_sums.get((l, h), 0) + m
    for hk in hooks:
        hk.remove()
    locality /= N_WALKS // 12
    nb = N_WALKS // 12
    xs, ys, tags = [], [], []
    for (l, h), s in head_sums.items():
        xs.append(locality[l, h].item())
        ys.append(org_of(s / nb, A))
        tags.append(f"{l}.{h}")
    r = torch.corrcoef(torch.stack([torch.tensor(xs), torch.tensor(ys)]))[0, 1]
    print(f"locality vs head-output organization across 144 heads: r = {r:+.3f}")
    top = sorted(zip(tags, xs, ys), key=lambda t: -t[1])[:8]
    print("most local heads:", [(t, f"{x:.2f}", f"{y:+.2f}") for t, x, y in top])

    # ---- pass 2: ablations (replace head contribution with batch-position mean) ----
    def run_ablate(kill: set):
        """kill: set of (layer, head). Mean-ablate via hook on c_proj input."""
        means = {}
        # first collect per-position means for killed heads
        h2 = []
        for l in {l for l, _ in kill}:
            def mk(l):
                def fn(mod, inp, out):
                    capt[f"z{l}"] = inp[0]
                return fn
            h2.append(model.transformer.h[l].attn.c_proj.register_forward_hook(mk(l)))
        acc = {}
        for b0 in range(0, N_WALKS, 24):
            model(ids[b0:b0 + 24].cuda())
            for l in {l for l, _ in kill}:
                acc[l] = acc.get(l, 0) + capt[f"z{l}"].sum(0)
        for hk in h2:
            hk.remove()
        means = {l: v / N_WALKS for l, v in acc.items()}

        h3 = []
        for l in {l for l, _ in kill}:
            def mkpatch(l):
                heads = [h for ll, h in kill if ll == l]
                def fn(mod, inp):
                    z = inp[0].clone()
                    for h in heads:
                        z[..., h * 64:(h + 1) * 64] = means[l][:, h * 64:(h + 1) * 64]
                    return (z,)
                return fn
            h3.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(mkpatch(l)))
        sums = torch.zeros(n, model.config.hidden_size, device="cuda")
        legal = [0, 0]
        for b0 in range(0, N_WALKS, 24):
            batch = ids[b0:b0 + 24].cuda()
            bw = walks[b0:b0 + 24]
            out = model(batch, output_hidden_states=True)
            sums += windowed_node_mean(out.hidden_states[11][:, 1:], bw, n)
            nl = out.logits.float()[..., node_ids.cuda()]
            cur = bw[:, T - WINDOW:T - 1].cuda()
            pred = nl[:, T - WINDOW + 1:T].argmax(-1)
            legal[0] += is_nbr[cur, pred].sum().item()
            legal[1] += cur.numel()
        for hk in h3:
            hk.remove()
        return org_of(sums / (N_WALKS // 24), A), legal[0] / legal[1]

    most_local = [tuple(map(int, t.split("."))) for t, _, _ in top]
    conds = {
        "baseline": set(),
        "top8 local heads": set(most_local),
        "8 random heads": {(3, 7), (6, 1), (9, 5), (1, 8), (7, 3), (10, 6), (0, 2), (8, 11)},
        "attn9+attn10 (all heads)": {(9, h) for h in range(12)} | {(10, h) for h in range(12)},
        "attn2 (all heads)": {(2, h) for h in range(12)},
    }
    results = {"locality_r": r.item(),
               "heads": {t: {"loc": x, "org": y} for t, x, y in zip(tags, xs, ys)}}
    print("\nablations (org@L11, legal rate):")
    for name, kill in conds.items():
        o, lg = run_ablate(kill)
        results[name] = {"org": o, "legal": lg}
        print(f"  {name:28s} org {o:+.3f}   legal {lg:.3f}")
    Path("runs_llm/gpt2-localheads.json").write_text(json.dumps(results))


if __name__ == "__main__":
    main()
