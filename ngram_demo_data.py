"""Data for the n-gram-circuit section of the atlas artifact (TL_CORPUS=tiny).

Picks depth-3-gated examples and computes, per example:
  - the TOKENIZED context tail (last 4 tokens shown as discrete pieces) + target
  - train-corpus conditional probabilities P(tgt | last-1), P(tgt | last-2),
    P(tgt | last-3) — the "why it's a 3/4-gram" evidence
  - attn2 vs attn3 P(target) (the depth gate itself)
  - attn3-seed0 per-head single-ablation P(target) (12 heads) and each head's top
    attention source OFFSETS at the prediction position (is it gathering recent tokens?)

Writes runs_lm/ngram_demo.json.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, einsum

from lm_eval import load_model
from text_data import N_CTX, RUNS, load_tokenizer, tokens, val_windows

DEVICE = "cuda"
V = 1024
N_EX = 4


def head_forward(model, x, ablate=None):
    h = model.embed(x)
    pats = []
    for li, layer in enumerate(model.layers):
        pat = layer.pattern(h)
        v = rearrange(layer.v(layer.norm(h)), "b t (n d) -> b t n d", n=layer.n_head)
        z = einsum(pat, v, "b n q k, b k n d -> b q n d")
        if ablate is not None and ablate[0] == li:
            z = z.clone()
            z[:, :, ablate[1], :] = 0
        pats.append(pat)
        h = torch.lerp(h, layer.o(rearrange(z, "b q n d -> b q (n d)")), layer.scale)
    return model.head(h), pats


def corpus_probs(train, ctx3, tgt):
    """P(tgt | last-1), P(tgt | last-2), P(tgt | last-3) by direct scan."""
    out = []
    t = np.asarray(train, dtype=np.int64)
    for order in (1, 2, 3):
        ctx = ctx3[3 - order:]
        m = np.ones(len(t) - order, dtype=bool)
        for k in range(order):
            m &= t[k:len(t) - order + k] == ctx[k]
        denom = int(m.sum())
        num = int((m & (t[order:] == tgt)).sum())
        out.append({"order": order, "count": denom,
                    "p": round(num / denom, 3) if denom else None})
    return out


def main():
    tok = load_tokenizer()
    data, _ = val_windows()
    train = tokens("train")
    m2 = load_model(RUNS / "attn2-seed0", None, DEVICE)
    m3 = load_model(RUNS / "attn3-seed0", None, DEVICE)
    dh = m3.layers[0].d_head
    gated = np.load(RUNS / "gated_depth3.npy")
    rng = np.random.default_rng(3)

    examples = []
    for idx in rng.permutation(gated):
        w, p = divmod(int(idx), N_CTX)
        if p < 10:
            continue
        buf = torch.from_numpy(np.array(data[w * N_CTX:w * N_CTX + N_CTX + 1],
                               dtype=np.int64))[None].to(DEVICE)
        x = buf[:, :-1]
        tgt = int(buf[0, p + 1])
        with torch.no_grad():
            p3 = torch.softmax(m3(x)[0, p], -1)[tgt].item()
            p2 = torch.softmax(m2(x)[0, p], -1)[tgt].item()
        if p3 < 0.75 or p2 > 0.15:
            continue
        ctx3 = np.array([int(x[0, p - 2]), int(x[0, p - 1]), int(x[0, p])])
        cp = corpus_probs(train, np.concatenate([[int(x[0, p - 3])], ctx3])[-3:] if False else ctx3, tgt)
        # require the 3-gram to be genuinely informative and the 1-gram not
        if not cp[2]["p"] or cp[2]["p"] < 0.6 or (cp[0]["p"] or 0) > 0.2:
            continue

        with torch.no_grad():
            _, pats = head_forward(m3, x)
            heads = {}
            for li in range(3):
                for hi in range(4):
                    al, _ = head_forward(m3, x, ablate=(li, hi))
                    ap = torch.softmax(al[0, p], -1)[tgt].item()
                    row = pats[li][0, hi, p]
                    topk = torch.topk(row.abs(), 2).indices.tolist()
                    heads[f"L{li}H{hi}"] = {
                        "ablated_p": round(ap, 3),
                        "top_offsets": [{"off": int(p - s),
                                         "tok": tok.decode([int(x[0, s])]),
                                         "score": round(float(row[s]), 3)} for s in topk]}
        examples.append({
            "window": w, "pos": p,
            "context_tail_tokens": [tok.decode([int(x[0, p - k])]) for k in (3, 2, 1, 0)],
            "target": tok.decode([tgt]),
            "prefix_text": tok.decode([int(t) for t in x[0, max(0, p - 28):p - 3]]),
            "p_attn2": round(p2, 3), "p_attn3": round(p3, 3),
            "corpus": cp, "heads": heads})
        print(f"ex {len(examples)}: ...{examples[-1]['context_tail_tokens']} -> "
              f"{examples[-1]['target']!r}  attn2 {p2:.2f} attn3 {p3:.2f}  "
              f"P123 {[c['p'] for c in cp]}", flush=True)
        if len(examples) >= N_EX:
            break

    (RUNS / "ngram_demo.json").write_text(json.dumps(examples, indent=1))
    print("saved", len(examples))


if __name__ == "__main__":
    main()
