"""Test the n-gram-order hypothesis (results_deeper.md): are depth-d gated tokens the
ones needing order-d corpus context — vs in-context copies (which no corpus n-gram
predicts)?

For sampled tokens from each gate set (+ base sample), compute train-corpus conditional
probabilities P(tgt|p1), P(tgt|p2,p1), P(tgt|p3,p2,p1) by a single targeted scan over
train.bin (searchsorted against the sample's needed context ids). Reports median and
fraction-with-P>0.5 per set. Run with TL_CORPUS=tiny for the archived grid.
"""

import numpy as np

from text_data import CORPUS, N_CTX, RUNS, tokens

V = 1024 if CORPUS == "tiny" else 5120
SAMPLE = 4000
CHUNK = 50_000_000


def targeted_counts(train, ctx_ids, full_ids, order):
    """Count occurrences of each needed context (order tokens) and context+target
    (order+1 tokens) via one pass. ids are base-V polynomial encodings (int64)."""
    ctx_sorted = np.sort(np.unique(ctx_ids))
    full_sorted = np.sort(np.unique(full_ids))
    ctx_counts = np.zeros(len(ctx_sorted), dtype=np.int64)
    full_counts = np.zeros(len(full_sorted), dtype=np.int64)
    n = len(train)
    for s in range(0, n - order, CHUNK):
        t = np.asarray(train[s:min(s + CHUNK + order, n)], dtype=np.int64)
        cid = np.zeros(len(t) - order, dtype=np.int64)
        for k in range(order):
            cid = cid * V + t[k:len(t) - order + k]
        fid = cid * V + t[order:]
        for ids_sorted, counts, ids in ((ctx_sorted, ctx_counts, cid),
                                        (full_sorted, full_counts, fid)):
            pos = np.searchsorted(ids_sorted, ids)
            pos[pos == len(ids_sorted)] = 0
            hit = ids_sorted[pos] == ids
            np.add.at(counts, pos[hit], 1)
    return dict(zip(ctx_sorted, ctx_counts)), dict(zip(full_sorted, full_counts))


def main():
    train = tokens("train")
    val = tokens("val")
    rng = np.random.default_rng(0)
    sets = {"base": rng.integers(3, (len(val) - 1) // N_CTX * N_CTX, SAMPLE)}
    for d in (2, 3, 4):
        f = RUNS / f"gated_depth{d}.npy"
        if f.exists():
            g = np.load(f)
            sets[f"depth{d}"] = rng.choice(g, min(SAMPLE, len(g)), replace=False)

    # gather (p3,p2,p1,tgt) for every sampled token (positions are window-relative;
    # exclude samples within 3 tokens of a window start to keep context inside stream)
    quads = {}
    for name, idx in sets.items():
        keep = idx % N_CTX >= 3
        idx = idx[keep]
        tpos = (idx // N_CTX) * N_CTX + idx % N_CTX + 1
        quads[name] = np.stack([np.asarray(val[tpos - 3], dtype=np.int64),
                                np.asarray(val[tpos - 2], dtype=np.int64),
                                np.asarray(val[tpos - 1], dtype=np.int64),
                                np.asarray(val[tpos], dtype=np.int64)], 1)

    print(f"scanning train ({len(train)/1e6:.0f}M tokens) for orders 1-3...", flush=True)
    allq = np.concatenate(list(quads.values()))
    probs = {}
    for order in (1, 2, 3):
        ctx = allq[:, 3 - order:3]
        cid = np.zeros(len(ctx), dtype=np.int64)
        for k in range(order):
            cid = cid * V + ctx[:, k]
        fid = cid * V + allq[:, 3]
        cc, fc = targeted_counts(train, cid, fid, order)
        probs[order] = (cid, fid, cc, fc)
        print(f"  order {order} counted", flush=True)

    off = 0
    print(f"\n{'set':10s}" + "".join(f"  P(tgt|{o}-gram): med / frac>0.5" for o in (1, 2, 3)))
    for name, q in quads.items():
        n = len(q)
        row = f"{name:10s}"
        for order in (1, 2, 3):
            cid, fid, cc, fc = probs[order]
            c = np.array([cc.get(i, 0) for i in cid[off:off + n]])
            f_ = np.array([fc.get(i, 0) for i in fid[off:off + n]])
            p = np.where(c > 0, f_ / np.maximum(c, 1), 0.0)
            row += f"      {np.median(p):.3f} / {np.mean(p > 0.5):.1%}"
        off += n
        print(row, flush=True)


if __name__ == "__main__":
    main()
