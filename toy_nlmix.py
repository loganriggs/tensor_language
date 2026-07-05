"""Natural-language co-training (user's session-4 idea 4 / prediction P8): does mixing
real natural-language documents into the training set install the positive map in the
toy softmax stack that anti-organizes on graph walks alone?

Setup: a ~5k-token BPE tokenizer trained on wikitext-103 (ids 0..V_NL-1). Graph node
labels live in a RESERVED range (V_NL .. V_NL+99) so text tokens and node tokens never
collide. Each batch is one third natural-language docs (256-token wikitext chunks),
one third grid walks, one third directed-ring walks — grid+dring is the pairing that
reliably pins the toy models ANTI (softmax -0.80, bilinear -0.55..-0.72), so any flip
to positive is attributable to the natural language.

Pre-registered P8: softmax-add-3L becomes non-anti (org >= 0) with NL co-training,
with no burst family and no queries — natural text is maximal re-predict pressure
(session-3 attribution: local/copy heads are worth ~0.93 nats on all wikitext tokens).

Usage: python toy_nlmix.py
"""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from graphs import N_VOCAB, N_CTX, MAX_NODES, TRAIN_POOLS, walk_pool, legal_tokens
from model import CycleModel

V_NL = 5000
NODE0 = V_NL                       # node ids 5000..5099
V_TOTAL = V_NL + N_VOCAB           # 5100
TOK_PATH = Path("runs_gen/nl_bpe.json")
NL_CACHE = Path("runs_gen/nl_chunks.pt")


def build_tokenizer_and_corpus():
    """Train a 5k BPE on wikitext-103 and pre-tokenize into 256-token chunks (cached)."""
    if NL_CACHE.exists():
        return torch.load(NL_CACHE)
    from datasets import load_dataset
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train[:6%]")
    texts = [t for t in ds["text"] if len(t) > 64]
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    trainer = trainers.BpeTrainer(vocab_size=V_NL, special_tokens=["[UNK]"])
    tok.train_from_iterator((t for t in texts), trainer=trainer)
    tok.save(str(TOK_PATH))
    ids = []
    for t in texts:
        ids.extend(tok.encode(t).ids)
    ids = torch.tensor(ids[: (len(ids) // N_CTX) * N_CTX]).view(-1, N_CTX)
    torch.save(ids, NL_CACHE)
    print(f"NL corpus: {ids.shape[0]} chunks of {N_CTX} tokens, vocab {V_NL}", flush=True)
    return ids


def node_walk(pool, n_seq, gen):
    """walk_pool but node labels are drawn from the reserved node id range."""
    toks, nodes, perm, pick = walk_pool(pool, n_seq, gen)
    return toks + NODE0, nodes, perm, pick


if __name__ == "__main__":
    device = "cuda"
    nl_chunks = build_tokenizer_and_corpus()
    results = {}
    grid_pool = TRAIN_POOLS["grid"]
    dring_pool = TRAIN_POOLS["dring"]
    for arch, kwargs, n_layer in (
        ("softmax-add-3L", dict(attention="softmax", residual="add"), 3),
        ("bilin-lerp-2L", dict(attention="bilinear", residual="lerp"), 2),
    ):
        seed = 0
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed + 600)
        nl_gen = torch.Generator().manual_seed(seed + 700)
        name = f"{arch}-grid+dring+NL-seed{seed}"
        if (Path("runs_gen") / name / "model.pt").exists():
            print(f"skip {name} (exists)", flush=True)
            continue
        model = CycleModel(V_TOTAL, 128, 1, n_layer, N_CTX, **kwargs).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        steps = 24000
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min((s + 1) / 100, 0.5 * (1 + math.cos(math.pi * s / steps))))
        for step in range(steps):
            gt = node_walk(grid_pool, 42, gen)[0]
            dr = node_walk(dring_pool, 42, gen)[0]
            idx = torch.randint(0, nl_chunks.size(0), (42,), generator=nl_gen)
            nl = nl_chunks[idx]
            tokens = torch.cat([gt, dr, nl]).to(device)
            logits = model(tokens[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, V_TOTAL), tokens[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if step % 6000 == 0:
                print(f"{name} step {step} loss {loss.item():.3f}", flush=True)
        out = Path("runs_gen") / name
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out / "model.pt")
        (out / "history.json").write_text(json.dumps({"config": {
            "d_model": 128, "n_layer": n_layer, "scale": 0.5, "norm": False,
            "n_vocab": V_TOTAL, "node0": NODE0, **kwargs}}))

        # measure grid organization: mean_reps expects node tokens; monkeypatch offset
        model.cpu().eval()
        import icl_reps
        from icl_reps import gram_adjacency_corr, adjacency, pc_spectrum_alignment
        torch.set_grad_enabled(True)   # icl_reps disables grads globally at import
        from geodata import walk_batch
        SHAPE = icl_reps.SHAPE
        gwb = torch.Generator().manual_seed(21)
        _, gnodes, gperm = walk_batch(512, SHAPE, "grid", gwb)
        stream = model.residuals(gperm[0][gnodes] + NODE0)[-1]
        reps = {}
        for t in icl_reps.CONTEXTS:
            lo = max(0, t - icl_reps.WINDOW)
            wn = gnodes[:, lo:t].reshape(-1)
            flat = stream[:, lo:t].reshape(-1, stream.size(-1))
            reps[t] = torch.stack([flat[wn == v].mean(0) for v in range(icl_reps.N_NODES)])
        A = adjacency("grid")
        c8, c256 = gram_adjacency_corr(reps[8], A), gram_adjacency_corr(reps[256], A)
        var, corr = pc_spectrum_alignment(reps[256], "grid")
        # legal rate on grid walks (node ids offset)
        g2 = torch.Generator().manual_seed(999)
        toks, nodes, perm, pick = node_walk(grid_pool, 128, g2)
        nb = torch.stack([grid_pool[i][0] for i in pick.tolist()])
        legal = F.pad(legal_tokens(nodes, perm, nb), (NODE0, V_TOTAL - N_VOCAB - NODE0))
        with torch.no_grad():
            lg = model(toks[:, :-1])
        hit = legal[:, :-1].gather(-1, lg.argmax(-1, keepdim=True)).squeeze(-1)
        legal_rate = hit[:, 128:].float().mean().item()
        print(f"== {name}: grid org ctx8 {c8:+.2f} ctx256 {c256:+.2f}  legal {legal_rate:.2f}  "
              f"PC12-harmonic corr {corr[0]:.2f}/{corr[1]:.2f}", flush=True)
        results[name] = {"org8": c8, "org256": c256, "legal": legal_rate,
                         "pc_corr": corr[:4], "pc_var": var[:4]}
        model.to(device)
        Path("runs_gen/nlmix_results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
