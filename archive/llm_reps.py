"""Park-protocol representation analysis on pretrained LLMs.

Random walks on token-labeled graphs (nodes = random common English words, one fixed
labeling per model), fed as plain word sequences. Measures, per layer x context:
  - organization (Gram-adjacency correlation of windowed mean reps, same as icl_reps)
  - behavior (is the LLM's top node-word prediction a graph neighbor?)
  - ownU/nbrU content coefficients in the model's own unembedding basis.

Usage: python llm_reps.py <hf_model_name> [--walks 96] [--8bit]
Writes runs_llm/<tag>/{org.json,behavior.json,coeffs.json,reps.pt,meta.json}
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

CONTEXTS = (8, 16, 32, 64, 128, 256, 400)
WINDOW = 50
N_STEPS = 400

# ~340 common words; per-model we keep those that are a single token with a leading space
WORDS = """time year people way day man thing woman life child world school state family
student group country problem hand part place case week company system program question
work night point home water room mother area money story fact month lot right study book
eye job word business issue side kind head house service friend father power hour game
line end member law car city name team minute idea body back parent face others level
office door health person art war history party result change morning reason research
girl guy moment air teacher force education foot boy age policy process music market
sense nation plan college interest death experience effect use class control care field
development role effort rate heart drug show leader light voice wife police mind price
report decision son view relationship town road arm difference value building action
model season society tax director position player record paper space ground form event
official matter center couple site project activity star table need court oil situation
cost industry figure street tree image phone data picture practice piece land product
doctor wall patient worker news test movie north love support technology bank military
current pressure security stage nature fire bed rule fish town animal machine wood
window bird chance dog dinner village energy weight future stone hope pain letter
mountain island computer summer winter language science glass king queen river snow
garden bridge doctor army castle beach forest flower ocean spring autumn cloud rain
thunder valley desert engine wheel button copper silver golden shadow candle mirror
bottle basket ladder hammer needle pencil carpet curtain pillow blanket kitchen ceiling
corner market temple palace prison harbor tunnel tower cellar meadow orchard pasture""".split()


def build_graph(name: str):
    if name == "grid45":
        rows, cols = 4, 5
        nbrs = []
        for r in range(rows):
            for c in range(cols):
                cur = []
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        cur.append(rr * cols + cc)
                nbrs.append(cur)
        return nbrs
    if name.startswith("ring"):
        n = int(name[4:])
        return [[(v - 1) % n, (v + 1) % n] for v in range(n)]
    raise ValueError(name)


def sample_walks(nbrs, n_walks, generator):
    n = len(nbrs)
    deg = torch.tensor([len(x) for x in nbrs])
    table = torch.full((n, int(deg.max())), -1, dtype=torch.long)
    for v, x in enumerate(nbrs):
        table[v, : len(x)] = torch.tensor(x)
    nodes = torch.empty(n_walks, N_STEPS, dtype=torch.long)
    nodes[:, 0] = torch.randint(n, (n_walks,), generator=generator)
    for t in range(1, N_STEPS):
        cur = nodes[:, t - 1]
        pick = (torch.rand(n_walks, generator=generator) * deg[cur]).long()
        nodes[:, t] = table[cur, pick]
    return nodes


def single_token_words(tokenizer):
    keep = []
    for w in dict.fromkeys(WORDS):
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1:
            keep.append((w, ids[0]))
    return keep


@torch.no_grad()
def run_model(model_name: str, n_walks: int, use_8bit: bool, batch_size: int):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tag = model_name.split("/")[-1]
    out_dir = Path("runs_llm") / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = {"torch_dtype": torch.bfloat16, "device_map": "cuda"}
    if use_8bit:
        from transformers import BitsAndBytesConfig
        kwargs = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True), "device_map": "cuda"}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()

    pool = single_token_words(tokenizer)
    print(f"{tag}: {len(pool)} single-token words in pool")
    bos = tokenizer.bos_token_id
    off = 1 if bos is not None else 0

    results = {"org": {}, "behavior": {}, "coeffs": {}, "meta": {"model": model_name,
               "n_walks": n_walks, "words": {}, "n_layers": model.config.num_hidden_layers}}
    reps_store = {}

    for graph in ("grid45", "ring12", "ring7"):
        nbrs = build_graph(graph)
        n = len(nbrs)
        gen = torch.Generator().manual_seed(0)
        # fixed labeling: node v -> word pool[perm[v]] for ALL walks (matches toy protocol)
        perm = torch.randperm(len(pool), generator=gen)[:n]
        words = [pool[i][0] for i in perm]
        node_ids = torch.tensor([pool[i][1] for i in perm], device="cuda")
        results["meta"]["words"][graph] = words
        A = torch.zeros(n, n)
        for v, x in enumerate(nbrs):
            A[v, x] = 1.0

        walks = sample_walks(nbrs, n_walks, gen)
        ids = node_ids.cpu()[walks]
        if off:
            ids = torch.cat([torch.full((n_walks, 1), bos, dtype=torch.long), ids], 1)

        L = model.config.num_hidden_layers + 1
        D = model.config.hidden_size
        sums = {t: torch.zeros(L, n, D, device="cuda") for t in CONTEXTS}
        counts = {t: torch.zeros(n, device="cuda") for t in CONTEXTS}
        legal_top = {t: [0, 0] for t in CONTEXTS}   # [hits, total]
        nbr_mass, node_mass = {t: 0.0 for t in CONTEXTS}, {t: 0.0 for t in CONTEXTS}

        is_nbr = A.bool().cuda()
        for b0 in range(0, n_walks, batch_size):
            batch_ids = ids[b0:b0 + batch_size].cuda()
            batch_nodes = walks[b0:b0 + batch_size].cuda()
            out = model(batch_ids, output_hidden_states=True)
            hs = out.hidden_states
            logits = out.logits.float()
            lse = torch.logsumexp(logits, -1)
            node_logits = logits[..., node_ids]          # B x T x n
            node_probs = (node_logits - lse.unsqueeze(-1)).exp()
            for t in CONTEXTS:
                lo = max(0, t - WINDOW)
                wn = batch_nodes[:, lo:t].reshape(-1)
                counts[t] += torch.bincount(wn, minlength=n).float()
                for l in range(L):
                    flat = hs[l][:, lo + off:t + off].reshape(-1, D).float()
                    sums[t][l].index_add_(0, wn, flat)
                # behavior in the same window: predict step p+1 from position p
                cur = batch_nodes[:, lo:t - 1]
                pred = node_logits[:, lo + off:t - 1 + off].argmax(-1)
                legal_top[t][0] += is_nbr[cur, pred].sum().item()
                legal_top[t][1] += cur.numel()
                probs = node_probs[:, lo + off:t - 1 + off]
                nbr_mass[t] += (probs * is_nbr[cur].float()).sum().item()
                node_mass[t] += probs.sum().item()

        org = torch.zeros(L, len(CONTEXTS))
        for ti, t in enumerate(CONTEXTS):
            H_all = sums[t] / counts[t].clamp(min=1).unsqueeze(-1)
            for l in range(L):
                Hc = H_all[l].cpu() - H_all[l].cpu().mean(0)
                offd = ~torch.eye(n, dtype=torch.bool)
                org[l, ti] = torch.corrcoef(torch.stack([(Hc @ Hc.T)[offd], A[offd]]))[0, 1]
        results["org"][graph] = org.tolist()
        results["behavior"][graph] = {
            "legal_top": {t: legal_top[t][0] / max(1, legal_top[t][1]) for t in CONTEXTS},
            "nbr_mass_frac": {t: nbr_mass[t] / max(1e-9, node_mass[t]) for t in CONTEXTS},
            "node_mass": {t: node_mass[t] / max(1, legal_top[t][1]) for t in CONTEXTS},
        }
        reps_store[graph] = (sums[max(CONTEXTS)] / counts[max(CONTEXTS)].clamp(min=1).unsqueeze(-1)).half().cpu()

        # ownU / nbrU at each layer, final context, in the model's unembedding basis
        U = model.get_output_embeddings().weight[node_ids].float()
        U = (U - U.mean(0)).cpu()
        Unbr = (A @ U) / A.sum(1, keepdim=True)
        coeffs = []
        H_all = (sums[max(CONTEXTS)] / counts[max(CONTEXTS)].clamp(min=1).unsqueeze(-1)).cpu().float()
        for l in range(L):
            Hc = H_all[l] - H_all[l].mean(0)
            X = torch.stack([U.reshape(-1), Unbr.reshape(-1)], 1)
            sol = torch.linalg.lstsq(X, Hc.reshape(-1, 1)).solution.squeeze()
            coeffs.append({"ownU": sol[0].item(), "nbrU": sol[1].item()})
        results["coeffs"][graph] = coeffs
        best = org[:, -1].argmax().item()
        print(f"  {graph}: org@400 last-layer {org[-1, -1]:+.2f}, best layer {best} "
              f"{org[best, -1]:+.2f}, legal_top@400 {results['behavior'][graph]['legal_top'][400]:.2f}")

    for key in ("org", "behavior", "coeffs", "meta"):
        (out_dir / f"{key}.json").write_text(json.dumps(results[key]))
    torch.save(reps_store, out_dir / "reps.pt")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--walks", type=int, default=96)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--8bit", dest="use_8bit", action="store_true")
    args = ap.parse_args()
    run_model(args.model, args.walks, args.use_8bit, args.batch)
