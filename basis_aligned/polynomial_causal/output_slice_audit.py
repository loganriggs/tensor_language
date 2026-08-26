"""Behavior-agnostic output bases as circuit-discovery tools.

Output bases are learned from discovery classes or from the unembedding spectrum.
Evaluation classes are disjoint. At matched basis rank, projected class directions
rank attention heads by the existing weights-only score. The rank-8 winner selected
only on discovery recall is frozen, then tested on held-out circuit-head recall and
selective removal against optimal constants.

Registered predictions:
  A. The discovery-selected learned basis reaches >= .50 of oracle recall@5 on
     evaluation classes and >= 2x the random-basis recall.
  B. Its predicted top-5 removals retain >= .50 of oracle class damage and have
     >= 2x random selectivity, averaged over powered evaluation classes.
  C. Head-recall ordering across methods agrees in sign between discovery and a
     second, disjoint set of evaluation classes.

This script is GPU-ready but does not enqueue itself.
"""

import json
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import tiktoken

HERE = Path(__file__).resolve().parent
BQ = HERE.parent / "bilinear_quotient"
QK = HERE.parent / "qk_mdl"
THESEUS = Path("/workspace/theseus-bench")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BQ))
sys.path.insert(0, str(QK))

from data import fineweb_rows
from tier2_model import load_elriggs, reference_forward

DEV = "cuda"
D = 1152
T = 256
BATCH = 8
NR = 240
BUDGETS = (2, 4, 8)
OUT = HERE / "output_slice_audit_results.json"
ENC = tiktoken.get_encoding("gpt2")

PATTERNS = {
    "question": r"^\?$| \?$",
    "pronouns": r"^ (he|she|they|He|She|They)$",
    "months": r"^ (January|February|March|April|May|June|July|August|September|October|November|December)$",
    "digits": r"^ ?[0-9]+$",
    "comma": r"^,$|^ ,$",
    "the": r"^ the$",
    "is": r"^ is$",
    "and": r"^ and$",
    "colon": r"^:$|^ :$",
    "semicolon": r"^;$|^ ;$",
    "dollar": r"^\$$|^ \$$",
    "open_paren": r"^\($|^ \($",
    "close_paren": r"^\)$|^ \)$",
    "to": r"^ to$",
    "said": r"^ said$",
    "days": r"^ (Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)$",
}
DISCOVERY = ("question", "pronouns", "months", "digits", "comma", "the", "is", "and")
EVALUATION = ("colon", "semicolon", "dollar", "open_paren", "close_paren", "to", "said", "days")


def vocab_mask(pattern):
    mask = torch.zeros(50257, dtype=torch.bool)
    for token in range(50257):
        if re.match(pattern, ENC.decode([token])):
            mask[token] = True
    return mask


def orthogonal_basis(columns, rank):
    u, _, _ = torch.linalg.svd(columns, full_matrices=False)
    return u[:, :rank]


def class_direction(unembed, mask):
    direction = unembed[mask.to(DEV)].mean(0)
    return direction / direction.norm().clamp_min(1e-12)


def projected(direction, basis):
    result = basis @ (basis.T @ direction)
    return result / result.norm().clamp_min(1e-12)


def head_ranking(model, direction):
    scores = torch.zeros(18, 9)
    for layer, block in enumerate(model.transformer.h):
        weight = block.attn.c_proj.weight.float()
        for head in range(9):
            scores[layer, head] = (direction @ weight[:, head * 128:(head + 1) * 128]).norm()
    order = scores.flatten().argsort(descending=True)
    return [f"{int(index) // 9}.{int(index) % 9}" for index in order.tolist()]


def ranking_metrics(ranking, truth, k=5):
    truth = set(truth)
    top = ranking[:k]
    hits = sum(head in truth for head in top)
    first = next((i + 1 for i, head in enumerate(ranking) if head in truth), len(ranking) + 1)
    return {"top5": top, "recall_at_5": hits / max(min(k, len(truth)), 1),
            "precision_at_5": hits / k, "reciprocal_rank": 1 / first}


def mean_metric(report, classes, key):
    return sum(report[name][key] for name in classes) / len(classes)


@torch.no_grad()
def removal_measure(model, rows, class_mask, heads, constants):
    selected = {tuple(map(int, head.split("."))) for head in heads}
    hooks = []

    def make_hook(layer):
        def hook(module, args):
            layer_heads = [head for current, head in selected if current == layer]
            if not layer_heads:
                return None
            value = args[0].clone()
            for head in layer_heads:
                value[:, :, head * 128:(head + 1) * 128] = constants[
                    f"head{layer}.{head}"
                ].to(DEV, dtype=value.dtype)
            return (value,) + tuple(args[1:])
        return hook

    for layer, block in enumerate(model.transformer.h):
        hooks.append(block.attn.c_proj.register_forward_pre_hook(make_hook(layer)))
    class_sum = background_sum = 0.0
    class_n = background_n = 0
    for start in range(0, len(rows), BATCH):
        batch = rows[start:start + BATCH].to(DEV)
        logits = reference_forward(model, batch[:, :-1]).float()
        targets = batch[:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                             reduction="none").view_as(targets)
        valid = torch.ones_like(targets, dtype=torch.bool)
        valid[:, :64] = False
        cmask = class_mask.to(DEV)[targets] & valid
        bmask = valid & ~cmask
        class_sum += float(ce[cmask].sum())
        background_sum += float(ce[bmask].sum())
        class_n += int(cmask.sum())
        background_n += int(bmask.sum())
    for hook in hooks:
        hook.remove()
    return {"class_ce": class_sum / max(class_n, 1),
            "background_ce": background_sum / max(background_n, 1),
            "class_positions": class_n}


@torch.no_grad()
def main():
    started = time.time()
    model, _ = load_elriggs("bilin18")
    unembed = model.lm_head.weight.float().to(DEV)[:50257]
    registry = json.loads((THESEUS / "registry/circuits.json").read_text())["certified"]
    truth = {name: registry[name]["heads"] for name in DISCOVERY + EVALUATION}
    masks = {name: vocab_mask(PATTERNS[name]) for name in DISCOVERY + EVALUATION}
    directions = {name: class_direction(unembed, masks[name])
                  for name in DISCOVERY + EVALUATION}

    discovery_matrix = torch.stack([directions[name] for name in DISCOVERY], dim=1)
    class_basis_full = orthogonal_basis(discovery_matrix, max(BUDGETS))
    _, _, output_pca_full = torch.pca_lowrank(unembed, q=max(BUDGETS), center=True, niter=4)
    gen = torch.Generator(device=DEV).manual_seed(0)
    random_full, _ = torch.linalg.qr(torch.randn(D, max(BUDGETS), device=DEV, generator=gen))
    full_bases = {"class_trained": class_basis_full,
                  "output_pca": output_pca_full[:, :max(BUDGETS)],
                  "random": random_full}

    rankings = {}
    for budget in BUDGETS:
        rankings[str(budget)] = {}
        for method, full_basis in full_bases.items():
            basis = full_basis[:, :budget]
            rankings[str(budget)][method] = {}
            for name in DISCOVERY + EVALUATION:
                rank = head_ranking(model, projected(directions[name], basis))
                rankings[str(budget)][method][name] = ranking_metrics(rank, truth[name])
        rankings[str(budget)]["oracle"] = {}
        for name in DISCOVERY + EVALUATION:
            rank = head_ranking(model, directions[name])
            rankings[str(budget)]["oracle"][name] = ranking_metrics(rank, truth[name])

    budget8 = rankings["8"]
    learned = ("class_trained", "output_pca")
    winner = max(learned, key=lambda method: mean_metric(budget8[method], DISCOVERY,
                                                         "recall_at_5"))

    rows = fineweb_rows(NR, skip=11000)[:, :T + 1].contiguous()
    constants = torch.load(BQ / "opt_ablation_consts_all.pt", map_location="cpu")
    removals = {}
    powered = []
    for name in EVALUATION:
        clean = removal_measure(model, rows, masks[name], [], constants)
        removals[name] = {"clean": clean}
        if clean["class_positions"] >= 40:
            powered.append(name)
        for method in (winner, "random", "oracle"):
            heads = budget8[method][name]["top5"]
            measured = removal_measure(model, rows, masks[name], heads, constants)
            class_rise = measured["class_ce"] - clean["class_ce"]
            background_rise = measured["background_ce"] - clean["background_ce"]
            removals[name][method] = {"heads": heads, "class_rise": class_rise,
                                      "background_rise": background_rise,
                                      "selectivity": class_rise / max(background_rise, 1e-6)}

    eval_recall = mean_metric(budget8[winner], EVALUATION, "recall_at_5")
    oracle_recall = mean_metric(budget8["oracle"], EVALUATION, "recall_at_5")
    random_recall = mean_metric(budget8["random"], EVALUATION, "recall_at_5")
    winner_damage = sum(removals[name][winner]["class_rise"] for name in powered) / max(len(powered), 1)
    oracle_damage = sum(removals[name]["oracle"]["class_rise"] for name in powered) / max(len(powered), 1)
    winner_sel = sum(removals[name][winner]["selectivity"] for name in powered) / max(len(powered), 1)
    random_sel = sum(removals[name]["random"]["selectivity"] for name in powered) / max(len(powered), 1)

    pred_a = eval_recall >= 0.5 * oracle_recall and eval_recall >= 2 * random_recall
    pred_b = winner_damage >= 0.5 * oracle_damage and winner_sel >= 2 * random_sel
    discovery_gap = (mean_metric(budget8[winner], DISCOVERY, "recall_at_5")
                     - mean_metric(budget8["random"], DISCOVERY, "recall_at_5"))
    evaluation_gap = eval_recall - random_recall
    pred_c = discovery_gap * evaluation_gap > 0
    output = {"config": {"discovery": DISCOVERY, "evaluation": EVALUATION,
                          "budgets": BUDGETS, "removal_rows": NR},
              "selected_method": winner, "rankings": rankings,
              "removals": removals, "powered_removal_classes": powered,
              "summary": {"evaluation_recall": eval_recall,
                          "oracle_recall": oracle_recall,
                          "random_recall": random_recall,
                          "winner_class_rise": winner_damage,
                          "oracle_class_rise": oracle_damage,
                          "winner_selectivity": winner_sel,
                          "random_selectivity": random_sel},
              "predictions": {"recall_win": pred_a, "removal_win": pred_b,
                              "ordering_generalizes": pred_c},
              "runtime_s": time.time() - started}
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["summary"], indent=2), flush=True)
    print(json.dumps(output["predictions"], indent=2), flush=True)
    print(f"wrote {OUT} in {output['runtime_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
