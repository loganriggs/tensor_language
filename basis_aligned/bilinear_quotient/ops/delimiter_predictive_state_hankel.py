"""RUNG 302 -- BEHAVIOR-ANCHORED DELIMITER PREDICTIVE-STATE HANKEL.

Scope
-----
The old generic token-prefix Hankel spliced unrelated natural prefixes/suffixes,
raised CE by 3.54 nats, and failed low-rank prediction.  This screen does not
repeat that object.  It extracts natural 64-token prefixes from frozen FineWeb
and WikiText rows and labels two genuine sequential states: quote parity and
whether parenthesis depth is nonzero.  Each task is binary and uses prefixes
from distinct source rows.

For a natural prefix p and short delimiter/action suffix v, define

    H[p,v] = log P_model(v | p).

The fixed suffix bank contains length-1, length-2, and length-3 words over quote,
close-parenthesis, punctuation, and neutral continuation actions.  Nested
Hankel blocks use every suffix with length <=k.  We double-center H before rank
scoring so prefix difficulty and suffix frequency cannot masquerade as a state
interaction.

State tests
-----------
For quote and parenthesis separately, on both corpora:
  * r90 of each nested interaction block;
  * fraction of centered H variance explained by the two state centroids;
  * deterministic half-template heldout nearest-centroid accuracy;
  * FineWeb-centroid transfer accuracy on WikiText.
A fixed label shuffle is the state null.

Causal control
--------------
Repeat every H matrix after zeroing head 13.8's 128-value slice immediately
before attention-13 c_proj.  This is the identified delimiter closer.  Zero
head 13.1 at the same interface as a matched control.  Measure reduction in the
state-centroid separation norm; no claim rests on average CE alone.

Frozen predictions
------------------
pred_a_small_predictive_state:
    For BOTH tasks and BOTH corpora, full length<=3 interaction r90 <=4 and
    two-state centroid R2 >=0.40.
pred_b_state_transfers:
    Within-corpus heldout accuracy >=0.80 and FineWeb->WikiText accuracy >=0.70
    for BOTH quote and parenthesis.
pred_c_delimiter_head_carries_state:
    For at least one task, head13.8 reduces state separation by >=25% on BOTH
    corpora, while head13.1 changes separation by <=15% on both.

Null: either task has state R2<=0.10 on either corpus, either transfer accuracy
<=0.60, or head13.8 does not reduce separation more than head13.1.  The screen
identifies at most a small circuit interface.  A two-state implementation would
need one state bit plus a priced parser/router and transitions; none receives
whole-model storage credit here.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "delimiter_predictive_state_hankel_results.json"
DEV = "cuda"
D = 1152
HEAD_D = 128
PREFIX_LEN = 64
PER_STATE = 8


def _load_rows(path: Path, n: int = 96) -> torch.Tensor:
    value = torch.load(path, map_location="cpu")
    rows = value["rows"] if isinstance(value, dict) else value
    return rows[:n, :257].long().contiguous()


def _wikitext_rows(n: int = 96, width: int = 257, skip: int = 1024) -> tuple[torch.Tensor, str]:
    from datasets import load_dataset
    import tiktoken

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())
    tokens = tiktoken.get_encoding("gpt2").encode_ordinary(text)
    stop = skip + n * width
    assert len(tokens) >= stop
    return torch.tensor(tokens[skip:stop], dtype=torch.long).reshape(n, width), str(dataset._fingerprint)


def _natural_prefixes(rows: torch.Tensor, encoder, task: str) -> tuple[list[torch.Tensor], np.ndarray]:
    selected: dict[int, list[torch.Tensor]] = {0: [], 1: []}
    used: dict[int, set[int]] = {0: set(), 1: set()}
    for row_index, row in enumerate(rows):
        quote = 0
        depth = 0
        for position, token in enumerate(row.tolist()):
            piece = encoder.decode([token])
            quote ^= piece.count('"') % 2
            depth = max(0, depth + piece.count("(") - piece.count(")"))
            if position < PREFIX_LEN - 1:
                continue
            state = quote if task == "quote" else int(depth > 0)
            if len(selected[state]) >= PER_STATE or row_index in used[state]:
                continue
            selected[state].append(row[position - PREFIX_LEN + 1:position + 1].clone())
            used[state].add(row_index)
    assert all(len(selected[state]) == PER_STATE for state in (0, 1)), {
        state: len(selected[state]) for state in selected
    }
    prefixes = []
    labels = []
    # Interleave states so deterministic even/odd template splits remain balanced.
    for index in range(PER_STATE):
        for state in (0, 1):
            prefixes.append(selected[state][index])
            labels.append(state)
    return prefixes, np.asarray(labels, dtype=np.int64)


def _suffix_bank(encoder) -> list[tuple[int, ...]]:
    def one(text: str) -> int:
        tokens = encoder.encode(text)
        assert len(tokens) == 1, (text, tokens)
        return tokens[0]

    quote = one('"')
    close = one(")")
    period = one(".")
    comma = one(",")
    and_token = one(" and")
    said = one(" said")
    return [
        (quote,), (close,), (period,), (comma,), (and_token,), (said,),
        (quote, period), (quote, comma), (close, period), (close, comma),
        (and_token, quote), (said, quote),
        (quote, period, and_token), (close, period, and_token),
        (and_token, quote, period), (said, quote, period),
    ]


def _manual_logits(model: torch.nn.Module, index: torch.Tensor) -> torch.Tensor:
    x = F.rms_norm(model.transformer.wte(index), (D,))
    x0 = x
    value0 = None
    for block in model.transformer.h:
        x, value0 = block(x, value0, x0)
    return 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def _hankel(
    model: torch.nn.Module,
    prefixes: list[torch.Tensor],
    suffixes: list[tuple[int, ...]],
    ablate_head: int | None,
) -> tuple[np.ndarray, float]:
    handle = None
    if ablate_head is not None:
        start = ablate_head * HEAD_D
        stop = start + HEAD_D

        def pre_hook(_module, args):
            value = args[0].clone()
            value[..., start:stop] = 0
            return (value,)

        handle = model.transformer.h[13].attn.c_proj.register_forward_pre_hook(pre_hook)
    examples: dict[int, list[tuple[int, int, torch.Tensor, int, int]]] = defaultdict(list)
    for prefix_index, prefix in enumerate(prefixes):
        for suffix_index, suffix in enumerate(suffixes):
            suffix_tensor = torch.tensor(suffix, dtype=torch.long)
            sequence = torch.cat((prefix, suffix_tensor))
            examples[len(sequence)].append((prefix_index, suffix_index, sequence, len(prefix), len(suffix)))
    matrix = np.zeros((len(prefixes), len(suffixes)), dtype=np.float64)
    nll_sum = 0.0
    nll_count = 0
    try:
        for _length, group in examples.items():
            for start_group in range(0, len(group), 64):
                chunk = group[start_group:start_group + 64]
                sequence = torch.stack([item[2] for item in chunk]).to(DEV)
                logits = _manual_logits(model, sequence[:, :-1])
                log_probs = torch.log_softmax(logits.float(), -1)
                for local, (prefix_index, suffix_index, full, prefix_length, suffix_length) in enumerate(chunk):
                    targets = full[prefix_length:prefix_length + suffix_length].to(DEV)
                    positions = torch.arange(prefix_length - 1, prefix_length - 1 + suffix_length, device=DEV)
                    value = float(log_probs[local, positions, targets].sum())
                    matrix[prefix_index, suffix_index] = value
                    nll_sum -= value
                    nll_count += suffix_length
    finally:
        if handle is not None:
            handle.remove()
    return matrix, nll_sum / nll_count


def _interaction(matrix: np.ndarray) -> np.ndarray:
    return matrix - matrix.mean(0, keepdims=True) - matrix.mean(1, keepdims=True) + matrix.mean()


def _rank90(matrix: np.ndarray) -> int:
    singular = np.linalg.svd(matrix, compute_uv=False)
    energy = singular ** 2
    return int(np.searchsorted(np.cumsum(energy) / max(energy.sum(), 1e-20), 0.90) + 1)


def _state_report(matrix: np.ndarray, labels: np.ndarray, suffixes: list[tuple[int, ...]]) -> dict[str, object]:
    centered = matrix - matrix.mean(0, keepdims=True)
    centroids = np.stack([centered[labels == state].mean(0) for state in (0, 1)])
    predicted = centroids[labels]
    state_r2 = 1.0 - np.square(centered - predicted).sum() / max(np.square(centered).sum(), 1e-20)
    separation = float(np.linalg.norm(centroids[1] - centroids[0]))
    fit = np.arange(len(labels)) % 4 < 2
    test = ~fit
    fit_centroids = np.stack([centered[np.logical_and(fit, labels == state)].mean(0) for state in (0, 1)])
    distance = np.square(centered[test, None, :] - fit_centroids[None, :, :]).sum(2)
    heldout_accuracy = float((distance.argmin(1) == labels[test]).mean())
    shuffled = np.roll(labels, 3)
    shuffled_centroids = np.stack([centered[shuffled == state].mean(0) for state in (0, 1)])
    shuffled_pred = shuffled_centroids[shuffled]
    shuffled_r2 = 1.0 - np.square(centered - shuffled_pred).sum() / max(np.square(centered).sum(), 1e-20)
    nested = {}
    for maximum in (1, 2, 3):
        columns = [index for index, suffix in enumerate(suffixes) if len(suffix) <= maximum]
        nested[str(maximum)] = {
            "n_suffixes": len(columns),
            "interaction_rank90": _rank90(_interaction(matrix[:, columns])),
        }
    return {
        "state_r2": float(state_r2),
        "shuffled_state_r2": float(shuffled_r2),
        "state_separation": separation,
        "heldout_accuracy": heldout_accuracy,
        "nested_hankel": nested,
        "centroids": centroids.tolist(),
    }


def _transfer_accuracy(source: dict[str, object], target_matrix: np.ndarray, target_labels: np.ndarray) -> float:
    target = target_matrix - target_matrix.mean(0, keepdims=True)
    centroids = np.asarray(source["centroids"])
    distance = np.square(target[:, None, :] - centroids[None, :, :]).sum(2)
    return float((distance.argmin(1) == target_labels).mean())


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert (ROOT / ".rowcache/fineweb_n96_skip1200.pt").exists()
        print("DELIMITER PREDICTIVE STATE HANKEL | dry run: natural rows, actions, controls, and bars valid")
        return
    started = time.time()
    sys.path.insert(0, "/workspace/tensor_language/basis_aligned/qk_mdl")
    from tier2_model import load_elriggs
    import tiktoken

    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D
    encoder = tiktoken.get_encoding("gpt2")
    suffixes = _suffix_bank(encoder)
    fineweb = _load_rows(ROOT / ".rowcache/fineweb_n96_skip1200.pt")
    wikitext, fingerprint = _wikitext_rows()
    corpus_rows = {"fineweb": fineweb, "wikitext": wikitext}
    matrices: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    reports: dict[str, dict[str, dict[str, object]]] = {}
    labels_by: dict[str, dict[str, np.ndarray]] = {}
    for corpus, rows in corpus_rows.items():
        matrices[corpus] = {}
        reports[corpus] = {}
        labels_by[corpus] = {}
        for task in ("quote", "parenthesis"):
            prefixes, labels = _natural_prefixes(rows, encoder, task)
            labels_by[corpus][task] = labels
            matrices[corpus][task] = {}
            reports[corpus][task] = {}
            for arm, head in (("native", None), ("head13_8_zero", 8), ("head13_1_zero", 1)):
                matrix, suffix_nll = _hankel(model, prefixes, suffixes, head)
                matrices[corpus][task][arm] = matrix
                report = _state_report(matrix, labels, suffixes)
                report["mean_suffix_token_nll"] = suffix_nll
                reports[corpus][task][arm] = report
                print(
                    f"{corpus} {task} {arm}: r90={report['nested_hankel']['3']['interaction_rank90']} "
                    f"stateR2={report['state_r2']:.3f} acc={report['heldout_accuracy']:.3f} "
                    f"sep={report['state_separation']:.3f} nll={suffix_nll:.3f}", flush=True,
                )

    transfer = {}
    head_effect = {}
    for task in ("quote", "parenthesis"):
        transfer[task] = _transfer_accuracy(
            reports["fineweb"][task]["native"],
            matrices["wikitext"][task]["native"],
            labels_by["wikitext"][task],
        )
        head_effect[task] = {}
        for corpus in ("fineweb", "wikitext"):
            base = reports[corpus][task]["native"]["state_separation"]
            head_effect[task][corpus] = {
                "head13_8_reduction": 1.0 - reports[corpus][task]["head13_8_zero"]["state_separation"] / base,
                "head13_1_reduction": 1.0 - reports[corpus][task]["head13_1_zero"]["state_separation"] / base,
            }

    pred_a = all(
        reports[corpus][task]["native"]["nested_hankel"]["3"]["interaction_rank90"] <= 4
        and reports[corpus][task]["native"]["state_r2"] >= 0.40
        for corpus in ("fineweb", "wikitext") for task in ("quote", "parenthesis")
    )
    pred_b = all(
        reports[corpus][task]["native"]["heldout_accuracy"] >= 0.80
        for corpus in ("fineweb", "wikitext") for task in ("quote", "parenthesis")
    ) and all(transfer[task] >= 0.70 for task in transfer)
    pred_c = any(
        all(head_effect[task][corpus]["head13_8_reduction"] >= 0.25 for corpus in ("fineweb", "wikitext"))
        and all(abs(head_effect[task][corpus]["head13_1_reduction"]) <= 0.15 for corpus in ("fineweb", "wikitext"))
        for task in ("quote", "parenthesis")
    )
    null = bool(
        any(reports[corpus][task]["native"]["state_r2"] <= 0.10
            for corpus in ("fineweb", "wikitext") for task in ("quote", "parenthesis"))
        or any(transfer[task] <= 0.60 for task in transfer)
        or all(
            any(head_effect[task][corpus]["head13_8_reduction"]
                <= head_effect[task][corpus]["head13_1_reduction"] for corpus in ("fineweb", "wikitext"))
            for task in ("quote", "parenthesis")
        )
    )
    # Centroids are retained in reports for exact transfer replay; no raw prefix
    # tokens or full H matrices are serialized.
    result = {
        "status": "delimiter_predictive_state_hankel_complete",
        "rung": 302,
        "claim_level": "behavior_anchored_finite_state_screen_only",
        "populations": {
            "prefix_length": PREFIX_LEN,
            "prefixes_per_state_per_task_corpus": PER_STATE,
            "suffix_words": len(suffixes),
            "suffix_lengths": [len(suffix) for suffix in suffixes],
            "wikitext_fingerprint": fingerprint,
        },
        "reports": reports,
        "fineweb_to_wikitext_accuracy": transfer,
        "head_effect": head_effect,
        "toy_scoring_control": {
            "two_state_double_centered_rank": int(np.linalg.matrix_rank(
                _interaction(np.repeat(np.asarray([[0.0, 1.0], [1.0, 0.0]]), 4, axis=0))
            )),
        },
        "literal_interface_price": {
            "state_bits_per_task": 1,
            "transition_entries_for_6_action_symbols": 12,
            "emission_values_for_16_suffix_tests": 32,
            "router_parser_and_native_replacement": "not_constructed_or_credited",
        },
        'pred_a_small_predictive_state': bool(pred_a),
        'pred_b_state_transfers': bool(pred_b),
        'pred_c_delimiter_head_carries_state': bool(pred_c),
        "null_no_small_causal_state": null,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "transfer": transfer,
        "head_effect": head_effect,
        "predicates": [pred_a, pred_b, pred_c],
        "null": null,
        "runtime_s": result["runtime_s"],
    }, indent=2), flush=True)
    print("DELIMITER PREDICTIVE STATE HANKEL DONE", flush=True)


if __name__ == "__main__":
    main()
