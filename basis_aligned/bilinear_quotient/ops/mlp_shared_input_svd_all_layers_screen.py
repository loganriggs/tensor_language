"""RUNG 318 -- ALL-LAYER SHARED-INPUT WEIGHT-SVD SCREEN.

Rungs 314--317 established a fully gated MLP0 result: concatenating its native
Left and Right maps and taking one shared right-singular subspace yields a
literal rank-768 input encoder with excellent contextual and causal transfer.
Test whether that is a repeated bilinear law or a front-layer exception.

For each of all 18 MLPs independently, factor A=[Left;Right] as

    z = V_p^T x; Left(x) = (Left V_p) z; Right(x) = (Right V_p) z

at p=512 and p=768.  Native Down, bias, and elementwise product remain.  This
is weight-only: neither corpus chooses a basis or layer.  Score untouched rows
176:188 from FineWeb skip11000 and 12 WikiText rows after token 80000.

Frozen predictions
------------------
pred_a_p768_is_broad:
    At least 9/18 p768 layers add <=.02 CE on BOTH corpora.
pred_b_p768_distribution_is_tight:
    Median p768 damage <=.012 on each corpus and the maximum p768 damage over
    every layer/corpus is <=.08.
pred_c_p512_has_a_real_lower_fidelity_regime:
    At least 6/18 p512 layers add <=.04 on BOTH corpora, and for every such
    layer matched p768 mean nonnegative damage is no worse than p512.

Null: at most 3 p768 layers qualify at <=.02 on both corpora, OR any p768
layer/corpus damage is >=.20.  This is an independent-site screen only.  No
adaptive layer selection follows it: if broad generality is earned, the first
physical composition is prospectively fixed to layers {0,8,17} at p768.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mlp_shared_input_svd_all_layers_screen_results.json"
DEV = "cuda"
D = 1152
H = 4608
LAYERS = 18
RANKS = (512, 768)
EVAL_ROWS = 12
FINEWEB_SLICE = (176, 188)
WIKI_SKIP = 80000
NATIVE_MLP = 3 * H * D + D


def _price(rank: int) -> int:
    return D * rank + 2 * H * rank + H * D + D


def _manual_logits(model, index):
    x = F.rms_norm(model.transformer.wte(index), (D,))
    x0 = x
    value0 = None
    for block in model.transformer.h:
        x, value0 = block(x, value0, x0)
    return 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def _score(model, rows, layer: int | None = None, program=None) -> float:
    handle = None
    if program is not None:
        encoder = program["encoder"]
        left = program["left"]
        right = program["right"]
        down = program["down"]
        bias = program["bias"]

        def hook(_module, args, output):
            x = args[0].float()
            z = x @ encoder.T
            hidden = (z @ left.T) * (z @ right.T)
            return (hidden @ down.T + bias).to(output.dtype)

        handle = model.transformer.h[layer].mlp.register_forward_hook(hook)
    total, count = 0.0, 0
    try:
        for start in range(0, len(rows), 2):
            batch = rows[start:start + 2]
            index, target = batch[:, :-1].to(DEV), batch[:, 1:].to(DEV)
            logits = _manual_logits(model, index)
            total += float(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                target.reshape(-1), reduction="sum"
            ))
            count += target.numel()
    finally:
        if handle is not None:
            handle.remove()
    return total / count


@torch.no_grad()
def _programs(mlp):
    left = mlp.Left.weight.detach().float()
    right = mlp.Right.weight.detach().float()
    down = mlp.Down.weight.detach().float()
    bias = mlp.Down_bias.detach().float()
    stacked = torch.cat((left, right), dim=0)
    gram = stacked.T @ stacked
    values, vectors = torch.linalg.eigh(0.5 * (gram + gram.T))
    order = torch.argsort(values, descending=True)
    values, vectors = values[order], vectors[:, order]
    total = values.clamp_min(0).sum()
    result = {}
    energy = {}
    for rank in RANKS:
        basis = vectors[:, :rank]
        coefficient = stacked @ basis
        result[rank] = {
            "encoder": basis.T.contiguous(),
            "left": coefficient[:H].contiguous(),
            "right": coefficient[H:].contiguous(),
            "down": down,
            "bias": bias,
        }
        energy[str(rank)] = float(values[:rank].clamp_min(0).sum() / total)
    return result, energy


def _median(values) -> float:
    return float(torch.tensor(values, dtype=torch.float64).median())


def _mean_nonnegative(arm) -> float:
    return (max(0.0, arm["fineweb_damage"]) + max(0.0, arm["wikitext_damage"])) / 2


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        cache = ROOT / ".rowcache/fineweb_n192_skip11000.pt"
        assert cache.exists() and FINEWEB_SLICE[1] <= 192
        assert _price(512) == 10_617_984 and NATIVE_MLP - _price(512) == 5_308_416
        assert _price(768) == 13_272_192 and NATIVE_MLP - _price(768) == 2_654_208
        assert set((0, 8, 17)) <= set(range(LAYERS))
        print("ALL-LAYER SHARED-INPUT SVD | dry run: rows, ranks, prices, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    from mlp0_signed_response_rank_screen import _wikitext_rows
    from tier2_model import load_elriggs

    cached = torch.load(ROOT / ".rowcache/fineweb_n192_skip11000.pt", map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    fineweb = cached[FINEWEB_SLICE[0]:FINEWEB_SLICE[1], :257].long().contiguous()
    assert fineweb.shape == (EVAL_ROWS, 257)
    wikitext, fingerprint = _wikitext_rows(EVAL_ROWS, skip=WIKI_SKIP)

    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D and len(model.transformer.h) == LAYERS
    native = {
        "fineweb": _score(model, fineweb),
        "wikitext": _score(model, wikitext),
    }
    arms = {}
    energy = {}
    for layer in range(LAYERS):
        programs, layer_energy = _programs(model.transformer.h[layer].mlp)
        energy[str(layer)] = layer_energy
        arms[str(layer)] = {}
        for rank in RANKS:
            program = programs[rank]
            for name, value in program.items():
                expected = {
                    "encoder": (rank, D), "left": (H, rank), "right": (H, rank),
                    "down": (D, H), "bias": (D,),
                }[name]
                assert tuple(value.shape) == expected
            ce_fw = _score(model, fineweb, layer, program)
            ce_wt = _score(model, wikitext, layer, program)
            arms[str(layer)][str(rank)] = {
                "fineweb_damage": ce_fw - native["fineweb"],
                "wikitext_damage": ce_wt - native["wikitext"],
                "saving_scalars_if_installed": NATIVE_MLP - _price(rank),
            }
            print(f"L{layer:02d} p{rank}: FW/WT "
                  f"{ce_fw-native['fineweb']:+.6f}/{ce_wt-native['wikitext']:+.6f}",
                  flush=True)
        del programs
        torch.cuda.empty_cache()

    p768_qualifying = [layer for layer in range(LAYERS)
                       if max(arms[str(layer)]["768"]["fineweb_damage"],
                              arms[str(layer)]["768"]["wikitext_damage"]) <= .02]
    p512_qualifying = [layer for layer in range(LAYERS)
                       if max(arms[str(layer)]["512"]["fineweb_damage"],
                              arms[str(layer)]["512"]["wikitext_damage"]) <= .04]
    p768_fw = [arms[str(layer)]["768"]["fineweb_damage"] for layer in range(LAYERS)]
    p768_wt = [arms[str(layer)]["768"]["wikitext_damage"] for layer in range(LAYERS)]
    p768_max = max(p768_fw + p768_wt)
    monotone_qualifiers = all(
        _mean_nonnegative(arms[str(layer)]["768"])
        <= _mean_nonnegative(arms[str(layer)]["512"]) + 1e-9
        for layer in p512_qualifying
    )
    pred_a = len(p768_qualifying) >= 9
    pred_b = _median(p768_fw) <= .012 and _median(p768_wt) <= .012 and p768_max <= .08
    pred_c = len(p512_qualifying) >= 6 and monotone_qualifiers
    null = len(p768_qualifying) <= 3 or p768_max >= .20

    result = {
        "status": "mlp_shared_input_svd_all_layers_screen_complete",
        "rung": 318,
        "claim_level": "independent_all_layer_weight_only_two_corpus_screen",
        "convention": "CE added above native; lower is better",
        "evaluation": {
            "fineweb_cache": "fineweb_n192_skip11000.pt",
            "fineweb_rows_half_open": list(FINEWEB_SLICE),
            "wikitext_skip": WIKI_SKIP,
            "wikitext_rows": EVAL_ROWS,
            "wikitext_fingerprint": str(fingerprint),
        },
        "native_ce": native,
        "weight_energy_capture": energy,
        "arms": arms,
        "p768_qualifying_layers": p768_qualifying,
        "p512_qualifying_layers": p512_qualifying,
        "p768_median_fineweb_damage": _median(p768_fw),
        "p768_median_wikitext_damage": _median(p768_wt),
        "p768_max_damage": p768_max,
        "p768_no_worse_on_every_p512_qualifier": monotone_qualifiers,
        'pred_a_p768_is_broad': bool(pred_a),
        'pred_b_p768_distribution_is_tight': bool(pred_b),
        'pred_c_p512_has_a_real_lower_fidelity_regime': bool(pred_c),
        "null_p768_is_not_a_repeated_law": bool(null),
        "prospective_first_composition_if_earned": {"layers": [0, 8, 17], "rank": 768},
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "p768_qualifying": p768_qualifying,
        "p512_qualifying": p512_qualifying,
        "p768_medians": [_median(p768_fw), _median(p768_wt)],
        "p768_max": p768_max,
        "predicates": [pred_a, pred_b, pred_c],
        "null": null,
        "runtime_s": result["runtime_s"],
    }, indent=2), flush=True)
    print("ALL-LAYER SHARED-INPUT SVD SCREEN DONE", flush=True)


if __name__ == "__main__":
    main()
