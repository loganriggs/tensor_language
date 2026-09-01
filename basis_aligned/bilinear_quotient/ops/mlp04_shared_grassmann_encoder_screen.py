"""RUNG 370 -- ONE SHARED GRASSMANN ENCODER FOR MLP0/4.

Take the chordal midpoint of the two independently optimal contextual p768
encoder rowspaces, then refit each paired Left/Right coefficient under its own
input covariance.  Compare independent, one-sided, and random shared controls.

Frozen predictions
------------------
pred_a_midpoint_matches_independent_pair:
    Mean damage <= independent+.003 on both corpora and <=.012 each.
pred_b_midpoint_preserves_row_tails:
    p95 <= independent+.008, max <= independent+.015 on both, max <=.040.
pred_c_shared_geometry_identity_and_price_hold:
    Rowspace overlap>=.72; midpoint mean<=best one-sided+.003 and <=50% of
    random on both; exact one-shared-encoder identity, shapes, price, splits.

Null: midpoint mean>=.030 on either corpus or no better than random on both.
Screen only; a pass licenses one physical QK64 composition without tuning.
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
OUT = ROOT / "mlp04_shared_grassmann_encoder_screen_results.json"
DEV = "cuda"
D = 1152
H = 4608
RANK = 768
LAYERS = (0, 4)
FIT_SLICE = (24, 48)
FINEWEB_SLICE = (176, 192)
WIKI_SKIP = 271_392
EVAL_ROWS = 16
INDEPENDENT_PAIR_SCALARS = 2 * (RANK * D + 2 * H * RANK + H * D + D)
SHARED_PAIR_SCALARS = RANK * D + 2 * (2 * H * RANK + H * D + D)


@torch.no_grad()
def _score_rows(model, rows, programs):
    handles = []
    shared_encoder_pointers = []
    for layer, program in programs.items():
        encoder, left, right = program["encoder"], program["left"], program["right"]
        down, bias = program["down"], program["bias"]
        shared_encoder_pointers.append(encoder.data_ptr())

        def hook(_module, args, output, encoder=encoder, left=left, right=right,
                 down=down, bias=bias):
            x = args[0].float()
            z = x @ encoder.T
            hidden = (z @ left.T) * (z @ right.T)
            return (hidden @ down.T + bias).to(output.dtype)

        handles.append(model.transformer.h[layer].mlp.register_forward_hook(hook))
    result = []
    try:
        for start in range(0, len(rows), 2):
            batch = rows[start:start + 2]
            index, target = batch[:, :-1].to(DEV), batch[:, 1:].to(DEV)
            x = F.rms_norm(model.transformer.wte(index), (D,))
            x0, value0 = x, None
            for block in model.transformer.h:
                x, value0 = block(x, value0, x0)
            logits = (30 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30)).float()
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
                                 reduction="none").reshape(len(batch), -1).mean(1)
            result.append(ce.cpu())
    finally:
        for handle in handles:
            handle.remove()
    return torch.cat(result).double(), shared_encoder_pointers


def _summary(values):
    return {"mean": float(values.mean()), "p95": float(torch.quantile(values, .95)),
            "max": float(values.max())}


@torch.no_grad()
def _orthonormal_rowspace(encoder):
    q, _r = torch.linalg.qr(encoder.T, mode="reduced")
    return q


@torch.no_grad()
def _program_for_q(mlp, covariance, q, shared_encoder):
    stacked = torch.cat((mlp.Left.weight.detach().float(),
                         mlp.Right.weight.detach().float()), dim=0)
    gram = q.T @ covariance @ q
    right_side = stacked @ covariance @ q
    coefficient = torch.linalg.solve(0.5 * (gram + gram.T), right_side.T).T
    return {
        "encoder": shared_encoder,
        "left": coefficient[:H].contiguous(),
        "right": coefficient[H:].contiguous(),
        "down": mlp.Down.weight.detach().float(),
        "bias": mlp.Down_bias.detach().float(),
    }


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert (ROOT / ".rowcache/fineweb_n192_skip11000.pt").exists()
        assert (ROOT / ".rowcache/fineweb_n192_skip7000.pt").exists()
        assert FINEWEB_SLICE[1] <= 192 and WIKI_SKIP + EVAL_ROWS * 257 == 275_504
        assert INDEPENDENT_PAIR_SCALARS - SHARED_PAIR_SCALARS == 884_736
        assert 511_758_646 - 884_736 == 510_873_910
        print("MLP04 SHARED GRASSMANN ENCODER | dry run: fits, controls, price, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    import mlp_late_context_metric_shared_input_screen as M
    from mixed56_context_metric_qk_newcorpus_ood import _wikitext103_train_rows
    from mlp_shared_input_svd_all_layers_screen import _manual_logits
    from tier2_model import load_elriggs

    fit_cache = torch.load(ROOT / ".rowcache/fineweb_n192_skip11000.pt", map_location="cpu")
    fit_cache = fit_cache["rows"] if isinstance(fit_cache, dict) else fit_cache
    eval_cache = torch.load(ROOT / ".rowcache/fineweb_n192_skip7000.pt", map_location="cpu")
    eval_cache = eval_cache["rows"] if isinstance(eval_cache, dict) else eval_cache
    fit_rows = fit_cache[FIT_SLICE[0]:FIT_SLICE[1], :257].long().contiguous()
    fineweb = eval_cache[FINEWEB_SLICE[0]:FINEWEB_SLICE[1], :257].long().contiguous()
    wikitext, fingerprint, token_count = _wikitext103_train_rows(
        n=EVAL_ROWS, width=257, skip=WIKI_SKIP)
    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D and len(model.transformer.h) == 18
    M.LAYERS = LAYERS
    covariances = M._covariances(model, fit_rows, _manual_logits)

    independent, q_by_layer = {}, {}
    for layer in LAYERS:
        program, _basis, _diag = M._rrr_program(
            model.transformer.h[layer].mlp, covariances[layer], rank=RANK)
        independent[layer] = program
        q_by_layer[layer] = _orthonormal_rowspace(program["encoder"])
    q0, q4 = q_by_layer[0], q_by_layer[4]
    overlap = float((q0.T @ q4).square().sum() / RANK)
    psum = q0 @ q0.T + q4 @ q4.T
    values, vectors = torch.linalg.eigh(0.5 * (psum + psum.T))
    qmid = vectors[:, torch.argsort(values, descending=True)[:RANK]].contiguous()
    generator = torch.Generator(device=DEV).manual_seed(29)
    qrandom, _ = torch.linalg.qr(torch.randn(D, RANK, device=DEV, generator=generator),
                                mode="reduced")

    families = {"independent": independent}
    for name, q in (("midpoint", qmid), ("one_sided_0", q0),
                    ("one_sided_4", q4), ("random", qrandom)):
        encoder = q.T.contiguous()
        families[name] = {layer: _program_for_q(model.transformer.h[layer].mlp,
                                                covariances[layer], q, encoder)
                          for layer in LAYERS}
    native_fw, _ = _score_rows(model, fineweb, {})
    native_wt, _ = _score_rows(model, wikitext, {})
    summaries, pointer_identity = {}, {}
    for name, programs in families.items():
        ce_fw, pointers_fw = _score_rows(model, fineweb, programs)
        ce_wt, pointers_wt = _score_rows(model, wikitext, programs)
        summaries[name] = {"fineweb": _summary(ce_fw - native_fw),
                           "wikitext": _summary(ce_wt - native_wt)}
        pointer_identity[name] = {
            "fineweb_encoder_data_ptrs": pointers_fw,
            "wikitext_encoder_data_ptrs": pointers_wt,
            "one_shared_tensor": len(set(pointers_fw + pointers_wt)) == 1,
        }
        print(f"{name}: FW {summaries[name]['fineweb']} WT {summaries[name]['wikitext']}",
              flush=True)

    midpoint, independent_s = summaries["midpoint"], summaries["independent"]
    one_sided_best = {corpus: min(summaries["one_sided_0"][corpus]["mean"],
                                  summaries["one_sided_4"][corpus]["mean"])
                      for corpus in ("fineweb", "wikitext")}
    pred_a = all(midpoint[c]["mean"] <= independent_s[c]["mean"] + .003
                 and midpoint[c]["mean"] <= .012 for c in ("fineweb", "wikitext"))
    pred_b = all(midpoint[c]["p95"] <= independent_s[c]["p95"] + .008
                 and midpoint[c]["max"] <= independent_s[c]["max"] + .015
                 and midpoint[c]["max"] <= .040 for c in ("fineweb", "wikitext"))
    shapes_exact = all(
        program["encoder"].shape == (RANK, D)
        and program["left"].shape == program["right"].shape == (H, RANK)
        and program["down"].shape == (D, H)
        for programs in families.values() for program in programs.values())
    pred_c = (overlap >= .72 and pointer_identity["midpoint"]["one_shared_tensor"]
              and all(midpoint[c]["mean"] <= one_sided_best[c] + .003
                      and midpoint[c]["mean"] <= .50 * summaries["random"][c]["mean"]
                      for c in ("fineweb", "wikitext"))
              and shapes_exact and INDEPENDENT_PAIR_SCALARS - SHARED_PAIR_SCALARS == 884_736
              and fingerprint == "7dabb830ac9ebb0d" and token_count == 675_457)
    null = (any(midpoint[c]["mean"] >= .030 for c in ("fineweb", "wikitext"))
            or all(midpoint[c]["mean"] >= summaries["random"][c]["mean"]
                   for c in ("fineweb", "wikitext")))
    result = {
        "status": "mlp04_shared_grassmann_encoder_screen_complete",
        "rung": 370,
        "claim_level": "split_fit_two_layer_shared_encoder_two_corpus_screen_only",
        "convention": "CE added above native; lower is better",
        "mathematical_object": "rank768 chordal midpoint of independent encoder rowspace projectors",
        "fit_rows": {"cache": "fineweb_n192_skip11000.pt",
                     "half_open": list(FIT_SLICE)},
        "evaluation": {"fineweb_cache": "fineweb_n192_skip7000.pt",
                       "fineweb_rows_half_open": list(FINEWEB_SLICE),
                       "wikitext103_train_span_half_open": [WIKI_SKIP,
                                                             WIKI_SKIP + EVAL_ROWS * 257],
                       "dataset_fingerprint": fingerprint},
        "layers": list(LAYERS),
        "rank": RANK,
        "independent_encoder_rowspace_overlap": overlap,
        "summaries": summaries,
        "shared_encoder_pointer_identity": pointer_identity,
        "independent_pair_scalars": INDEPENDENT_PAIR_SCALARS,
        "shared_pair_scalars": SHARED_PAIR_SCALARS,
        "saving_from_one_shared_encoder": INDEPENDENT_PAIR_SCALARS - SHARED_PAIR_SCALARS,
        "prospective_full_program_scalars": 510_873_910,
        'pred_a_midpoint_matches_independent_pair': bool(pred_a),
        'pred_b_midpoint_preserves_row_tails': bool(pred_b),
        'pred_c_shared_geometry_identity_and_price_hold': bool(pred_c),
        "null_shared_midpoint_encoder_not_useful": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"overlap": overlap, "summaries": summaries,
                      "predicates": [pred_a, pred_b, pred_c], "null": null,
                      "runtime_s": result["runtime_s"]}, indent=2), flush=True)
    print("MLP04 SHARED GRASSMANN ENCODER SCREEN DONE", flush=True)


if __name__ == "__main__":
    main()
