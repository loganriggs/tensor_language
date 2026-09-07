#!/usr/bin/env python3
"""Complete source-MLP response atlas on the failed v15 construction family."""
# BQGATE: EXPERIMENT pred_a_authority_replay_closure_finiteness_and_price pred_b_complete_five_mlp_write_restores_iswas pred_c_whole_write_beats_failed_rank16 pred_d_one_complete_mlp_is_dominant pred_e_native_gain_is_not_worse_than_old_gain
import hashlib
import json
import math
import os
from pathlib import Path

import run_temporal_iswas_rank16_source_task_pair_weight_groups_v1 as base
import run_temporal_iswas_rank16_source_mask26_construction_holdout_v1 as holdout

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_complete_source_mlp_write_atlas_v1.json"
PARENT = ROOT / "circuits/followups/temporal_iswas_rank16_source_mask26_construction_holdout_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_complete_source_mlp_write_atlas_v1_result.json"
EXPECTED = {
    "prior": "a97a124aa157afc9f0cd53fdc064ebd495b79879ba298c61acadc49441fed10d",
    "holdout_result": "8c75ccd9d627ae8c1782a3d6ceb3dfcfb2535797943e4dbe297ef9550d764900",
    "holdout_runner": "87fd06f9a9208b49905106e1216d00f59811e39500720a8dd535a9f0e9e7945f",
}
WHOLE_ARMS = {
    **{f"whole_{site}_gain100": (1 << index, 1.0) for index, site in enumerate(base.SUPPORT)},
    "whole_all_gain100": (31, 1.0),
    "whole_all_gain115": (31, 1.15),
}
BASE_ARMS = (
    "full_rank16", "shared_mean", "task_contrast", "mean_plus_contrast",
    "own_task", "cross_task", "mean_plus_contrast_complement",
)
EXECUTED_ARMS = BASE_ARMS + tuple(WHOLE_ARMS)
MAX_FORWARDS = 64


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def report_replay(left, right):
    values = []
    for task in base.TASKS:
        values.append(abs(left["behavior_signed_projection"][task] - right["behavior_signed_projection"][task]))
        for metric in ("signed_projection", "relative_squared_error", "norm_ratio"):
            values.append(abs(left["response"][task][metric] - right["response"][task][metric]))
    values.append(abs(left["control"]["median_kl"] - right["control"]["median_kl"]))
    return max(values)


def main():
    candidate_id = "temporal_auxiliary.iswas_v15_complete_source_mlp_write_atlas_v1"
    dry = {
        "candidate_id": candidate_id, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "whole_arms": WHOLE_ARMS,
        "model_forwards_max": MAX_FORWARDS, "fit_updates": 0,
        "model_updates": 0, "transformer_backwards": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(OUT)
    observed = {
        "prior": sha(PRIOR), "holdout_result": sha(PARENT),
        "holdout_runner": sha(ROOT / "ops/run_temporal_iswas_rank16_source_mask26_construction_holdout_v1.py"),
    }
    if observed != EXPECTED:
        raise RuntimeError(f"complete-source atlas authority changed: {observed}")
    parent = json.loads(PARENT.read_text())
    if parent["terminal"] != "parent_construction_failure":
        raise RuntimeError("parent no longer licenses complete-module split")

    base.PRIOR = PRIOR
    base.OUT = OUT
    base.MAX_FORWARDS = MAX_FORWARDS
    base.ARMS = EXECUTED_ARMS
    base.EXPECTED_CONTROL_TASKS = set(base.TASKS)
    base.EXPECTED = {
        "prior": EXPECTED["prior"], "weight_compiler_audit": sha(base.AUDIT),
        "mask30_holdout": sha(base.HOLDOUT), "mask30_control_attribution": sha(base.ATTRIBUTION),
        "task_mode_helper": sha(base.HELPER), "mathematical_review": sha(base.MATH_REVIEW),
    }
    holdout.POPULATION.clear()
    base.oodctx.capture = holdout.holdout_capture
    original_run_blocks = base.run_blocks
    original_write = base.atomic_create_json

    def run_blocks(backend, batch, rows, base_out, delta_hidden, bases, maps, blocks, arm, *, control=False):
        if arm not in WHOLE_ARMS:
            return original_run_blocks(
                backend, batch, rows, base_out, delta_hidden, bases, maps, blocks, arm, control=control,
            )
        mask, gain = WHOLE_ARMS[arm]
        handles = []
        response = {"attention": {}, "mlp": {}, "mlp_input": {}}
        for bit, site in enumerate(base.SUPPORT):
            if not mask & (1 << bit):
                continue
            layer = int(site[3:])
            base_value = base_out["mlp"][layer]
            delta = delta_hidden[site]
            down = backend.model.transformer.h[layer].mlp.Down.weight.detach().float().T

            def patch(_module, _arguments, output, base_value=base_value, delta=delta, down=down):
                changed = output.clone()
                for index, _row in enumerate(rows):
                    stop = int(batch.semantic_positions[index]) + 1
                    complete = delta[index, :stop].float() @ down
                    changed[index, :stop] = base_value[index, :stop].to(changed) + (gain * complete).to(changed)
                return changed

            handles.append(backend.model.transformer.h[layer].mlp.register_forward_hook(patch))
        for response_site in base.edge.RESPONSE_SITES:
            kind, layer, _head = base.atlas.site_parts(response_site)
            if kind == "attn":
                handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(
                    lambda _module, arguments, layer=layer: response["attention"].__setitem__(layer, arguments[0].detach().float().clone())))
            else:
                handles.append(backend.model.transformer.h[layer].mlp.register_forward_pre_hook(
                    lambda _module, arguments, layer=layer: response["mlp_input"].__setitem__(layer, arguments[0].detach().float().clone())))
                handles.append(backend.model.transformer.h[layer].mlp.register_forward_hook(
                    lambda _module, _arguments, output, layer=layer: response["mlp"].__setitem__(layer, output.detach().float().clone())))
        try:
            output = backend.native(batch, capture=True)
        finally:
            for handle in handles:
                handle.remove()
        return output, response

    def write_result(_path, result):
        reports = result["reports"]
        rank16 = reports["full_rank16"]
        whole = reports["whole_all_gain100"]
        whole115 = reports["whole_all_gain115"]
        replay = report_replay(rank16, parent["reports"]["full_rank16"])
        singleton = {site: reports[f"whole_{site}_gain100"] for site in base.SUPPORT}
        whole_iswas = whole["behavior_signed_projection"]["iswas"]
        singleton_fraction = {
            site: report["behavior_signed_projection"]["iswas"] / max(abs(whole_iswas), 1e-30)
            for site, report in singleton.items()
        }
        top_site = max(base.SUPPORT, key=lambda site: (singleton_fraction[site], -base.SUPPORT.index(site)))
        pred_a = bool(
            replay <= 1e-5 and max(result["union_complement_closure_rse_by_site"].values()) <= 1e-6
            and finite([reports, singleton_fraction]) and result["price"]["model_forwards_observed"] <= MAX_FORWARDS
            and holdout.POPULATION["temporal_rows"] == 62 and holdout.POPULATION["iswas_rows"] == 32
        )
        pred_b = (
            whole["behavior_signed_projection"]["iswas"] >= .75
            and whole["response"]["iswas"]["signed_projection"] >= .8
            and whole["response"]["iswas"]["relative_squared_error"] <= .2
        )
        pred_c = (
            whole["behavior_signed_projection"]["iswas"] >= rank16["behavior_signed_projection"]["iswas"] + .50
            and whole["response"]["iswas"]["signed_projection"] >= rank16["response"]["iswas"]["signed_projection"] + .10
        )
        pred_d = singleton_fraction[top_site] >= .60
        pred_e = (
            whole["behavior_signed_projection"]["iswas"] >= whole115["behavior_signed_projection"]["iswas"] - .03
            and whole["response"]["iswas"]["signed_projection"] >= whole115["response"]["iswas"]["signed_projection"] - .03
        )
        predictions = {
            "pred_a_authority_replay_closure_finiteness_and_price": bool(pred_a),
            "pred_b_complete_five_mlp_write_restores_iswas": bool(pred_b),
            "pred_c_whole_write_beats_failed_rank16": bool(pred_c),
            "pred_d_one_complete_mlp_is_dominant": bool(pred_d),
            "pred_e_native_gain_is_not_worse_than_old_gain": bool(pred_e),
        }
        if not pred_a:
            terminal = "invalid"
        elif not pred_b or not pred_c:
            terminal = "source_graph_construction_failure"
        elif pred_d:
            terminal = "localized_subspace_construction_failure"
        else:
            terminal = "distributed_subspace_construction_failure"
        result.update({
            "schema": "temporal_iswas_v15_complete_source_mlp_write_atlas_result_v1",
            "candidate_id": candidate_id, "authority_sha256": EXPECTED,
            "parent_terminal": parent["terminal"], "rank16_parent_replay_max_abs_error": replay,
            "whole_reports": {arm: reports[arm] for arm in WHOLE_ARMS},
            "singleton_iswas_behavior_fraction_of_whole": singleton_fraction,
            "top_singleton_site": top_site, "predictions": predictions, "terminal": terminal,
        })
        original_write(OUT, result)

    base.run_blocks = run_blocks
    base.atomic_create_json = write_result
    base.main()


if __name__ == "__main__":
    main()
