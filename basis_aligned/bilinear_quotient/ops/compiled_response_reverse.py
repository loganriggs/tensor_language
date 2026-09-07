"""Shared reverse-direction scorer for compiled response source graphs."""
# BQGATE: LIBRARY
from __future__ import annotations

import circuit_das_subspace as das
import run_temporal_five_mlp_source_clamped_compiled_graph_v1 as source
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_compiled_coordinate_upstream_atlas_v1 as atlas
import run_temporal_five_mlp_factor_pruned_causal_program_v1 as pruned
import run_temporal_five_mlp_frozen_response_ood_regularization_v2 as ood
import run_temporal_five_mlp_family_feasible_kl_refinement_v1 as klfit
import run_temporal_five_mlp_generic_subspace_complement_program_v1 as comp


def prepare(backend, ctx):
    fresh = ctx["fresh"]; torch = backend.torch
    _do, donor_full = atlasrun.capture_native(backend, fresh["donor_batch"])
    _co, control_donor_full = atlasrun.capture_native(backend, fresh["control_donor_batch"])
    reverse_full_output, _ = atlasrun.run_patch(backend, fresh["donor_batch"], fresh["base"][1], comp.SITES)
    return {"donor_full": donor_full, "control_donor_full": control_donor_full,
            "donor_state": atlasrun.states(torch, backend, fresh["donor"][0], fresh["rows"]),
            "reverse_full_state": atlasrun.states(torch, backend, reverse_full_output, fresh["rows"]),
            "control_donor_state": atlasrun.states(torch, backend, fresh["control_donor"][0], fresh["controls"])}


def evaluate(backend, ctx, reverse, support):
    fresh = ctx["fresh"]; torch = backend.torch
    _go, gai, gmi, counts = source.restricted_graph(backend, fresh["donor_batch"], fresh["batch"], reverse["donor_full"], support, source=True)
    _co, cai, cmi, ccounts = source.restricted_graph(backend, fresh["control_donor_batch"], fresh["control_batch"], reverse["control_donor_full"], support, source=True)
    control_ctx = {"rows": fresh["controls"], "base_logits": das.head_logits(backend, reverse["control_donor_state"]).float()}
    reports = {}
    for label, qs in ctx["projectors"].items():
        reference = atlas.compiled_coordinates(backend, fresh["rows"], qs, fresh["donor"][2], fresh["donor"][5], fresh["base"][5], fresh["donor"][3], fresh["base"][3], ctx["rank64_bases"][label])
        generated = atlas.compiled_coordinates(backend, fresh["rows"], qs, fresh["donor"][2], fresh["donor"][5], gai, fresh["donor"][3], gmi, ctx["rank64_bases"][label])
        coordinate = atlas.coordinate_report(torch, atlas.vectors(backend, fresh["donor_batch"], qs, reference), atlas.vectors(backend, fresh["donor_batch"], qs, generated))
        target_output = pruned.run_factors(backend, fresh["donor_batch"], fresh["donor"][1], qs,
            {s: v for s, v in generated.items() if atlasrun.site_parts(s)[0] == "attn"},
            {s: v for s, v in generated.items() if atlasrun.site_parts(s)[0] == "mlp"}, use_attention=True, use_mlp=True)
        target = ood.target_report(backend, fresh["rows"], reverse["donor_state"], reverse["reverse_full_state"],
                                   atlasrun.states(torch, backend, target_output, fresh["rows"]), ctx["reader"], fresh["temporal_n"])
        cgenerated = atlas.compiled_coordinates(backend, fresh["controls"], qs, fresh["control_donor"][2], fresh["control_donor"][5], cai, fresh["control_donor"][3], cmi, ctx["rank64_bases"][label])
        control_output = pruned.run_factors(backend, fresh["control_donor_batch"], fresh["control_donor"][1], qs,
            {s: v for s, v in cgenerated.items() if atlasrun.site_parts(s)[0] == "attn"},
            {s: v for s, v in cgenerated.items() if atlasrun.site_parts(s)[0] == "mlp"}, use_attention=True, use_mlp=True)
        control = klfit.control_metrics(backend, control_ctx, das.head_logits(backend, atlasrun.states(torch, backend, control_output, fresh["controls"])).float())
        reports[label] = {"coordinate": coordinate, "target": target, "control": control}
    return reports, counts, ccounts


def feasible(reports):
    return all(min(r["coordinate"]["signed_projection"].values()) >= .75 and r["coordinate"]["mean_residual"] <= .2
               and min(r["target"]["behavior_signed_projection"].values()) >= .8 and r["target"]["worst_target_residual"] <= .15
               and max(r["control"]["margin_rms_fraction"].values()) <= .1 and r["control"]["median_kl"] <= .02
               and r["control"]["top1_flip_fraction"] <= .05 for r in reports.values())


def summary(reports):
    return {"coordinate_projection_min": min(min(r["coordinate"]["signed_projection"].values()) for r in reports.values()),
            "coordinate_mean_residual_max": max(r["coordinate"]["mean_residual"] for r in reports.values()),
            "target_projection_min": min(min(r["target"]["behavior_signed_projection"].values()) for r in reports.values()),
            "target_worst_max": max(r["target"]["worst_target_residual"] for r in reports.values()),
            "control_margin_max": max(max(r["control"]["margin_rms_fraction"].values()) for r in reports.values()),
            "control_median_kl_max": max(r["control"]["median_kl"] for r in reports.values()),
            "control_flip_max": max(r["control"]["top1_flip_fraction"] for r in reports.values())}
