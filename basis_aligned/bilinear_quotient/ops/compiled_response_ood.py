"""Shared v12/v11 OOD context for compiled response-graph experiments."""
# BQGATE: LIBRARY
from __future__ import annotations

import json

import circuit_candidate_temporal_auxiliary_fresh_cues_v12 as ood_t
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v11 as ood_i
import circuit_das_subspace as das
import compiled_response_graph as graph
import run_temporal_iswas_three_mlp_response_program_fresh_v12_v1 as population
import run_temporal_five_mlp_attention_value_weight_pullback_v1 as pullback
import run_temporal_five_mlp_generic_subspace_complement_program_v1 as comp
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun


def core(backend):
    """Frozen projectors/readers and training-derived MLP covectors (nine forwards)."""
    replay = graph.reconstruct_projectors(backend); training = graph.training_context(backend)
    replay["training"] = training
    replay["rank64_bases"] = graph.rank64_bases(backend, replay["projectors"], training)
    return replay


def capture(backend, temporal_capability_path, iswas_capability_path):
    """Capture capable temporal-v12/iswas-v11 targets and temporal P controls (seven forwards)."""
    tcap = json.loads(temporal_capability_path.read_text()); icap = json.loads(iswas_capability_path.read_text())
    temporal = sum((population.capable_rows(ood_t, tcap, panel, 12) for panel in ("A1", "A2")), [])
    iswas = sum((population.capable_rows(ood_i, icap, panel, 12) for panel in ("A1", "A2")), [])
    rows = temporal + iswas; controls = [row for row in ood_t.build_rows() if row["transform_id"] == "P"][:16]
    batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    cbatch, cdonor_batch = das._batch(backend, controls, side="base"), das._batch(backend, controls, side="donor")
    base = pullback.capture_with_attention_inputs(backend, batch); donor = pullback.capture_with_attention_inputs(backend, donor_batch)
    cbase = pullback.capture_with_attention_inputs(backend, cbatch); cdonor = pullback.capture_with_attention_inputs(backend, cdonor_batch)
    _bo, base_full = atlasrun.capture_native(backend, batch); _co, control_base_full = atlasrun.capture_native(backend, cbatch)
    full_output, _ = atlasrun.run_patch(backend, batch, donor[1], comp.SITES); torch = backend.torch
    return {"rows": rows, "temporal_n": len(temporal), "controls": controls, "batch": batch,
            "donor_batch": donor_batch, "control_batch": cbatch, "control_donor_batch": cdonor_batch,
            "base": base, "donor": donor, "control_base": cbase, "control_donor": cdonor,
            "base_full": base_full, "control_base_full": control_base_full,
            "base_state": atlasrun.states(torch, backend, base[0], rows),
            "full_state": atlasrun.states(torch, backend, full_output, rows),
            "control_base_state": atlasrun.states(torch, backend, cbase[0], controls)}


def prepare(backend, temporal_capability_path, iswas_capability_path):
    replay = core(backend); replay["fresh"] = capture(backend, temporal_capability_path, iswas_capability_path)
    return replay
