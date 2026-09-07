"""Shared preparation and pool operations for the temporal/is-was compiled response graph."""
# BQGATE: LIBRARY
from __future__ import annotations

import json

import circuit_candidate_temporal_auxiliary_fresh_cues_v13 as fresh_t
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12 as fresh_i
import circuit_das_subspace as das
import run_temporal_iswas_three_mlp_response_program_fresh_v12_v1 as population
import run_temporal_five_mlp_target_contrast_response_basis_v1 as v1
import run_temporal_five_mlp_generic_subspace_complement_program_v1 as comp
import run_temporal_five_mlp_matched_control_upstream_atlas_v2 as matched
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_upstream_tensor_ranked_greedy_program_v1 as greedy
import run_temporal_five_mlp_crossfit_response_weight_interface_v1 as interface
import run_temporal_five_mlp_family_feasible_kl_refinement_v1 as klfit
import run_temporal_five_mlp_attention_value_weight_pullback_v1 as pullback
import run_temporal_five_mlp_mlp_input_covector_crossfit_v1 as covector
import run_temporal_five_mlp_compiled_coordinate_upstream_atlas_v1 as atlas


ALL_UPSTREAM = tuple(
    site
    for layer in range(11)
    for site in tuple(f"L{layer}H{head}" for head in range(9)) + (f"MLP{layer}",)
)
BANDS = {"early": tuple(range(0, 3)), "middle": tuple(range(3, 6)),
         "late": tuple(range(6, 9)), "terminal": tuple(range(9, 11))}


def site_layer(site):
    return int(site[3:]) if site.startswith("MLP") else int(site[1:].split("H")[0])


def expanded_pool(top, bands):
    """Preserve execution order while adding every non-pool component in named layer bands."""
    names = tuple(bands)
    if len(names) != len(set(names)) or any(name not in BANDS for name in names):
        raise ValueError("unknown or repeated component band")
    layers = {layer for name in names for layer in BANDS[name]}
    selected = set(top) | {site for site in ALL_UPSTREAM if site_layer(site) in layers}
    return tuple(site for site in ALL_UPSTREAM if site in selected)


def reconstruct_projectors(backend):
    """Replay the two frozen response projectors (exactly eight model forwards)."""
    old_tcap, old_icap = json.loads(comp.TCAP.read_text()), json.loads(comp.ICAP.read_text())
    old_tr, old_ir, *_ = greedy.rows_and_controls(old_tcap, old_icap); controls = matched.control_rows()
    te, to = old_tr[::2] + old_ir[::2], old_tr[1::2] + old_ir[1::2]
    ceb, _, ce0, _, ce1, _, _ = interface.cap_inputs(backend, controls["discovery"])
    cob, _, co0, _, co1, _, _ = interface.cap_inputs(backend, controls["validation"])
    teb, _, te0, _, te1, _, _ = interface.cap_inputs(backend, te)
    tob, _, to0, _, to1, _, _ = interface.cap_inputs(backend, to)
    cbe, _ = comp.fit_bases(backend, ceb, ce0, ce1); cbo, _ = comp.fit_bases(backend, cob, co0, co1)
    qe, _ = v1.fit_target(backend, teb, te0, te1, cbe); qo, _ = v1.fit_target(backend, tob, to0, to1, cbo)
    projectors = {"even_fit": qe, "odd_fit": qo}; iface = json.loads(interface.OUT.read_text())
    hashes_ok = all([interface.thash(qe[site]), interface.thash(qo[site])] == iface["records"][site]["fit_basis_sha256"] for site in comp.SITES)
    reader, orientation, reader_ok = greedy.physical_reader(backend, json.loads(comp.WEIGHTS.read_text()))
    return {"projectors": projectors, "hashes_ok": hashes_ok, "reader": reader,
            "reader_orientation": orientation, "reader_ok": reader_ok}


def training_context(backend):
    """Capture the old A1/A2 base gates used to derive fixed rank-64 MLP readers (one forward)."""
    tcap, icap = json.loads(klfit.STC.read_text()), json.loads(klfit.SIC.read_text())
    temporal = sum((population.capable_rows(klfit.sealed_t, tcap, panel, 12) for panel in ("A1", "A2")), [])
    iswas = sum((population.capable_rows(klfit.sealed_i, icap, panel, 12) for panel in ("A1", "A2")), [])
    rows = temporal + iswas; batch = das._batch(backend, rows, side="base")
    _output, _cache, _attention, inputs, error, _attention_inputs = pullback.capture_with_attention_inputs(backend, batch)
    panels = {panel: [i for i, row in enumerate(rows) if row["transform_id"] == panel] for panel in ("A1", "A2")}
    return {"rows": rows, "batch": batch, "mlp_inputs": inputs, "capture_error": error, "panels": panels}


def rank64_bases(backend, projectors, training):
    output = {}
    for label, qs in projectors.items():
        ids = training["panels"]["A1" if label == "even_fit" else "A2"]; output[label] = {}
        for site, q in qs.items():
            if atlasrun.site_parts(site)[0] != "mlp": continue
            amap = covector.effective_map(backend, site, q, training["mlp_inputs"])
            output[label][site], _energy = covector.fit_input_basis(
                backend.torch, training["batch"], amap, ids, 64)
    return output


def fresh_context(backend):
    """Capture v13/v12 targets, P controls, full base clamp caches, and the 8-site target reference."""
    tcap, icap = json.loads(atlas.TC.read_text()), json.loads(atlas.IC.read_text())
    temporal = sum((population.capable_rows(fresh_t, tcap, panel, 12) for panel in ("A1", "A2")), [])
    iswas = sum((population.capable_rows(fresh_i, icap, panel, 12) for panel in ("A1", "A2")), [])
    rows = temporal + iswas; controls = [row for row in fresh_t.build_rows() if row["transform_id"] == "P"][:16]
    batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    cbatch, cdonor_batch = das._batch(backend, controls, side="base"), das._batch(backend, controls, side="donor")
    base = pullback.capture_with_attention_inputs(backend, batch)
    donor = pullback.capture_with_attention_inputs(backend, donor_batch)
    cbase = pullback.capture_with_attention_inputs(backend, cbatch)
    cdonor = pullback.capture_with_attention_inputs(backend, cdonor_batch)
    _base_full_output, base_full = atlasrun.capture_native(backend, batch)
    _control_full_output, control_base_full = atlasrun.capture_native(backend, cbatch)
    full_output, _io = atlasrun.run_patch(backend, batch, donor[1], comp.SITES)
    torch = backend.torch
    return {"rows": rows, "temporal_n": len(temporal), "controls": controls,
            "batch": batch, "donor_batch": donor_batch, "control_batch": cbatch,
            "control_donor_batch": cdonor_batch, "base": base, "donor": donor,
            "control_base": cbase, "control_donor": cdonor, "base_full": base_full,
            "control_base_full": control_base_full,
            "base_state": atlasrun.states(torch, backend, base[0], rows),
            "full_state": atlasrun.states(torch, backend, full_output, rows),
            "control_base_state": atlasrun.states(torch, backend, cbase[0], controls)}


def prepare(backend):
    replay = reconstruct_projectors(backend); training = training_context(backend)
    replay["training"] = training; replay["rank64_bases"] = rank64_bases(backend, replay["projectors"], training)
    replay["fresh"] = fresh_context(backend)
    return replay
