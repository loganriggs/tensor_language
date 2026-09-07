"""Deterministic complete-family pooled projector used by compiled response programs."""
# BQGATE: LIBRARY
from __future__ import annotations

import json

import run_temporal_five_mlp_target_contrast_response_basis_v1 as v1
import run_temporal_five_mlp_generic_subspace_complement_program_v1 as comp
import run_temporal_five_mlp_matched_control_upstream_atlas_v2 as matched
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_upstream_tensor_ranked_greedy_program_v1 as greedy
import run_temporal_five_mlp_crossfit_response_weight_interface_v1 as interface
import run_temporal_five_mlp_mlp_input_covector_crossfit_v1 as covector
import compiled_response_graph as graph


def fit(backend):
    """Fit one rank-eight projector on all v13/v12 target and matched-control rows (four forwards)."""
    tcap, icap = json.loads(comp.TCAP.read_text()), json.loads(comp.ICAP.read_text())
    temporal, iswas, *_ = greedy.rows_and_controls(tcap, icap)
    target_rows = temporal + iswas
    controls = matched.control_rows(); control_rows = controls["discovery"] + controls["validation"]
    cbatch, _, cbase, _, cdonor, _, _ = interface.cap_inputs(backend, control_rows)
    tbatch, _, tbase, _, tdonor, _, _ = interface.cap_inputs(backend, target_rows)
    control_basis, _ = comp.fit_bases(backend, cbatch, cbase, cdonor)
    projector, _ = v1.fit_target(backend, tbatch, tbase, tdonor, control_basis)
    return {"projector": projector, "target_rows": target_rows, "control_rows": control_rows}


def fit_mlp_input_bases(backend, projector):
    """Fit the rank-64 activation-conditioned MLP input bases on all old panels (one forward)."""
    training = graph.training_context(backend); ids = list(range(len(training["rows"]))); output = {}
    for site, q in projector.items():
        if atlasrun.site_parts(site)[0] != "mlp": continue
        amap = covector.effective_map(backend, site, q, training["mlp_inputs"])
        output[site], _energy = covector.fit_input_basis(backend.torch, training["batch"], amap, ids, 64)
    return training, output
