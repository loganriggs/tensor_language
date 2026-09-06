#!/usr/bin/env python3
"""Remove the live L9H1/H4/H7 subject write and measure H3 necessity."""

# BQGATE: EXPERIMENT pred_a_authority_capture_self_clamp_and_price pred_b_complete_l9_removal_is_material pred_c_greedy_set_is_necessary_for_h3_response pred_d_complement_is_unnecessary_for_h3_response pred_e_greedy_set_is_necessary_for_behavior
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import attention_path_mediation_eval as mediation
import attention_source_destination_eval as attention_eval
import attention_source_group_eval as source_groups
import circuit_candidate_temporal_auxiliary_fresh_cues_v4 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import residual_source_onset_eval as onset

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_l9_subject_to_h3_greedy_necessity_v1.json"
GREEDY = ROOT / "circuits/followups/temporal_auxiliary_will_had_l9_subject_to_h3_greedy_v1_result.json"
GREEDY_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_l9_subject_to_h3_greedy_v1.py"
SUBSPACE = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v4_capability_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v4.py"
MEDIATION = ROOT / "ops/attention_path_mediation_eval.py"
ATTENTION = ROOT / "ops/attention_source_destination_eval.py"
SOURCE = ROOT / "ops/attention_source_group_eval.py"
ONSET = ROOT / "ops/residual_source_onset_eval.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_l9_subject_to_h3_greedy_necessity_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.l9_subject_to_h3_greedy_necessity_v1"
EXPECTED = {
    "prior": "967afdd3938cf0c80e7dca72a422788a09cb6d60fa4abf1172b226d69669251a",
    "greedy": "b345449b2bc71e24658f4f4f61eda1b5bb41867974e83f813eb6c2d649e65b8b",
    "greedy_runner": "8444e2bad1433220ab6019db8a267989f33b3720bae2a715c221a0ba51666200",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "capability": "63b69e3bc57a0a8a9afcffa252737614f9a8a41b6732b9fff655d9da128ef8b2",
    "builder": "31e40a5e8a8b285ce7afdb6327276c0aa28b4759083586d0310b0857c8b86764",
    "mediation": "05d07578129db0cb8bf5e46d08e6cd6e9c2bb9c3883675586c83c3efd24b8fda",
    "attention": "608ae6bf74af96663ec022b907c53d371670a36e5d7ec4fd1667b3c6add58dfd",
    "source": "6ecf5f40b92f94cb32bccf1a703e527a3d468281936e63d4c7e91e8af66b4348",
    "onset": "c276450cc9ec7c2b0a05e2be0e88bac3df9af7003e370b99b66552083c4f4b45",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
}
ARMS = {"writer_native": None, "clamp_greedy": (1, 4, 7),
        "clamp_complement": (0, 2, 3, 5, 6, 8), "clamp_all": tuple(range(9)),
        "base_self_clamp": tuple(range(9))}
MAX_FORWARDS, MAX_EVALUATIONS, RECORDS = 22, 704, 320


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def pair_error(first, second):
    return max(abs(float(a) - float(b)) for left, right in zip(first.answer_foil, second.answer_foil)
               for a, b in zip(left, right))


def run_clamped(backend, item, heads, *, enable_writer):
    handles, dynamic = [], {}
    if enable_writer:
        handles.append(backend.model.transformer.h[8].attn.c_proj.register_forward_pre_hook(
            item["writer_hook"]))
    attention = backend.model.transformer.h[9].attn
    head_count, head_dim = backend.model.config.n_head, backend.model.config.n_embd // backend.model.config.n_head

    def capture(_module, arguments):
        current = arguments[0]
        v1 = arguments[1] if len(arguments) > 1 else None
        pattern, value, reconstructed = attention_eval._attention_terms(
            backend, attention, current, v1)
        dynamic.update(reconstructed=reconstructed)

    def clamp(_module, arguments):
        flattened = arguments[0]
        native = flattened.view(len(item["base_batch"].row_ids), flattened.shape[1],
                                head_count, head_dim)
        dynamic["reconstruction_max_abs"] = float(
            (dynamic["reconstructed"].float() - native.float()).abs().max())
        changed = native.clone()
        for index, positions in enumerate(item["destinations"]):
            for position in positions:
                for head in heads:
                    changed[index, position, head] = item["base9"]["head_output"][
                        index, position, head].to(changed)
        return (changed.reshape_as(flattened),) + tuple(arguments[1:])

    handles.extend([attention.register_forward_pre_hook(capture),
                    attention.c_proj.register_forward_pre_hook(clamp)])
    try:
        output, h3 = attention_eval.capture_layer_attention(backend, item["base_batch"], 11)
    finally:
        for handle in handles:
            handle.remove()
    return output, h3, float(dynamic.get("reconstruction_max_abs", math.inf))


def projected_removal(backend, writer_h3, clamped_h3, batch, q):
    values = []
    for index, query in enumerate(batch.semantic_positions):
        delta = (writer_h3["head_output"][index, query, 3].float()
                 - clamped_h3["head_output"][index, query, 3].float())
        values.append(float(backend.torch.linalg.vector_norm(delta @ q)))
    return sum(values) / len(values)


def main():
    paths = {"prior": PRIOR, "greedy": GREEDY, "greedy_runner": GREEDY_RUNNER,
             "subspace": SUBSPACE, "capability": CAPABILITY, "builder": BUILDER,
             "mediation": MEDIATION, "attention": ATTENTION, "source": SOURCE,
             "onset": ONSET, "producer": PRODUCER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("authority or implementation hash changed")
    prior, greedy, subspace, capability = [json.loads(path.read_text()) for path in
        (PRIOR, GREEDY, SUBSPACE, CAPABILITY)]
    if (prior.get("candidate_id") != CANDIDATE_ID or greedy.get("terminal") != "screen"
            or capability.get("terminal") != "manifest"):
        raise RuntimeError("authority terminal changed")
    rows = [row for row in candidate.build_rows() if row["transform_id"] in {"A1", "A2"}]
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False, "arms": list(ARMS),
              "model_forwards_max": MAX_FORWARDS, "example_evaluations_max": MAX_EVALUATIONS,
              "records": RECORDS, "fit_updates": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    q = backend.torch.linalg.qr(backend.torch.tensor(
        subspace["axis_artifacts"]["two_task_dim_union_rank2"],
        device=backend.device).float(), mode="reduced").Q
    items, reconstruction, identity = [], 0.0, 0.0
    forwards = evaluations = 0
    for panel in ("A1", "A2"):
        panel_rows = [row for row in rows if row["transform_id"] == panel]
        base_batch = das._batch(backend, panel_rows, side="base")
        donor_batch = das._batch(backend, panel_rows, side="donor")
        base_output, base8 = attention_eval.capture_layer_attention(backend, base_batch, 8)
        donor_output, donor8 = attention_eval.capture_layer_attention(backend, donor_batch, 8)
        base9_output, base9 = attention_eval.capture_layer_attention(backend, base_batch, 9)
        base11_output, base11 = attention_eval.capture_layer_attention(backend, base_batch, 11)
        destinations = onset.positions_for_group(base_batch, donor_batch, "subject_onset")
        writer_hook = mediation.fixed_source_delta_hook(
            backend, base_batch, donor_batch, base8, donor8, destinations,
            ("cue",), selected_heads=(1,))
        handle = backend.model.transformer.h[8].attn.c_proj.register_forward_pre_hook(writer_hook)
        try:
            writer_output, writer_h3 = attention_eval.capture_layer_attention(backend, base_batch, 11)
        finally:
            handle.remove()
        forwards += 5; evaluations += 5 * len(panel_rows)
        identity = max(identity, pair_error(base_output, base9_output),
                       pair_error(base_output, base11_output))
        reconstruction = max(reconstruction, *(float(x["reconstruction_max_abs"])
            for x in (base8, donor8, base9, base11, writer_h3)))
        items.append({"panel": panel, "rows": panel_rows, "base_batch": base_batch,
            "base_output": base_output, "donor_output": donor_output, "base9": base9,
            "writer_output": writer_output, "writer_h3": writer_h3,
            "destinations": destinations, "writer_hook": writer_hook})

    outputs, h3_removal, records, self_clamp = {}, {}, [], 0.0
    for item in items:
        panel = item["panel"]
        outputs[panel] = {"writer_native": item["writer_output"]}
        h3_removal[panel] = {"writer_native": 0.0}
        records.extend(dict(record, panel=panel) for record in source_groups.recovery_records(
            item["rows"], item["base_output"], item["donor_output"], item["writer_output"],
            arm="writer_native"))
        for arm in ("clamp_greedy", "clamp_complement", "clamp_all", "base_self_clamp"):
            output, h3, error = run_clamped(
                backend, item, ARMS[arm], enable_writer=arm != "base_self_clamp")
            forwards += 1; evaluations += len(item["rows"]); reconstruction = max(reconstruction, error)
            outputs[panel][arm] = output
            if arm == "base_self_clamp":
                self_clamp = max(self_clamp, pair_error(output, item["base_output"]))
            else:
                h3_removal[panel][arm] = projected_removal(
                    backend, item["writer_h3"], h3, item["base_batch"], q)
            records.extend(dict(record, panel=panel) for record in source_groups.recovery_records(
                item["rows"], item["base_output"], item["donor_output"], output, arm=arm))

    summaries = {panel: {arm: source_groups.summarize([r for r in records
        if r["panel"] == panel and r["arm"] == arm]) for arm in ARMS}
        for panel in ("A1", "A2")}
    behavioral_removal = {panel: {arm: summaries[panel]["writer_native"]["mean_recovery"]
        - summaries[panel][arm]["mean_recovery"] for arm in
        ("clamp_greedy", "clamp_complement", "clamp_all")} for panel in ("A1", "A2")}
    h3_fraction = {panel: {arm: h3_removal[panel][arm] / h3_removal[panel]["clamp_all"]
        if h3_removal[panel]["clamp_all"] > 1e-12 else None for arm in
        ("clamp_greedy", "clamp_complement", "clamp_all")} for panel in ("A1", "A2")}
    pred_a = bool(reconstruction <= 5e-4 and identity <= 1e-4 and self_clamp <= 1e-4
                  and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS
                  and len(records) == RECORDS and all(math.isfinite(r["recovery"]) for r in records))
    pred_b = all(behavioral_removal[p]["clamp_all"]
                 >= 0.03 * abs(summaries[p]["writer_native"]["mean_recovery"])
                 and h3_removal[p]["clamp_all"] > 1e-8 for p in ("A1", "A2"))
    pred_c = all(h3_fraction[p]["clamp_greedy"] >= 0.90 for p in ("A1", "A2"))
    pred_d = all(h3_fraction[p]["clamp_complement"] <= 0.15 for p in ("A1", "A2"))
    pred_e = all(abs(behavioral_removal[p]["clamp_greedy"]
                         - behavioral_removal[p]["clamp_all"])
                 <= 0.20 * abs(behavioral_removal[p]["clamp_all"]) for p in ("A1", "A2"))
    predictions = {"pred_a_authority_capture_self_clamp_and_price": pred_a,
        "pred_b_complete_l9_removal_is_material": pred_b,
        "pred_c_greedy_set_is_necessary_for_h3_response": pred_c,
        "pred_d_complement_is_unnecessary_for_h3_response": pred_d,
        "pred_e_greedy_set_is_necessary_for_behavior": pred_e}
    terminal = ("invalid" if not pred_a else "screen" if all(predictions.values())
                else "redundant" if pred_b else "null")
    result = {"schema": "temporal_auxiliary_l9_subject_to_h3_greedy_necessity_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "instrument": {"reconstruction_max_abs": reconstruction,
            "capture_identity_max_abs": identity, "base_self_clamp_max_abs": self_clamp},
        "summaries": summaries, "behavioral_removal": behavioral_removal,
        "projected_h3_removal": h3_removal, "fraction_of_all_h3_removal": h3_fraction,
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
                  "records": len(records), "fit_updates": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "summaries",
          "behavioral_removal", "projected_h3_removal", "fraction_of_all_h3_removal",
          "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
