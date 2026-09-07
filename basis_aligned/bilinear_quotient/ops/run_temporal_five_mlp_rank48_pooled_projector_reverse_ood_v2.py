#!/usr/bin/env python3
"""Disjoint OOD repair of the pooled rank48 reverse projector experiment."""
# BQGATE: EXPERIMENT pred_a_authority_disjointness_finiteness_and_exact_price pred_b_pooled_reverse_coordinates_transfer pred_c_pooled_reverse_behavior_passes pred_d_pooled_reverse_controls_are_selective pred_e_pooling_resolves_the_split_failure
from datetime import datetime, timezone
import hashlib, json, os
from pathlib import Path

from circuit_fast_screen_managed_runner import atomic_create_json
import compiled_response_ood as oodctx
import run_temporal_five_mlp_rank48_pooled_projector_reverse_fresh_v1 as engine

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_rank48_pooled_projector_reverse_ood_v2.json"
V1_RESULT = ROOT / "circuits/followups/temporal_five_mlp_rank48_pooled_projector_reverse_fresh_v1_result.json"
V1_RUNNER = ROOT / "ops/run_temporal_five_mlp_rank48_pooled_projector_reverse_fresh_v1.py"
OOD_HELPER = ROOT / "ops/compiled_response_ood.py"
TCAP = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v12_capability_v1_result.json"
ICAP = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v11_capability_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_five_mlp_rank48_pooled_projector_reverse_ood_v2_result.json"
EXPECTED = {"v1_result": "a2348920adc50c8a8d156693af038c5fbade6e9c4a1f8cd943cc332a047b7e3d",
            "v1_runner": "6cbf975c5f8f858249921a08715ed7a2cef653b0c1fa8cb4e1916f87e50f6f17",
            "ood_helper": "96b9e3aef3b64364c7fc4c49d0af9ff35827eadb4ca5bfbdf5d2271cb26ad808",
            "tcap": "4758b02cd026c85289dc3eaf352cc496d238057c6f8b52dfc6fe49ae17893324",
            "icap": "6dd757b066304d1f81ea1e52e0db601fea05adeac516a49cc84ab42bc73a86a2"}
REGISTERED_PREDICTIONS = (
    "pred_a_authority_disjointness_finiteness_and_exact_price",
    "pred_b_pooled_reverse_coordinates_transfer",
    "pred_c_pooled_reverse_behavior_passes",
    "pred_d_pooled_reverse_controls_are_selective",
    "pred_e_pooling_resolves_the_split_failure",
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def main():
    paths = {"v1_result": V1_RESULT, "v1_runner": V1_RUNNER, "ood_helper": OOD_HELPER, "tcap": TCAP, "icap": ICAP}
    observed = {key: sha(path) for key, path in paths.items()}
    if observed != EXPECTED: raise RuntimeError(f"pooled OOD repair authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_rank48_pooled_projector_reverse_ood_v2",
           "dryrun": True, "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "model_forwards_exact": 19, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    original_writer = engine.atomic_create_json

    def write_result(_path, result):
        result["schema"] = "temporal_five_mlp_rank48_pooled_projector_reverse_ood_result_v2"
        result["candidate_id"] = "temporal_auxiliary.five_mlp_rank48_pooled_projector_reverse_ood_v2"
        result["authority_sha256"] = EXPECTED
        result["evaluation_scope"] = {"fit": "temporal_v13_iswas_v12", "evaluation": "temporal_v12_iswas_v11_ood"}
        result["finished_utc"] = now()
        result["predictions"]["pred_e_pooling_resolves_the_split_failure"] = bool(
            min(result["target"]["behavior_signed_projection"].values()) > .7981567
            and result["control"]["top1_flip_fraction"] == 0.0)
        result["terminal"] = ("invalid" if not result["predictions"]["pred_a_authority_disjointness_finiteness_and_exact_price"]
                              else "pooled_reverse_ood_rank48_program" if all(result["predictions"].values())
                              else "pooled_projector_ood_failure")
        original_writer(OUT, result)

    engine.OUT = OUT
    engine.atomic_create_json = write_result
    engine.graph.fresh_context = lambda backend: oodctx.capture(backend, TCAP, ICAP)
    engine.main()


if __name__ == "__main__": main()
