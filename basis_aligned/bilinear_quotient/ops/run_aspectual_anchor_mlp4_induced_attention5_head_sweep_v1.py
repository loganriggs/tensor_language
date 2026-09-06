#!/usr/bin/env python3
"""Hash-bound attention5 specialization of the all-head endpoint sweep."""

from __future__ import annotations

import hashlib
from pathlib import Path

import circuit_fast_screen_producer as producer
import run_aspectual_anchor_block4_contextual_source_writer_factorial_v1 as block4
import run_aspectual_anchor_mlp4_to_l9h1_h4_path_mediation_v1 as path


SCRIPT_FILE = Path(globals()["__file__"]).resolve()
ROOT = SCRIPT_FILE.parents[1]
TEMPLATE = ROOT / "ops/run_aspectual_anchor_mlp4_induced_l9_head_sweep_v1.py"
EXPECTED_TEMPLATE_SHA256 = "38a945b0b08d588d49e107abd19d8b09ec744c289c27859fe56a84eccbf6a126"
WRITER_FACTORS = ("left_change", "right_change")
STATIC_PREDICATES = {
    "pred_a_exact_instrument": "native capture and all-head closure",
    "pred_b_writer_recurrence": "fixed writer recurrence",
    "pred_c_attention5_recurrence": "all-head onset recurrence",
    "pred_d_head_localization": "endpoint sufficiency plus necessity",
    "pred_e_exact_coverage": "all frozen arms and rows",
}


class Attention5Backend(path.PathBackend):
    """Replace the inherited L9 manual capture with exact L5 module hooks."""

    def capture_writer(
        self,
        base_batch: producer.ModelBatch,
        donor_batch: producer.ModelBatch,
        base_capture,
        donor_capture,
        factors: tuple[str, ...],
    ):
        if factors not in ((), WRITER_FACTORS):
            raise RuntimeError("writer factor set changed")
        projected, tensor_error = self.projected_terms(base_capture, donor_capture)
        positions = block4.source_positions(base_batch, donor_batch)
        captured = {}

        def patch_mlp4(_module, _arguments, output):
            changed = output.clone()
            for i, bank in enumerate(positions):
                for position in bank:
                    delta = self.torch.zeros_like(
                        changed[i, position], dtype=self.torch.float32
                    )
                    for factor in factors:
                        delta += projected[factor][i, position]
                    changed[i, position] = (
                        changed[i, position].float() + delta
                    ).to(changed.dtype)
            return changed

        def capture_heads(_module, arguments):
            flattened = arguments[0]
            head_dim = self.model.config.n_embd // self.model.config.n_head
            captured["head_output"] = flattened.view(
                len(base_batch.row_ids),
                flattened.shape[1],
                self.model.config.n_head,
                head_dim,
            ).detach().clone()

        writer_handle = self.model.transformer.h[4].mlp.register_forward_hook(patch_mlp4)
        capture_handle = self.model.transformer.h[5].attn.c_proj.register_forward_pre_hook(
            capture_heads
        )
        try:
            output = self.native(base_batch, capture=False)
        finally:
            capture_handle.remove()
            writer_handle.remove()
        if set(captured) != {"head_output"}:
            raise RuntimeError("attention5 head capture missing")
        captured["reconstruction_max_abs"] = 0.0
        return output, captured, tensor_error


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"expected exactly one template occurrence: {old!r}")
    return source.replace(old, new)


def main() -> None:
    payload = TEMPLATE.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != EXPECTED_TEMPLATE_SHA256:
        raise RuntimeError(
            f"L9 head template changed: expected={EXPECTED_TEMPLATE_SHA256} observed={observed}"
        )
    source = payload.decode("utf-8")
    replacements = (
        (
            "aspectual_anchor_mlp4_induced_l9_head_sweep_v1.json",
            "aspectual_anchor_mlp4_induced_attention5_head_sweep_v1.json",
        ),
        (
            "aspectual_anchor_mlp4_to_l9h1_h4_path_mediation_v1_result.json",
            "aspectual_anchor_mlp4_induced_block5_crossing_factorial_v1_result.json",
        ),
        (
            "run_aspectual_anchor_mlp4_to_l9h1_h4_path_mediation_v1.py",
            "run_aspectual_anchor_mlp4_induced_block5_crossing_factorial_v1.py",
        ),
        (
            "aspectual_anchor_mlp4_induced_l9_head_sweep_v1_result.json",
            "aspectual_anchor_mlp4_induced_attention5_head_sweep_v1_result.json",
        ),
        (
            "aspectual_anchor.has_vs_had.mlp4_induced_l9_head_sweep_v1",
            "aspectual_anchor.has_vs_had.mlp4_induced_attention5_head_sweep_v1",
        ),
        (
            "d25d9ca7c01b0ad68f7b834941ab652c4a2dfe3bcc8e6e3bda548b6f802349b6",
            "e8e19829a024f60044da1ba81a78eea75691f932ecbf5e3cb8ab0d09a3b86d9f",
        ),
        (
            "649cc961fd4203a9d7489344bbf169754081a288b5d575bcefcab2caf41da9ab",
            "de2aaeb66b493a7b087f44d7c9e506861eb84cde17ef786d00613504b298ae63",
        ),
        (
            "5449d45505267f2ebc92c9285b7a984fce773f834c2071e1d7e6d1a9f1949372",
            "b81fd30c8c302448ab5b8547356b336fd3eba9a8dda75b11595ea9c70cf8741f",
        ),
        (
            "class HeadSweepBackend(path.PathBackend):",
            "class HeadSweepBackend(Attention5Backend):",
        ),
        (
            "if parent.get(\"terminal\") != \"null\":\n        raise ExperimentError(\"parent terminal changed\")\n    if parent[\"score\"][\"bank_to_writer_retained_fraction\"] >= 0.40:\n        raise ExperimentError(\"parent null basis changed\")",
            "if parent.get(\"terminal\") != \"screen\":\n        raise ExperimentError(\"parent terminal changed\")\n    if parent[\"score\"][\"dominant_factor\"] != \"attention5\":\n        raise ExperimentError(\"parent attention5 decision changed\")",
        ),
        ("    \"h1_h4\",\n", ""),
        ("                \"h1_h4\": (1, 4),\n", ""),
        ("len(ARMS) != 21", "len(ARMS) != 20"),
        ("MODEL_FORWARDS_MAX = 50", "MODEL_FORWARDS_MAX = 48"),
        ("EXAMPLE_EVALUATIONS_MAX = 1600", "EXAMPLE_EVALUATIONS_MAX = 1536"),
        (
            "if name not in {\"h1\", \"h4\"}\n        and attributions[name][\"endpoint_average\"] >= 0.03",
            "if attributions[name][\"endpoint_average\"] >= 0.03",
        ),
        (
            "pred_b = (\n        abs(writer_target - 0.33379277118533013) <= 0.02\n        and abs(targets[\"h1_h4\"] - 0.13009089135863688) <= 0.02\n        and all(\n            summaries[arm][\"families\"][family][\"mean_recovery\"] > 0.0\n            and summaries[arm][\"families\"][family][\"direction_fraction\"] >= 0.80\n            for arm in (\"writer_two_term\", \"h1_h4\")\n            for family in (\"A1\", \"A2\")\n        )\n    )",
            "pred_b = abs(writer_target - 0.33379277118533013) <= 0.02 and all(summaries[\"writer_two_term\"][\"families\"][family][\"mean_recovery\"] > 0.0 and summaries[\"writer_two_term\"][\"families\"][family][\"direction_fraction\"] >= 0.80 for family in (\"A1\", \"A2\"))",
        ),
        (
            "pred_c = all_retained >= 0.50 and all(\n        summaries[\"all_heads\"][\"families\"][family][\"mean_recovery\"] > 0.0\n        and summaries[\"all_heads\"][\"families\"][family][\"direction_fraction\"] >= 0.80\n        for family in (\"A1\", \"A2\")\n    )",
            "pred_c = abs(all_target - 0.055312640920517714) <= 0.01 and all(summaries[\"all_heads\"][\"families\"][family][\"mean_recovery\"] >= 0.04 and summaries[\"all_heads\"][\"families\"][family][\"direction_fraction\"] >= 0.80 for family in (\"A1\", \"A2\"))",
        ),
        ("\"pred_b_writer_and_h1h4_recurrence\": pred_b", "\"pred_b_writer_recurrence\": pred_b"),
        ("\"pred_c_all_head_mediation\": pred_c", "\"pred_c_attention5_recurrence\": pred_c"),
        ("\"pred_d_additional_head\": pred_d", "\"pred_d_head_localization\": pred_d"),
    )
    for old, new in replacements:
        source = replace_once(source, old, new)
    for old, new in (
        ("transformer.h[9]", "transformer.h[5]"),
        ("additional_l9_heads_carry_mlp4_induced_signal", "attention5_head_localization"),
        ("no_additional_direct_l9_head_route", "attention5_head_localization_failed"),
        ("head_sweep_instrument_recurrence_or_coverage_invalid", "attention5_head_sweep_instrument_recurrence_or_coverage_invalid"),
        ("aspectual_anchor_mlp4_induced_l9_head_sweep", "aspectual_anchor_mlp4_induced_attention5_head_sweep"),
        ("licensed_additional_heads", "licensed_attention5_heads"),
        ("factor the source positions read by the licensed additional L9 heads", "factor the source positions read by the licensed attention5 heads"),
    ):
        if old not in source:
            raise RuntimeError(f"missing template token: {old!r}")
        source = source.replace(old, new)
    namespace = {
        "__name__": "__main__", "__file__": str(SCRIPT_FILE),
        "__package__": None, "__cached__": None,
        "Attention5Backend": Attention5Backend,
    }
    exec(compile(source, str(SCRIPT_FILE), "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
