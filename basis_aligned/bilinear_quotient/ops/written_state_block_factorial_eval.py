"""Exact full-sequence block factorial after an upstream source-resolved write."""

# BQGATE: LIBRARY
from __future__ import annotations

import attention_path_mediation_eval as mediation
import attention_source_group_eval as source_score
import block_component_state_eval as components


class WrittenStateBlockFactorialError(RuntimeError):
    pass


def pair_error(first, second):
    return max(
        abs(float(a) - float(b))
        for pair_a, pair_b in zip(first.answer_foil, second.answer_foil)
        for a, b in zip(pair_a, pair_b)
    )


def capture_crossing(
    backend,
    base_batch,
    donor_batch,
    writer_base_capture,
    writer_donor_capture,
    writer_destinations,
    *,
    block_index,
    writer_layer=8,
    writer_heads=(1,),
    writer_groups=("cue",),
    input_group="subject_onset",
):
    """Capture native and writer-input components around one downstream block."""
    output_boundary = int(block_index) + 1
    writer_output, writer_states = mediation.capture_source_written_states(
        backend,
        base_batch,
        donor_batch,
        writer_base_capture,
        writer_donor_capture,
        writer_destinations,
        maximum_boundary=output_boundary,
        writer_layer=writer_layer,
        writer_heads=writer_heads,
        writer_groups=writer_groups,
    )
    base_output, base_states, base_components, base_error = components.capture(
        backend,
        base_batch,
        block_index,
        lambda: backend.forward_states(base_batch, maximum_boundary=output_boundary),
    )
    hybrid_output, hybrid_states, hybrid_components, hybrid_error = components.capture(
        backend,
        base_batch,
        block_index,
        lambda: backend.forward_states(
            base_batch,
            maximum_boundary=output_boundary,
            donor_batch=donor_batch,
            donor_states=writer_states,
            boundary=block_index,
            group_name=input_group,
        ),
    )
    return {
        "writer_output": writer_output,
        "writer_states": writer_states,
        "base_output": base_output,
        "base_states": base_states,
        "base_components": base_components,
        "hybrid_output": hybrid_output,
        "hybrid_states": hybrid_states,
        "hybrid_components": hybrid_components,
        "block_reconstruction_max_abs": max(float(base_error), float(hybrid_error)),
    }


def run_full_sequence_factorial(backend, items, *, block_index):
    output_boundary = int(block_index) + 1
    subsets = components.subsets()
    records = []
    forward_calls = 0
    example_evaluations = 0
    full_state_closure_max_abs = 0.0
    empty_base_closure_max_abs = 0.0
    direct_full_scored_logit_max_abs = 0.0
    for subset in subsets:
        arm = components.arm_id(subset)
        for item in items:
            crossing = item["crossing"]
            if not subset:
                output = item["base_output"]
                empty_base_closure_max_abs = max(
                    empty_base_closure_max_abs,
                    pair_error(output, crossing["base_output"]),
                )
            else:
                state = components.assemble(
                    crossing["base_components"], crossing["hybrid_components"], subset
                )
                if subset == components.COMPONENTS:
                    full_state_closure_max_abs = max(
                        full_state_closure_max_abs,
                        float(
                            (state.float() - crossing["hybrid_states"][output_boundary].float())
                            .abs().max()
                        ),
                    )
                output, _ = backend.forward_states(
                    item["base_batch"],
                    maximum_boundary=output_boundary,
                    donor_batch=item["donor_batch"],
                    donor_states=tuple(state for _ in range(output_boundary + 1)),
                    boundary=output_boundary,
                    group_name="all_positions",
                )
                forward_calls += 1
                example_evaluations += len(item["rows"])
                if subset == components.COMPONENTS:
                    direct_full_scored_logit_max_abs = max(
                        direct_full_scored_logit_max_abs,
                        pair_error(output, crossing["hybrid_output"]),
                    )
            records.extend(
                source_score.recovery_records(
                    item["rows"], item["base_output"], item["donor_output"], output, arm=arm
                )
            )
    summaries = {
        components.arm_id(subset): source_score.summarize_by_family(
            [record for record in records if record["arm"] == components.arm_id(subset)]
        )
        for subset in subsets
    }
    shapley = {}
    shapley_efficiency_max_abs = 0.0
    families = tuple(dict.fromkeys(record["family"] for record in records))
    for family in families:
        accounting = components.factorial_accounting(
            {
                subset: summaries[components.arm_id(subset)][family]["mean_recovery"]
                for subset in subsets
            }
        )
        shapley[family] = accounting["shapley"]
        shapley_efficiency_max_abs = max(
            shapley_efficiency_max_abs, float(accounting["efficiency_error"])
        )
    return {
        "records": records,
        "summaries": summaries,
        "shapley": shapley,
        "forward_calls": forward_calls,
        "example_evaluations": example_evaluations,
        "full_state_closure_max_abs": full_state_closure_max_abs,
        "empty_base_closure_max_abs": empty_base_closure_max_abs,
        "direct_full_scored_logit_max_abs": direct_full_scored_logit_max_abs,
        "shapley_efficiency_max_abs": shapley_efficiency_max_abs,
    }
