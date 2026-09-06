"""Reusable residual-depth interventions on declared semantic source groups."""

# BQGATE: LIBRARY
from __future__ import annotations

import math
import statistics

import attention_source_group_eval as source_groups
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer


class ResidualOnsetError(RuntimeError):
    pass


def positions_for_group(base_batch, donor_batch, group_name):
    if group_name == "all_positions":
        if tuple(base_batch.semantic_positions) != tuple(donor_batch.semantic_positions):
            raise ResidualOnsetError("all-position intervention requires aligned query positions")
        return tuple(tuple(range(int(query) + 1)) for query in base_batch.semantic_positions)
    if group_name not in source_groups.GROUP_ORDER:
        raise ResidualOnsetError("unknown semantic source group")
    partitions = source_groups.batch_partitions(base_batch, donor_batch)
    selected = tuple(tuple(mapping[group_name]) for mapping in partitions)
    if any(not positions for positions in selected):
        raise ResidualOnsetError("selected source group is empty in at least one row")
    return selected


class ResidualGroupBackend(producer.Bilin18TorchBackend):
    def forward_states(
        self,
        batch,
        *,
        maximum_boundary,
        donor_batch=None,
        donor_states=None,
        boundary=None,
        group_name=None,
    ):
        maximum_boundary = int(maximum_boundary)
        if not 0 <= maximum_boundary <= len(self.model.transformer.h):
            raise ResidualOnsetError("maximum residual boundary is invalid")
        patching = boundary is not None
        if patching:
            boundary = int(boundary)
            if (
                not 0 <= boundary <= maximum_boundary
                or donor_batch is None
                or donor_states is None
                or group_name is None
            ):
                raise ResidualOnsetError("residual source intervention is incomplete")
            positions = positions_for_group(batch, donor_batch, group_name)
            if len(donor_states) != maximum_boundary + 1:
                raise ResidualOnsetError("donor residual-state coverage changed")
        elif donor_batch is not None or donor_states is not None or group_name is not None:
            raise ResidualOnsetError("native state capture has intervention inputs")
        torch, F, model = self.torch, self.F, self.model
        tokens, lengths = self._tensor_batch(batch)

        def patch(value, at_boundary):
            if not patching or boundary != at_boundary:
                return value
            changed = value.clone()
            donor_value = donor_states[at_boundary]
            for index, row_positions in enumerate(positions):
                for position in row_positions:
                    changed[index, position] = donor_value[index, position].to(
                        device=value.device, dtype=value.dtype
                    )
            return changed

        captured = []
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
            x0 = x
            captured.append(x.detach().clone())
            x = patch(x, 0)
            v1 = None
            for layer, block in enumerate(model.transformer.h):
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, v1 = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1)
                x = live + attention
                x = x + block.mlp(F.rms_norm(x, (model.config.n_embd,)))
                if layer + 1 <= maximum_boundary:
                    captured.append(x.detach().clone())
                    x = patch(x, layer + 1)
            logits = 30.0 * torch.tanh(
                model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30.0
            )
            values = tuple(
                (
                    float(logits[index, length - 1, batch.answer_ids[index]].float()),
                    float(logits[index, length - 1, batch.foil_ids[index]].float()),
                )
                for index, length in enumerate(lengths)
            )
        if len(captured) != maximum_boundary + 1:
            raise ResidualOnsetError("residual capture coverage changed")
        return producer.BatchOutput(values, {}), tuple(captured)


def recovery_records(rows, base_output, donor_output, patched_output, *, group, boundary):
    records = []
    for row, base_pair, donor_pair, patched_pair in zip(
        rows,
        base_output.answer_foil,
        donor_output.answer_foil,
        patched_output.answer_foil,
    ):
        base_margin = float(base_pair[0]) - float(base_pair[1])
        donor_margin = float(donor_pair[0]) - float(donor_pair[1])
        patched_margin = float(patched_pair[0]) - float(patched_pair[1])
        if any(not math.isfinite(value) for value in (base_margin, donor_margin, patched_margin)):
            raise ResidualOnsetError("nonfinite recovery input")
        recovery = kernel.signed_pairwise_donor_recovery(
            -base_margin, donor_margin, -patched_margin
        )
        records.append(
            {
                "group": str(group),
                "boundary": int(boundary),
                "family": str(row["transform_id"]),
                "direction": str(row["direction_id"]),
                "row_id": str(row["row_id"]),
                "recovery": recovery,
            }
        )
    return records


def summarize(records):
    values = [float(record["recovery"]) for record in records]
    if not values or any(not math.isfinite(value) for value in values):
        raise ResidualOnsetError("missing or nonfinite residual-onset recovery")
    return {
        "count": len(values),
        "mean_recovery": statistics.fmean(values),
        "mean_absolute_recovery": statistics.fmean(abs(value) for value in values),
        "direction_fraction": sum(value > 0.0 for value in values) / len(values),
    }


def curve(records, group, boundaries, families=("A1", "A2"), *, recovery_bar=0.5, direction_bar=0.8):
    points = []
    for boundary in boundaries:
        family_summaries = {
            family: summarize(
                [
                    record
                    for record in records
                    if record["group"] == group
                    and record["boundary"] == boundary
                    and record["family"] == family
                ]
            )
            for family in families
        }
        passed = all(
            family_summaries[family]["mean_recovery"] >= recovery_bar
            and family_summaries[family]["direction_fraction"] >= direction_bar
            for family in families
        )
        points.append(
            {
                "boundary": int(boundary),
                "families": family_summaries,
                "mean_target_recovery": statistics.fmean(
                    family_summaries[family]["mean_recovery"] for family in families
                ),
                "passed": passed,
            }
        )
    return points


def earliest_passing(points):
    return next((int(point["boundary"]) for point in points if point["passed"]), None)


def sweep_precomputed_states(
    backend,
    items,
    *,
    boundaries,
    group_name,
    maximum_boundary,
    recovery_bar,
    direction_bar,
):
    """Patch precomputed causal states across boundaries and score one source group.

    Each item supplies rows, base/donor batches and outputs, plus ``donor_states``.
    The donor states may come from a natural donor or from an upstream intervention;
    this function deliberately does not assume how they were produced.
    """
    selected_boundaries = tuple(int(boundary) for boundary in boundaries)
    if (
        not selected_boundaries
        or len(selected_boundaries) != len(set(selected_boundaries))
        or tuple(sorted(selected_boundaries)) != selected_boundaries
        or any(not 0 <= boundary <= int(maximum_boundary) for boundary in selected_boundaries)
    ):
        raise ResidualOnsetError("boundary sweep must be unique, ordered, and in range")
    prepared = tuple(items)
    if not prepared:
        raise ResidualOnsetError("boundary sweep has no family items")
    records = []
    forward_calls = 0
    example_evaluations = 0
    base_scored_logit_max_abs_by_boundary = {str(boundary): 0.0 for boundary in selected_boundaries}
    for boundary in selected_boundaries:
        for item in prepared:
            required = {
                "rows", "base_batch", "donor_batch", "base_output", "donor_output", "donor_states"
            }
            if not required.issubset(item):
                raise ResidualOnsetError("boundary sweep item is incomplete")
            patched_output, _ = backend.forward_states(
                item["base_batch"],
                maximum_boundary=maximum_boundary,
                donor_batch=item["donor_batch"],
                donor_states=item["donor_states"],
                boundary=boundary,
                group_name=group_name,
            )
            forward_calls += 1
            example_evaluations += len(item["rows"])
            base_scored_logit_max_abs_by_boundary[str(boundary)] = max(
                base_scored_logit_max_abs_by_boundary[str(boundary)],
                max(
                    abs(float(actual) - float(base))
                    for actual_pair, base_pair in zip(
                        patched_output.answer_foil, item["base_output"].answer_foil
                    )
                    for actual, base in zip(actual_pair, base_pair)
                ),
            )
            records.extend(
                recovery_records(
                    item["rows"],
                    item["base_output"],
                    item["donor_output"],
                    patched_output,
                    group=group_name,
                    boundary=boundary,
                )
            )
    families = tuple(dict.fromkeys(str(row["transform_id"]) for item in prepared for row in item["rows"]))
    points = curve(
        records,
        group_name,
        selected_boundaries,
        families=families,
        recovery_bar=recovery_bar,
        direction_bar=direction_bar,
    )
    return {
        "records": records,
        "curve": points,
        "forward_calls": forward_calls,
        "example_evaluations": example_evaluations,
        "base_scored_logit_max_abs_by_boundary": base_scored_logit_max_abs_by_boundary,
    }
