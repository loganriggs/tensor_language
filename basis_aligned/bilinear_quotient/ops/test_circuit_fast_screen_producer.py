from __future__ import annotations

import ast
from pathlib import Path

import pytest

import circuit_battery_integration_contract as battery
import circuit_experiment_spec as framework
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer
import circuit_fast_screen_spec as screen


SOURCE = Path(__file__).with_name("circuit_fast_screen_producer.py")


def rows(groups: int = 1, *, c_answer_changes: bool = True) -> list[dict[str, object]]:
    output = []
    for group in range(groups):
        for index, family in enumerate(screen.TRANSFORMS):
            changes = family not in ({"P"} if c_answer_changes else {"P", "C"})
            first, second = 10 + 2 * index, 11 + 2 * index
            if group % 2:
                first, second = second, first
            output.append({
                "row_id": f"FIT:g{group}:{family}", "group_id": f"FIT:g{group}",
                "split": "FIT", "task_id": "fixture.behavior", "transform_id": family,
                "answer_changes": changes,
                "capability_cell_id": f"{family}/{first}/{second if changes else first}",
                "base_ids": [100 + group, 200 + index, 300],
                "donor_ids": [400 + group, 500, 600 + index, 700],
                "base_answer_id": first, "base_foil_id": second,
                "donor_answer_id": second if changes else first,
                "donor_foil_id": first if changes else second,
                "base_position": 1, "donor_position": 2,
            })
    return output


def task(*, c_answer_changes: bool = True) -> battery.BatteryTaskSpec:
    return battery.BatteryTaskSpec(
        task_id="fixture.behavior", generator_role="fixture", answer_role="pair",
        transforms=(
            battery.TransformSpec("A1", "a1", True, "toward_donor"),
            battery.TransformSpec("A2", "a2", True, "toward_donor"),
            battery.TransformSpec("P", "p", False, "invariant"),
            battery.TransformSpec("C", "c", c_answer_changes, "registered_active"),
        ),
    )


def spec(authority, *, c_answer_changes: bool = True) -> screen.CircuitFastScreenSpec:
    count = len(authority)
    maximum_evaluations = 264 * (count // 4)
    return screen.CircuitFastScreenSpec(
        experiment_id="fixture-screen",
        hypothesis=screen.CandidateHypothesis(
            behavior="fixture.behavior", answer_score=screen.ANSWER_SCORE,
            information_read="marked state", proposed_operation="transport it",
            proposed_write="answer-relevant state", candidate_sites=screen.CEILING_SITE_IDS,
            alternative_explanation="lexical shortcut", circuit_prediction="selective transfer",
            opposing_null_prediction="no selective transfer",
        ),
        task=task(c_answer_changes=c_answer_changes),
        authority_sha256=framework.canonical_sha256(authority),
        expected_fit_rows=count, batch_size=count // 4,
        semantic_position=screen.SemanticPositionSpec(
            "marked", "base_position", "donor_position"
        ),
        fields=screen.AuthorityFieldSpec(), bars=kernel.FIXED_BARS,
        declared_max_price=battery.ExactPhasePrice(
            "FIT", 264, maximum_evaluations, 0, 0, 8 * maximum_evaluations
        ),
    )


class FakeBackend:
    def __init__(self, *, weak_capability: bool = False, attention_parent: bool = False):
        self.weak_capability = weak_capability
        self.attention_parent = attention_parent
        self.native_calls = []
        self.patch_calls = []

    def native(self, batch, *, capture):
        self.native_calls.append((batch, capture))
        pairs = []
        captured = {}
        for row_id, tokens, position in zip(
            batch.row_ids, batch.token_rows, batch.semantic_positions
        ):
            bad = self.weak_capability and batch.side == "base" \
                and (row_id.endswith("g0:A2") or row_id.endswith("g2:A2"))
            pairs.append((0.0, 1.0) if bad else (1.0, 0.0))
            if capture:
                for site_id in screen.CEILING_SITE_IDS:
                    captured[(row_id, site_id)] = tokens[position]
                for layer in range(18):
                    for head in range(9):
                        captured[(row_id, f"attn:{layer:02d}:head:{head:02d}")] = tokens[position]
        return producer.BatchOutput(tuple(pairs), captured)

    def patched(self, batch, *, site, donor_cache):
        self.patch_calls.append((batch, site))
        # Exercise the same row-position gather/scatter contract on unequal lengths.
        donors = tuple((0, 0, donor_cache[(row_id, site.site_id)], 0)
                       for row_id in batch.row_ids)
        changed = producer.replace_declared_positions(
            batch.token_rows, batch.semantic_positions, donors, (2,) * len(donors)
        )
        for before, after, position in zip(
            batch.token_rows, changed, batch.semantic_positions
        ):
            assert [i for i, pair in enumerate(zip(before, after)) if pair[0] != pair[1]] \
                   == [position]
        target_recovery = 0.60
        if self.attention_parent and site.site_id == "attn:00":
            target_recovery = 0.90
        if site.evidence_kind == "head":
            target_recovery = 0.95
        pairs = []
        for row_id in batch.row_ids:
            family = row_id.rsplit(":", 1)[1]
            if family in {"A1", "A2"}:
                own_margin = 1.0 - 2.0 * target_recovery
            elif family == "P":
                own_margin = 0.95
            else:  # unrelated C transfer stays near zero
                own_margin = 0.90
            pairs.append((own_margin, 0.0))
        return producer.BatchOutput(tuple(pairs), {})


def test_position_gather_scatter_changes_only_declared_row_positions() -> None:
    recipient = ((1, 2, 3), (4, 5, 6, 7, 8))
    donor = ((9, 10, 11, 12), (13, 14))
    changed = producer.replace_declared_positions(recipient, (1, 4), donor, (2, 0))
    assert changed == ((1, 11, 3), (4, 5, 6, 7, 13))
    assert tuple(len(row) for row in changed) == (3, 5)


def test_all_55_routes_screen_and_residual_module_scores_agree() -> None:
    authority = rows()
    backend = FakeBackend()
    ticks = iter((10.0, 12.5))
    result = producer.run_science(
        spec(authority), authority, backend=backend, clock=lambda: next(ticks)
    )
    assert result.terminal == "screen"
    assert result.head_stage == "skipped_parent_not_attention"
    assert len(result.site_results) == 55
    by_site = {item.site.site_id: item for item in result.site_results}
    assert by_site["resid:00"].target_recovery == by_site["attn:00"].target_recovery
    assert by_site["attn:00"].target_recovery == by_site["mlp:00"].target_recovery
    assert result.timing.forward_calls == 228
    assert result.timing.example_evaluations == 228
    assert result.timing.seconds == 2.5
    assert len(result.native_logits) == 8
    assert len(result.intervention_logits) == 220
    assert (result.native_logits[0].answer_logit, result.native_logits[0].foil_logit) \
           == (1.0, 0.0)


def test_same_answer_c_control_runs_all_sites_as_normalized_invariance() -> None:
    authority = rows(c_answer_changes=False)
    result = producer.run_science(
        spec(authority, c_answer_changes=False), authority, backend=FakeBackend()
    )
    assert result.terminal == "screen"
    assert len(result.site_results) == 55
    assert all(item.c_direction_fraction is None for item in result.site_results)
    assert all(abs((item.c_absolute_recovery or 0.0) - 0.05) < 1e-12
               for item in result.site_results)


def test_capability_cell_failure_stops_before_interventions() -> None:
    authority = rows()
    backend = FakeBackend(weak_capability=True)
    result = producer.run_science(spec(authority), authority, backend=backend)
    assert result.terminal == "null"
    assert result.reason == "native_behavior_incapable"
    assert result.head_stage == "capability_stop"
    assert len(backend.native_calls) == 8
    assert backend.patch_calls == []
    assert any(cell.family == "A2" and not cell.passed for cell in result.capability_cells)
    assert result.timing.forward_calls == 8


def test_ordered_capability_cell_cannot_hide_in_passing_family_pool() -> None:
    authority = rows(8)
    result = producer.run_science(
        spec(authority), authority, backend=FakeBackend(weak_capability=True)
    )
    assert result.terminal == "null"
    a2_cells = [cell for cell in result.capability_cells if cell.family == "A2"]
    assert [cell.accuracy for cell in a2_cells] == [0.75, 1.0]
    assert [(cell.base_accuracy, cell.donor_accuracy) for cell in a2_cells] \
           == [(0.5, 1.0), (1.0, 1.0)]
    assert sum(cell.correct_count for cell in a2_cells) / sum(
        cell.expected_count for cell in a2_cells
    ) == 0.875
    assert not a2_cells[0].passed and a2_cells[1].passed


def test_heads_expand_only_for_selected_passing_attention_parent() -> None:
    authority = rows()
    backend = FakeBackend(attention_parent=True)
    result = producer.run_science(spec(authority), authority, backend=backend)
    assert result.terminal == "screen"
    assert result.head_stage == "expanded"
    assert len(result.site_results) == 64
    head_sites = [
        item.site.site_id
        for item in result.site_results
        if item.site.evidence_kind == "head"
    ]
    assert head_sites == [f"attn:00:head:{head:02d}" for head in range(9)]
    assert result.selected_site == kernel.SiteRef("head", "attn:00:head:00")
    assert result.timing.forward_calls == 264
    assert result.timing.example_evaluations == 264


def test_real_backend_head_helper_scatter_is_slice_and_position_local() -> None:
    class Tensor:
        def __init__(self, values):
            self.values = values
            self.shape = (len(values),)

        def to(self, **_kwargs):
            return self.values

    class State:
        device = "fake"
        dtype = "fake"

        def __init__(self):
            self.values = [[[0] * 6 for _ in range(3)]]

        def clone(self):
            copy = State()
            copy.values = [[list(position) for position in row] for row in self.values]
            return copy

        def __setitem__(self, key, value):
            row, position, interval = key
            self.values[row][position][interval] = value

    backend = object.__new__(producer.Bilin18TorchBackend)
    backend.model = type("Model", (), {
        "config": type("Config", (), {"n_embd": 6, "n_head": 3})()
    })()
    batch = producer.ModelBatch(("row",), "base", ((1, 2, 3),), (1,), (2,), (1,))
    state = State()
    changed = backend._replace_head(  # noqa: SLF001 - exact regression for real helper
        state, batch, "attn:00:head:01", {("row", "attn:00:head:01"): Tensor([7, 8])}
    )
    assert state.values == [[[0] * 6 for _ in range(3)]]
    assert changed.values == [
        [[0, 0, 0, 0, 0, 0], [0, 0, 7, 8, 0, 0], [0, 0, 0, 0, 0, 0]]
    ]


def test_import_and_dryrun_have_no_torch_fastload_or_cuda_access(monkeypatch) -> None:
    tree = ast.parse(SOURCE.read_text())
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imports.isdisjoint({"torch", "fastload"})
    authority = rows()

    def forbidden(name, *args, **kwargs):
        if name.startswith("torch") or name == "fastload":
            raise AssertionError("dryrun imported model dependency")
        return original(name, *args, **kwargs)

    original = producer.importlib.import_module
    monkeypatch.setattr(producer.importlib, "import_module", forbidden)
    receipt = producer.compile_dryrun(spec(authority), authority)
    assert receipt["model_loaded"] is False
    assert receipt["gpu_accessed"] is False
    assert receipt["model_forwards"] == 0


def test_nonfinite_backend_evidence_is_invalid() -> None:
    authority = rows()

    class BadBackend(FakeBackend):
        def __init__(self):
            super().__init__()

        def native(self, batch, *, capture):
            output = super().native(batch, capture=capture)
            return producer.BatchOutput(((float("nan"), 0.0),), output.captured)

    backend = BadBackend()
    result = producer.run_science(spec(authority), authority, backend=backend)
    assert result.terminal == "invalid"
    assert result.reason == "execution_invalid"
    assert len(backend.native_calls) == 1
