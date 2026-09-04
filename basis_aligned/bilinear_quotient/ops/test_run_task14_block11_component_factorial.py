#!/usr/bin/env python3
"""CPU-only tests for the Task 14 fixed-component block-11 factorial."""

from __future__ import annotations

import json
import pytest

import circuit_prior_art
import run_task14_block11_component_factorial as subject


class FakeBackend:
    def __init__(self, *, break_empty_replay: bool = False) -> None:
        rows, parent, _prior = subject.load_sources()
        self.native, self.interventions = subject._parent_maps(parent)  # noqa: SLF001
        self.family_by_row = {
            str(row["row_id"]): str(row["transform_id"]) for row in rows
        }
        self.break_empty_replay = break_empty_replay
        self.capture_calls = []
        self.patch_calls = []

    def capture_native(self, batch):
        self.capture_calls.append(batch)
        pairs = tuple(self.native[(row_id, batch.side)] for row_id in batch.row_ids)
        captured = {
            (row_id, f"block11:{batch.side}:{factor}"): object()
            for row_id in batch.row_ids for factor in subject.FACTORS
        }
        return subject.producer.BatchOutput(pairs, captured)

    def patched(self, batch, *, mask, component_cache):
        self.patch_calls.append((batch, mask, component_cache))
        pairs = []
        recovery = {1: 0.25, 2: 0.55, 3: 0.85, 5: 0.35, 6: 0.60}
        for row_id in batch.row_ids:
            family = self.family_by_row[row_id]
            if mask == 0:
                pair = self.native[(row_id, "base")]
                if self.break_empty_replay:
                    pair = (pair[0] + 0.01, pair[1])
            elif mask == 4:
                pair = self.interventions[("mlp:11", row_id)]
            elif mask == 7:
                pair = self.interventions[("resid:12", row_id)]
            elif family in {"P", "C"}:
                pair = self.native[(row_id, "base")]
            else:
                base = self.native[(row_id, "base")]
                donor = self.native[(row_id, "donor")]
                base_margin = base[0] - base[1]
                donor_margin = donor[0] - donor[1]
                patched_margin = base_margin - recovery[mask] * (
                    base_margin + donor_margin
                )
                pair = (patched_margin, 0.0)
            pairs.append(pair)
        return subject.producer.BatchOutput(tuple(pairs), {})


def test_prior_art_and_dryrun_bind_exact_sources_without_model_access() -> None:
    prior = json.loads(subject.PRIOR_ART.read_text())
    assert circuit_prior_art.validate_source_files(prior, subject.ROOT) == \
        subject.PRIOR_ART_SHA256
    dryrun = subject.compile_dryrun()
    assert dryrun["model_loaded"] is False
    assert dryrun["gpu_accessed"] is False
    assert dryrun["queue_touched"] is False
    assert dryrun["price"] == {
        "forward_calls": 40,
        "example_evaluations": 1280,
        "backward_calls": 0,
        "model_updates": 0,
        "evidence_bytes": 10240,
    }
    assert dryrun["arms"] == [
        "empty", "R", "A", "R+A", "M", "R+M", "A+M", "R+A+M",
    ]


def test_cli_requires_explicit_valid_arguments_and_supports_dry_run() -> None:
    args = subject.parse_args(["--dry-run"])
    assert args.dry_run is True
    with pytest.raises(SystemExit):
        subject.parse_args(["--unknown-argument"])


def test_fixed_component_source_map_and_assembly_use_each_bit_once() -> None:
    assert {
        factor: subject.component_source(3, factor) for factor in subject.FACTORS
    } == {"R": "donor", "A": "donor", "M": "base"}

    class Vector:
        def __init__(self, value):
            self.value = value

        def to(self, **_kwargs):
            return self

        def __add__(self, other):
            return Vector(self.value + other.value)

    class State:
        device = "fake"
        dtype = "fake"

        def __init__(self):
            self.values = [[Vector(0), Vector(0), Vector(0)]]

        def clone(self):
            copy = State()
            copy.values = [[Vector(item.value) for item in row] for row in self.values]
            return copy

        def __setitem__(self, key, value):
            row, position = key
            self.values[row][position] = value

    backend = object.__new__(subject.Block11FactorialTorchBackend)
    batch = subject.producer.ModelBatch(
        ("row",), "base", ((1, 2, 3),), (1,), (2,), (1,),
    )
    cache = {
        ("row", f"block11:{side}:{factor}"): Vector(value)
        for side, offset in (("base", 0), ("donor", 10))
        for factor, value in zip(subject.FACTORS, (1 + offset, 2 + offset, 4 + offset))
    }
    changed = backend._assemble(State(), batch, 3, cache)  # noqa: SLF001
    assert [item.value for item in changed.values[0]] == [0, 27, 0]


def test_fake_full_run_scores_all_eight_arms_and_closes_mobius_exactly() -> None:
    backend = FakeBackend()
    ticks = iter((10.0, 12.0))
    result = subject.run_science(backend=backend, clock=lambda: next(ticks))
    assert result["terminal"] == "screen"
    assert all(result["predictions"].values())
    assert result["timing"] == {
        "forward_calls": 40,
        "example_evaluations": 1280,
        "evidence_bytes": 10240,
        "seconds": 2.0,
    }
    assert len(backend.capture_calls) == 8
    assert len(backend.patch_calls) == 32
    assert list(result["arm_scores"]) == [
        "empty", "R", "A", "R+A", "M", "R+M", "A+M", "R+A+M",
    ]
    assert result["interactions"]["closure_max_abs"] < 1.0e-12
    assert set(result["interactions"]["equal_weight_A1_A2_terms"]) == {
        "R", "A", "R+A", "M", "R+M", "A+M", "R+A+M",
    }


def test_parent_replay_failure_is_instrument_invalid_not_scientific_null() -> None:
    result = subject.run_science(backend=FakeBackend(break_empty_replay=True))
    assert result["terminal"] == "invalid"
    assert result["predictions"]["pred_a_instrument_replays_parent"] is False
