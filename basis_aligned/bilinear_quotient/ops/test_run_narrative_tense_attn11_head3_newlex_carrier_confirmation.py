"""Focused CPU tests for the genuinely new narrative carrier confirmation."""
from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import subprocess
import sys

import torch

import circuit_fast_screen_candidate_narrative_tense as old
import circuit_fast_screen_candidate_narrative_tense_fresh_unchanged_carrier as prior
import circuit_fast_screen_candidate_narrative_tense_newlex_carrier_confirmation as authority
import run_narrative_tense_attn11_head3_newlex_carrier_confirmation as run


def test_new_lexical_roles_and_endpoints_are_disjoint_and_frozen():
    assert len(authority.ELIGIBLE_SINGULARS) == 40
    assert set(authority.SUBJECTS).isdisjoint(authority.ALTERNATES)
    assert set(authority.SUBJECTS + authority.ALTERNATES).isdisjoint(
        authority.TAIL_LOCATION_POOL)
    for group, (subject, alternate, tail, location) in enumerate(authority.LEXICAL_TUPLES):
        donor_location = authority.LOCATIONS[(group + 1) % 16]
        assert len({subject, alternate, tail, location, donor_location}) == 5
    old_words = set(old._SUBJECTS + old._ALTERNATES + old._PLACES + old._FOCUS) \
        | set(prior.FRESH_VOCABULARY)
    assert set(authority.ELIGIBLE_SINGULARS).isdisjoint(old_words)
    rows = authority.build_rows()
    assert len(rows) == 64
    assert authority.authority_sha256() == run.AUTHORITY_SHA256
    endpoints = [(row[f"{side}_text"], tuple(row[f"{side}_ids"]))
                 for row in rows for side in ("base", "donor")]
    assert len(set(endpoints)) == 128
    assert len({row["row_id"] for row in rows}) == 64


def test_exact_pair_changes_balance_and_algorithmic_carrier_partition():
    expected = {"A1": (0, 3), "A2": (3, 6, 7), "P": (2,), "C": (2, 6)}
    rows = run.build_rows()
    cells = Counter(row["capability_cell_id"] for row in rows)
    for row in rows:
        base, donor = row["base_ids"], row["donor_ids"]
        changed = tuple(i for i, values in enumerate(zip(base, donor)) if values[0] != values[1])
        assert changed == expected[row["transform_id"]]
        assert len(base) == len(donor) and base[-1] == donor[-1]
        assert all(base[index] == donor[index] for index in row["R_positions"])
        assert set(row["R_positions"]).isdisjoint(row["complement_positions"])
        assert set(row["R_positions"]) | set(row["complement_positions"]) \
            == set(range(len(base)))
    assert sorted(value for cell, value in cells.items() if cell.startswith(("A1/", "A2/"))) \
        == [8, 8, 8, 8]
    assert sorted(value for cell, value in cells.items() if cell.startswith(("P/", "C/"))) \
        == [4] * 8


def test_runner_uses_shared_exact_subset_primitive():
    native = {"p": torch.tensor([[.2, .8]]),
              "u": torch.tensor([[[1., 2.], [3., 4.]]])}
    donor = {"p": torch.tensor([[.6, .4]]),
             "u": torch.tensor([[[5., 6.], [7., 8.]]])}
    native["head"] = torch.einsum("bk,bkd->bd", native["p"], native["u"])
    donor["head"] = torch.einsum("bk,bkd->bd", donor["p"], donor["u"])
    observed = run._subset_head(native, donor, (0,), "value", torch)
    expected = native["head"] - .2 * native["u"][:, 0] + .2 * donor["u"][:, 0]
    assert torch.allclose(observed, expected)
    assert torch.equal(run._subset_head(native, donor, (), "joint", torch), native["head"])


def test_plan_price_bars_and_no_model_path():
    plan = run.compile_plan()
    assert plan["price"] == {"model_forwards": 6, "example_evaluations": 768,
                             "backwards": 0, "parameter_updates": 0}
    assert plan["bars"]["minimum_native_accuracy_each_direction_side_cell"] == .875
    assert plan["registered_predictions"] == list(run.REGISTERED_PREDICTIONS)
    assert plan["subset_primitive"].endswith("replace_head_source_subset")
    env = dict(os.environ, BQLIB_NO_MODEL="1", PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run([sys.executable, str(Path(run.__file__))], env=env,
                               check=True, capture_output=True, text=True)
    assert '"model_loaded": false' in completed.stdout
