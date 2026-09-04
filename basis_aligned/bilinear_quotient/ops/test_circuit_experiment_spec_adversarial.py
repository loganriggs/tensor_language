"""Adversarial fixtures for the proposed CircuitExperimentSpec compiler/runtime (Claude, 2026-09-04; red-team companion to
polynomial_causal/CIRCUIT_EXPERIMENT_SPEC_REDTEAM_2026-09-04.md). Written BEFORE the implementation, against the interface named in
the 2026-09-04 audit: compile_experiment / validate_call_evidence / project_result / stage_and_publish / managed_main.

Two layers:
  1. PLANTED-ATTACK GENERATORS (pure Python, run today): a synthetic authority (rows, splits, groups), a call manifest, primitive
     evidence, and one mutation per attack. Each generator self-tests that the attack is NON-VACUOUS (it preserves whatever the
     naive check looks at and changes only what the real invariant must see).
  2. API TESTS (importorskip until ops/circuit_experiment_spec.py exists): each feeds one planted attack to the API and requires a
     refusal. Expected exception type is any Exception whose message mentions the listed keyword (case-insensitive) -- Codex may
     rename the class; the keyword is the contract.

Standard library only. No model, no filesystem outside tmp_path, no GPU.
"""
import copy
import hashlib
import json
import os
import pathlib
import random

import pytest

API_MODULE = "circuit_experiment_spec"


ADAPTER_SURFACE = ("build_synthetic_spec", "with_projector", "with_artifact", "package_evidence", "verify_package", "dump_package",
                   "load_package", "begin_run", "dump_spec", "render_managed_entry", "stage_dir_for", "managed_main",
                   "validate_call_evidence", "project_result", "stage_and_publish", "compile_experiment")


def _api():
    """ops/circuit_experiment_spec.py exists since 2026-09-04 01:19Z (Codex, in progress). The API tests need the small adapter surface
    named in ADAPTER_SURFACE (documented in _spec below); until it is wired they SKIP with the missing names listed, so this file is
    never red noise in Codex's runs -- and never silently green either."""
    api = pytest.importorskip(API_MODULE)
    missing = [n for n in ADAPTER_SURFACE if not hasattr(api, n)]
    if missing:
        pytest.skip(f"adapter surface not wired yet: {missing}")
    return api


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _raises_with(keyword):
    class _Ctx:
        def __enter__(self):
            self.ctx = pytest.raises(Exception); self.info = self.ctx.__enter__(); return self.info
        def __exit__(self, et, ev, tb):
            out = self.ctx.__exit__(et, ev, tb)
            if out:
                assert keyword.lower() in str(self.info.value).lower(), f"refused, but not for the registered reason {keyword!r}: {self.info.value}"
            return out
    return _Ctx()


# ----------------------------------------------------------------------------------------------------------------------
# 1. Planted-attack generators
# ----------------------------------------------------------------------------------------------------------------------

def make_authority(n_fit=12, n_select=6, seed=0):
    """Synthetic authority: rows with content-derived IDs, two splits, groups of 3, one semantic field."""
    rng = random.Random(seed); rows = []
    for split, n in (("FIT", n_fit), ("SELECT", n_select)):
        for i in range(n):
            content = f"{split}-doc{i}-" + "".join(rng.choice("abcdefgh") for _ in range(6))
            rows.append({"row_id": _sha(content.encode())[:16], "split": split, "group_id": f"{split}-g{i // 3}", "answer_pos": 5 + (i % 4), "content": content})
    return rows


def split_hash(rows, split):
    return _sha(_canon(sorted(r["row_id"] for r in rows if r["split"] == split)))


def attack_split_swap(rows):
    """Swap one FIT row and one SELECT row between splits (relabel + regroup). Preserves every per-split COUNT and per-group size;
    changes only the per-split content hash."""
    out = copy.deepcopy(rows)
    f = next(r for r in out if r["split"] == "FIT"); s = next(r for r in out if r["split"] == "SELECT")
    f["split"], s["split"] = "SELECT", "FIT"; f["group_id"], s["group_id"] = s["group_id"], f["group_id"]
    return out


def test_generator_split_swap_is_count_preserving():
    rows = make_authority(); bad = attack_split_swap(rows)
    for sp in ("FIT", "SELECT"):
        assert sum(r["split"] == sp for r in rows) == sum(r["split"] == sp for r in bad)
        assert split_hash(rows, sp) != split_hash(bad, sp)
    from collections import Counter
    assert Counter(r["group_id"] for r in rows) == Counter(r["group_id"] for r in bad)


def make_calls(rows, arms=("native", "counterfactual"), batch=4):
    """Deterministic total order: FIT rows first, per arm, batches of `batch` in authority order; the last batch is literal."""
    calls = []
    for split in ("FIT", "SELECT"):
        ids = [r["row_id"] for r in rows if r["split"] == split]
        for arm in arms:
            for b0 in range(0, len(ids), batch):
                chunk = ids[b0:b0 + batch]
                calls.append({"call_id": f"{split}:{arm}:{b0 // batch}", "split": split, "arm": arm, "rows": chunk, "physical_width": batch})
    return calls


def attack_call_delete(calls):
    out = copy.deepcopy(calls); del out[len(out) // 2]; return out


def attack_call_reorder(calls):
    out = copy.deepcopy(calls); out[0], out[1] = out[1], out[0]; return out


def attack_batch_recompose(calls):
    """Same rows, same arms, same total call count and price; two adjacent batches trade one row each (batch composition drift)."""
    out = copy.deepcopy(calls)
    a, b = out[0], out[1]; a["rows"][-1], b["rows"][0] = b["rows"][0], a["rows"][-1]
    return out


def test_generator_batch_recompose_preserves_price_and_multiset():
    calls = make_calls(make_authority()); bad = attack_batch_recompose(calls)
    assert len(calls) == len(bad)
    assert sorted(sum((c["rows"] for c in calls), [])) == sorted(sum((c["rows"] for c in bad), []))
    assert _sha(_canon(calls)) != _sha(_canon(bad))


def make_evidence(calls, seed=0, nonfinite_at=None):
    """One primitive record per (call, row): a finite logit-margin; optional planted NaN at (call_index, row_index)."""
    rng = random.Random(seed); ev = []
    for ci, c in enumerate(calls):
        for ri, rid in enumerate(c["rows"]):
            val = rng.uniform(-3, 3) * (1.0 if c["arm"] == "native" else 0.6)
            if nonfinite_at == (ci, ri):
                val = float("nan")
            ev.append({"call_id": c["call_id"], "arm": c["arm"], "row_id": rid, "margin": val, "request_sha": _sha(_canon([c["call_id"], c["rows"], c["physical_width"]]))})
    return ev


def attack_dead_arm(calls, ev):
    """Make the counterfactual arm's evidence byte-identical to the native arm's for every row (a dead control)."""
    native = {(e["row_id"]): e["margin"] for e in ev if e["arm"] == "native"}
    out = copy.deepcopy(ev)
    for e in out:
        if e["arm"] == "counterfactual":
            e["margin"] = native[e["row_id"]]
    return out


def attack_move_primitive(ev):
    """Move one record from one call to another call of the same arm (keeps arm/row multiset per arm; breaks the call binding)."""
    out = copy.deepcopy(ev); a = out[0]; b = next(e for e in out if e["arm"] == a["arm"] and e["call_id"] != a["call_id"])
    a["call_id"], b["call_id"] = b["call_id"], a["call_id"]
    return out


def projector_mean_gap(ev):
    """Reference pure projector: mean(native margin) - mean(counterfactual margin), order-independent."""
    n = [e["margin"] for e in ev if e["arm"] == "native"]; c = [e["margin"] for e in ev if e["arm"] == "counterfactual"]
    return {"pred_a_instrument": len(n) == len(c) > 0, "score_gap": sum(n) / len(n) - sum(c) / len(c)}


def projector_order_dependent(ev):
    """Impure projector: reports the FIRST record's arm as a 'score' -- differs under a permutation of the evidence."""
    return {"pred_a_instrument": True, "score_gap": 1.0 if ev[0]["arm"] == "native" else -1.0}


def test_generator_order_dependent_projector_is_detectable():
    calls = make_calls(make_authority()); ev = make_evidence(calls)
    perm = ev[::-1]
    assert projector_mean_gap(ev)["score_gap"] == pytest.approx(projector_mean_gap(perm)["score_gap"])
    assert projector_order_dependent(ev) != projector_order_dependent(perm)


def test_generator_dead_arm_keeps_counts():
    calls = make_calls(make_authority()); ev = make_evidence(calls); bad = attack_dead_arm(calls, ev)
    assert len(ev) == len(bad) and projector_mean_gap(bad)["score_gap"] == 0.0


# ----------------------------------------------------------------------------------------------------------------------
# 2. API tests (skip until the module exists). Spec construction goes through one helper so a rename costs one edit.
# ----------------------------------------------------------------------------------------------------------------------

def _spec(api, tmp_path, rows, calls, *, predicates=None, arms=None, science_names=("pred_a_instrument", "score_gap")):
    """Build the smallest spec the API accepts for a synthetic authority. Codex: adapt ONLY this function to the real constructors."""
    auth_path = tmp_path / "authority.json"; auth_path.write_text(json.dumps(rows))
    return api.build_synthetic_spec(experiment_id="redteam", rows_path=str(auth_path), calls=calls, arms=arms or [{"name": "native", "role": "native", "direction": "undirected"}, {"name": "counterfactual", "role": "counterfactual", "direction": "undirected"}],
                                    predicates=predicates or [], science_names=list(science_names), projector=projector_mean_gap)


def test_split_swap_preserving_counts_rejected(tmp_path):
    api = _api(); rows = make_authority(); bad = attack_split_swap(rows)
    spec = _spec(api, tmp_path, rows, make_calls(rows))
    compiled = api.compile_experiment(spec, authority_inputs={"rows": _canon(rows)})
    with _raises_with("split"):
        api.compile_experiment(spec, authority_inputs={"rows": _canon(bad)})  # same counts, different per-split content hash
    assert compiled.split_content_hashes["FIT"] == split_hash(rows, "FIT")


def test_arm_role_missing_rejected(tmp_path):
    api = _api(); rows = make_authority()
    with _raises_with("role"):
        _spec(api, tmp_path, rows, make_calls(rows), arms=[{"name": "native"}, {"name": "counterfactual"}])


def test_dead_arm_identical_evidence_is_hard_abort(tmp_path):
    api = _api(); rows = make_authority(); calls = make_calls(rows)
    compiled = api.compile_experiment(_spec(api, tmp_path, rows, calls), authority_inputs={"rows": _canon(rows)})
    ev = make_evidence(calls); audit = api.validate_call_evidence(compiled, ev); assert audit.ok
    audit_bad = api.validate_call_evidence(compiled, attack_dead_arm(calls, ev))
    assert not audit_bad.ok and any("dead" in f.lower() or "identical" in f.lower() for f in audit_bad.failures)


def test_call_deletion_reorder_and_batch_recompose_rejected(tmp_path):
    api = _api(); rows = make_authority(); calls = make_calls(rows)
    compiled = api.compile_experiment(_spec(api, tmp_path, rows, calls), authority_inputs={"rows": _canon(rows)})
    for attack in (attack_call_delete, attack_call_reorder, attack_batch_recompose):
        ev = make_evidence(attack(calls))
        assert not api.validate_call_evidence(compiled, ev).ok, attack.__name__


def test_batch_composition_in_contract_hash(tmp_path):
    api = _api(); rows = make_authority(); calls = make_calls(rows)
    c1 = api.compile_experiment(_spec(api, tmp_path, rows, calls), authority_inputs={"rows": _canon(rows)})
    c2 = api.compile_experiment(_spec(api, tmp_path, rows, attack_batch_recompose(calls)), authority_inputs={"rows": _canon(rows)})
    assert c1.contract_hash != c2.contract_hash and c1.max_price == c2.max_price


def test_nonfinite_in_nonfinal_call_hard_aborts(tmp_path):
    api = _api(); rows = make_authority(); calls = make_calls(rows)
    compiled = api.compile_experiment(_spec(api, tmp_path, rows, calls), authority_inputs={"rows": _canon(rows)})
    bad_mid = make_evidence(calls, nonfinite_at=(1, 0))
    audit = api.validate_call_evidence(compiled, bad_mid)
    assert not audit.ok and audit.terminal == "hard_abort"
    last = len(calls) - 1; bad_last = make_evidence(calls, nonfinite_at=(last, 0))
    audit2 = api.validate_call_evidence(compiled, bad_last)
    assert audit2.terminal in ("hard_abort", "final_nonfinite_diagnostic")   # policy decides; never 'ok'


def test_instrument_failure_dominates_science_success(tmp_path):
    api = _api(); rows = make_authority(); calls = make_calls(rows)
    preds = [{"predicate_id": "instrument_margin_bound", "kind": "instrument", "evaluator": lambda ev: all(abs(e["margin"]) < 1e-9 for e in ev), "disposition": "hard_abort"}]
    compiled = api.compile_experiment(_spec(api, tmp_path, rows, calls, predicates=preds), authority_inputs={"rows": _canon(rows)})
    ev = make_evidence(calls)                          # science would be a clean positive gap; the instrument bound FAILS
    pkg = api.project_result(compiled, api.package_evidence(compiled, ev))
    assert pkg.terminal == "hard_abort" and pkg.scores.get("score_gap") is None or pkg.terminal == "hard_abort"


def test_science_priority_above_instrument_rejected_at_compile(tmp_path):
    api = _api(); rows = make_authority(); calls = make_calls(rows)
    preds = [{"predicate_id": "instrument_x", "kind": "instrument", "priority": 5, "evaluator": lambda ev: True, "disposition": "hard_abort"},
             {"predicate_id": "science_y", "kind": "science", "priority": 1, "evaluator": lambda ev: True, "disposition": "diagnostic"}]
    with _raises_with("priority"):
        api.compile_experiment(_spec(api, tmp_path, rows, calls, predicates=preds), authority_inputs={"rows": _canon(rows)})


def test_projector_depends_on_evidence_order_rejected(tmp_path):
    api = _api(); rows = make_authority(); calls = make_calls(rows)
    spec = _spec(api, tmp_path, rows, calls); spec = api.with_projector(spec, projector_order_dependent)
    compiled = api.compile_experiment(spec, authority_inputs={"rows": _canon(rows)})
    with _raises_with("pure"):
        api.project_result(compiled, api.package_evidence(compiled, make_evidence(calls)))


def test_projector_reads_environment_rejected(tmp_path, monkeypatch):
    api = _api(); rows = make_authority(); calls = make_calls(rows)
    def leaky(ev):
        return {"pred_a_instrument": True, "score_gap": float(len(os.environ.get("REDTEAM_LEAK", "")))}
    monkeypatch.setenv("REDTEAM_LEAK", "x")
    spec = api.with_projector(_spec(api, tmp_path, rows, calls), leaky)
    compiled = api.compile_experiment(spec, authority_inputs={"rows": _canon(rows)})
    with _raises_with("pure"):
        api.project_result(compiled, api.package_evidence(compiled, make_evidence(calls)))


def test_primitive_moved_between_calls_rejected(tmp_path):
    api = _api(); rows = make_authority(); calls = make_calls(rows)
    compiled = api.compile_experiment(_spec(api, tmp_path, rows, calls), authority_inputs={"rows": _canon(rows)})
    assert not api.validate_call_evidence(compiled, attack_move_primitive(make_evidence(calls))).ok


def test_summary_mutation_without_primitive_change_rejected(tmp_path):
    api = _api(); rows = make_authority(); calls = make_calls(rows)
    compiled = api.compile_experiment(_spec(api, tmp_path, rows, calls), authority_inputs={"rows": _canon(rows)})
    pkg = api.project_result(compiled, api.package_evidence(compiled, make_evidence(calls)))
    tampered = api.load_package(api.dump_package(pkg)); tampered.scores["score_gap"] += 0.5
    with _raises_with("regenerat"):
        api.verify_package(compiled, tampered)


def test_literal_final_batch_requires_physical_width(tmp_path):
    api = _api(); rows = make_authority(); calls = make_calls(rows)
    for c in calls:
        c["physical_width"] = None
    with _raises_with("width"):
        api.compile_experiment(_spec(api, tmp_path, rows, calls), authority_inputs={"rows": _canon(rows)})


def test_forward_request_width_drift_rejected(tmp_path):
    api = _api(); rows = make_authority(); calls = make_calls(rows)
    compiled = api.compile_experiment(_spec(api, tmp_path, rows, calls), authority_inputs={"rows": _canon(rows)})
    ev = make_evidence(calls)
    drift = copy.deepcopy(calls); drift[-1]["physical_width"] = 8       # padded tail instead of the literal last batch
    ev[-1]["request_sha"] = _sha(_canon([drift[-1]["call_id"], drift[-1]["rows"], 8]))
    assert not api.validate_call_evidence(compiled, ev).ok


def test_diagnostic_named_like_science_rejected(tmp_path):
    api = _api(); rows = make_authority(); calls = make_calls(rows)
    preds = [{"predicate_id": "pred_b_looks_like_science", "kind": "evidence", "evaluator": lambda ev: True, "disposition": "diagnostic"}]
    with _raises_with("pred_"):
        api.compile_experiment(_spec(api, tmp_path, rows, calls, predicates=preds), authority_inputs={"rows": _canon(rows)})


def test_prereg_tamper_after_compile_rejected(tmp_path):
    api = _api(); rows = make_authority(); calls = make_calls(rows)
    prereg = tmp_path / "PREREG.md"; prereg.write_text("# prereg\n")
    spec = api.with_artifact(_spec(api, tmp_path, rows, calls), role="prereg", path=str(prereg), sha256=_sha(prereg.read_bytes()), kind="prereg")
    compiled = api.compile_experiment(spec, authority_inputs={"rows": _canon(rows)})
    prereg.write_text("# prereg (edited after compile)\n")
    with _raises_with("hash"):
        api.begin_run(compiled)      # the runtime re-verifies every non-outcome artifact before the first forward request


def test_managed_dryrun_touches_no_outcome_and_no_model(tmp_path, monkeypatch, capsys):
    api = _api(); rows = make_authority(); calls = make_calls(rows)
    outcome = tmp_path / "prior_results.json"; outcome.write_text(json.dumps({"preds": {"pred_a_instrument": True}}))
    spec = api.with_artifact(_spec(api, tmp_path, rows, calls), role="prior", path=str(outcome), sha256=_sha(outcome.read_bytes()), kind="outcome")
    spec_path = tmp_path / "spec.json"; spec_path.write_text(api.dump_spec(spec))
    opened = []
    real_open = open
    def spy(f, *a, **k):
        opened.append(str(f)); return real_open(f, *a, **k)
    monkeypatch.setattr("builtins.open", spy)
    monkeypatch.setitem(__import__("sys").modules, "torch", None)      # any `import torch` raises ImportError during dry run
    with pytest.raises(SystemExit) as ex:
        api.managed_main(spec_path, {"BQLIB_DRYRUN": "1", "BQLIB_NO_MODEL": "1"})
    assert ex.value.code in (0, None)
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert json.loads(out)["status"] == "dry_run_passed"
    assert str(outcome) not in opened, "dry run opened an outcome-bearing artifact"


def test_generated_entry_passes_ops_gate(tmp_path):
    api = _api(); rows = make_authority(); calls = make_calls(rows)
    entry = tmp_path / "redteam_entry.py"; entry.write_text(api.render_managed_entry(_spec(api, tmp_path, rows, calls)))
    import subprocess, sys
    gate = pathlib.Path(__file__).with_name("gate.py")
    r = subprocess.run([sys.executable, str(gate), str(entry)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_stage_dir_same_filesystem_as_target(tmp_path):
    api = _api(); rows = make_authority(); calls = make_calls(rows)
    compiled = api.compile_experiment(_spec(api, tmp_path, rows, calls), authority_inputs={"rows": _canon(rows)})
    pkg = api.project_result(compiled, api.package_evidence(compiled, make_evidence(calls)))
    target = tmp_path / "out"; target.mkdir()
    stage = api.stage_dir_for(target)
    assert os.stat(stage).st_dev == os.stat(target).st_dev


def test_publish_refuses_complete_outcome(tmp_path):
    api = _api(); rows = make_authority(); calls = make_calls(rows)
    compiled = api.compile_experiment(_spec(api, tmp_path, rows, calls), authority_inputs={"rows": _canon(rows)})
    pkg = api.project_result(compiled, api.package_evidence(compiled, make_evidence(calls)))
    target = tmp_path / "out"; target.mkdir()
    api.stage_and_publish(compiled, pkg, target=target)
    with _raises_with("complete"):
        api.stage_and_publish(compiled, pkg, target=target)      # a post-publication adapter return must not republish
