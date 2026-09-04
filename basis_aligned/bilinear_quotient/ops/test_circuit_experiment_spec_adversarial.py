"""Adversarial fixtures for the CircuitExperimentSpec compiler/runtime (Claude, 2026-09-04; red-team companion to
polynomial_causal/CIRCUIT_EXPERIMENT_SPEC_REDTEAM_2026-09-04.md, attacks A1-A9 / B1-B4).

Rewired 2026-09-04 (later): the API tests no longer skip.  They run against Codex's ACTUAL public names through the thin
adapter in ops/_adv_adapter.py (spec dataclasses + compile_experiment, validate_call_prefix, write/validate_nonfinite_masks,
validate_science_projection, stage_package/publish_staged_package/validate_complete_package, capture_frozen_artifacts/
validate_dryrun_closure/load_verified_modules/dispatch).  The adapter adds NO validation of its own.  Where Codex lacks the
semantics an attack requires, the test FAILS with a message naming the gap -- it is never shimmed green.

Two layers:
  1. PLANTED-ATTACK GENERATORS (pure Python): a synthetic authority (rows, splits, groups), a call manifest, primitive
     evidence, and one mutation per attack.  Each generator self-tests that the attack is NON-VACUOUS.
  2. API TESTS: each feeds one planted attack to Codex's API and requires a refusal.

Scratch space is <repo>/.adv_tmp (same filesystem as the repo, never /tmp -- attack B4); every test cleans up after itself.
Standard library + numpy (Codex's evidence format is .npy).  No model, no GPU.
"""
import copy
import hashlib
import io
import json
import os
import pathlib
import random
import subprocess
import sys

import pytest

import _adv_adapter as api
import circuit_artifact_package as cap
import circuit_managed_entry as managed


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def workdir(request):
    """Same-filesystem scratch directory under <repo>/.adv_tmp (not /tmp)."""
    path = api.make_workdir(request.node.name[:40])
    yield path
    api.remove_workdir(path)


def _expect_refusal(fn, keywords, *, missing):
    """Run fn.  It must raise, and the message must contain one of `keywords` (Codex may rename the class; the wording of
    the reason is the contract).  If it does NOT raise, fail with `missing` -- the statement of the absent semantics."""
    try:
        value = fn()
    except Exception as error:      # noqa: BLE001
        text = str(error).lower()
        assert any(k.lower() in text for k in keywords), \
            f"refused, but not for the registered reason {keywords!r}: {type(error).__name__}: {error}"
        return error
    pytest.fail(f"{missing} (call returned {type(value).__name__} instead of refusing)")


# ----------------------------------------------------------------------------------------------------------------------
# 1. Planted-attack generators
# ----------------------------------------------------------------------------------------------------------------------

def make_authority(n_fit=12, n_select=6, seed=0, width=8):
    """Synthetic authority: rows with content-derived IDs, two splits, groups of 3, one semantic field, a fixed-width
    token sequence (Codex batches by exact common length).  Rows are emitted in Codex's canonical order (length, canonical
    JSON) so that the generator's authority-order schedule and Codex's compiled manifest coincide (checked below)."""
    rng = random.Random(seed); rows = []
    for split, n in (("FIT", n_fit), ("SELECT", n_select)):
        for i in range(n):
            content = f"{split}-doc{i}-" + "".join(rng.choice("abcdefgh") for _ in range(6))
            ids = [int(ch, 16) for ch in _sha(content.encode())[:width]]
            rows.append({"row_id": _sha(content.encode())[:16], "split": split, "group_id": f"{split}-g{i // 3}",
                         "answer_pos": 5 + (i % 4), "content": content, "ids": ids})
    rows.sort(key=lambda r: (len(r["ids"]), _canon(r)))
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
    """Deterministic total order: FIT rows first, per arm, batches of `batch` in authority order; the last batch is literal
    (physical_width == its literal row count)."""
    calls = []
    for split in ("FIT", "SELECT"):
        ids = [r["row_id"] for r in rows if r["split"] == split]
        for arm in arms:
            for b0 in range(0, len(ids), batch):
                chunk = ids[b0:b0 + batch]
                calls.append({"call_id": f"{split}:{arm}:{b0 // batch}", "split": split, "arm": arm, "rows": chunk, "physical_width": len(chunk)})
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


SCIENCE_NAMES = ("pred_a_instrument", "pred_b_gap_positive", "pred_c_gap_bounded", "score_gap")


def projector_mean_gap(ev):
    """Reference pure projector: mean(native margin) - mean(counterfactual margin), order-independent."""
    n = [e["margin"] for e in ev if e["arm"] == "native"]; c = [e["margin"] for e in ev if e["arm"] == "counterfactual"]
    gap = sum(n) / len(n) - sum(c) / len(c)
    return {"pred_a_instrument": len(n) == len(c) > 0, "pred_b_gap_positive": gap > 0, "pred_c_gap_bounded": abs(gap) < 10.0, "score_gap": gap}


def projector_order_dependent(ev):
    """Impure projector: reports the FIRST record's arm as a 'score' -- differs under a permutation of the evidence."""
    out = projector_mean_gap(ev); out["score_gap"] = 1.0 if ev[0]["arm"] == "native" else -1.0; return out


def test_generator_order_dependent_projector_is_detectable():
    calls = make_calls(make_authority()); ev = make_evidence(calls)
    perm = ev[::-1]
    assert projector_mean_gap(ev)["score_gap"] == pytest.approx(projector_mean_gap(perm)["score_gap"])
    assert projector_order_dependent(ev) != projector_order_dependent(perm)


def test_generator_dead_arm_keeps_counts():
    calls = make_calls(make_authority()); ev = make_evidence(calls); bad = attack_dead_arm(calls, ev)
    assert len(ev) == len(bad) and projector_mean_gap(bad)["score_gap"] == 0.0


# ----------------------------------------------------------------------------------------------------------------------
# 2. API tests against Codex's modules.  Spec construction goes through one helper.
# ----------------------------------------------------------------------------------------------------------------------

ARMS = [{"name": "native", "role": "native", "direction": "undirected"}, {"name": "counterfactual", "role": "counterfactual", "direction": "undirected"}]


def _spec(workdir, rows, calls, *, predicates=None, arms=None, science_names=SCIENCE_NAMES):
    """Build the smallest spec Codex's dataclasses accept for a synthetic authority (authority.json lives in workdir)."""
    auth_path = workdir / api.ROWS_FILE; auth_path.write_text(json.dumps(rows))
    return api.build_synthetic_spec(experiment_id="redteam", rows_path=str(auth_path), calls=calls, arms=arms or ARMS,
                                    predicates=predicates or [], science_names=list(science_names), projector=projector_mean_gap)


def _compile(workdir, rows, calls, **kw):
    return api.compile_experiment(_spec(workdir, rows, calls, **kw), authority_inputs={"rows": _canon(rows)})


def test_adapter_manifest_is_the_generator_schedule(workdir):
    """Sanity: Codex's compiled manifest IS the generator's schedule, so every attack below mutates the real manifest."""
    rows = make_authority(); calls = make_calls(rows)
    compiled = _compile(workdir, rows, calls)
    assert compiled.calls == calls
    assert compiled.max_price == len(calls) == 10
    assert api.validate_call_evidence(compiled, make_evidence(calls), workdir=workdir).ok
    # the typed boundary carries the fixture's declarations losslessly (board 02:12) and the contract hash covers them
    spec = compiled.synth.spec
    for family in spec.calls:
        assert [(a.name, a.role, a.direction) for a in family.arm_specs] == [(a["name"], a["role"], a["direction"]) for a in ARMS]
        assert family.arms == tuple(a.name for a in family.arm_specs)
    assert spec.authority_tables[0].expected_records_sha256 == compiled.compiled["authority"]["rows"]["records_sha256"]
    assert compiled.synth.evaluators[spec.science.projector_role] is projector_mean_gap and spec.predicates == ()
    roleless = api.build_synthetic_spec(experiment_id="redteam", rows_path=str(workdir / api.ROWS_FILE), calls=calls,
                                        arms=[{"name": "native"}, {"name": "counterfactual"}], predicates=[],
                                        science_names=list(SCIENCE_NAMES), projector=projector_mean_gap)
    assert roleless.spec.calls[0].arm_specs[0].role is None and roleless.spec.calls[0].arms == spec.calls[0].arms
    assert api.canon(api.cs.spec_json(roleless.spec)) != api.canon(api.cs.spec_json(spec))


# --- A2 -----------------------------------------------------------------------------------------------------------------

def test_split_swap_preserving_counts_rejected(workdir):
    rows = make_authority(); bad = attack_split_swap(rows)
    spec = _spec(workdir, rows, make_calls(rows))
    good = api.compile_experiment(spec, authority_inputs={"rows": _canon(rows)})
    swapped = _expect_refusal(
        lambda: api.compile_experiment(spec, authority_inputs={"rows": _canon(bad)}), ("split", "record", "digest"),
        missing="A2: the spec pins expected_records_sha256 (= canonical_sha256(rows), the value compile_authority_tables emits as "
                "records_sha256) but the compiler compares only expected_counts/expected_total, so a count-preserving FIT/SELECT swap "
                f"compiled clean (pinned {spec.spec.authority_tables[0].expected_records_sha256[:12]}..., swapped rows hash "
                f"{api.cs.canonical_sha256(bad)[:12]}...)")
    assert swapped is not None
    assert good.compiled["authority"]["rows"]["counts_by_split"] == {"FIT": 12, "SELECT": 6}
    assert good.compiled["authority"]["rows"]["records_sha256"] == spec.spec.authority_tables[0].expected_records_sha256


# --- A1 -----------------------------------------------------------------------------------------------------------------

def test_arm_role_missing_rejected(workdir):
    rows = make_authority()
    _expect_refusal(lambda: _spec(workdir, rows, make_calls(rows), arms=[{"name": "native"}, {"name": "counterfactual"}]), ("role",),
                    missing="A1: the typed arm declarations reach the spec as CallFamilySpec.arm_specs (ArmSpec(name, role=None, "
                            "direction=None) for this role-missing input, distinct from the role-bearing ARMS in spec JSON and hash), "
                            "but validate_spec/compile_call_manifest read only the arm NAMES, so a spec whose arms declare no "
                            "counterfactual/native role validated clean")


def test_dead_arm_identical_evidence_is_hard_abort(workdir):
    rows = make_authority(); calls = make_calls(rows)
    compiled = _compile(workdir, rows, calls)
    ev = make_evidence(calls); audit = api.validate_call_evidence(compiled, ev, workdir=workdir); assert audit.ok, audit.failures
    audit_bad = api.validate_call_evidence(compiled, attack_dead_arm(calls, ev), workdir=workdir)
    assert not audit_bad.ok and any("dead" in f.lower() or "identical" in f.lower() for f in audit_bad.failures), (
        "A1: no Codex validator compares retained evidence ACROSS arms of one call family; a counterfactual arm whose margins are "
        f"byte-identical to the native arm audited clean (checks run: {sorted(set(c.split('[')[0] for c in audit_bad.checks))})")


# --- A6 / A9 ------------------------------------------------------------------------------------------------------------

def test_call_deletion_reorder_and_batch_recompose_rejected(workdir):
    rows = make_authority(); calls = make_calls(rows)
    compiled = _compile(workdir, rows, calls)
    for attack in (attack_call_delete, attack_call_reorder, attack_batch_recompose):
        ev = make_evidence(attack(calls))
        audit = api.validate_call_evidence(compiled, ev, workdir=workdir)
        assert not audit.ok, attack.__name__
        assert any("prefix" in f or "census" in f for f in audit.failures), (attack.__name__, audit.failures)


def test_batch_composition_in_contract_hash(workdir):
    """A9: the ordered call->rows map must be inside the contract hash; a recomposition with identical price must change it,
    and an observed recomposed schedule must be refused as evidence."""
    rows = make_authority(); calls = make_calls(rows)
    c1 = _compile(workdir, rows, calls)
    c2 = api.rebind_manifest(c1, attack_batch_recompose(calls))
    assert c1.contract_hash != c2.contract_hash and c1.max_price == c2.max_price
    assert c1.compiled["call_summary"]["manifest_sha256"] != c2.compiled["call_summary"]["manifest_sha256"]
    assert c1.compiled["call_summary"]["shape_counts"] == c2.compiled["call_summary"]["shape_counts"]
    # a runtime option cannot re-batch: the compiled manifest is a pure function of (spec, records)
    again = _compile(workdir, rows, calls)
    assert again.contract_hash == c1.contract_hash
    assert not api.validate_call_evidence(c1, make_evidence(attack_batch_recompose(calls)), workdir=workdir).ok


# --- A3 -----------------------------------------------------------------------------------------------------------------

def test_nonfinite_in_nonfinal_call_hard_aborts(workdir):
    rows = make_authority(); calls = make_calls(rows)
    compiled = _compile(workdir, rows, calls)
    last = len(calls) - 1
    audit_last = api.validate_call_evidence(compiled, make_evidence(calls, nonfinite_at=(last, 0)), workdir=workdir)
    assert audit_last.terminal in ("hard_abort", "final_nonfinite_diagnostic"), audit_last   # policy decides; never 'ok'
    audit_mid = api.validate_call_evidence(compiled, make_evidence(calls, nonfinite_at=(1, 0)), workdir=workdir)
    assert not audit_mid.ok and audit_mid.terminal == "hard_abort", (
        "A3: a NaN in NON-FINAL call 1 (final call finite) audited clean. Codex's validate_nonfinite_masks is scoped to ONE call "
        "directory under the run's terminal predicate; ArraySpec.finite_policy='final_nonfinite_diagnostic' is declared but read by "
        f"no module, and no invariant requires every earlier call to be finite. audit={audit_mid}")


# --- A4 -----------------------------------------------------------------------------------------------------------------

def test_instrument_failure_dominates_science_success(workdir):
    rows = make_authority(); calls = make_calls(rows)
    preds = [{"predicate_id": "instrument_margin_bound", "kind": "instrument", "evaluator": lambda ev: all(abs(e["margin"]) < 1e-9 for e in ev), "disposition": "hard_abort"}]
    compiled = _compile(workdir, rows, calls, predicates=preds)
    ev = make_evidence(calls)                          # science would be a clean positive gap; the instrument bound FAILS
    assert not preds[0]["evaluator"](ev)
    registered = compiled.synth.spec.predicates[0]
    assert registered.kind == "instrument" and registered.disposition == "hard_abort"
    assert compiled.synth.evaluators[registered.evaluator_role] is preds[0]["evaluator"]     # reaches the boundary intact
    pkg = api.project_result(compiled, api.package_evidence(compiled, ev))
    assert pkg.evaluators is not None and registered.evaluator_role in pkg.evaluators
    assert pkg.predicates_evaluated and pkg.terminal == "hard_abort" and pkg.scores.get("score_gap") is None, (
        "A4: the failing instrument evaluator is registered under evaluator_role "
        f"{registered.evaluator_role!r} (kind={registered.kind!r}, disposition={registered.disposition!r}) and handed to the boundary, "
        "but no Codex module evaluates a predicate or resolves a terminal from predicate_order, so a FAILING registered instrument "
        f"bound cannot dominate the science projection (predicate_order={compiled.compiled['predicate_order']}, "
        f"terminal={pkg.terminal!r}, score_gap={pkg.scores.get('score_gap')})")


def test_science_priority_above_instrument_rejected_at_compile(workdir):
    rows = make_authority(); calls = make_calls(rows)
    preds = [{"predicate_id": "instrument_x", "kind": "instrument", "priority": 5, "evaluator": lambda ev: True, "disposition": "hard_abort"},
             {"predicate_id": "science_y", "kind": "science", "priority": 1, "evaluator": lambda ev: True, "disposition": "diagnostic"}]
    _expect_refusal(lambda: _compile(workdir, rows, calls, predicates=preds), ("priority",),
                    missing="A4: the predicate kinds reach the spec typed (PredicateSpec.kind = 'instrument' / 'science') but nothing "
                            "orders kinds (instrument < authority < evidence < science); priority is a bare int and predicate_order a "
                            "bare sort, so a science predicate at priority 1 above an instrument predicate at 5 compiled clean")


# --- A5 -----------------------------------------------------------------------------------------------------------------

def test_projector_depends_on_evidence_order_rejected(workdir):
    rows = make_authority(); calls = make_calls(rows)
    spec = api.with_projector(_spec(workdir, rows, calls), projector_order_dependent)
    compiled = api.compile_experiment(spec, authority_inputs={"rows": _canon(rows)})
    _expect_refusal(lambda: api.project_result(compiled, api.package_evidence(compiled, make_evidence(calls))), ("pure",),
                    missing="A5: validate_science_projection recomputes the projector ONCE on the saved evidence order; an "
                            "order-dependent projector reproduces its own output and is accepted (no permutation/sandbox purity check)")


def test_projector_reads_environment_rejected(workdir, monkeypatch):
    rows = make_authority(); calls = make_calls(rows)
    def leaky(ev):
        out = projector_mean_gap(ev); out["score_gap"] = float(len(os.environ.get("REDTEAM_LEAK", ""))); return out
    monkeypatch.setenv("REDTEAM_LEAK", "x")
    spec = api.with_projector(_spec(workdir, rows, calls), leaky)
    compiled = api.compile_experiment(spec, authority_inputs={"rows": _canon(rows)})
    _expect_refusal(lambda: api.project_result(compiled, api.package_evidence(compiled, make_evidence(calls))), ("pure",),
                    missing="A5: the projector runs in the producer's own process/environment; one that reads os.environ reproduces "
                            "its own output under validate_science_projection and is accepted (no frozen-env sandbox)")


# --- A6 -----------------------------------------------------------------------------------------------------------------

def test_primitive_moved_between_calls_rejected(workdir):
    rows = make_authority(); calls = make_calls(rows)
    compiled = _compile(workdir, rows, calls)
    audit = api.validate_call_evidence(compiled, attack_move_primitive(make_evidence(calls)), workdir=workdir)
    assert not audit.ok and any("prefix" in f for f in audit.failures), audit


def test_summary_mutation_without_primitive_change_rejected(workdir):
    rows = make_authority(); calls = make_calls(rows)
    compiled = _compile(workdir, rows, calls)
    pkg = api.project_result(compiled, api.package_evidence(compiled, make_evidence(calls)))
    target = workdir / "pkg"; target.mkdir()
    paths = api.dump_package(pkg, target=target)
    result = api.load_package(paths)
    result["projection"]["score_gap"] += 0.5                      # hand-picked summary; primitives untouched
    paths.result.write_bytes(api.canon(result) + b"\n")
    _expect_refusal(lambda: api.verify_package(compiled, paths), ("binding",),
                    missing="A6: a tampered result.json passed validate_complete_package (receipt binding)")
    api.resign_receipt(paths)                                     # attacker with write access re-signs the receipt
    api.load_package(paths)                                       # now internally consistent ...
    _expect_refusal(lambda: api.verify_package(compiled, paths), ("regenerat", "differs from primitive evidence"),
                    missing="A6: a re-signed package with a mutated summary and unchanged primitives verified clean")


# --- A7 -----------------------------------------------------------------------------------------------------------------

def test_literal_final_batch_requires_physical_width(workdir):
    """Codex cannot express a family without a literal width: every compiled call carries int logical_batch_size and
    padded_sequence_length, records without a token sequence do not compile, and an observed record with a symbolic width
    is not a literal prefix."""
    rows = make_authority(); calls = make_calls(rows)
    compiled = _compile(workdir, rows, calls)
    for call in compiled.manifest:
        assert type(call["logical_batch_size"]) is int and type(call["padded_sequence_length"]) is int, call
    assert compiled.manifest[-1]["logical_batch_size"] == 2 and compiled.compiled["call_summary"]["shape_counts"] == {"2x8": 2, "4x8": 8}
    widthless = copy.deepcopy(rows); widthless[0]["ids"] = []
    _expect_refusal(lambda: _compile(workdir, widthless, calls), ("token sequence", "width"),
                    missing="A7: a record with no token sequence compiled to a call with no literal width")
    observed = copy.deepcopy(compiled.manifest); observed[-1]["padded_sequence_length"] = None
    with pytest.raises(cap.PackageError, match="prefix"):
        cap.validate_call_prefix(compiled.manifest, observed, [cap.call_directory_name(i, c["call_id"]) for i, c in enumerate(observed)])


def test_forward_request_width_drift_rejected(workdir):
    rows = make_authority(); calls = make_calls(rows)
    compiled = _compile(workdir, rows, calls)
    ev = make_evidence(calls)
    drift = copy.deepcopy(calls); drift[-1]["physical_width"] = 8       # padded tail instead of the literal last batch (2 rows)
    ev[-1]["request_sha"] = _sha(_canon([drift[-1]["call_id"], drift[-1]["rows"], 8]))
    audit = api.validate_call_evidence(compiled, ev, workdir=workdir, physical_pad={drift[-1]["call_id"]: 8})
    assert not audit.ok, (
        "A7: the final call's saved array has 8 rows against a compiled logical_batch_size of 2 and audited clean. "
        "shape_validation_mode is a policy STRING nothing enforces; no Codex validator compares saved evidence array shapes to the "
        "compiled (logical_batch_size, padded_sequence_length) and no forward-request hash is bound to the primitives. "
        f"checks={sorted(set(c.split('[')[0] for c in audit.checks))}")


# --- A8 -----------------------------------------------------------------------------------------------------------------

def test_diagnostic_named_like_science_rejected(workdir):
    rows = make_authority(); calls = make_calls(rows)
    preds = [{"predicate_id": "pred_b_looks_like_science", "kind": "evidence", "evaluator": lambda ev: True, "disposition": "diagnostic"}]
    _expect_refusal(lambda: _compile(workdir, rows, calls, predicates=preds), ("pred_",),
                    missing="A8: predicate_id is unconstrained; a DIAGNOSTIC named 'pred_b_looks_like_science' compiled clean next to "
                            "ScienceProjectionSpec.output_types, and ops/gate.py scores 'pred_*' keys by NAME")


# --- B2 -----------------------------------------------------------------------------------------------------------------

def test_prereg_tamper_after_compile_rejected(workdir):
    rows = make_authority(); calls = make_calls(rows)
    prereg = workdir / "PREREG.md"; prereg.write_text("# prereg\n")
    spec = api.with_artifact(_spec(workdir, rows, calls), role="prereg", path=str(prereg), sha256=_sha(prereg.read_bytes()), kind="prereg")
    compiled = api.compile_experiment(spec, authority_inputs={"rows": _canon(rows)})
    assert api.begin_run(compiled)["prereg"] == b"# prereg\n"
    prereg.write_text("# prereg (edited after compile)\n")
    _expect_refusal(lambda: api.begin_run(compiled), ("hash", "sha", "frozen artifact changed"),
                    missing="B2: the runtime did not re-verify a non-outcome artifact before the first forward request")


# --- B1 -----------------------------------------------------------------------------------------------------------------

def test_managed_dryrun_touches_no_outcome_and_no_model(workdir, monkeypatch, capsys):
    rows = make_authority(); calls = make_calls(rows)
    outcome = workdir / "prior_results.json"; outcome.write_text(json.dumps({"preds": {"pred_a_instrument": True}}))
    # declared sha deliberately WRONG: any dry-run path that hashes the outcome must fail loudly, not just be spied on
    wrong_sha = "f" * 64
    spec = api.with_artifact(_spec(workdir, rows, calls), role="prior", path=str(outcome), sha256=wrong_sha, kind="outcome")
    spec_path = workdir / api.SPEC_FILE; spec_path.write_text(api.dump_spec(spec))
    opened = []
    real_io_open, real_os_open = io.open, os.open
    def spy_io(f, *a, **k):
        opened.append(str(f)); return real_io_open(f, *a, **k)
    def spy_os(p, *a, **k):
        opened.append(str(p)); return real_os_open(p, *a, **k)
    monkeypatch.setattr("builtins.open", spy_io); monkeypatch.setattr(io, "open", spy_io); monkeypatch.setattr(os, "open", spy_os)
    monkeypatch.setitem(sys.modules, "torch", None)      # any `import torch` raises ImportError during dry run
    with pytest.raises(SystemExit) as ex:
        api.managed_main(spec_path, {"BQLIB_DRYRUN": "1", "BQLIB_NO_MODEL": "1"})
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert ex.value.code in (0, None), out
    assert json.loads(out)["status"] == "dry_run_passed"
    assert str(outcome) not in opened and not any(p.endswith("prior_results.json") for p in opened), "dry run opened an outcome-bearing artifact"
    assert sys.modules.get("torch") is None
    # the SCIENCE branch must verify the same artifact and refuse the wrong sha before importing the producer
    monkeypatch.undo()
    with pytest.raises(SystemExit) as ex2:
        api.managed_main(spec_path, {})
    assert ex2.value.code == 1 and "frozen artifact changed: prior" in capsys.readouterr().out


def test_generated_entry_passes_ops_gate(workdir):
    rows = make_authority(); calls = make_calls(rows)
    entry = workdir / "redteam_entry.py"; entry.write_text(api.render_managed_entry(_spec(workdir, rows, calls)))
    gate = pathlib.Path(__file__).with_name("gate.py")
    r = subprocess.run([sys.executable, str(gate), str(entry)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    env = dict(os.environ, PYTHONPATH=str(pathlib.Path(__file__).parent), BQLIB_DRYRUN="1", BQLIB_NO_MODEL="1")
    run = subprocess.run([sys.executable, str(entry)], capture_output=True, text=True, env=env, cwd=str(workdir))
    assert run.returncode == 0 and json.loads(run.stdout.strip().splitlines()[-1])["status"] == "dry_run_passed", run.stdout + run.stderr


# --- B4 / post-publication --------------------------------------------------------------------------------------------

def test_stage_dir_same_filesystem_as_target(workdir):
    target = workdir / "out"; target.mkdir()
    stage = api.stage_dir_for(target)
    assert os.stat(stage).st_dev == os.stat(target).st_dev
    assert stage.parent == target, "stage must live beside its target (a default tempfile.mkdtemp() lands in /tmp)"


def test_publish_refuses_complete_outcome(workdir):
    rows = make_authority(); calls = make_calls(rows)
    compiled = _compile(workdir, rows, calls)
    pkg = api.project_result(compiled, api.package_evidence(compiled, make_evidence(calls)))
    target = workdir / "out"; target.mkdir()
    paths = api.stage_and_publish(compiled, pkg, target=target)
    receipt_before = paths.receipt.read_bytes()
    _expect_refusal(lambda: api.stage_and_publish(compiled, pkg, target=target), ("complete", "occupied"),
                    missing="post-publication: a second publish over a complete package succeeded")
    assert paths.receipt.read_bytes() == receipt_before and api.load_package(paths)["projection"] == pkg.scores
    (workdir / "elsewhere").mkdir()
    stale = cap.stage_package(api.package_paths(workdir / "elsewhere", "redteam"), evidence_files={}, result={})
    with pytest.raises(cap.PackageError, match="complete"):
        cap.recover_stale_publication(stale, paths)


def test_ordinary_successful_producer_return_exits_zero(workdir, capsys):
    """Regression for R590: the managed wrapper raised after the producer had published and returned normally."""
    rows = make_authority(); calls = make_calls(rows)
    spec_path = workdir / api.SPEC_FILE; spec_path.write_text(api.dump_spec(_spec(workdir, rows, calls)))
    with pytest.raises(SystemExit) as ex:
        api.managed_main(spec_path, {})                     # science branch: producer publishes, then RETURNS
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert ex.value.code == 0, out
    report = json.loads(out); assert report["status"] == "published"
    paths = api.package_paths(workdir / "out", "redteam")
    result = api.load_package(paths)
    assert result["contract_hash"] == report["contract_hash"]
    # dispatch itself returned the producer's value rather than raising after the publish
    synth = api.load_spec(spec_path)
    value = managed.dispatch(synth.spec, base_dir=synth.base_dir, bindings=(managed.ModuleBinding("producer", "redteam_regression_direct"),),
                             producer_role="producer", environment={"BQLIB_DRYRUN": "1"})
    assert value["status"] == "dry_run_passed"
