"""Thin adapter binding Claude's adversarial fixtures to Codex's ACTUAL public names (2026-09-04).

Rule of this file: it contains NO validation logic of its own.  Every refusal a test observes must come from
circuit_experiment_spec / circuit_artifact_package / circuit_managed_entry.  Where this adapter has to *choose* which
Codex validator to call (e.g. which predicate_id a runtime would pass to validate_nonfinite_masks) the choice models the
documented R592 runtime behaviour and is stated in a comment.  Where Codex lacks a semantic entirely, the adapter does
not fake it: the attack test fails with a message naming the gap.

Temporary directories live under <repo>/.adv_tmp (same filesystem as the repo), never under /tmp (attack B4).
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
import uuid
from typing import Callable, Mapping, Sequence

import numpy as np

import circuit_artifact_package as pkg
import circuit_experiment_spec as cs
import circuit_managed_entry as managed

REPO = pathlib.Path(__file__).resolve().parents[1]
ADV_TMP = REPO / ".adv_tmp"
ROWS_FILE = "authority.json"
PRODUCER_FILE = "redteam_producer.py"
SPEC_FILE = "spec.json"
TERMINAL_NONFINITE = "nonfinite_observation"      # Codex's literal terminal name (circuit_artifact_package)
TERMINAL_OK = "ok"


# ----------------------------------------------------------------------------------------------------------------------
# scratch space (B4: same filesystem as the publication target)
# ----------------------------------------------------------------------------------------------------------------------

def make_workdir(label: str) -> pathlib.Path:
    ADV_TMP.mkdir(exist_ok=True)
    return pathlib.Path(tempfile.mkdtemp(prefix=f"{label}-", dir=ADV_TMP))


def remove_workdir(path: pathlib.Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
    try:
        ADV_TMP.rmdir()          # only succeeds when the last workdir is gone
    except OSError:
        pass


def canon(obj) -> bytes:
    return cs.canonical_json_bytes(obj)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ----------------------------------------------------------------------------------------------------------------------
# spec construction on Codex's dataclasses
# ----------------------------------------------------------------------------------------------------------------------

@dataclasses.dataclass
class SynthSpec:
    spec: cs.CircuitExperimentSpec
    base_dir: pathlib.Path
    arms: list                       # the fixture's arm declarations (role/direction) -- Codex's spec cannot carry them
    predicates: list                 # the fixture's predicate declarations (kind/evaluator) -- ditto
    science_names: list
    projector: Callable
    batch: int


def build_synthetic_spec(*, experiment_id, rows_path, calls, arms, predicates, science_names, projector, batch=4):
    rows_path = pathlib.Path(rows_path)
    base_dir = rows_path.parent
    rows = json.loads(rows_path.read_text())
    counts = {}
    for r in rows:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    arm_names = tuple(a["name"] for a in arms)
    families = []
    for split in ("FIT", "SELECT"):
        families.append(cs.CallFamilySpec(
            name=f"{split.lower()}_margin", split=split, arms=arm_names, batch_size=batch, call_kind="margin",
            guard="fit_always" if split == "FIT" else "selected_only", call_id_template="{split}:{arm}:{batch}",
            sequence_field="ids", row_id_field="row_id", axis_order="arm_batch", sort_policy="canonical_json"))
    preds = []
    for i, p in enumerate(predicates):
        preds.append(cs.PredicateSpec(
            predicate_id=p["predicate_id"], phase="FIT", priority=p.get("priority", i),
            evaluator_role=p.get("kind", "evidence"), required_arrays=("margin",), disposition=p["disposition"]))
    spec = cs.CircuitExperimentSpec(
        experiment_id=experiment_id, rung=0,
        artifacts=(cs.ArtifactRef("rows", rows_path.name, sha(rows_path.read_bytes()), "authority", dryrun_access=True),),
        phases=(cs.PhaseSpec("FIT"), cs.PhaseSpec("SELECT", opens_after="FIT")),
        authority_tables=(cs.AuthorityTableSpec("rows", ("row_id",), group_fields=("group_id",),
                                                expected_counts=counts, expected_total=len(rows)),),
        calls=tuple(families),
        arrays=(cs.ArraySpec("margin", ("margin",), "float64", ("batch",), True, "final_nonfinite_diagnostic"),),
        predicates=tuple(preds),
        science=cs.ScienceProjectionSpec(
            projector_role="projector", decision_role="decision",
            allowed_terminals=(TERMINAL_OK, "hard_abort", "final_nonfinite_diagnostic"),
            output_types={n: ("boolean" if n.startswith("pred_") else "number") for n in science_names}),
    )
    cs.validate_spec(spec)
    return SynthSpec(spec, base_dir, list(arms), list(predicates), list(science_names), projector, batch)


def with_projector(synth: SynthSpec, projector) -> SynthSpec:
    return dataclasses.replace(synth, projector=projector)


def with_artifact(synth: SynthSpec, *, role, path, sha256, kind, executable=False) -> SynthSpec:
    rel = os.path.relpath(path, synth.base_dir)
    ref = cs.ArtifactRef(role, rel, sha256, kind, executable=executable, dryrun_access=(kind != "outcome"))
    spec = dataclasses.replace(synth.spec, artifacts=synth.spec.artifacts + (ref,))
    cs.validate_spec(spec)
    return dataclasses.replace(synth, spec=spec)


_MAPPING_FIELDS = {"expected_counts", "output_types"}
_ITEM_TYPES = {"artifacts": cs.ArtifactRef, "phases": cs.PhaseSpec, "authority_tables": cs.AuthorityTableSpec,
               "calls": cs.CallFamilySpec, "arrays": cs.ArraySpec, "predicates": cs.PredicateSpec}


def _tuples(v):
    return tuple(_tuples(x) for x in v) if isinstance(v, list) else v


def _from_json(cls, data: Mapping):
    kwargs = {}
    for f in dataclasses.fields(cls):
        v = data[f.name]
        if f.name in _ITEM_TYPES:
            v = tuple(_from_json(_ITEM_TYPES[f.name], item) for item in v)
        elif f.name == "science":
            v = None if v is None else _from_json(cs.ScienceProjectionSpec, v)
        elif f.name not in _MAPPING_FIELDS:
            v = _tuples(v)
        kwargs[f.name] = v
    return cls(**kwargs)


def dump_spec(synth: SynthSpec) -> str:
    """Freeze the executable closure: render the producer, declare it as an executable source artifact, dump JSON."""
    producer = synth.base_dir / PRODUCER_FILE
    producer.write_text(_producer_source(synth))
    refs = tuple(r for r in synth.spec.artifacts if r.role != "producer")
    refs += (cs.ArtifactRef("producer", PRODUCER_FILE, sha(producer.read_bytes()), "source", executable=True, dryrun_access=True),)
    spec = dataclasses.replace(synth.spec, artifacts=refs)
    cs.validate_spec(spec)
    return json.dumps({"spec": cs.spec_json(spec), "adapter": {
        "arms": synth.arms, "predicates": [{k: v for k, v in p.items() if k != "evaluator"} for p in synth.predicates],
        "science_names": synth.science_names, "batch": synth.batch}}, sort_keys=True, indent=1)


def load_spec(spec_path: pathlib.Path) -> SynthSpec:
    data = json.loads(pathlib.Path(spec_path).read_text())
    spec = _from_json(cs.CircuitExperimentSpec, data["spec"])
    cs.validate_spec(spec)
    a = data["adapter"]
    return SynthSpec(spec, pathlib.Path(spec_path).parent, a["arms"], a["predicates"], a["science_names"],
                     default_projector, a["batch"])


def default_projector(records):
    """Pure reference projector used by rendered producers (mirrors the fixture's projector_mean_gap)."""
    n = [e["margin"] for e in records if e["arm"] == "native"]
    c = [e["margin"] for e in records if e["arm"] == "counterfactual"]
    gap = sum(n) / len(n) - sum(c) / len(c)
    return {"pred_a_instrument": len(n) == len(c) > 0, "pred_b_gap_positive": gap > 0,
            "pred_c_gap_bounded": abs(gap) < 10.0, "score_gap": gap}


# ----------------------------------------------------------------------------------------------------------------------
# compile
# ----------------------------------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Compiled:
    synth: SynthSpec
    compiled: dict
    contract_hash: str

    @property
    def manifest(self) -> list:
        return self.compiled["call_manifest"]

    @property
    def max_price(self) -> int:
        return self.compiled["call_summary"]["call_count"]

    @property
    def calls(self) -> list:
        """Generator-shaped view of Codex's literal manifest (rows = row_ids; physical_width = literal batch)."""
        return [{"call_id": c["call_id"], "split": c["split"], "arm": c["arm"], "rows": list(c["row_ids"]),
                 "physical_width": c["logical_batch_size"]} for c in self.manifest]

    @property
    def split_content_hashes(self):
        raise AttributeError("Codex's compiled contract carries records_sha256/ordered_identities_sha256 per TABLE, "
                             "not a per-split content hash, and AuthorityTableSpec pins neither")


def compile_experiment(synth: SynthSpec, *, authority_inputs: Mapping[str, bytes]) -> Compiled:
    rows = json.loads(authority_inputs["rows"])
    compiled = cs.compile_experiment(synth.spec, authority_tables={"rows": rows}, call_source_records=rows)
    return Compiled(synth, compiled, cs.canonical_sha256(compiled))


def rebind_manifest(c: Compiled, calls: Sequence[Mapping]) -> Compiled:
    """Same compiled contract with the call manifest replaced by a generator-shaped schedule (A9 hash sensitivity)."""
    by_id = {m["call_id"]: m for m in c.manifest}
    manifest = [dict(by_id[k["call_id"]], row_ids=list(k["rows"]), logical_batch_size=len(k["rows"])) for k in calls]
    compiled = dict(c.compiled, call_manifest=manifest, call_summary=cs.summarize_call_manifest(manifest))
    return Compiled(c.synth, compiled, cs.canonical_sha256(compiled))


# ----------------------------------------------------------------------------------------------------------------------
# evidence in Codex's on-disk shape: one directory per call, <index>_<call_id>/{call.json, margin.npy}
# ----------------------------------------------------------------------------------------------------------------------

def _npy_bytes(arr) -> bytes:
    buf = io.BytesIO(); np.save(buf, np.asarray(arr), allow_pickle=False); return buf.getvalue()


def synthetic_margins(manifest: Sequence[Mapping]) -> list:
    """Deterministic model-free primitives for rendered producers (one record per call x row)."""
    ev = []
    for c in manifest:
        for rid in c["row_ids"]:
            h = int(sha(f"{c['call_id']}|{rid}".encode())[:8], 16) / 0xFFFFFFFF
            ev.append({"call_id": c["call_id"], "arm": c["arm"], "row_id": rid,
                       "margin": (h * 6 - 3) * (1.0 if c["arm"] == "native" else 0.6)})
    return ev


@dataclasses.dataclass
class Evidence:
    records: list            # observed call records, in observation order (what a runtime writes per call)
    dir_names: list
    files: dict              # relative path -> bytes (Codex's evidence_files shape)
    primitives: list         # the generator's primitive records


def package_evidence(c: Compiled, ev: Sequence[Mapping], *, physical_pad: Mapping[str, int] | None = None) -> Evidence:
    """Materialise primitives per call.  The observed call record is DERIVED from the primitives in that directory
    (row_ids / logical_batch_size), all other fields copied from the compiled record, as a runtime would write it.
    `physical_pad` models a runtime that forwarded a padded tail: the saved array has that many rows (A7 drift)."""
    by_id = {m["call_id"]: m for m in c.manifest}
    order, groups = [], {}
    for e in ev:
        if e["call_id"] not in groups:
            order.append(e["call_id"]); groups[e["call_id"]] = []
        groups[e["call_id"]].append(e)
    records, dir_names, files = [], [], {}
    for i, cid in enumerate(order):
        rows = groups[cid]
        base = dict(by_id.get(cid, {"call_id": cid, "split": rows[0].get("split", "?"), "arm": rows[0]["arm"]}))
        rec = dict(base, row_ids=[e["row_id"] for e in rows], logical_batch_size=len(rows))
        arr = np.array([e["margin"] for e in rows], dtype=np.float64)
        if physical_pad and cid in physical_pad:
            arr = np.concatenate([arr, np.zeros(physical_pad[cid] - len(arr))])
        d = pkg.call_directory_name(i, cid)
        records.append(rec); dir_names.append(d)
        files[f"calls/{d}/call.json"] = canon(rec) + b"\n"
        files[f"calls/{d}/margin.npy"] = _npy_bytes(arr)
    return Evidence(records, dir_names, files, list(ev))


def _write_files(root: pathlib.Path, files: Mapping[str, bytes]) -> None:
    for rel, data in files.items():
        p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)


def _has_nonfinite(call_dir: pathlib.Path) -> bool:
    return any(a.dtype.kind == "f" and not np.isfinite(a).all() for a in pkg._raw_arrays(call_dir).values())


@dataclasses.dataclass
class Audit:
    ok: bool
    failures: list
    terminal: str
    checks: list


def validate_call_evidence(c: Compiled, ev: Sequence[Mapping], *, workdir: pathlib.Path,
                           physical_pad: Mapping[str, int] | None = None) -> Audit:
    """Run every Codex evidence validator that exists: literal manifest-prefix + directory census, and the
    nonfinite-mask validators.  Runtime modelling (R592): the run's terminal predicate is nonfinite_observation iff
    the LAST observed call holds a nonfinite float array; that predicate is validated on the last call directory; earlier
    directories are validated under the finite predicate (which, in Codex, only asserts mask artifacts are absent)."""
    evidence = package_evidence(c, ev, physical_pad=physical_pad)
    root = workdir / f"audit-{uuid.uuid4().hex[:8]}"
    _write_files(root, evidence.files)
    dirs = [root / "calls" / d for d in evidence.dir_names]
    failures, checks = [], []
    try:
        pkg.validate_call_prefix(c.manifest, evidence.records, evidence.dir_names)
        checks.append("validate_call_prefix")
    except pkg.PackageError as e:
        failures.append(f"validate_call_prefix: {e}")
    terminal = TERMINAL_NONFINITE if dirs and _has_nonfinite(dirs[-1]) else TERMINAL_OK
    if terminal == TERMINAL_NONFINITE:
        try:
            pkg.write_nonfinite_masks(dirs[-1]); checks.append("write_nonfinite_masks")
        except pkg.PackageError as e:
            failures.append(f"write_nonfinite_masks: {e}")
    for d in dirs:
        pid = terminal if d is dirs[-1] else TERMINAL_OK
        try:
            pkg.validate_nonfinite_masks(d, pid); checks.append(f"validate_nonfinite_masks[{d.name}:{pid}]")
        except pkg.PackageError as e:
            failures.append(f"validate_nonfinite_masks[{d.name}:{pid}]: {e}")
    ok = not failures
    out_terminal = "hard_abort" if failures else ("final_nonfinite_diagnostic" if terminal == TERMINAL_NONFINITE else TERMINAL_OK)
    return Audit(ok, failures, out_terminal, checks)


# ----------------------------------------------------------------------------------------------------------------------
# projection + package + publication
# ----------------------------------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Package:
    compiled: Compiled
    evidence: Evidence
    result: dict
    predicates_evaluated: bool = False       # Codex evaluates no predicate; see project_result

    @property
    def scores(self) -> dict:
        return self.result["projection"]

    @property
    def terminal(self) -> str:
        return self.result["terminal"]


def _projector_of(synth: SynthSpec):
    return lambda evidence: dict(synth.projector(evidence["records"]))


def project_result(c: Compiled, evidence: Evidence) -> Package:
    """Producer path: projection = projector(primitives); Codex's audit = validate_science_projection, which recomputes
    the projector ONCE on the same evidence object.  No predicate is evaluated because Codex's PredicateSpec has no
    evaluator (evaluator_role is a string) and no module resolves a terminal from predicate_order."""
    prim = {"records": evidence.primitives}
    projection = dict(c.synth.projector(evidence.primitives))
    pkg.validate_science_projection(prim, projection, _projector_of(c.synth))
    result = {"schema": "redteam_result_v1", "experiment_id": c.synth.spec.experiment_id, "contract_hash": c.contract_hash,
              "manifest_sha256": c.compiled["call_summary"]["manifest_sha256"], "projection": projection,
              "terminal": TERMINAL_OK, "predicate_order": c.compiled["predicate_order"]}
    return Package(c, evidence, result)


def package_paths(target: pathlib.Path, namespace: str) -> pkg.PackagePaths:
    return pkg.PackagePaths(root=target, result=target / "result.json", receipt=target / "receipt.json",
                            evidence=target / "evidence", namespace=namespace)


def stage_dir_for(target: pathlib.Path) -> pathlib.Path:
    """Codex has no separate stage-dir API: stage_package creates the stage.  Return the stage it creates (left on disk;
    the caller's workdir cleanup removes it)."""
    paths = package_paths(target, "probe")
    return pkg.stage_package(paths, evidence_files={"probe.bin": b"probe"}, result={"schema": "probe"})


def stage_and_publish(c: Compiled, p: Package, *, target: pathlib.Path) -> pkg.PackagePaths:
    paths = package_paths(target, c.synth.spec.experiment_id)
    stage = pkg.stage_package(paths, evidence_files=p.evidence.files, result=p.result)
    pkg.publish_staged_package(stage, paths)
    pkg.validate_complete_package(paths)
    return paths


def dump_package(p: Package, *, target: pathlib.Path) -> pkg.PackagePaths:
    return stage_and_publish(p.compiled, p, target=target)


def primitives_from_disk(paths: pkg.PackagePaths) -> list:
    """Rebuild the projector's input from the on-disk primitives (call.json row order zipped with margin.npy)."""
    out = []
    for d in sorted((paths.evidence / "calls").iterdir()):
        rec = json.loads((d / "call.json").read_bytes())
        arr = np.load(d / "margin.npy", allow_pickle=False)
        for rid, m in zip(rec["row_ids"], arr.tolist()):
            out.append({"call_id": rec["call_id"], "arm": rec["arm"], "row_id": rid, "margin": m})
    return out


def load_package(paths: pkg.PackagePaths) -> dict:
    return pkg.validate_complete_package(paths)


def verify_package(c: Compiled, paths: pkg.PackagePaths) -> dict:
    result = pkg.validate_complete_package(paths)
    return pkg.validate_science_projection({"records": primitives_from_disk(paths)}, result["projection"], _projector_of(c.synth))


def resign_receipt(paths: pkg.PackagePaths) -> None:
    """Attacker with write access: make receipt.result_sha256 match a tampered result.json (descriptors untouched)."""
    receipt = json.loads(paths.receipt.read_bytes())
    receipt["result_sha256"] = sha(paths.result.read_bytes())
    paths.receipt.write_bytes(canon(receipt) + b"\n")


# ----------------------------------------------------------------------------------------------------------------------
# managed entry
# ----------------------------------------------------------------------------------------------------------------------

def begin_run(c: Compiled) -> dict:
    """Runtime preflight before the first forward request: Codex's capture re-hashes every declared artifact."""
    managed.validate_dryrun_closure(c.synth.spec)
    return managed.capture_frozen_artifacts(c.synth.spec, base_dir=c.synth.base_dir, dryrun=False)


def _producer_source(synth: SynthSpec) -> str:
    return f'''"""Adapter-rendered producer for {synth.spec.experiment_id}: dry run compiles only; science publishes and RETURNS."""
import json, pathlib
import _adv_adapter as adapter
BASE = pathlib.Path({str(synth.base_dir)!r})

def _compiled():
    synth = adapter.load_spec(BASE / {SPEC_FILE!r})
    rows = json.loads((BASE / {ROWS_FILE!r}).read_text())
    return adapter.compile_experiment(synth, authority_inputs={{"rows": adapter.canon(rows)}})

def run_dryrun():
    compiled = _compiled()
    return {{"status": "dry_run_passed", "contract_hash": compiled.contract_hash, "model_forwards": 0,
            "maximum_forwards": compiled.max_price}}

def run_science():
    compiled = _compiled()
    evidence = adapter.package_evidence(compiled, adapter.synthetic_margins(compiled.manifest))
    package = adapter.project_result(compiled, evidence)
    target = BASE / "out"; target.mkdir(exist_ok=True)
    paths = adapter.stage_and_publish(compiled, package, target=target)
    return {{"status": "published", "receipt": str(paths.receipt), "contract_hash": compiled.contract_hash}}
'''


def managed_main(spec_path: pathlib.Path, environment: Mapping[str, str]) -> None:
    """CLI-shaped wrapper Codex does not ship: dispatch, print one JSON status line, exit 0 on an ORDINARY return
    (the R590 wrapper raised after a successful publish; this maps a normal producer return to exit 0)."""
    spec_path = pathlib.Path(spec_path)
    synth = load_spec(spec_path)
    try:
        report = managed.dispatch(
            synth.spec, base_dir=synth.base_dir,
            bindings=(managed.ModuleBinding("producer", f"redteam_producer_{uuid.uuid4().hex[:8]}"),),
            producer_role="producer", environment=dict(environment))
    except Exception as error:      # noqa: BLE001 - the CLI boundary maps any failure to exit 1
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}, sort_keys=True))
        raise SystemExit(1) from error
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0)


def render_managed_entry(synth: SynthSpec) -> str:
    """Per-experiment queue entry in the R590 hand-written pattern (Codex ships no renderer): dumps the spec, then
    emits a gate-shaped script whose only runtime call is circuit_managed_entry.dispatch."""
    spec_path = synth.base_dir / SPEC_FILE
    spec_path.write_text(dump_spec(synth))
    preds = [n for n in synth.science_names if n.startswith("pred_")]
    pred_lines = "\n".join(f"    {n!r}: {('registered science output ' + n)!r}," for n in preds)
    return f'''#!/usr/bin/env python3
# BQLANE: cpu
# BQGATE: EXPERIMENT {" ".join(preds)}
"""Managed entry for {synth.spec.experiment_id} (adapter-rendered; runtime = circuit_managed_entry.dispatch)."""
import json
import os
import pathlib
import sys
import _adv_adapter as adapter
import circuit_managed_entry as managed

SPEC_PATH = pathlib.Path({str(spec_path)!r})
PREDICATES = {{
{pred_lines}
}}


def main():
    synth = adapter.load_spec(SPEC_PATH)
    report = managed.dispatch(
        synth.spec, base_dir=synth.base_dir,
        bindings=(managed.ModuleBinding('producer', 'redteam_entry_producer'),),
        producer_role='producer', environment=os.environ)
    print(json.dumps({{'status': report.get('status'), 'science': sorted(PREDICATES)}}, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
'''
