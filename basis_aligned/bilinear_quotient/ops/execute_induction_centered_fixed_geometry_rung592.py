#!/usr/bin/env python3
"""Immutable-byte managed preflight and scientific dispatch adapter for R592."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"
POLY = ROOT.parent / "polynomial_causal"
PRODUCER = OPS / "induction_centered_fixed_geometry_rung592.py"
OWNER_TEST = OPS / "test_induction_centered_fixed_geometry_rung592.py"
FAKE_RUNTIME_TEST = OPS / "test_induction_centered_fixed_geometry_rung592_fake_runtime.py"
RUNTIME = OPS / "induction_centered_fixed_geometry_rung592_runtime.py"
DRYRUN = ROOT / "induction_centered_fixed_geometry_rung592_dryrun.json"
PREREG = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_PREREGISTRATION.md"
AMENDMENT = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_PREREGISTRATION_AMENDMENT.md"
DIAGNOSTIC_AMENDMENT = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_DIAGNOSTIC_PREFIX_AMENDMENT.md"
MASK_AMENDMENT = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_NONFINITE_MASK_AMENDMENT.md"
PREREG_REVIEW = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_PREREGISTRATION_REVIEW.md"
AMENDMENT_REVIEW = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_AMENDMENT_INDEPENDENT_REVIEW.md"
DIAGNOSTIC_REVIEW = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_DIAGNOSTIC_PREFIX_AMENDMENT_INDEPENDENT_REVIEW.md"
MASK_REVIEW = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_NONFINITE_MASK_AMENDMENT_INDEPENDENT_REVIEW.md"

FROZEN_HASHES = {
    PRODUCER: "c52e1225c128de98b01d33649eb4227ff99e63177a8cbd85b9fd0556b4bf5aee",
    OWNER_TEST: "85f73a6b35f4e9960320bf23996ebc595d02dcd5a76f34ceaef51a6d502c7d54",
    FAKE_RUNTIME_TEST: "f5ea5e005991d57f6d23b5df44d1eccb500ec59469c20f70745ce37f1f6980c0",
    RUNTIME: "df2d59245dc5bd407c96af0a8a6d1c98a70ae25f1925c4540dbd47bb956254a1",
    DRYRUN: "a2c6e760b9b87d70b5a444a11d5bd9f76b0090330fc31e67bdee710aa31e517d",
    PREREG: "870fec55da7207a6e850e64ea705d4f9bb96b2cef40326b2cf59732466dd341a",
    AMENDMENT: "5e9fe2bcf41b88c199b5dfab2ba3ec7d0fa8f4b4b2952173c1984391e4d53094",
    DIAGNOSTIC_AMENDMENT: "f153fa3df6d7d00e951d2e7d2f0a270e6383f9133d0d34049a9eee57640b2c62",
    MASK_AMENDMENT: "f93ce1e524e6a0298a0b28f036ac35c75621c5bc80cf4cc0cac7bbe7589a99dc",
    PREREG_REVIEW: "9b76b91995374697b8a828ce042e59d81bfddcbaa5f6e843cb0f32f6b01e57f7",
    AMENDMENT_REVIEW: "21bdc310b4798d3ae6d47fc2ed7dfee969afd871bc90db381db634e2c4cae2f5",
    DIAGNOSTIC_REVIEW: "e7373c2249e0456327d386559d4f3fa68e0661ed076a35fb120ad9d8effaa675",
    MASK_REVIEW: "b1990a81565cdd63e283ba8896cd9a57b7e8ab81064435a90ea9304d1a5a6c60",
}

OUTCOME_NAMESPACES = (
    ROOT / "induction_centered_fixed_geometry_rung592_results.json",
    ROOT / "induction_centered_fixed_geometry_rung592_receipt.json",
    ROOT / "induction_centered_fixed_geometry_rung592_evidence",
    ROOT / "induction_centered_fixed_geometry_rung592_invalid_diagnostic.json",
    ROOT / "induction_centered_fixed_geometry_rung592_invalid_receipt.json",
    ROOT / "induction_centered_fixed_geometry_rung592_invalid_evidence",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_bytes(bindings: Mapping[Path, str] = FROZEN_HASHES) -> dict[str, str]:
    observed = {}
    for path, expected in bindings.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen R592 file changed or missing: {path}")
        observed[str(path)] = expected
    return observed


def require_unused_namespaces(paths: Sequence[Path] = OUTCOME_NAMESPACES) -> None:
    occupied = [str(path) for path in paths if path.exists()]
    if occupied:
        raise RuntimeError(f"R592 outcome namespace already exists: {occupied}")


def load_frozen_producer():
    os.environ["BQLIB_NO_MODEL"] = "1"
    source = PRODUCER.read_bytes()
    if hashlib.sha256(source).hexdigest() != FROZEN_HASHES[PRODUCER]:
        raise RuntimeError("R592 producer changed before immutable import")
    name = "r592_hash_pinned_managed_producer"
    spec = importlib.util.spec_from_loader(name, loader=None, origin=str(PRODUCER))
    if spec is None:
        raise RuntimeError("cannot construct frozen R592 producer")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(PRODUCER)
    sys.modules[name] = module
    exec(compile(source, str(PRODUCER), "exec"), module.__dict__)
    return module


def run_model_free_validation() -> dict[str, object]:
    producer = load_frozen_producer()
    observed = producer.build_dryrun()
    committed = json.loads(DRYRUN.read_text(encoding="utf-8"))
    if observed != committed:
        raise RuntimeError("R592 dry run differs from frozen artifact")
    if observed["model_forwards"] != 0 or observed["model_backwards"] != 0 or observed["model_weights_updated"] is not False:
        raise RuntimeError("R592 dry run reports model work")
    if any(observed[key] for key in ("select_opened", "final_opened", "ood_opened")):
        raise RuntimeError("R592 dry run opened a sealed split")
    return observed


def preflight(*, namespace_paths: Sequence[Path] = OUTCOME_NAMESPACES) -> dict[str, object]:
    observed = verify_frozen_bytes()
    require_unused_namespaces(namespace_paths)
    dryrun = run_model_free_validation()
    return {
        "schema": "execute_induction_centered_fixed_geometry_rung592_preflight_v1",
        "status": "prospective_candidate_different_agent_exact_review_required",
        "frozen_sha256": observed,
        "fit_call_manifest_sha256": dryrun["fit_call_manifest_sha256"],
        "select_call_manifest_sha256": dryrun["select_call_manifest_sha256"],
        "registered_fit_forwards": 639,
        "registered_select_forwards": 322,
        "registered_max_forwards": 961,
        "model_forwards": 0,
        "model_backwards": 0,
        "model_weights_updated": False,
        "select_opened": False,
        "final_opened": False,
        "ood_opened": False,
    }


def scientific_command() -> tuple[str, list[str]]:
    """Embed verified producer bytes so dispatch cannot reopen a swapped path."""
    source = PRODUCER.read_bytes()
    if hashlib.sha256(source).hexdigest() != FROZEN_HASHES[PRODUCER]:
        raise RuntimeError("R592 producer changed before immutable dispatch")
    encoded = base64.b64encode(source).decode("ascii")
    logical_path = str(PRODUCER)
    launcher = (
        "import base64,sys;"
        f"_p={logical_path!r};sys.argv=[_p];"
        f"_b=base64.b64decode({encoded!r});"
        "exec(compile(_b,_p,'exec'),"
        f"{{'__name__':'__main__','__file__':_p,'__package__':None,"
        f"'__r592_immutable_sha256__':{FROZEN_HASHES[PRODUCER]!r}}})"
    )
    return sys.executable, [sys.executable, "-I", "-c", launcher]


def dispatch(environment: Mapping[str, str], *, exec_function=os.execv,
             namespace_paths: Sequence[Path] = OUTCOME_NAMESPACES) -> dict[str, object]:
    plan = preflight(namespace_paths=namespace_paths)
    mode = environment.get("BQLIB_DRYRUN")
    if mode == "1":
        plan["mode"] = "model_free_dryrun"
        return plan
    if mode is not None:
        raise RuntimeError("BQLIB_DRYRUN must be absent or exactly '1'")
    executable, argv = scientific_command()
    exec_function(executable, argv)
    raise RuntimeError("R592 scientific os.execv unexpectedly returned")


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        raise SystemExit("R592 preflight adapter accepts no command-line arguments")
    report = dispatch(os.environ)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
