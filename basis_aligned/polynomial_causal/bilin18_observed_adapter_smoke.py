"""Synthetic production-shape closure of one observed P/P/N coordinate transaction.

This is an implementation smoke, not a data role or scientific evaluation.  It uses
only token ID zero, runs initial-denominator route Q, and prints a JSON receipt.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

import bilin18_frozen_ship_program as frozen
import bilin18_observed_adapter as observed
import bilin18_observed_model_facade as facade
import early_mlp_suffix_transport_v1_capabilities as capabilities
import early_mlp_suffix_transport_v1_inherited as inherited
import early_mlp_suffix_transport_v1_runtime as runtime


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = (
    "basis_aligned/polynomial_causal/bilin18_observed_adapter_smoke.py",
    "basis_aligned/polynomial_causal/bilin18_observed_adapter.py",
    "basis_aligned/polynomial_causal/bilin18_observed_model_facade.py",
    "basis_aligned/polynomial_causal/bilin18_frozen_ship_program.py",
    "basis_aligned/polynomial_causal/early_mlp_suffix_transport_v1_capabilities.py",
    "basis_aligned/polynomial_causal/early_mlp_suffix_transport_v1_runtime.py",
    "basis_aligned/polynomial_causal/early_mlp_suffix_transport_v1_inherited.py",
)


def _git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def _source_closure() -> tuple[str, dict[str, str]]:
    commit = _git("rev-parse", "HEAD").decode().strip()
    hashes: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        current = (ROOT / relative).read_bytes()
        committed = _git("show", f"{commit}:{relative}")
        if current != committed:
            raise RuntimeError(f"smoke source differs from HEAD: {relative}")
        hashes[relative] = hashlib.sha256(current).hexdigest()
    return commit, hashes


def run_smoke() -> dict:
    started = time.time()
    source_commit, source_hashes = _source_closure()
    initial = inherited.load_canonical_initialization()
    bases = initial.clone_bases()
    program = initial.make_program("L").to("cuda")
    coordinator = runtime.ScopeCoordinator()
    issuer = runtime.logical_identity_sha256({
        "smoke": source_commit, "kind": "production_adapter_initial_Q",
    })
    synthetic_rows = runtime.logical_identity_sha256({
        "role": "synthetic_zero_tokens", "shape": [4, 256],
    })
    synthetic_fit = runtime.logical_identity_sha256({"tensor": "all_zero_token_ids"})
    teacher_map = runtime.logical_identity_sha256({
        "mapping": "identity", "synthetic": True,
    })
    context = capabilities.RunContext(
        source_commit=source_commit,
        inherited_snapshot_sha256=initial.authority.snapshot_sha256,
        rows_receipt_sha256=synthetic_rows,
        fit_role_tensor_sha256=synthetic_fit,
        identity_teacher_mapping_sha256=teacher_map,
    )
    torch.cuda.reset_peak_memory_stats()
    model, model_receipt = facade.load_bilin18(
        device="cuda", verify_weights_sha256=True,
    )
    ship, ship_receipt = frozen.load_frozen_ship(device="cuda", verify_bytes=True)
    adapter = observed.ObservedBilin18Adapter(model, ship, production=True)
    broker = adapter.make_capability_broker(
        issuer_id=issuer, coordinator=coordinator, run_context=context, bases=bases,
    )
    hook = runtime.StudentCorrectionHook(
        bases, issuer_id=issuer, coordinator=coordinator,
    )
    hook.configure(program=program, states={0: "P", 1: "P"})
    tokens = torch.zeros((4, 256), dtype=torch.long, device="cuda")
    indices = (0, 1, 2, 3)
    identity = runtime.TraceIdentity.from_inputs(
        inputs=tokens, ordered_batch_indices=indices, source_commit=source_commit,
        inherited_snapshot_sha256=initial.authority.snapshot_sha256,
        rows_receipt_sha256=synthetic_rows,
        fit_role_tensor_sha256=synthetic_fit,
        program_snapshot_sha256=runtime.program_snapshot_sha256(program),
        teacher_mapping_sha256=teacher_map,
        phase="initial_denominator", route="Q", control="true",
        teacher_kind="coordinate_labels", trial=0, epoch=0,
        optimizer_step=0, batch_ordinal=0,
        student_states=((0, "P"), (1, "P"), (2, "N")),
    )
    session = broker.begin_student(identity, hook, tokens, indices)
    step, student_closure, observed_closure = adapter.run_student(
        session=session, hook=hook, identity=identity, tokens=tokens,
    )
    teacher = broker.run_coordinate_teacher(identity, step)
    moments, teacher_closure = teacher.consume_moments()
    torch.cuda.synchronize()
    ledger = broker.ledger_snapshot

    if [moment.count for moment in moments] != [768, 768]:
        raise RuntimeError("coordinate moment support did not close")
    if ledger.outstanding_identity_sha256 is not None or (
        ledger.student_identity_count,
        ledger.teacher_identity_count,
        ledger.completed_identity_count,
    ) != (1, 1, 1):
        raise RuntimeError("capability ledger did not close")
    if not coordinator.idle or hook.program is not None or hook.states != {}:
        raise RuntimeError("scope coordinator or correction hook remained live")

    return {
        "status": "implementation_smoke_only",
        "scientific_authority": False,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "source_hashes": source_hashes,
        "input": {"kind": "synthetic_constant_tokens", "token_id": 0, "shape": [4, 256]},
        "model": asdict(model_receipt),
        "frozen_ship": asdict(ship_receipt),
        "inherited_snapshot_sha256": initial.authority.snapshot_sha256,
        "identity_sha256": identity.sha256,
        "student_closure": asdict(student_closure),
        "observed_closure": asdict(observed_closure),
        "teacher_closure": asdict(teacher_closure),
        "moment_counts": [moment.count for moment in moments],
        "ledger": asdict(ledger),
        "coordinator_idle": coordinator.idle,
        "hook_cleared": hook.program is None and hook.states == {},
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "runtime_seconds": round(time.time() - started, 3),
        "interpretation": (
            "The complete observed P/P/N student plus coordinate-teacher capability "
            "transaction closes on synthetic inputs. No corpus row, teacher target, "
            "CE, KL, selection, or scientific gate was evaluated."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run_smoke(), indent=2))
