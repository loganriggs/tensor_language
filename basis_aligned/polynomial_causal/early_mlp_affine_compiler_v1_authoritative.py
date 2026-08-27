#!/usr/bin/env python3
"""Fit, freeze, and score the preregistered executable affine compiler v1."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
PREREG = HERE / "early_mlp_affine_compiler_v1_preregistration.json"
PREREG_SHA256 = "f4da5a8085f9aad3b0cf22f377b04b6dfc39d7950089ba8b9a2c50af587cbc5f"
ROWS_RECEIPT = BQ / "early_mlp_affine_compiler_v1_rows_receipt.json"
ROWS_RECEIPT_SHA256 = "762528ea02cd98071ea55e6b4e904a8fc453f3eb4e545946b8e7149aaf8caa04"
V3_AUTHORITY = BQ / "joint_early_mlp_pca_composition_authoritative_v3_authority.json"
V3_RESULT = BQ / "joint_early_mlp_pca_composition_authoritative_v3_results.json"
V3_MANIFEST = BQ / "joint_early_mlp_pca_composition_authoritative_v3_manifest.json"
V3_BASIS = BQ / "joint_early_mlp_pca_composition_authoritative_v3_bases.pt"
V3_BASIS_RECEIPT = BQ / "joint_early_mlp_pca_composition_authoritative_v3_basis_receipt.json"
PINS = {
    PREREG: PREREG_SHA256,
    ROWS_RECEIPT: ROWS_RECEIPT_SHA256,
    V3_AUTHORITY: "38cf5a349e4426b4ed3227ad11f37499ffa9c3959de5d85d606580aa32c39f1e",
    V3_RESULT: "c3408feb031165b747346107841e2e82066aa80ea223ecde845f085d30006587",
    V3_MANIFEST: "cae4a3092213d3e1d0983448576f9a4c3966a092d731d56ab16169a8c82b7588",
    V3_BASIS: "0eee01f39087548a479486d068404f78c4bdc2fd930932add162212da31fe4d9",
    V3_BASIS_RECEIPT: "b81adb4c78255613997de4cbfc8ffd9e8eec233b40950915a14005ba3efcba0f",
}
ARTIFACT = BQ / "early_mlp_affine_compiler_v1_programs.pt"
ARTIFACT_RECEIPT = BQ / "early_mlp_affine_compiler_v1_programs_receipt.json"
RESULT = BQ / "early_mlp_affine_compiler_v1_results.json"
MANIFEST = BQ / "early_mlp_affine_compiler_v1_manifest.json"
AUTHORITY = BQ / "early_mlp_affine_compiler_v1_authority.json"
LOCK = Path("/workspace/runs/.early_mlp_affine_compiler_v1.lock")
OUTPUTS = (ARTIFACT, ARTIFACT_RECEIPT, RESULT, MANIFEST, AUTHORITY)
SHIP_HASH = "21ddc9ffdb7703aa570f88c5c7f4fa9fe007a988a1a7a3fd91058ee76a25ab8e"
FIT_SEED = 271828
BOOTSTRAP_SEED = 31415926
BOOTSTRAP_DRAWS = 2000

sys.path.insert(0, str(HERE))
import affine_compiler_runtime_v1 as runtime  # noqa: E402
import code_ood_oracle as code_oracle  # noqa: E402
import early_mlp_affine_compiler_v1 as compiler  # noqa: E402
import frozen_ship_oracle_v2 as frozen  # noqa: E402
import joint_early_mlp_oracle_factorial_authoritative as exact_runner  # noqa: E402
import joint_early_mlp_pca_composition_authoritative_v3 as v3  # noqa: E402
import prepare_affine_compiler_rows_v1 as fresh_rows  # noqa: E402
import prepare_fineweb_oracle_rows as old_rows  # noqa: E402


SOURCE_CLOSURE = (
    Path(__file__),
    HERE / "early_mlp_affine_compiler_v1.py",
    HERE / "affine_compiler_runtime_v1.py",
    HERE / "prepare_affine_compiler_rows_v1.py",
    HERE / "test_early_mlp_affine_compiler_v1.py",
    HERE / "test_affine_compiler_runtime_v1.py",
    HERE / "test_prepare_affine_compiler_rows_v1.py",
    *exact_runner.SOURCE_CLOSURE,
)
PROTECTED = tuple(dict.fromkeys((
    *PINS,
    *exact_runner.PROTECTED_EXISTING,
    Path("/workspace/runs/bilin18_frozen_ship_v2.pt"),
    Path("/workspace/runs/bilin18_frozen_ship_v2_manifest.json"),
)))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return code_oracle.tensor_sha256(value.detach().cpu().contiguous())


def write_json_atomic(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_torch_atomic(value: Any, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def protected_snapshot() -> dict[str, str | None]:
    return {str(path): file_sha256(path) if path.is_file() else None for path in PROTECTED}


def verify_pins() -> None:
    for path, expected in PINS.items():
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"pinned affine compiler input changed: {path}")
    authority = json.loads(V3_AUTHORITY.read_text())
    if authority.get("authorized_for_scored_experiments") is not True:
        raise RuntimeError("v3 basis authority is not admitted")
    if authority.get("authorized_for_training") is not False:
        raise RuntimeError("v3 basis training guard unexpectedly changed")
    row_receipt = json.loads(ROWS_RECEIPT.read_text())
    required = {
        "authorized_for_training": True,
        "authorized_for_scored_experiments": True,
        "training_license_sites": [0, 1],
        "preregistration_sha256": PREREG_SHA256,
    }
    for key, expected in required.items():
        if row_receipt.get(key) != expected:
            raise RuntimeError(f"fresh compiler authority changed at {key}")


def verify_source_closure() -> dict[str, str]:
    hashes = {}
    for path in dict.fromkeys(SOURCE_CLOSURE):
        relative = path.resolve().relative_to(ROOT.resolve())
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(relative)], cwd=ROOT,
            capture_output=True, text=True,
        )
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", str(relative)], cwd=ROOT,
        )
        if tracked.returncode != 0 or dirty.returncode != 0:
            raise RuntimeError(f"behavior source is not committed and clean: {relative}")
        hashes[str(relative)] = file_sha256(path)
    return hashes


def document_block_permutation(document_ids: Sequence[str], seed: int) -> torch.Tensor:
    """Permute whole row blocks among documents having the same row count."""

    groups: dict[str, list[int]] = {}
    for index, document in enumerate(document_ids):
        if not isinstance(document, str) or not document:
            raise ValueError("document IDs must be nonempty strings")
        groups.setdefault(document, []).append(index)
    strata: dict[int, list[str]] = {}
    for document, indices in groups.items():
        strata.setdefault(len(indices), []).append(document)
    generator = torch.Generator().manual_seed(seed)
    output = torch.arange(len(document_ids), dtype=torch.long)
    moved = 0
    for count, documents in sorted(strata.items()):
        documents = sorted(documents)
        if len(documents) < 2:
            continue
        order = torch.randperm(len(documents), generator=generator).tolist()
        if order == list(range(len(documents))):
            order = order[1:] + order[:1]
        for target_index, source_index in enumerate(order):
            target_rows = groups[documents[target_index]]
            source_rows = groups[documents[source_index]]
            if len(target_rows) != count or len(source_rows) != count:
                raise RuntimeError("document permutation stratum changed size")
            output[torch.tensor(target_rows)] = torch.tensor(source_rows)
            moved += sum(a != b for a, b in zip(target_rows, source_rows, strict=True))
    if moved == 0 or sorted(output.tolist()) != list(range(len(document_ids))):
        raise RuntimeError("document-block permutation is degenerate")
    return output


def expand_capture_permutation(row_permutation: torch.Tensor) -> torch.Tensor:
    offsets = torch.arange(64, dtype=torch.long)
    return (row_permutation[:, None] * 64 + offsets[None, :]).reshape(-1)


def cpu_state(state: Mapping[str, Any]) -> dict[str, Any]:
    output = {}
    for key, value in state.items():
        output[key] = (
            value.detach().cpu().float().contiguous() if torch.is_tensor(value) else value
        )
    return output


def device_programs(
    programs: Mapping[str, Mapping[int, Mapping[str, Any]]], device: Any
) -> dict[str, dict[int, dict[str, Any]]]:
    return {
        name: {
            site: {
                key: (value.to(device).float() if torch.is_tensor(value) else value)
                for key, value in state.items()
            }
            for site, state in sites.items()
        }
        for name, sites in programs.items()
    }


def mean_state(target: torch.Tensor) -> dict[str, Any]:
    return {
        "mean": torch.zeros(compiler.D_MODEL),
        "scale": torch.ones(compiler.D_MODEL),
        "bias": target.float().mean(dim=0),
        "left": torch.zeros(compiler.D_MODEL, 8),
        "right": torch.zeros(8, compiler.COEFFICIENT_DIM),
        "rank": 0,
        "lambda": None,
        "control": "train_mean",
    }


def gauge_variant(
    programs: Mapping[str, Mapping[int, Mapping[str, Any]]],
    bases: Mapping[int, torch.Tensor],
) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[int, torch.Tensor], dict[str, Any]]:
    """Build the deterministic orthogonal-gauge replay of the main program."""

    moved_program = {"main": {}}
    moved_bases = {}
    diagnostics = {}
    for site in (0, 1):
        generator = torch.Generator().manual_seed(FIT_SEED + 100 + site)
        raw = torch.randn(
            compiler.COEFFICIENT_DIM, compiler.COEFFICIENT_DIM,
            generator=generator, dtype=torch.float64,
        )
        rotation = torch.linalg.qr(raw, mode="reduced").Q
        state = programs["main"][site]
        moved, moved_basis = compiler.transport_output_gauge(
            state, bases[site], rotation
        )
        moved_program["main"][site] = cpu_state(moved)
        moved_bases[site] = moved_basis.float().contiguous()
        x = torch.randn(
            32, compiler.D_MODEL,
            generator=torch.Generator().manual_seed(FIT_SEED + 200 + site),
        )
        original_physical = compiler.affine_predict(x, state) @ bases[site].double().T
        moved_physical = compiler.affine_predict(x, moved) @ moved_basis.double().T
        difference = moved_physical - original_physical
        diagnostics[str(site)] = {
            "physical_max_abs_error": float(difference.abs().max()),
            "physical_rms_error": float(difference.square().mean().sqrt()),
            "rank": int(state["rank"]),
            "price_before": compiler.affine_program_price(
                int(state["rank"]), include_basis=True
            ),
            "price_after": compiler.affine_program_price(
                int(state["rank"]), include_basis=True
            ),
        }
        if diagnostics[str(site)]["physical_max_abs_error"] > 3e-5:
            raise RuntimeError(f"gauge physical replay failed at MLP{site}")
    return moved_program, moved_bases, diagnostics


@torch.no_grad()
def capture_labels(
    sa: Any,
    hook: runtime.CompilerCorrectionHook,
    rows: torch.Tensor,
    twall: Mapping[int, Any],
    all_attention: frozenset[int],
    *,
    capture_site: int,
    upstream_states: Mapping[int, str],
    program_name: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, int]]:
    hook.configure(upstream_states, program_name=program_name, capture_site=capture_site)
    with runtime.OriginalMLPCallGuard(sa.H, {capture_site}) as guard:
        for start in range(0, len(rows), 8):
            idx = rows[start:start + 8, :256].to(sa.DEV).contiguous()
            sa.fwd_arm(idx, all_attention, twall, frozenset(range(18)))
    guard.assert_contract(require_allowed_calls=True)
    x, y = hook.captured()
    expected = len(rows) * 64
    if x.shape != (expected, compiler.D_MODEL) or y.shape != (
        expected, compiler.COEFFICIENT_DIM
    ):
        raise RuntimeError(f"unexpected capture shape at MLP{capture_site}: {x.shape} {y.shape}")
    return x, y, dict(guard.counts)


def fit_state(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    validation_x: torch.Tensor,
    validation_y: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state, frontier = compiler.fit_ridge_frontier(
        train_x, train_y, validation_x, validation_y
    )
    return cpu_state(state), frontier


def fit_all_programs(
    sa: Any,
    hook: runtime.CompilerCorrectionHook,
    rows: Mapping[str, torch.Tensor],
    document_ids: Mapping[str, Sequence[str]],
    twall: Mapping[int, Any],
    all_attention: frozenset[int],
) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, Any]]:
    diagnostics: dict[str, Any] = {"captures": {}, "frontiers": {}}
    x0_train, y0_train, calls = capture_labels(
        sa, hook, rows["compiler_fit"], twall, all_attention,
        capture_site=0, upstream_states={}, program_name="main",
    )
    diagnostics["captures"]["site0_fit_original_calls"] = calls
    x0_val, y0_val, calls = capture_labels(
        sa, hook, rows["compiler_validation"], twall, all_attention,
        capture_site=0, upstream_states={}, program_name="main",
    )
    diagnostics["captures"]["site0_validation_original_calls"] = calls
    main0, diagnostics["frontiers"]["main0"] = fit_state(
        x0_train, y0_train, x0_val, y0_val
    )
    mean0 = mean_state(y0_train)
    fit_perm = expand_capture_permutation(document_block_permutation(
        document_ids["compiler_fit"], FIT_SEED
    ))
    val_perm = expand_capture_permutation(document_block_permutation(
        document_ids["compiler_validation"], FIT_SEED + 1
    ))
    shuffle0, diagnostics["frontiers"]["shuffle0"] = fit_state(
        x0_train, y0_train[fit_perm], x0_val, y0_val[val_perm]
    )
    programs: dict[str, dict[int, dict[str, Any]]] = {
        "main": {0: main0}, "mean": {0: mean0}, "shuffle": {0: shuffle0}
    }
    hook.programs = device_programs(programs, sa.DEV)

    for program_index, program_name in enumerate(("main", "mean", "shuffle")):
        x1_train, y1_train, calls = capture_labels(
            sa, hook, rows["compiler_fit"], twall, all_attention,
            capture_site=1, upstream_states={0: "Q"}, program_name=program_name,
        )
        diagnostics["captures"][f"site1_{program_name}_fit_original_calls"] = calls
        x1_val, y1_val, calls = capture_labels(
            sa, hook, rows["compiler_validation"], twall, all_attention,
            capture_site=1, upstream_states={0: "Q"}, program_name=program_name,
        )
        diagnostics["captures"][f"site1_{program_name}_validation_original_calls"] = calls
        if program_name == "main":
            state1, frontier = fit_state(x1_train, y1_train, x1_val, y1_val)
        elif program_name == "mean":
            state1, frontier = mean_state(y1_train), []
        else:
            state1, frontier = fit_state(
                x1_train, y1_train[fit_perm], x1_val, y1_val[val_perm]
            )
        programs[program_name][1] = state1
        diagnostics["frontiers"][f"{program_name}1"] = frontier
        hook.programs = device_programs(programs, sa.DEV)
        diagnostics.setdefault("local_validation", {})[f"{program_name}1"] = (
            compiler.coefficient_metrics(
                compiler.affine_predict(x1_val, state1), y1_val
            )
        )
    diagnostics["local_validation"]["main0"] = compiler.coefficient_metrics(
        compiler.affine_predict(x0_val, main0), y0_val
    )
    diagnostics["local_validation"]["mean0"] = compiler.coefficient_metrics(
        compiler.affine_predict(x0_val, mean0), y0_val
    )
    diagnostics["local_validation"]["shuffle0"] = compiler.coefficient_metrics(
        compiler.affine_predict(x0_val, shuffle0), y0_val
    )
    return programs, diagnostics


def artifact_payload(
    programs: Mapping[str, Mapping[int, Mapping[str, Any]]],
    diagnostics: Mapping[str, Any],
    bases_payload: Mapping[str, Any],
    realization_hash: str,
    source_commit: str,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    pricing = {
        name: {
            str(site): compiler.affine_program_price(
                max(int(state.get("rank", 0)), 8), include_basis=True
            )
            for site, state in sites.items()
        }
        for name, sites in programs.items()
    }
    return {
        "schema_version": 1,
        "status": "frozen_before_final_scoring",
        "authority": "isolated_compiler_experiment",
        "authorized_for_scored_experiments": False,
        "authorized_for_training": False,
        "preregistration_sha256": PREREG_SHA256,
        "rows_receipt_sha256": ROWS_RECEIPT_SHA256,
        "ship_realization_sha256": realization_hash,
        "basis_artifact_sha256": PINS[V3_BASIS],
        "basis_tensor_sha256": {
            str(site): bases_payload["sites"][site]["basis_sha256"] for site in (0, 1)
        },
        "programs": programs,
        "diagnostics": diagnostics,
        "pricing": pricing,
        "source_commit": source_commit,
        "source_hashes": dict(source_hashes),
        "forbidden_artifact_contents": [
            "original MLP weights", "cached labels", "clean activations",
            "token IDs", "position IDs", "row-index lookup tables",
        ],
    }


def validate_artifact(path: Path = ARTIFACT) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("status") != "frozen_before_final_scoring":
        raise RuntimeError("compiler artifact is not frozen")
    if payload.get("preregistration_sha256") != PREREG_SHA256:
        raise RuntimeError("compiler artifact preregistration binding changed")
    if set(payload.get("programs", {})) != {"main", "mean", "shuffle"}:
        raise RuntimeError("compiler artifact program set changed")
    for name, sites in payload["programs"].items():
        if set(sites) != {0, 1}:
            raise RuntimeError(f"compiler artifact sites changed for {name}")
        for site, state in sites.items():
            expected = {
                "mean": (compiler.D_MODEL,), "scale": (compiler.D_MODEL,),
                "bias": (compiler.COEFFICIENT_DIM,),
            }
            for key, shape in expected.items():
                value = state[key]
                if not torch.is_tensor(value) or tuple(value.shape) != shape:
                    raise RuntimeError(f"invalid {name}/MLP{site}/{key}")
            if state["left"].shape[0] != compiler.D_MODEL:
                raise RuntimeError("compiler left factor shape changed")
            if state["right"].shape != (
                state["left"].shape[1], compiler.COEFFICIENT_DIM
            ):
                raise RuntimeError("compiler right factor shape changed")
    receipt = json.loads(ARTIFACT_RECEIPT.read_text())
    if receipt.get("artifact_sha256") != file_sha256(path):
        raise RuntimeError("compiler artifact receipt binding changed")
    if receipt.get("status") != "frozen_before_final_scoring":
        raise RuntimeError("compiler artifact receipt is not frozen")
    return payload, receipt


def score_arm(
    sa: Any,
    hook: runtime.CompilerCorrectionHook,
    rows: torch.Tensor,
    rare_vocab: torch.Tensor,
    twall: Mapping[int, Any],
    all_attention: frozenset[int],
    arm: tuple[str, str, str],
) -> tuple[dict[str, Any], dict[int, int]]:
    states = {0: arm[0], 1: arm[1], 2: arm[2]}
    hook.configure(states, program_name="main")
    allowed = {
        site for site, state in states.items()
        if (site in (0, 1) and state == "O") or (site == 2 and state == "E")
    }
    with runtime.OriginalMLPCallGuard(sa.H, allowed) as guard:
        scored = sa._score_content_rows(
            rows, twall, all_attention, frozenset(range(18)),
            rare_vocab=rare_vocab, retain_row_ce=True,
        )
    guard.assert_contract(require_allowed_calls=bool(allowed))
    return scored, dict(guard.counts)


def score_control(
    sa: Any,
    hook: runtime.CompilerCorrectionHook,
    rows: torch.Tensor,
    rare_vocab: torch.Tensor,
    twall: Mapping[int, Any],
    all_attention: frozenset[int],
    program_name: str,
) -> tuple[dict[str, Any], dict[int, int]]:
    hook.configure({0: "Q", 1: "Q", 2: "N"}, program_name=program_name)
    with runtime.OriginalMLPCallGuard(sa.H, set()) as guard:
        scored = sa._score_content_rows(
            rows, twall, all_attention, frozenset(range(18)),
            rare_vocab=rare_vocab, retain_row_ce=True,
        )
    guard.assert_contract(require_allowed_calls=False)
    return scored, dict(guard.counts)


def run_claimed(protected_before: Mapping[str, str | None]) -> None:
    verify_pins()
    source_hashes = verify_source_closure()
    old_receipt, old_frozen_rows = old_rows.validate_receipt()
    compiler_row_receipt, rows_full = fresh_rows.load_and_validate()
    rows = {role: tensor[:, :257].contiguous() for role, tensor in rows_full.items()}
    document_ids = {
        role: [record["document_id"] for record in
               compiler_row_receipt["document_provenance"]["sets"][role]]
        for role in fresh_rows.ROLE_SPECS
    }
    code_rows, _ = code_oracle.load_frozen_corpus()
    frozen.validate_frozen_ship_pair(old_receipt)
    bases_payload, _ = v3.validate_basis_pair()
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "status": "running_affine_compiler_v1",
        "authority": "isolated_compiler_experiment",
        "authorized_for_scored_experiments": False,
        "authorized_for_training": False,
        "preregistration_sha256": PREREG_SHA256,
        "rows_receipt_sha256": ROWS_RECEIPT_SHA256,
        "source_commit": source_commit,
        "source_hashes": source_hashes,
        "protected_before": dict(protected_before),
    }
    write_json_atomic(manifest, MANIFEST)

    torch.manual_seed(exact_runner.SHIP_SEED)
    torch.cuda.manual_seed_all(exact_runner.SHIP_SEED)
    sys.path.insert(0, str(BQ))
    import ship_error_attrib as sa  # noqa: PLC0415

    start_time = time.time()

    def callback(twall: dict, all_attention: frozenset[int], _: float) -> None:
        realization_hash, frozen_manifest = frozen.restore_ship_realization(
            sa, twall, all_attention, old_receipt, code_rows
        )
        if realization_hash != SHIP_HASH:
            raise RuntimeError("frozen ship realization changed")
        component_before = exact_runner.component_tree_sha256(sa, twall, all_attention)
        if component_before != realization_hash:
            raise RuntimeError("restored component tree differs from frozen ship")
        prior_hook = sa.add_oracle_correction
        bases_device = {
            site: bases_payload["sites"][site]["basis"].to(sa.DEV).float()
            for site in (0, 1)
        }
        hook = runtime.CompilerCorrectionHook(bases_device, {})
        sa.add_oracle_correction = hook
        try:
            programs, diagnostics = fit_all_programs(
                sa, hook, rows, document_ids, twall, all_attention
            )
            if ARTIFACT.exists() or ARTIFACT_RECEIPT.exists():
                raise RuntimeError("refusing to overwrite compiler program freeze")
            payload = artifact_payload(
                programs, diagnostics, bases_payload, realization_hash,
                source_commit, source_hashes,
            )
            write_torch_atomic(payload, ARTIFACT)
            receipt = {
                "schema_version": 1,
                "status": "frozen_before_final_scoring",
                "authority": "isolated_compiler_experiment",
                "authorized_for_scored_experiments": False,
                "authorized_for_training": False,
                "preregistration_sha256": PREREG_SHA256,
                "rows_receipt_sha256": ROWS_RECEIPT_SHA256,
                "ship_realization_sha256": realization_hash,
                "artifact_path": str(ARTIFACT.resolve()),
                "artifact_sha256": file_sha256(ARTIFACT),
                "artifact_bytes": ARTIFACT.stat().st_size,
                "freeze_rule": "Written and validated before any compiler_final arm.",
                "source_commit": source_commit,
                "source_hashes": source_hashes,
            }
            write_json_atomic(receipt, ARTIFACT_RECEIPT)
            payload, receipt = validate_artifact()
            hook.programs = device_programs(payload["programs"], sa.DEV)

            final_rows = rows["compiler_final"]
            rare_vocab = sa._token_masks(final_rows)
            evaluations: dict[str, Any] = {}
            call_counters: dict[str, Any] = {}
            row_ce_by_arm = {}
            for arm in compiler.ARM_STATES:
                scored, counters = score_arm(
                    sa, hook, final_rows, rare_vocab, twall, all_attention, arm
                )
                name = compiler.arm_name(arm)
                evaluations[name] = scored
                call_counters[name] = counters
                row_ce_by_arm[arm] = scored["row_global_ce"]
                print(f"affine compiler final arm={name} done", flush=True)
            controls = {}
            for name in ("mean", "shuffle"):
                scored, counters = score_control(
                    sa, hook, final_rows, rare_vocab, twall, all_attention, name
                )
                controls[name] = scored
                call_counters[f"control_{name}"] = counters
                print(f"affine compiler final control={name} done", flush=True)
            gauge_programs, gauge_bases, gauge_diagnostics = gauge_variant(
                payload["programs"], {
                    site: bases_payload["sites"][site]["basis"] for site in (0, 1)
                },
            )
            gauge_hook = runtime.CompilerCorrectionHook(
                {site: basis.to(sa.DEV) for site, basis in gauge_bases.items()},
                device_programs(gauge_programs, sa.DEV),
            )
            sa.add_oracle_correction = gauge_hook
            gauge_score, gauge_calls = score_control(
                sa, gauge_hook, final_rows, rare_vocab, twall, all_attention, "main"
            )
            sa.add_oracle_correction = hook
            call_counters["gauge_QQN"] = gauge_calls
            gauge_row_difference = (
                torch.tensor(gauge_score["row_global_ce"], dtype=torch.float64)
                - torch.tensor(evaluations["QQN"]["row_global_ce"], dtype=torch.float64)
            )
            gauge_diagnostics["final_ce_replay"] = {
                "max_abs_row_ce_difference": float(gauge_row_difference.abs().max()),
                "mean_abs_row_ce_difference": float(gauge_row_difference.abs().mean()),
                "tolerance": 2e-5,
            }
            gauge_pass = (
                gauge_diagnostics["final_ce_replay"]["max_abs_row_ce_difference"] <= 2e-5
            )
            if not gauge_pass:
                raise RuntimeError(
                    f"gauge final CE replay failed: {gauge_diagnostics['final_ce_replay']}"
                )
            analysis = compiler.compiler_lattice_analysis(
                row_ce_by_arm, document_ids["compiler_final"],
                mean_control_rows=controls["mean"]["row_global_ce"],
                shuffle_control_rows=controls["shuffle"]["row_global_ce"],
                draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED,
            )
            baseline = evaluations["NNN"]["ce"]
            joint = evaluations["QQN"]["ce"]
            collateral = {
                cell: joint[cell] - baseline[cell] for cell in ("copy", "novel_freq")
            }
            collateral_pass = all(value <= 0.01 for value in collateral.values())
            hook.configure({})
            with runtime.OriginalMLPCallGuard(sa.H, set()) as guard:
                replay = sa._score_content_rows(
                    final_rows, twall, all_attention, frozenset(range(18)),
                    rare_vocab=rare_vocab, retain_row_ce=True,
                )
            guard.assert_contract(require_allowed_calls=False)
            difference = torch.tensor(replay["row_global_ce"]) - torch.tensor(
                evaluations["NNN"]["row_global_ce"]
            )
            replay_gate = {
                "max_abs_row_ce_difference": float(difference.abs().max()),
                "mean_abs_row_ce_difference": float(difference.abs().mean()),
            }
            if replay_gate["max_abs_row_ce_difference"] != 0.0:
                raise RuntimeError(f"compiler baseline replay changed: {replay_gate}")
            component_after = exact_runner.component_tree_sha256(sa, twall, all_attention)
            if component_after != component_before:
                raise RuntimeError("component tree changed during compiler run")
            statistical = analysis["decisions"]
            decisions = {
                **statistical,
                "collateral": collateral_pass,
                "gauge_replay": gauge_pass,
                "integrity": gauge_pass,
                "all_registered_gates": (
                    statistical["all_statistical_gates"] and collateral_pass and gauge_pass
                ),
            }
            result = {
                "schema_version": 1,
                "status": "completed_payload_awaiting_authority_receipt",
                "authority": "isolated_compiler_experiment",
                "authorized_for_scored_experiments": False,
                "authorized_for_training": False,
                "interpretation_guardrail": json.loads(PREREG.read_text())[
                    "scope_guardrail"
                ],
                "preregistration_sha256": PREREG_SHA256,
                "rows_receipt_sha256": ROWS_RECEIPT_SHA256,
                "program_artifact_sha256": receipt["artifact_sha256"],
                "program_receipt_sha256": file_sha256(ARTIFACT_RECEIPT),
                "ship_realization_sha256": realization_hash,
                "component_tree_before_sha256": component_before,
                "component_tree_after_sha256": component_after,
                "component_tree_unchanged": True,
                "baseline_replay": replay_gate,
                "config": {
                    "arms": [compiler.arm_name(arm) for arm in compiler.ARM_STATES],
                    "controls": ["mean", "shuffle"],
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_unit": "FineWeb source document cluster",
                    "final_rows": len(final_rows),
                    "final_unique_documents": len(set(document_ids["compiler_final"])),
                },
                "evaluations": evaluations,
                "controls": controls,
                "gauge_replay": {
                    "evaluation": gauge_score,
                    "diagnostics": gauge_diagnostics,
                },
                "original_call_counters": call_counters,
                "analysis": analysis,
                "collateral_worsening": collateral,
                "decisions": decisions,
                "program_summary": {
                    name: {
                        str(site): {
                            "rank": state.get("rank"),
                            "lambda": state.get("lambda"),
                            "price": payload["pricing"][name][str(site)],
                        }
                        for site, state in sites.items()
                    }
                    for name, sites in payload["programs"].items()
                },
                "frozen_ship_artifact_sha256": frozen_manifest["artifact_sha256"],
                "source_commit": source_commit,
                "source_hashes": source_hashes,
                "runtime_s": round(time.time() - start_time, 1),
            }
            write_json_atomic(result, RESULT)
            manifest.update({
                "status": "completed_payload_awaiting_authority_receipt",
                "program_artifact_sha256": receipt["artifact_sha256"],
                "program_receipt_sha256": file_sha256(ARTIFACT_RECEIPT),
                "result_sha256": file_sha256(RESULT),
                "ship_realization_sha256": realization_hash,
                "component_tree_sha256": component_after,
                "runtime_s": result["runtime_s"],
            })
            write_json_atomic(manifest, MANIFEST)
        finally:
            hook.clear()
            sa.add_oracle_correction = prior_hook
            exact_runner.require_inert_correction_state(sa)

    sa.run_oracle_content_screen = callback
    sa.main(oracle_content_screen=True)


def mark_failed(error: BaseException, protected_after: Mapping[str, str | None]) -> None:
    if AUTHORITY.exists():
        raise RuntimeError("refusing to invalidate an existing compiler authority") from error
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest.update({
        "schema_version": 1,
        "status": "failed_affine_compiler_v1",
        "authorized_for_scored_experiments": False,
        "authorized_for_training": False,
        "failure_type": type(error).__name__,
        "failure_message": str(error),
        "protected_after": dict(protected_after),
        "recovery": "Preserve artifacts and use a new versioned namespace for retry.",
    })
    write_json_atomic(manifest, MANIFEST)


def finalize(protected_before: Mapping[str, str | None], protected_after: Mapping[str, str | None]) -> None:
    if protected_before != protected_after:
        raise RuntimeError("compiler run changed protected prior artifacts")
    if AUTHORITY.exists():
        raise RuntimeError("refusing to overwrite compiler authority")
    payload, receipt = validate_artifact()
    result = json.loads(RESULT.read_text())
    manifest = json.loads(MANIFEST.read_text())
    if result.get("status") != "completed_payload_awaiting_authority_receipt":
        raise RuntimeError("compiler result is not ready for authority")
    if result.get("component_tree_unchanged") is not True:
        raise RuntimeError("compiler component-tree gate failed")
    if result.get("baseline_replay", {}).get("max_abs_row_ce_difference") != 0.0:
        raise RuntimeError("compiler baseline replay gate failed")
    authority = {
        "schema_version": 1,
        "status": "completed_affine_compiler_v1",
        "authority": "isolated_compiler_experiment",
        "authorized_for_scored_experiments": True,
        "authorized_for_training": False,
        "training_license_sites": [],
        "preregistration_sha256": PREREG_SHA256,
        "rows_receipt_sha256": ROWS_RECEIPT_SHA256,
        "ship_realization_sha256": result["ship_realization_sha256"],
        "result_path": str(RESULT.resolve()),
        "result_sha256": file_sha256(RESULT),
        "manifest_path": str(MANIFEST.resolve()),
        "manifest_sha256": file_sha256(MANIFEST),
        "program_artifact_path": str(ARTIFACT.resolve()),
        "program_artifact_sha256": receipt["artifact_sha256"],
        "program_receipt_path": str(ARTIFACT_RECEIPT.resolve()),
        "program_receipt_sha256": file_sha256(ARTIFACT_RECEIPT),
        "source_commit": manifest["source_commit"],
        "source_hashes": manifest["source_hashes"],
        "protected_paths_unchanged": True,
        "authorization_rule": "This last-written receipt alone authorizes the bound scored payload; a failed statistical decision remains an authoritative failure.",
    }
    write_json_atomic(authority, AUTHORITY)


def main() -> None:
    existing = [str(path) for path in OUTPUTS if path.exists()]
    if existing:
        raise RuntimeError(f"refusing to overwrite affine compiler outputs: {existing}")
    try:
        LOCK.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"affine compiler launch already claimed: {LOCK}") from error
    before = protected_snapshot()
    run_error: BaseException | None = None
    try:
        run_claimed(before)
    except BaseException as error:
        run_error = error
    after = protected_snapshot()
    try:
        if run_error is not None:
            mark_failed(run_error, after)
            raise run_error
        try:
            finalize(before, after)
        except BaseException as error:
            mark_failed(error, after)
            raise
    finally:
        LOCK.rmdir()


if __name__ == "__main__":
    main()
