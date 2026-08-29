#!/usr/bin/env python3
"""Source-closed discovery runner for the frozen real shared-output RRR v1.

The module is import-pure.  Production execution first publishes an immutable
no-outcome authority, then loads rows/checkpoint tensors, fits the registered 24
programs, scores all three discovery roles, and publishes result then receipt.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language")
HERE = ROOT / "basis_aligned" / "polynomial_causal"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bilin18_observed_model_facade as facade
import simultaneous_shared_output_rrr as core


PREREG = HERE / "SHARED_OUTPUT_RRR_REAL_V1_PREREGISTRATION.md"
RUNNER = HERE / "run_shared_output_rrr_real_v1.py"
TEST = HERE / "test_run_shared_output_rrr_real_v1.py"
CORE = HERE / "simultaneous_shared_output_rrr.py"
CORE_TEST = HERE / "test_simultaneous_shared_output_rrr.py"
FACADE = HERE / "bilin18_observed_model_facade.py"
FACADE_TEST = HERE / "test_bilin18_observed_model_facade.py"
TT_MODEL = ROOT / "jacclust" / "tt_model.py"
TT_INIT = ROOT / "jacclust" / "__init__.py"
FRONTIER_SOURCE = ROOT / "basis_aligned" / "bilinear_quotient" / "ops" / "frontier_at_map512.py"
FRONTIER_RESULT = ROOT / "basis_aligned" / "bilinear_quotient" / "ops" / "frontier_at_map512_results.json"

AUTHORITY = HERE / "shared_output_rrr_real_v1_authority.json"
RESULTS = HERE / "shared_output_rrr_real_v1_results.json"
FAILURE = HERE / "shared_output_rrr_real_v1_failure.json"
RECEIPT = HERE / "shared_output_rrr_real_v1_receipt.json"
LOCK = Path("/workspace/runs/.shared_output_rrr_real_v1.lock")

ROW_ROOT = ROOT / "basis_aligned" / "bilinear_quotient" / ".rowcache"
FIT_PATH = ROW_ROOT / "fineweb_n96_skip80.pt"
ROLE_PATHS = {
    "skip7000": ROW_ROOT / "fineweb_n192_skip7000.pt",
    "skip11000": ROW_ROOT / "fineweb_n192_skip11000.pt",
    "skip1200": ROW_ROOT / "fineweb_n96_skip1200.pt",
}
FILE_PINS = {
    str(FIT_PATH.relative_to(ROOT)): "94bc1fb3e3a6a061541e555295e0af8c50ae6068fdff84e95a69c25844091eda",
    str(ROLE_PATHS["skip7000"].relative_to(ROOT)): "d66c1ee7807bc6b9bd7d0ddba5cdd7e3bc64926b00320a10675a2f817d67128c",
    str(ROLE_PATHS["skip11000"].relative_to(ROOT)): "b1564bfd071418f401a816cb01e3d26b082a3e73ba858838f1c83c250db4d868",
    str(ROLE_PATHS["skip1200"].relative_to(ROOT)): "21707551f35d13818c10ac59e12e9445ef076d0522371fe779691bfab719d34f",
    str(FRONTIER_SOURCE.relative_to(ROOT)): "893fdc9dca7859cb01d54b4ac36d48bfb95dcd98eca0769a83fa86bbe52fe8a7",
    str(FRONTIER_RESULT.relative_to(ROOT)): "83039fba204e1339fe330e5e026a13cbf207c668751d4d480711675a6a5a890a",
}
SOURCE_PATHS = tuple(str(path.relative_to(ROOT)) for path in (
    PREREG, RUNNER, TEST, CORE, CORE_TEST, FACADE, FACADE_TEST, TT_MODEL, TT_INIT,
))

D = 1152
N_SITES = 36
VOCAB = 50_257
CONTEXT = 256
SCORE_START = 64
SCORED_PER_ROW = 192
COVERAGE = 5_419
RIDGE_SCALE = 0.01
RIDGE = RIDGE_SCALE * COVERAGE / D
RANKS = (64, 128, 256, 512)
DIRECT_GLOBAL_RANK = 494
DIRECT_TYPED_RANK = 481
ROLE_ROWS = {"skip7000": 192, "skip11000": 192, "skip1200": 96}
FIT_CAPTURE_BATCH = 256
EVAL_BATCH = 4
FIT_OUTER_CALLS = math.ceil(COVERAGE / FIT_CAPTURE_BATCH)
EVAL_CALLS_PER_ARM = sum(value // EVAL_BATCH for value in ROLE_ROWS.values())
MAX_WALL_SECONDS = 75 * 60
MAX_ALLOCATED_CUDA_BYTES = 16 * 1024 ** 3
COMMON_TABLE_FLOATS = N_SITES * COVERAGE * D

# The historical source records all MLPs then all attentions.  Preserve it exactly;
# execution maps physical layer events back to these indices.
SITE_ORDER = tuple([("mlp", site) for site in range(18)] + [
    ("attn", site) for site in range(18)
])
SITE_TO_INDEX = {site: index for index, site in enumerate(SITE_ORDER)}

FRONTIER_ANCHORS = {
    "legacy_svd_q64": {
        "skip7000": 6.01167, "skip11000": 5.98477, "skip1200": 6.00165,
    },
    "legacy_svd_q512": {
        "skip7000": 5.96702, "skip11000": 5.93645, "skip1200": 5.96095,
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()


def source_closure() -> dict[str, Any]:
    commit = git("rev-parse", "HEAD")
    hashes: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if completed.returncode:
            raise RuntimeError(f"shared-RRR source is not committed: {relative}")
        digest = hashlib.sha256(completed.stdout).hexdigest()
        if file_sha256(ROOT / relative) != digest:
            raise RuntimeError(f"shared-RRR live source differs from commit: {relative}")
        hashes[relative] = digest
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=ROOT, check=True,
    )
    body = {"commit": commit, "paths": hashes}
    return {**body, "sha256": logical_sha256(body)}


def verify_source_closure(value: Mapping[str, Any]) -> None:
    body = {"commit": value.get("commit"), "paths": value.get("paths")}
    if set(value) != {"commit", "paths", "sha256"} or set(value["paths"]) != set(
        SOURCE_PATHS
    ) or value["sha256"] != logical_sha256(body):
        raise RuntimeError("shared-RRR source closure schema changed")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", str(value["commit"]), "origin/main"],
        cwd=ROOT, check=True,
    )
    for relative, digest in value["paths"].items():
        if file_sha256(ROOT / relative) != digest:
            raise RuntimeError(f"shared-RRR source drift: {relative}")


def input_bindings() -> dict[str, str]:
    observed = {relative: file_sha256(ROOT / relative) for relative in FILE_PINS}
    if observed != FILE_PINS:
        raise RuntimeError("shared-RRR pinned input bytes changed")
    return observed


def validate_frontier_anchors(path: Path = FRONTIER_RESULT) -> None:
    value = json.loads(path.read_text())
    results = value.get("results", {})
    for arm, role_values in FRONTIER_ANCHORS.items():
        frontier_key = "m64_full" if arm.endswith("q64") else "m512_full"
        for role, expected in role_values.items():
            observed = results.get(role, {}).get(frontier_key, {}).get("all")
            if observed != expected:
                raise RuntimeError("shared-RRR frontier known-answer bytes changed")


def arm_descriptors() -> tuple[dict[str, Any], ...]:
    arms: list[dict[str, Any]] = []
    for rank in RANKS:
        arms.extend((
            {"name": f"global_q{rank}", "family": "global", "rank": rank},
            {"name": f"typed_q{rank}", "family": "typed", "rank": rank},
            {"name": f"independent_q{rank}", "family": "independent", "rank": rank},
            {"name": f"price_global_q{rank}", "family": "price_independent",
             "rank": rank, "budget_bases": 1},
            {"name": f"price_typed_q{rank}", "family": "price_independent",
             "rank": rank, "budget_bases": 2},
        ))
    arms.extend((
        {"name": "global_q494", "family": "global", "rank": 494},
        {"name": "typed_q481", "family": "typed", "rank": 481},
        {"name": "legacy_svd_q64", "family": "legacy_svd", "rank": 64,
         "promotive": False},
        {"name": "legacy_svd_q512", "family": "legacy_svd", "rank": 512,
         "promotive": False},
    ))
    if len(arms) != 24 or len({arm["name"] for arm in arms}) != 24:
        raise RuntimeError("registered shared-RRR arm bank changed")
    return tuple(arms)


def expected_call_schedule() -> dict[str, Any]:
    return {
        "fit_native_outer": FIT_OUTER_CALLS,
        "native_reference_outer": EVAL_CALLS_PER_ARM,
        "compiled_outer_per_arm": EVAL_CALLS_PER_ARM,
        "compiled_arm_count": 24,
        "compiled_outer_total": 24 * EVAL_CALLS_PER_ARM,
        "outer_total": FIT_OUTER_CALLS + 25 * EVAL_CALLS_PER_ARM,
        "native_component_calls_per_kind": 18 * (FIT_OUTER_CALLS + EVAL_CALLS_PER_ARM),
        "compiled_component_calls": 24 * EVAL_CALLS_PER_ARM * N_SITES,
        "optimizer_calls": 0,
        "backward_calls": 0,
    }


def checkpoint_binding(*, verify_hash: bool = True) -> dict[str, Any]:
    return asdict(facade.validate_snapshot(verify_weights_sha256=verify_hash))


def authority_payload(
    source: Mapping[str, Any], inputs: Mapping[str, str], checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = {
        "fit_path": str(FIT_PATH), "roles": {key: str(value) for key, value in ROLE_PATHS.items()},
        "row_truncation": 257, "score_positions": [64, 255],
        "covered_types": COVERAGE, "dimension": D, "site_order": [list(x) for x in SITE_ORDER],
        "ranks": list(RANKS), "ridge_scale": RIDGE_SCALE, "effective_ridge": RIDGE,
        "arms": list(arm_descriptors()), "call_schedule": expected_call_schedule(),
        "resource_ceiling": {"wall_seconds": MAX_WALL_SECONDS,
                             "peak_allocated_cuda_bytes": MAX_ALLOCATED_CUDA_BYTES},
        "authority_scope": "discovery_only_no_validation_final_or_semantic_coordinates",
    }
    body = {
        "schema": "shared_output_rrr_real_v1_authority",
        "status": "frozen_before_any_row_tensor_or_model_load",
        "source_closure": dict(source), "input_file_sha256s": dict(inputs),
        "checkpoint": dict(checkpoint), "protocol": protocol,
        "outputs": {"results": str(RESULTS), "failure": str(FAILURE), "receipt": str(RECEIPT)},
    }
    return {**body, "authority_sha256": logical_sha256(body)}


@dataclass(frozen=True)
class RunClaim:
    fd: int
    inode: int
    nonce: str


def acquire_lock(path: Path = LOCK) -> RunClaim:
    path.parent.mkdir(parents=True, exist_ok=True)
    nonce = secrets.token_hex(16)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    try:
        os.write(fd, json.dumps({"pid": os.getpid(), "nonce": nonce}).encode())
        os.fsync(fd)
        inode = os.fstat(fd).st_ino
    except BaseException:
        os.close(fd)
        path.unlink(missing_ok=True)
        raise
    return RunClaim(fd=fd, inode=inode, nonce=nonce)


def require_claim(claim: RunClaim, path: Path = LOCK) -> None:
    if os.fstat(claim.fd).st_ino != claim.inode or path.stat().st_ino != claim.inode:
        raise RuntimeError("shared-RRR run lock was replaced")
    payload = json.loads(path.read_text())
    if payload != {"pid": os.getpid(), "nonce": claim.nonce}:
        raise RuntimeError("shared-RRR run lock claim changed")


def release_lock(claim: RunClaim, path: Path = LOCK) -> None:
    try:
        require_claim(claim, path)
        path.unlink()
    finally:
        os.close(claim.fd)


def atomic_create_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def output_namespace() -> tuple[Path, ...]:
    return AUTHORITY, RESULTS, FAILURE, RECEIPT


def require_pristine_namespace() -> None:
    spent = [str(path) for path in (*output_namespace(), LOCK) if path.exists()]
    if spent:
        raise RuntimeError(f"shared-RRR v1 namespace is spent: {spent}")


def require_published_authority(value: Mapping[str, Any]) -> None:
    if not AUTHORITY.is_file() or json.loads(AUTHORITY.read_text()) != dict(value):
        raise RuntimeError("shared-RRR published authority changed")
    body = {key: item for key, item in value.items() if key != "authority_sha256"}
    if value.get("authority_sha256") != logical_sha256(body):
        raise RuntimeError("shared-RRR authority self-hash changed")


def verify_frozen_inputs(value: Mapping[str, Any], *, verify_checkpoint_hash: bool) -> None:
    verify_source_closure(value["source_closure"])
    if input_bindings() != value["input_file_sha256s"]:
        raise RuntimeError("shared-RRR input binding changed")
    if checkpoint_binding(verify_hash=verify_checkpoint_hash) != value["checkpoint"]:
        raise RuntimeError("shared-RRR checkpoint binding changed")


class PhysicalCallLedger:
    """Exact logical and physical call census for the registered phases."""

    def __init__(self) -> None:
        self.outer = Counter()
        self.site = defaultdict(Counter)
        self.compiled_outer = Counter()
        self.compiled_site = defaultdict(Counter)

    def record_native_site(self, phase: str, kind: str, site: int) -> None:
        if phase not in {"fit", "native_reference"} or kind not in {"attn", "mlp"}:
            raise RuntimeError("unexpected native shared-RRR call")
        self.site[(phase, kind)][site] += 1

    def record_native_outer(self, phase: str) -> None:
        self.outer[phase] += 1

    def record_compiled_site(self, arm: str, kind: str, site: int) -> None:
        if arm not in {value["name"] for value in arm_descriptors()}:
            raise RuntimeError("unknown compiled shared-RRR arm")
        self.compiled_site[(arm, kind)][site] += 1

    def record_compiled_outer(self, arm: str) -> None:
        self.compiled_outer[arm] += 1

    def receipt(self) -> dict[str, Any]:
        for phase, expected in (("fit", FIT_OUTER_CALLS),
                                ("native_reference", EVAL_CALLS_PER_ARM)):
            if self.outer[phase] != expected:
                raise RuntimeError(f"shared-RRR {phase} outer census changed")
            for kind in ("attn", "mlp"):
                if dict(self.site[(phase, kind)]) != {site: expected for site in range(18)}:
                    raise RuntimeError(f"shared-RRR {phase} {kind} census changed")
        arm_names = [arm["name"] for arm in arm_descriptors()]
        if dict(self.compiled_outer) != {arm: EVAL_CALLS_PER_ARM for arm in arm_names}:
            raise RuntimeError("shared-RRR compiled outer census changed")
        for arm in arm_names:
            for kind in ("attn", "mlp"):
                if dict(self.compiled_site[(arm, kind)]) != {
                    site: EVAL_CALLS_PER_ARM for site in range(18)
                }:
                    raise RuntimeError(f"shared-RRR compiled {arm}/{kind} census changed")
        outer_returns = FIT_OUTER_CALLS + EVAL_CALLS_PER_ARM + len(arm_names) * EVAL_CALLS_PER_ARM
        return {
            "schema": "shared_output_rrr_real_v1_call_ledger",
            "registered": expected_call_schedule(),
            "native_outer": dict(self.outer),
            "native_sites": {
                f"{phase}:{kind}": {str(k): v for k, v in sorted(counts.items())}
                for (phase, kind), counts in sorted(self.site.items())
            },
            "compiled_outer": dict(self.compiled_outer),
            "compiled_sites": {
                f"{arm}:{kind}": {str(k): v for k, v in sorted(counts.items())}
                for (arm, kind), counts in sorted(self.compiled_site.items())
            },
            "outer_returns": outer_returns, "logit_returns": outer_returns,
            "optimizer_calls": 0, "backward_calls": 0,
        }


def _rows_tensor(raw: Any) -> torch.Tensor:
    if isinstance(raw, dict):
        if set(raw) != {"rows"}:
            raise RuntimeError("shared-RRR row wrapper has unexpected keys")
        raw = raw["rows"]
    if not torch.is_tensor(raw) or raw.dtype != torch.long or raw.ndim != 2 or raw.shape[1] < 257:
        raise RuntimeError("shared-RRR rows have invalid schema")
    return raw[:, :257].contiguous()


def load_rows_after_authority(path: Path, expected_file_sha256: str, expected_rows: int,
                              frozen: Mapping[str, Any]) -> torch.Tensor:
    require_published_authority(frozen)
    before = file_sha256(path)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    rows = _rows_tensor(raw)
    if before != expected_file_sha256 or file_sha256(path) != before or rows.shape != (
        expected_rows, 257
    ):
        raise RuntimeError("shared-RRR row file changed during load")
    return rows


def native_dispatchers(ledger: PhysicalCallLedger, phase: str):
    def attention(event: facade.AttentionEvent) -> tuple[torch.Tensor, torch.Tensor]:
        ledger.record_native_site(phase, "attn", event.site)
        return event.block.attn(event.state, event.first_value)

    def mlp(event: facade.EarlyMLPEvent) -> torch.Tensor:
        ledger.record_native_site(phase, "mlp", event.site)
        return event.block.mlp(event.state)

    return attention, mlp


def capture_native_tables(model: torch.nn.Module, covered: torch.Tensor,
                          ledger: PhysicalCallLedger, device: str) -> torch.Tensor:
    stores: list[list[torch.Tensor]] = [[] for _ in range(N_SITES)]
    attention, mlp = native_dispatchers(ledger, "fit")
    for start in range(0, covered.numel(), FIT_CAPTURE_BATCH):
        tokens = covered[start:start + FIT_CAPTURE_BATCH].to(device).unsqueeze(1)
        captured: dict[int, torch.Tensor] = {}

        def capture_attention(event: facade.AttentionEvent):
            write, value = attention(event)
            captured[SITE_TO_INDEX[("attn", event.site)]] = write[:, 0].detach().float().cpu()
            return write, value

        def capture_mlp(event: facade.EarlyMLPEvent):
            write = mlp(event)
            captured[SITE_TO_INDEX[("mlp", event.site)]] = write[:, 0].detach().float().cpu()
            return write

        facade.forward_with_dispatch(
            model, tokens, capture_attention, capture_mlp, require_production=False,
        )
        ledger.record_native_outer("fit")
        if set(captured) != set(range(N_SITES)):
            raise RuntimeError("native table capture missed a site")
        for site in range(N_SITES):
            stores[site].append(captured[site])
    table = torch.stack([torch.cat(chunks) for chunks in stores])
    if table.shape != (N_SITES, COVERAGE, D) or table.dtype != torch.float32 or not bool(
        torch.isfinite(table).all()
    ):
        raise RuntimeError("shared-RRR native table has invalid schema")
    return table


@dataclass
class SpectralState:
    gram: torch.Tensor
    crosses: tuple[torch.Tensor, ...]
    y2: tuple[float, ...]
    solved: tuple[torch.Tensor, ...]
    independent_values: tuple[torch.Tensor, ...]
    independent_vectors: tuple[torch.Tensor, ...]
    global_values: torch.Tensor
    global_vectors: torch.Tensor
    typed_values: dict[str, torch.Tensor]
    typed_vectors: dict[str, torch.Tensor]
    legacy_svd: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...] | None = None


def _descending_eigh(merit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    merit = 0.5 * (merit + merit.T)
    values, vectors = torch.linalg.eigh(merit)
    values, vectors = values.flip(0).contiguous(), vectors.flip(1).contiguous()
    tolerance = 1e-10 * max(float(values.abs().max()), 1.0)
    if float(values.min()) < -tolerance:
        raise RuntimeError("shared-RRR merit has a materially negative eigenvalue")
    return values.clamp_min(0), vectors


def build_spectral_state(embedding: torch.Tensor, table: torch.Tensor) -> SpectralState:
    x = embedding.detach().cpu().float().double()
    y = table.double()
    if x.shape != (COVERAGE, D) or y.shape != (N_SITES, COVERAGE, D):
        raise RuntimeError("shared-RRR fit tensors have wrong shapes")
    gram = x.T @ x
    crosses = tuple(x.T @ y[site] for site in range(N_SITES))
    y2 = tuple(float(y[site].square().sum()) for site in range(N_SITES))
    chol = torch.linalg.cholesky(0.5 * (gram + gram.T) + RIDGE * torch.eye(D, dtype=torch.float64))
    solved = tuple(torch.cholesky_solve(cross, chol) for cross in crosses)
    merits = tuple(0.5 * (cross.T @ solution + solution.T @ cross)
                   for cross, solution in zip(crosses, solved, strict=True))
    independent = tuple(_descending_eigh(merit) for merit in merits)
    global_values, global_vectors = _descending_eigh(sum(merits[1:], merits[0].clone()))
    typed_values: dict[str, torch.Tensor] = {}
    typed_vectors: dict[str, torch.Tensor] = {}
    for label, indices in {"mlp": range(18), "attn": range(18, 36)}.items():
        selected = [merits[index] for index in indices]
        typed_values[label], typed_vectors[label] = _descending_eigh(
            sum(selected[1:], selected[0].clone())
        )
    # The two legacy anchors are prefixes of the same coefficient-SVD.  Compute each
    # site's decomposition once rather than repeating 36 large SVDs for each anchor.
    legacy_svd = tuple(torch.linalg.svd(value, full_matrices=False) for value in solved)
    return SpectralState(
        gram=gram, crosses=crosses, y2=y2, solved=solved,
        independent_values=tuple(item[0] for item in independent),
        independent_vectors=tuple(item[1] for item in independent),
        global_values=global_values, global_vectors=global_vectors,
        typed_values=typed_values, typed_vectors=typed_vectors,
        legacy_svd=legacy_svd,
    )


@dataclass
class FactorProgram:
    name: str
    descriptor: dict[str, Any]
    bases: dict[str, torch.Tensor]
    site_groups: tuple[str, ...]
    input_maps: tuple[torch.Tensor, ...]
    ranks_by_site: tuple[int, ...]
    diagnostics: dict[str, Any]


def _basis_diagnostics(basis: torch.Tensor, values: torch.Tensor, rank: int) -> dict[str, Any]:
    identity = torch.eye(rank, dtype=basis.dtype)
    orth = float((basis.T @ basis - identity).abs().max()) if rank else 0.0
    projector = basis @ basis.T
    idem = float((projector @ projector - projector).abs().max()) if rank else 0.0
    gap = float(values[rank - 1] - values[rank]) if rank < values.numel() else None
    return {"rank": rank, "orthogonality_max_abs": orth,
            "projector_idempotence_max_abs": idem, "boundary_eigengap": gap}


def fit_program(descriptor: Mapping[str, Any], state: SpectralState) -> FactorProgram:
    name, family, rank = str(descriptor["name"]), str(descriptor["family"]), int(descriptor["rank"])
    bases: dict[str, torch.Tensor] = {}
    groups: list[str] = []
    maps: list[torch.Tensor] = []
    ranks: list[int] = []
    spectra_for_groups: dict[str, tuple[torch.Tensor, int]] = {}
    if family == "global":
        basis = state.global_vectors[:, :rank]
        bases["global"] = basis
        groups = ["global"] * N_SITES
        maps = [solution @ basis for solution in state.solved]
        ranks = [rank] * N_SITES
        spectra_for_groups["global"] = (state.global_values, rank)
    elif family == "typed":
        for label in ("mlp", "attn"):
            bases[label] = state.typed_vectors[label][:, :rank]
            spectra_for_groups[label] = (state.typed_values[label], rank)
        groups = ["mlp"] * 18 + ["attn"] * 18
        maps = [state.solved[index] @ bases[groups[index]] for index in range(N_SITES)]
        ranks = [rank] * N_SITES
    elif family in {"independent", "price_independent"}:
        if family == "independent":
            ranks = [rank] * N_SITES
        else:
            allocation = core.allocate_equal_storage_independent_ranks(
                state.independent_values, n_output_bases=int(descriptor["budget_bases"]),
                input_dim=D, output_dim=D, shared_rank=rank,
            )
            ranks = list(allocation.ranks_by_site)
        for index, site_rank in enumerate(ranks):
            label = f"site{index}"
            basis = state.independent_vectors[index][:, :site_rank]
            bases[label] = basis
            groups.append(label)
            maps.append(state.solved[index] @ basis)
            spectra_for_groups[label] = (state.independent_values[index], site_rank)
    elif family == "legacy_svd":
        ranks = [rank] * N_SITES
        for index, coefficient in enumerate(state.solved):
            u, singular, vh = (
                state.legacy_svd[index] if state.legacy_svd is not None
                else torch.linalg.svd(coefficient, full_matrices=False)
            )
            label = f"site{index}"
            bases[label] = vh[:rank].T.contiguous()
            groups.append(label)
            maps.append((u[:, :rank] * singular[:rank]).contiguous())
            spectra_for_groups[label] = (singular.square(), rank)
    else:
        raise RuntimeError(f"unknown shared-RRR family: {family}")

    bases32 = {key: value.float().contiguous() for key, value in bases.items()}
    maps32 = tuple(value.float().contiguous() for value in maps)
    map_floats = sum(value.numel() for value in maps32) + sum(value.numel() for value in bases32.values())
    multiplies = sum(2 * D * value for value in ranks)
    explained = 0.0
    for label, (values, retained) in spectra_for_groups.items():
        del label
        explained += float(values[:retained].sum())
    total_y2 = sum(state.y2)
    diagnostics = {
        "explained_penalized_merit": explained,
        "penalized_residual_fraction": (total_y2 - explained) / total_y2,
        "groups": {label: _basis_diagnostics(bases[label], values, retained)
                   for label, (values, retained) in spectra_for_groups.items()},
        "map_float_count": map_floats,
        "map_float_bytes": 4 * map_floats,
        "common_table_float_count": COMMON_TABLE_FLOATS,
        "full_program_float_count": COMMON_TABLE_FLOATS + map_floats,
        "full_program_float_bytes": 4 * (COMMON_TABLE_FLOATS + map_floats),
        "dense_multiplies_per_uncovered_token": multiplies,
        "ranks_by_site": ranks,
        "basis_sha256s": {key: tensor_sha256(value) for key, value in bases32.items()},
        "input_map_sha256s": [tensor_sha256(value) for value in maps32],
        "finite": all(bool(torch.isfinite(value).all()) for value in (*maps32, *bases32.values())),
    }
    expected_bases = 1 if family == "global" else 2 if family == "typed" else N_SITES
    if family == "price_independent":
        target = core.grouped_map_price(N_SITES, int(descriptor["budget_bases"]), D, D, rank)
        if map_floats != target.grouped_float_count:
            raise RuntimeError("exact-price independent program missed its budget")
    elif map_floats != core.grouped_map_price(N_SITES, expected_bases, D, D, rank).grouped_float_count:
        raise RuntimeError("shared-RRR literal factor price changed")
    if not diagnostics["finite"]:
        raise RuntimeError("shared-RRR program contains nonfinite factors")
    return FactorProgram(name=name, descriptor=dict(descriptor), bases=bases32,
                         site_groups=tuple(groups), input_maps=maps32,
                         ranks_by_site=tuple(ranks), diagnostics=diagnostics)


class AutonomousProgram:
    """All-36-site context-free program; neither callback invokes a native component."""

    def __init__(self, model: torch.nn.Module, table: torch.Tensor, token_to_row: torch.Tensor,
                 factors: FactorProgram, ledger: PhysicalCallLedger, device: str):
        self.model = model
        self.table = table.to(device)
        self.token_to_row = token_to_row.to(device)
        self.bases = {key: value.to(device) for key, value in factors.bases.items()}
        self.maps = tuple(value.to(device) for value in factors.input_maps)
        self.groups = factors.site_groups
        self.arm = factors.name
        self.ledger = ledger

    def _write(self, index: int, tokens: torch.Tensor) -> torch.Tensor:
        embedding = self.model.transformer.wte(tokens)
        mapped = (embedding @ self.maps[index]) @ self.bases[self.groups[index]].T
        rows = self.token_to_row[tokens]
        covered = rows >= 0
        if bool(covered.any()):
            mapped = mapped.clone()
            mapped[covered] = self.table[index, rows[covered]]
        return mapped

    def attention(self, event: facade.AttentionEvent) -> tuple[torch.Tensor, torch.Tensor]:
        self.ledger.record_compiled_site(self.arm, "attn", event.site)
        write = self._write(SITE_TO_INDEX[("attn", event.site)], event.tokens)
        sentinel = torch.zeros(
            (*event.tokens.shape, event.block.attn.n_head, event.block.attn.head_dim),
            dtype=write.dtype, device=write.device,
        )
        return write, sentinel

    def mlp(self, event: facade.EarlyMLPEvent) -> torch.Tensor:
        self.ledger.record_compiled_site(self.arm, "mlp", event.site)
        return self._write(SITE_TO_INDEX[("mlp", event.site)], event.tokens)


def _empty_metric() -> dict[str, list[float | int]]:
    return {"covered": [0.0, 0], "uncovered": [0.0, 0]}


def _finish_metric(metric: Mapping[str, Sequence[float | int]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    total_loss, total_count = 0.0, 0
    for label in ("covered", "uncovered"):
        loss, count = float(metric[label][0]), int(metric[label][1])
        if count <= 0:
            raise RuntimeError(f"shared-RRR {label} score has zero denominator")
        result[label] = {"ce": loss / count, "tokens": count}
        total_loss += loss
        total_count += count
    result["all"] = {"ce": total_loss / total_count, "tokens": total_count}
    return result


def score_role(model: torch.nn.Module, rows: torch.Tensor, covered_mask: torch.Tensor,
               attention, mlp, *, after_outer, device: str) -> dict[str, Any]:
    metric = _empty_metric()
    for start in range(0, rows.shape[0], EVAL_BATCH):
        batch = rows[start:start + EVAL_BATCH].to(device)
        tokens, targets = batch[:, :CONTEXT], batch[:, 1:CONTEXT + 1]
        logits = facade.forward_with_dispatch(model, tokens, attention, mlp, require_production=True)
        after_outer()
        losses = F.cross_entropy(
            logits[:, SCORE_START:].reshape(-1, logits.shape[-1]),
            targets[:, SCORE_START:].reshape(-1), reduction="none",
        ).reshape(tokens.shape[0], SCORED_PER_ROW)
        mask = covered_mask[tokens[:, SCORE_START:]].reshape(-1)
        flat = losses.double().reshape(-1)
        for label, choose in (("covered", mask), ("uncovered", ~mask)):
            metric[label][0] += float(flat[choose].sum())
            metric[label][1] += int(choose.sum())
    return _finish_metric(metric)


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor_sha256(value).encode())
    return digest.hexdigest()


def require_resources(started: float) -> tuple[float, int]:
    elapsed = time.time() - started
    allocated = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    if elapsed > MAX_WALL_SECONDS or allocated > MAX_ALLOCATED_CUDA_BYTES:
        raise RuntimeError(f"shared-RRR resource ceiling exceeded: {elapsed}, {allocated}")
    return elapsed, allocated


def _result_gates(arms: Mapping[str, Any], coverage: int) -> dict[str, Any]:
    roles = tuple(ROLE_PATHS)
    passing_ranks = []
    for rank in RANKS:
        global_arm = arms[f"global_q{rank}"]["roles"]
        independent = arms[f"independent_q{rank}"]["roles"]
        price = arms[f"price_global_q{rank}"]["roles"]
        if all(global_arm[role]["all"]["ce"] <= independent[role]["all"]["ce"] + 0.01
               and global_arm[role]["all"]["ce"] <= price[role]["all"]["ce"] - 0.01
               for role in roles):
            passing_ranks.append(rank)
    e22 = all(
        arms["typed_q481"]["roles"][role]["all"]["ce"]
        <= arms["global_q494"]["roles"][role]["all"]["ce"] - 0.01
        for role in roles
    )
    covered_values = [payload["roles"][role]["covered"]["ce"]
                      for payload in arms.values() for role in roles]
    spread = max(covered_values) - min(covered_values)
    anchors = {
        name: {role: abs(arms[name]["roles"][role]["all"]["ce"] - expected) <= 0.002
               for role, expected in role_values.items()}
        for name, role_values in FRONTIER_ANCHORS.items()
    }
    return {
        "e2_1_pass": bool(passing_ranks), "e2_1_passing_ranks": passing_ranks,
        "e2_2_pass": e22, "coverage_control": coverage == COVERAGE,
        "covered_ce_spread": spread, "covered_identity_control": spread <= 1e-6,
        "legacy_frontier_anchor_controls": anchors,
        "legacy_frontier_controls_all_pass": all(all(value.values()) for value in anchors.values()),
        "exact_price_controls": all(
            payload["diagnostics"]["map_float_count"] == core.grouped_map_price(
                N_SITES, int(payload["descriptor"]["budget_bases"]), D, D,
                int(payload["descriptor"]["rank"]),
            ).grouped_float_count
            for payload in arms.values() if payload["descriptor"]["family"] == "price_independent"
        ),
    }


def comparison_ledger(arms: Mapping[str, Any]) -> dict[str, Any]:
    roles = tuple(ROLE_PATHS)
    result: dict[str, Any] = {}
    for rank in RANKS:
        for family, price_prefix in (("global", "price_global"), ("typed", "price_typed")):
            name = f"{family}_q{rank}"
            result[name] = {
                role: {
                    "minus_same_rank_independent_ce": arms[name]["roles"][role]["all"]["ce"]
                    - arms[f"independent_q{rank}"]["roles"][role]["all"]["ce"],
                    "minus_exact_price_independent_ce": arms[name]["roles"][role]["all"]["ce"]
                    - arms[f"{price_prefix}_q{rank}"]["roles"][role]["all"]["ce"],
                } for role in roles
            }
    result["typed_q481_minus_global_q494"] = {
        role: arms["typed_q481"]["roles"][role]["all"]["ce"]
        - arms["global_q494"]["roles"][role]["all"]["ce"] for role in roles
    }
    return result


def semantic_validate_call_ledger(value: Mapping[str, Any]) -> None:
    arm_names = [item["name"] for item in arm_descriptors()]
    expected_keys = {
        "schema", "registered", "native_outer", "native_sites", "compiled_outer",
        "compiled_sites", "outer_returns", "logit_returns", "optimizer_calls", "backward_calls",
    }
    if set(value) != expected_keys or value.get("schema") != (
        "shared_output_rrr_real_v1_call_ledger"
    ) or value.get("registered") != expected_call_schedule() or value.get("native_outer") != {
        "fit": FIT_OUTER_CALLS, "native_reference": EVAL_CALLS_PER_ARM,
    } or value.get("compiled_outer") != {
        name: EVAL_CALLS_PER_ARM for name in arm_names
    } or value.get("optimizer_calls") != 0 or value.get("backward_calls") != 0:
        raise RuntimeError("shared-RRR call ledger header changed")
    total_returns = FIT_OUTER_CALLS + (len(arm_names) + 1) * EVAL_CALLS_PER_ARM
    if value.get("outer_returns") != total_returns or value.get("logit_returns") != total_returns:
        raise RuntimeError("shared-RRR outer/logit return census changed")
    expected_native = {}
    for phase, count in (("fit", FIT_OUTER_CALLS),
                         ("native_reference", EVAL_CALLS_PER_ARM)):
        for kind in ("attn", "mlp"):
            expected_native[f"{phase}:{kind}"] = {str(site): count for site in range(18)}
    expected_compiled = {
        f"{name}:{kind}": {str(site): EVAL_CALLS_PER_ARM for site in range(18)}
        for name in arm_names for kind in ("attn", "mlp")
    }
    if value.get("native_sites") != expected_native or value.get(
        "compiled_sites"
    ) != expected_compiled:
        raise RuntimeError("shared-RRR exact site call census changed")


def semantic_validate_metric_ledger(value: Mapping[str, Any], expected_tokens: int) -> None:
    if set(value) != {"covered", "uncovered", "all"}:
        raise RuntimeError("shared-RRR metric ledger keys changed")
    for key in ("covered", "uncovered", "all"):
        metric = value[key]
        if set(metric) != {"ce", "tokens"} or not isinstance(metric["tokens"], int) or (
            metric["tokens"] <= 0
        ) or not math.isfinite(float(metric["ce"])):
            raise RuntimeError("shared-RRR metric ledger is malformed")
    if value["all"]["tokens"] != expected_tokens or value["covered"]["tokens"] + value[
        "uncovered"
    ]["tokens"] != expected_tokens:
        raise RuntimeError("shared-RRR metric denominator changed")


def expected_program_price(descriptor: Mapping[str, Any]) -> tuple[int, tuple[int, ...] | None]:
    family, rank = str(descriptor["family"]), int(descriptor["rank"])
    if family == "global":
        bases = 1
    elif family == "typed":
        bases = 2
    elif family in {"independent", "legacy_svd"}:
        bases = N_SITES
    elif family == "price_independent":
        return core.grouped_map_price(
            N_SITES, int(descriptor["budget_bases"]), D, D, rank,
        ).grouped_float_count, None
    else:
        raise RuntimeError("shared-RRR result contains an unknown family")
    price = core.grouped_map_price(N_SITES, bases, D, D, rank).grouped_float_count
    return price, tuple([rank] * N_SITES)


def replay_price_allocation(raw_spectra: Any, descriptor: Mapping[str, Any]) -> tuple[int, ...]:
    if not isinstance(raw_spectra, list) or len(raw_spectra) != N_SITES:
        raise RuntimeError("shared-RRR independent spectra schema changed")
    spectra = tuple(torch.tensor(item, dtype=torch.float64) for item in raw_spectra)
    if any(item.shape != (D,) for item in spectra):
        raise RuntimeError("shared-RRR independent spectra shape changed")
    replay = core.allocate_equal_storage_independent_ranks(
        spectra, n_output_bases=int(descriptor["budget_bases"]), input_dim=D,
        output_dim=D, shared_rank=int(descriptor["rank"]),
    )
    return replay.ranks_by_site


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def semantic_validate_diagnostics(value: Mapping[str, Any], descriptor: Mapping[str, Any]) -> None:
    required = {
        "explained_penalized_merit", "penalized_residual_fraction", "groups",
        "map_float_count", "map_float_bytes", "common_table_float_count",
        "full_program_float_count", "full_program_float_bytes",
        "dense_multiplies_per_uncovered_token", "ranks_by_site", "basis_sha256s",
        "input_map_sha256s", "finite",
    }
    if set(value) != required or value.get("common_table_float_count") != COMMON_TABLE_FLOATS:
        raise RuntimeError("shared-RRR diagnostics schema changed")
    if not math.isfinite(float(value["explained_penalized_merit"])) or not math.isfinite(
        float(value["penalized_residual_fraction"])
    ):
        raise RuntimeError("shared-RRR fit diagnostics are nonfinite")
    family = descriptor["family"]
    expected_groups = ({"global"} if family == "global" else {"mlp", "attn"}
                       if family == "typed" else {f"site{site}" for site in range(N_SITES)})
    if set(value["groups"]) != expected_groups or set(value["basis_sha256s"]) != expected_groups or (
        not all(_is_sha256(item) for item in value["basis_sha256s"].values())
    ) or len(value["input_map_sha256s"]) != N_SITES or not all(
        _is_sha256(item) for item in value["input_map_sha256s"]
    ):
        raise RuntimeError("shared-RRR factor hash closure changed")
    ranks = tuple(value["ranks_by_site"])
    if len(ranks) != N_SITES or any(
        not isinstance(rank, int) or rank < 0 or rank > D for rank in ranks
    ):
        raise RuntimeError("shared-RRR diagnostic rank vector changed")
    group_ranks = ({"global": ranks[0]} if family == "global" else
                   {"mlp": ranks[0], "attn": ranks[18]} if family == "typed" else
                   {f"site{site}": ranks[site] for site in range(N_SITES)})
    for label, item in value["groups"].items():
        if set(item) != {"rank", "orthogonality_max_abs", "projector_idempotence_max_abs",
                         "boundary_eigengap"} or item["rank"] != group_ranks[label] or any(
            not math.isfinite(float(item[key])) or float(item[key]) < 0
            for key in ("orthogonality_max_abs", "projector_idempotence_max_abs")
        ) or (item["boundary_eigengap"] is not None and not math.isfinite(
            float(item["boundary_eigengap"])
        )):
            raise RuntimeError("shared-RRR projector diagnostic changed")


def semantic_validate_resources(elapsed: Any, allocated: Any) -> None:
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(
        float(elapsed)
    ) or elapsed < 0 or elapsed > MAX_WALL_SECONDS or isinstance(allocated, bool) or not isinstance(
        allocated, int
    ) or allocated < 0 or allocated > MAX_ALLOCATED_CUDA_BYTES:
        raise RuntimeError("shared-RRR resource receipt changed")


def semantic_validate_physical_calls(value: Mapping[str, Any]) -> None:
    expected = {str(site): FIT_OUTER_CALLS + EVAL_CALLS_PER_ARM for site in range(18)}
    if value != {"attn": expected, "mlp": expected}:
        raise RuntimeError("shared-RRR physical native call census changed")


@torch.inference_mode()
def execute_discovery(frozen: Mapping[str, Any], *, device: str = "cuda") -> dict[str, Any]:
    require_published_authority(frozen)
    validate_frontier_anchors()
    started = time.time()
    fit_rows = load_rows_after_authority(
        FIT_PATH, FILE_PINS[str(FIT_PATH.relative_to(ROOT))], 96, frozen,
    )
    model, loaded = facade.load_bilin18(device=device, dtype=torch.float32, verify_weights_sha256=True)
    if asdict(loaded) != frozen["checkpoint"]:
        raise RuntimeError("shared-RRR loaded checkpoint differs from authority")
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    state_before = model_state_sha256(model)
    ledger = PhysicalCallLedger()
    fit_physical = {"attn": Counter(), "mlp": Counter()}
    fit_handles = []
    for site, block in enumerate(model.transformer.h):
        fit_handles.append(block.attn.register_forward_hook(
            lambda _m, _a, _o, site=site: fit_physical["attn"].update([site])
        ))
        fit_handles.append(block.mlp.register_forward_hook(
            lambda _m, _a, _o, site=site: fit_physical["mlp"].update([site])
        ))
    covered = torch.unique(fit_rows[:, :CONTEXT].reshape(-1), sorted=True)
    if covered.shape != (COVERAGE,):
        raise RuntimeError("shared-RRR fit coverage changed")
    try:
        table = capture_native_tables(model, covered, ledger, device)
    finally:
        for handle in fit_handles:
            handle.remove()
    if any(dict(fit_physical[kind]) != {site: FIT_OUTER_CALLS for site in range(18)}
           for kind in fit_physical):
        raise RuntimeError("shared-RRR fit physical native census changed")
    embedding = model.transformer.wte.weight.detach()[covered.to(device)].float().cpu()
    spectral = build_spectral_state(embedding, table)
    # Materialize every fit-only choice before any evaluation-role tensor is even
    # deserialized.  This closes the discovery-role leakage path despite all stages
    # sharing one process.
    programs = []
    for descriptor in arm_descriptors():
        require_resources(started)
        programs.append(fit_program(descriptor, spectral))
    role_rows = {
        role: load_rows_after_authority(
            path, FILE_PINS[str(path.relative_to(ROOT))], ROLE_ROWS[role], frozen,
        ) for role, path in ROLE_PATHS.items()
    }
    table_device = table.to(device)
    token_to_row = torch.full((VOCAB,), -1, dtype=torch.long)
    token_to_row[covered] = torch.arange(COVERAGE)
    covered_mask = token_to_row >= 0

    physical = {kind: Counter(counts) for kind, counts in fit_physical.items()}
    handles = []
    for site, block in enumerate(model.transformer.h):
        handles.append(block.attn.register_forward_hook(
            lambda _m, _a, _o, site=site: physical["attn"].update([site])
        ))
        handles.append(block.mlp.register_forward_hook(
            lambda _m, _a, _o, site=site: physical["mlp"].update([site])
        ))
    try:
        native_attention, native_mlp = native_dispatchers(ledger, "native_reference")
        native_roles = {
            role: score_role(
                model, rows, covered_mask, native_attention, native_mlp,
                after_outer=lambda: ledger.record_native_outer("native_reference"), device=device,
            ) for role, rows in role_rows.items()
        }
        arm_results: dict[str, Any] = {}
        for program in programs:
            require_resources(started)
            runner = AutonomousProgram(model, table_device, token_to_row, program, ledger, device)
            before_physical = {kind: dict(counts) for kind, counts in physical.items()}
            roles = {
                role: score_role(
                    model, rows, covered_mask, runner.attention, runner.mlp,
                    after_outer=lambda name=program.name: ledger.record_compiled_outer(name),
                    device=device,
                ) for role, rows in role_rows.items()
            }
            if any(dict(physical[kind]) != before_physical[kind] for kind in physical):
                raise RuntimeError(f"compiled arm {program.name} invoked a native component")
            arm_results[program.name] = {
                "descriptor": program.descriptor, "diagnostics": program.diagnostics,
                "roles": roles,
            }
            del runner, program
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    finally:
        for handle in handles:
            handle.remove()
    expected_native = FIT_OUTER_CALLS + EVAL_CALLS_PER_ARM
    if any(dict(physical[kind]) != {site: expected_native for site in range(18)}
           for kind in physical):
        raise RuntimeError("shared-RRR physical native module census changed")
    call_receipt = ledger.receipt()
    state_after = model_state_sha256(model)
    if state_after != state_before:
        raise RuntimeError("shared-RRR model state changed")
    elapsed, allocated = require_resources(started)
    result = {
        "schema": "shared_output_rrr_real_v1_results",
        "status": "discovery_complete_no_validation_or_generalization_authority",
        "authority_sha256": frozen["authority_sha256"],
        "coverage": COVERAGE, "covered_tokens_sha256": tensor_sha256(covered),
        "table_sha256": tensor_sha256(table), "gram_sha256": tensor_sha256(spectral.gram),
        "cross_sha256s": [tensor_sha256(value) for value in spectral.crosses],
        "independent_marginal_spectra": [value.tolist() for value in spectral.independent_values],
        "ridge": RIDGE, "site_order": [list(value) for value in SITE_ORDER],
        "native_reference": native_roles, "arms": arm_results,
        "comparisons": comparison_ledger(arm_results),
        "call_ledger": call_receipt,
        "physical_native_module_calls": {
            kind: {str(site): count for site, count in sorted(counts.items())}
            for kind, counts in physical.items()
        },
        "model_state_before_sha256": state_before, "model_state_after_sha256": state_after,
        "gates": _result_gates(arm_results, COVERAGE),
        "elapsed_seconds": elapsed, "maximum_allocated_cuda_bytes": allocated,
        "optimizer_calls": 0, "backward_calls": 0,
        "authority_scope": "discovery_only_no_validation_final_or_semantic_coordinates",
    }
    semantic_validate_result(result, frozen)
    return result


def semantic_validate_result(value: Mapping[str, Any], frozen: Mapping[str, Any]) -> None:
    required = {
        "schema", "status", "authority_sha256", "coverage", "covered_tokens_sha256",
        "table_sha256", "gram_sha256", "cross_sha256s", "independent_marginal_spectra",
        "ridge", "site_order", "native_reference", "arms", "comparisons", "call_ledger",
        "physical_native_module_calls",
        "model_state_before_sha256", "model_state_after_sha256", "gates",
        "elapsed_seconds", "maximum_allocated_cuda_bytes", "optimizer_calls",
        "backward_calls", "authority_scope",
    }
    if set(value) != required or value["schema"] != "shared_output_rrr_real_v1_results" or (
        value["authority_sha256"] != frozen["authority_sha256"]
    ) or value["status"] != "discovery_complete_no_validation_or_generalization_authority" or (
        value["coverage"] != COVERAGE
    ) or value["ridge"] != RIDGE or value[
        "site_order"
    ] != [list(item) for item in SITE_ORDER]:
        raise RuntimeError("shared-RRR result top-level schema changed")
    descriptors = {item["name"]: item for item in arm_descriptors()}
    if set(value["arms"]) != set(descriptors) or set(value["native_reference"]) != set(ROLE_PATHS):
        raise RuntimeError("shared-RRR result arm or role set changed")
    expected_tokens = {role: rows * SCORED_PER_ROW for role, rows in ROLE_ROWS.items()}
    for role, metrics in value["native_reference"].items():
        semantic_validate_metric_ledger(metrics, expected_tokens[role])
    raw_spectra = value["independent_marginal_spectra"]
    for name, payload in value["arms"].items():
        if payload["descriptor"] != descriptors[name] or set(payload) != {
            "descriptor", "diagnostics", "roles"
        } or set(payload["roles"]) != set(ROLE_PATHS):
            raise RuntimeError("shared-RRR arm schema changed")
        diagnostics = payload["diagnostics"]
        semantic_validate_diagnostics(diagnostics, payload["descriptor"])
        expected_price, expected_ranks = expected_program_price(payload["descriptor"])
        ranks = tuple(diagnostics.get("ranks_by_site", ()))
        if payload["descriptor"]["family"] == "price_independent":
            expected_ranks = replay_price_allocation(raw_spectra, payload["descriptor"])
        if diagnostics["finite"] is not True or diagnostics["full_program_float_count"] != (
            diagnostics["map_float_count"] + COMMON_TABLE_FLOATS
        ) or diagnostics["map_float_bytes"] != 4 * diagnostics["map_float_count"] or (
            diagnostics["full_program_float_bytes"] != 4 * diagnostics["full_program_float_count"]
        ) or diagnostics["map_float_count"] != expected_price or len(ranks) != N_SITES or any(
            not isinstance(rank, int) or rank < 0 or rank > D for rank in ranks
        ) or (expected_ranks is not None and ranks != expected_ranks) or diagnostics[
            "dense_multiplies_per_uncovered_token"
        ] != sum(2 * D * rank for rank in ranks):
            raise RuntimeError("shared-RRR price schema changed")
        for role, metrics in payload["roles"].items():
            semantic_validate_metric_ledger(metrics, expected_tokens[role])
    semantic_validate_call_ledger(value["call_ledger"])
    semantic_validate_physical_calls(value["physical_native_module_calls"])
    if value["model_state_before_sha256"] != value["model_state_after_sha256"] or (
        value["optimizer_calls"], value["backward_calls"]
    ) != (0, 0) or value["authority_scope"] != (
        "discovery_only_no_validation_final_or_semantic_coordinates"
    ):
        raise RuntimeError("shared-RRR execution integrity changed")
    semantic_validate_resources(value["elapsed_seconds"], value["maximum_allocated_cuda_bytes"])
    if value["comparisons"] != comparison_ledger(value["arms"]):
        raise RuntimeError("shared-RRR comparison ledger changed")
    recomputed = _result_gates(value["arms"], COVERAGE)
    if value["gates"] != recomputed:
        raise RuntimeError("shared-RRR registered gates changed")
    if not all(_is_sha256(value[key]) for key in (
        "covered_tokens_sha256", "table_sha256", "gram_sha256",
        "model_state_before_sha256", "model_state_after_sha256",
    )) or len(value["cross_sha256s"]) != N_SITES or not all(
        _is_sha256(item) for item in value["cross_sha256s"]
    ):
        raise RuntimeError("shared-RRR result hash closure changed")


def receipt_payload(frozen: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema": "shared_output_rrr_real_v1_receipt",
        "status": "complete_discovery_receipt_last",
        "authority_path": str(AUTHORITY), "authority_file_sha256": file_sha256(AUTHORITY),
        "authority_sha256": frozen["authority_sha256"],
        "results_path": str(RESULTS), "results_file_sha256": file_sha256(RESULTS),
        "results_logical_sha256": logical_sha256(result),
        "source_closure_sha256": frozen["source_closure"]["sha256"],
        "input_file_sha256s": frozen["input_file_sha256s"],
        "failure_absent": not FAILURE.exists(),
        "authority_scope": "discovery_only_no_validation_final_or_semantic_coordinates",
    }
    return {**body, "receipt_sha256": logical_sha256(body)}


def semantic_validate_receipt(value: Mapping[str, Any], frozen: Mapping[str, Any],
                              result: Mapping[str, Any]) -> None:
    if value != receipt_payload(frozen, result):
        raise RuntimeError("shared-RRR receipt joins changed")
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if value.get("receipt_sha256") != logical_sha256(body) or value.get(
        "failure_absent"
    ) is not True:
        raise RuntimeError("shared-RRR receipt self-hash changed")


def publish_failure(claim: RunClaim, error: BaseException) -> None:
    # A preflight refusal (not committed/pushed, bad input pins, absent checkpoint)
    # must not open any scientific namespace.  Failure authority begins only once
    # the no-outcome authority exists.
    if not AUTHORITY.exists() or RECEIPT.exists() or FAILURE.exists():
        return
    require_claim(claim, LOCK)
    if RECEIPT.exists() or FAILURE.exists():
        return
    payload = {
        "schema": "shared_output_rrr_real_v1_failure",
        "status": "terminal_failure_no_receipt",
        "error_type": type(error).__name__, "error": str(error),
        "authority_exists": AUTHORITY.exists(), "results_exists": RESULTS.exists(),
        "receipt_exists": False,
        "authority_file_sha256": file_sha256(AUTHORITY) if AUTHORITY.exists() else None,
        "results_file_sha256": file_sha256(RESULTS) if RESULTS.exists() else None,
    }
    atomic_create_json(payload, FAILURE)


def run(*, device: str = "cuda") -> dict[str, Any]:
    require_pristine_namespace()
    claim = acquire_lock(LOCK)
    try:
        source = source_closure()
        inputs = input_bindings()
        checkpoint = checkpoint_binding(verify_hash=True)
        frozen = authority_payload(source, inputs, checkpoint)
        require_claim(claim, LOCK)
        if any(path.exists() for path in output_namespace()):
            raise RuntimeError("shared-RRR output appeared before authority")
        atomic_create_json(frozen, AUTHORITY)
        require_published_authority(frozen)
        result = execute_discovery(frozen, device=device)
        semantic_validate_result(result, frozen)
        verify_frozen_inputs(frozen, verify_checkpoint_hash=True)
        require_claim(claim, LOCK)
        if RESULTS.exists() or FAILURE.exists() or RECEIPT.exists():
            raise RuntimeError("shared-RRR terminal namespace changed before result")
        atomic_create_json(result, RESULTS)
        if json.loads(RESULTS.read_text()) != result:
            raise RuntimeError("shared-RRR result replay changed")
        # Receipt is the final artifact write.  Recheck exact source/input bytes and
        # lock ownership immediately before it; the checkpoint was fully rehashed
        # immediately before result and its stat identity is bound in the authority.
        verify_source_closure(frozen["source_closure"])
        if input_bindings() != frozen["input_file_sha256s"]:
            raise RuntimeError("shared-RRR input changed before receipt")
        require_claim(claim, LOCK)
        receipt = receipt_payload(frozen, result)
        semantic_validate_receipt(receipt, frozen, result)
        if receipt["failure_absent"] is not True or RECEIPT.exists():
            raise RuntimeError("shared-RRR receipt preconditions changed")
        atomic_create_json(receipt, RECEIPT)
        return receipt
    except BaseException as error:
        publish_failure(claim, error)
        raise
    finally:
        release_lock(claim, LOCK)


def main() -> None:
    receipt = run()
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
