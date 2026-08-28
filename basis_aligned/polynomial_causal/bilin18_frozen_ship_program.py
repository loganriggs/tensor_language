"""Validated, explicit execution of the canonical frozen bilin18 ship.

Unlike historical experiment runners, this module loads no model or rows at import,
does not rebuild the ship, and performs no network access.  It consumes the frozen
state/manifest pair and supplies the two dispatchers required by
``bilin18_observed_model_facade.forward_with_dispatch``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

import bilin18_observed_model_facade as facade


FROZEN_STATE = Path("/workspace/runs/bilin18_frozen_ship_v2.pt")
FROZEN_MANIFEST = Path("/workspace/runs/bilin18_frozen_ship_v2_manifest.json")
FROZEN_LOCK = Path("/workspace/runs/.bilin18_frozen_ship_v2.lock")
ROW_RECEIPT = Path(
    "/workspace/tensor_language/basis_aligned/bilinear_quotient/"
    ".rowcache/fineweb_oracle_v2_receipt.json"
)
SCHEMA_VERSION = 2
SHIP_SEED = 27_182_818
ARTIFACT_SHA256 = "fe21ead35b1dcb3c0914a36b04d7be36e9c3f179c57bc63eee62bd78d34fe9df"
ARTIFACT_BYTES = 1_468_244_695
REALIZATION_SHA256 = "21ddc9ffdb7703aa570f88c5c7f4fa9fe007a988a1a7a3fd91058ee76a25ab8e"
ROW_RECEIPT_SHA256 = "8c510aed5586b4d950f0688a0e575c7695e51525d69f80c6bc39817c1454e9cb"
SPEC = {
    5: frozenset({7}),
    8: frozenset({1, 2, 3, 7}),
    10: frozenset({2, 3, 4, 5, 6}),
    13: frozenset({0, 5, 8}),
    14: frozenset({4, 6, 7}),
    16: frozenset({0, 3, 4, 5}),
    17: frozenset({0, 1, 2}),
}


@dataclass(frozen=True)
class FrozenShipReceipt:
    artifact_sha256: str
    artifact_bytes: int
    realization_sha256: str
    row_receipt_sha256: str
    source_commit: str
    device: str


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode() + b"\0")
            digest.update(json.dumps(list(tensor.shape)).encode() + b"\0")
            # ``memoryview`` avoids materializing a second multi-gigabyte bytes
            # object for the canonical tensor tree while preserving the exact
            # C-order byte stream used by the freezing pipeline.
            digest.update(memoryview(tensor.numpy()))
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item, key=lambda candidate: str(candidate)):
                update(str(key))
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(("list" if isinstance(item, list) else "tuple").encode() + b"\0")
            for child in item:
                update(child)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            digest.update(type(item).__name__.encode() + b"\0")
            digest.update(repr(item).encode() + b"\0")
        else:
            raise TypeError(f"unsupported frozen ship value: {type(item)}")

    update(value)
    return digest.hexdigest()


def _device_tree(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.detach().to(device).contiguous().clone()
    if isinstance(value, dict):
        return {key: _device_tree(child, device) for key, child in value.items()}
    if isinstance(value, list):
        return [_device_tree(child, device) for child in value]
    if isinstance(value, tuple):
        return tuple(_device_tree(child, device) for child in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported frozen ship value: {type(value)}")


@contextmanager
def _pair_claim(lock: Path = FROZEN_LOCK):
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise RuntimeError("canonical frozen ship is already being created or loaded") from error
    try:
        yield
    finally:
        lock.rmdir()


def load_frozen_ship(
    *, device: str | torch.device = "cuda", verify_bytes: bool = True,
) -> tuple["FrozenShipProgram", FrozenShipReceipt]:
    """Load and validate the canonical pair without importing experiment runners."""

    target = torch.device(device)
    with _pair_claim():
        if not FROZEN_STATE.is_file() or not FROZEN_MANIFEST.is_file() or not (
            ROW_RECEIPT.is_file()
        ):
            raise RuntimeError("canonical frozen ship pair or row receipt is absent")
        manifest = json.loads(FROZEN_MANIFEST.read_text())
        expected = {
            "schema_version": SCHEMA_VERSION,
            "artifact_path": str(FROZEN_STATE),
            "artifact_sha256": ARTIFACT_SHA256,
            "artifact_bytes": ARTIFACT_BYTES,
            "ship_realization_sha256": REALIZATION_SHA256,
            "ship_seed": SHIP_SEED,
            "row_receipt_sha256": ROW_RECEIPT_SHA256,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise RuntimeError(f"frozen ship manifest field changed: {key}")
        if FROZEN_STATE.stat().st_size != ARTIFACT_BYTES:
            raise RuntimeError("frozen ship artifact size changed")
        if verify_bytes and _file_sha256(FROZEN_STATE) != ARTIFACT_SHA256:
            raise RuntimeError("frozen ship artifact bytes changed")
        row_receipt = json.loads(ROW_RECEIPT.read_text())
        if _logical_sha256(row_receipt) != ROW_RECEIPT_SHA256:
            raise RuntimeError("frozen ship row receipt changed")
        payload = torch.load(FROZEN_STATE, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != SCHEMA_VERSION or payload.get("ship_seed") != SHIP_SEED:
            raise RuntimeError("frozen ship payload metadata changed")
        state = payload.get("state")
        if _tree_sha256(state) != REALIZATION_SHA256 or payload.get(
            "ship_realization_sha256"
        ) != REALIZATION_SHA256:
            raise RuntimeError("frozen ship realization changed")
        moved = _device_tree(state, target)
    program = FrozenShipProgram(moved, production=True)
    receipt = FrozenShipReceipt(
        artifact_sha256=ARTIFACT_SHA256,
        artifact_bytes=ARTIFACT_BYTES,
        realization_sha256=REALIZATION_SHA256,
        row_receipt_sha256=ROW_RECEIPT_SHA256,
        source_commit=str(manifest.get("source_commit")),
        device=str(target),
    )
    return program, receipt


class FrozenShipProgram:
    """Stateless dispatchers for one validated frozen realization."""

    def __init__(self, state: Mapping[str, Any], *, production: bool) -> None:
        if not isinstance(state, Mapping) or not all(
            key in state for key in ("TWALL", "SHIP", "CORR", "all_attention")
        ):
            raise RuntimeError("frozen ship state is malformed")
        self.twall = state["TWALL"]
        self.ship = state["SHIP"]
        self.corr = state["CORR"]
        self.all_attention = frozenset(int(site) for site in state["all_attention"])
        self.production = bool(production)
        self.d_model = int(self.ship["r0"][0].shape[0])
        required_early = {"t0", "r0", "t1", "r1", "r2"}
        if not required_early.issubset(self.ship):
            raise RuntimeError("frozen ship lacks an early deployed program")
        if self.production:
            if self.d_model != 1152 or self.all_attention != frozenset(range(18)):
                raise RuntimeError("frozen ship topology differs from production")
            if set(self.twall) != set(range(18)):
                raise RuntimeError("frozen ship attention map is incomplete")
            for site in range(4, 18):
                if f"u{site}" not in self.ship:
                    raise RuntimeError(f"frozen ship lacks MLP{site}")

    def attention(
        self, event: facade.AttentionEvent,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.production:
            raise RuntimeError("synthetic frozen ship has no attention dispatcher")
        site, block, state = event.site, event.block, event.state
        if site not in self.all_attention or site not in self.twall:
            raise RuntimeError(f"frozen attention{site} is not deployed")
        batch, sequence, width = state.shape
        if width != 1152:
            raise RuntimeError("frozen attention state width changed")
        attention = block.attn
        factors = self.twall[site]

        def projected(name: str) -> torch.Tensor:
            return (state.float() @ factors[name].T).view(batch, sequence, 9, 128)

        query, key = projected("q"), projected("k")
        query2, key2 = projected("q2"), projected("k2")
        if site in SPEC:
            native = {
                "q": attention.c_q(state).view(batch, sequence, 9, 128).float(),
                "k": attention.c_k(state).view(batch, sequence, 9, 128).float(),
                "q2": attention.c_q2(state).view(batch, sequence, 9, 128).float(),
                "k2": attention.c_k2(state).view(batch, sequence, 9, 128).float(),
            }
            for head in SPEC[site]:
                query[:, :, head] = native["q"][:, :, head]
                key[:, :, head] = native["k"][:, :, head]
                query2[:, :, head] = native["q2"][:, :, head]
                key2[:, :, head] = native["k2"][:, :, head]

        cosine, sine = attention.rotary(query)
        query = facade.TT.apply_rotary_emb(F.rms_norm(query, (128,)), cosine, sine)
        key = facade.TT.apply_rotary_emb(F.rms_norm(key, (128,)), cosine, sine)
        query2 = facade.TT.apply_rotary_emb(F.rms_norm(query2, (128,)), cosine, sine)
        key2 = facade.TT.apply_rotary_emb(F.rms_norm(key2, (128,)), cosine, sine)
        pattern = (
            torch.einsum("bqhd,bkhd->bhqk", query, key) / 128.0
        ) * (
            torch.einsum("bqhd,bkhd->bhqk", query2, key2) / 128.0
        )
        causal = torch.tril(torch.ones(
            sequence, sequence, device=state.device, dtype=torch.bool,
        ))
        pattern = pattern.masked_fill(~causal, 0.0)
        value = attention.c_v(state).view(batch, sequence, 9, 128)
        first_value = value if event.first_value is None else event.first_value
        mixed = (1 - attention.lamb) * value + attention.lamb * first_value.view_as(value)
        output = torch.einsum("bhqk,bkhd->bqhd", pattern.to(mixed.dtype), mixed)
        write = attention.c_proj(output.reshape(batch, sequence, 1152))
        return write, first_value

    def mlp(self, event: facade.EarlyMLPEvent) -> torch.Tensor:
        site, state = event.site, event.state
        batch, sequence, width = state.shape
        if width != self.d_model or len(event.prior_writes) != site:
            raise RuntimeError(f"frozen MLP{site} received a nonsequential event")
        tokens = event.tokens.reshape(-1)
        if site == 0:
            weight, mean_input, mean_output = self.ship["r0"]
            value = self.ship["t0"][tokens].view(batch, sequence, width)
            value = value + (
                mean_output + (event.attention_write.float().reshape(-1, width) - mean_input) @ weight
            ).view(batch, sequence, width)
        elif site in (1, 2):
            weight, mean_input, mean_output = self.ship[f"r{site}"]
            features = torch.cat([
                event.attention_write.float(), event.prior_writes[site - 1].float(),
            ], dim=-1).reshape(-1, 2 * width)
            value = mean_output + (features - mean_input) @ weight
            if site == 1:
                value = self.ship["t1"][tokens] + value
            elif bool(self.corr["on"]):
                value = value + self.corr["b"] + (
                    (features - mean_input) @ self.corr["V"]
                ) @ self.corr["U"].T
            value = value.view(batch, sequence, width)
        elif site == 3:
            native = event.block.mlp(state).float()
            centered = native - self.ship["mean3"]
            basis = self.ship["p3"]
            value = self.ship["mean3"] + (
                centered.reshape(-1, width) @ basis.T
            ) @ basis
            value = value.view(batch, sequence, width)
        elif 4 <= site <= 17:
            factors = self.ship[f"u{site}"]
            hidden = (state.float() @ factors["l"].T) * (
                state.float() @ factors["r"].T
            )
            value = hidden @ factors["d"].T + factors["b"]
        else:
            raise RuntimeError(f"frozen ship has no MLP{site}")
        return value.to(state.dtype)
