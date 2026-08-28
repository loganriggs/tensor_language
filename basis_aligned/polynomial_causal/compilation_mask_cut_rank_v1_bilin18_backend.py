"""Lazy production bilin18 backend for the fixed-program cut-rank assay.

The shared program is the historical section-1786 construction, rebuilt once:

* current-token context-free rows on the 5,419 fit-covered token types;
* centered rank-64 truncation independently at every attention/MLP site;
* output-distribution nearest-neighbour fallback indices, retained as a hashed
  control realization;
* a rank-64 embedding-to-row map refit inside each truncated table basis; and
* dense token rows whose covered entries are the truncated table and whose
  uncovered entries are the learned rank-64 prediction.

Every one of the 64 masks selects from this one immutable bank.  There are no
mask-specific fits and all scalar gains are identity.  Import and ``create_backend``
perform no row, checkpoint, model, CUDA, or program work.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import bilin18_observed_model_facade as facade
import compilation_mask_cut_rank_v1_gpu_adapter as adapter
import compilation_mask_cut_rank_v1_measurements as measurement


HERE = Path(__file__).resolve().parent
FIT_ROW_RECEIPT = adapter.DEFAULT_ROW_RECEIPT
FIT_ROW_ROLE = "n96_skip80"
SOURCE_PATHS = (
    "basis_aligned/polynomial_causal/compilation_mask_cut_rank_v1_bilin18_backend.py",
    "basis_aligned/polynomial_causal/bilin18_observed_model_facade.py",
    "jacclust/tt_model.py",
)
PROGRAM_FAMILY = "section1786_contextfree_rank64_table_learned_rank64_map"
FALLBACK_CONTROL = "output_distribution_nearest_covered_token"
EXECUTED_UNCOVERED_PATH = "rank64_embedding_to_row_map"


def _sha256_text(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _logical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def tensor_content_sha256(value: torch.Tensor, *, chunk_bytes: int = 64 << 20) -> str:
    """Hash exact shape/dtype and bytes without retaining a host-sized copy."""

    if not torch.is_tensor(value) or value.layout != torch.strided or not value.is_contiguous() or (
        value.requires_grad
    ) or type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ValueError("program tensor hashing requires a contiguous detached strided tensor")
    digest = hashlib.sha256(json.dumps({
        "shape": list(value.shape), "dtype": str(value.dtype),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    raw = value.detach().view(torch.uint8).reshape(-1)
    for start in range(0, raw.numel(), chunk_bytes):
        block = raw[start:start + chunk_bytes].cpu().numpy().tobytes(order="C")
        digest.update(block)
    return digest.hexdigest()


def model_tree_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        value = state[name].detach().contiguous()
        header = json.dumps({
            "name": name, "shape": list(value.shape), "dtype": str(value.dtype),
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        raw = value.view(torch.uint8).reshape(-1)
        for start in range(0, raw.numel(), 64 << 20):
            digest.update(raw[start:start + (64 << 20)].cpu().numpy().tobytes(order="C"))
    return digest.hexdigest()


def _tensor_metadata(value: torch.Tensor) -> tuple[Any, ...]:
    return (
        tuple(value.shape), str(value.dtype), str(value.device), value.data_ptr(), value._version,
    )


class ModelTreeGuard:
    """Exact initial/final hash with cheap per-cell version/pointer checks."""

    def __init__(self, model: torch.nn.Module, *, initial_sha256: str | None = None) -> None:
        self._model = model
        self.sha256 = initial_sha256 or model_tree_sha256(model)
        if not _sha256_text(self.sha256):
            raise ValueError("initial model-tree hash is malformed")
        self._metadata = {
            name: _tensor_metadata(value)
            for name, value in model.state_dict().items()
        }

    def verify_metadata(self) -> str:
        state = self._model.state_dict()
        if set(state) != set(self._metadata) or any(
            _tensor_metadata(state[name]) != expected
            for name, expected in self._metadata.items()
        ):
            raise RuntimeError("bilin18 model state changed during program execution")
        return self.sha256

    def verify_exact(self) -> str:
        self.verify_metadata()
        if model_tree_sha256(self._model) != self.sha256:
            raise RuntimeError("bilin18 model content changed without a version match")
        return self.sha256


@dataclass(frozen=True, slots=True)
class ProgramDimensions:
    model_width: int = 1152
    tokenizer_vocab: int = 50_257
    logit_vocab: int = 50_304
    layer_count: int = 18
    table_rank: int = 64
    map_rank: int = 64
    ridge: float = 1e-2
    expected_covered_token_count: int = 5_419
    build_batch_size: int = 256
    eval_batch_size: int = 8

    def __post_init__(self) -> None:
        integers = (
            self.model_width, self.tokenizer_vocab, self.logit_vocab, self.layer_count,
            self.table_rank, self.map_rank, self.expected_covered_token_count,
            self.build_batch_size, self.eval_batch_size,
        )
        if any(type(value) is not int or value <= 0 for value in integers) or not (
            0 < self.table_rank <= min(self.expected_covered_token_count, self.model_width)
        ) or not 0 < self.map_rank <= self.model_width or not isinstance(
            self.ridge, float
        ) or not self.ridge > 0:
            raise ValueError("program dimensions are malformed")


PRODUCTION_DIMENSIONS = ProgramDimensions()


def centered_rank_truncate(table: torch.Tensor, rank: int) -> torch.Tensor:
    """Historical float64 centered SVD followed by float32 reconstruction."""

    if not torch.is_tensor(table) or table.ndim != 2 or table.dtype != torch.float32 or (
        not table.is_contiguous()
    ) or type(rank) is not int or not 0 < rank <= min(table.shape):
        raise ValueError("context-free table/rank is malformed")
    source = table.double()
    mean = source.mean(dim=0, keepdim=True)
    left, singular, right = torch.linalg.svd(source - mean, full_matrices=False)
    return (mean + (left[:, :rank] * singular[:rank]) @ right[:rank]).float().contiguous()


def rank_truncated_ridge_map(
    covered_embeddings: torch.Tensor, covered_rows: torch.Tensor, *, rank: int,
    ridge: float,
) -> torch.Tensor:
    """Fit the exact section-1786 map inside the already-truncated row basis."""

    if not torch.is_tensor(covered_embeddings) or not torch.is_tensor(
        covered_rows
    ) or covered_embeddings.ndim != 2 or covered_rows.ndim != 2 or (
        covered_embeddings.shape[0] != covered_rows.shape[0]
    ) or covered_embeddings.shape[1] != covered_rows.shape[1] or type(rank) is not int or (
        not 0 < rank <= covered_embeddings.shape[1]
    ) or not isinstance(ridge, float) or not ridge > 0:
        raise ValueError("ridge-map inputs are malformed")
    embeddings = covered_embeddings.double()
    rows = covered_rows.double()
    count, width = embeddings.shape
    normal = embeddings.T @ embeddings + ridge * torch.eye(
        width, device=embeddings.device, dtype=torch.float64,
    ) * (count / width)
    coefficient = torch.linalg.solve(normal, embeddings.T @ rows)
    left, singular, right = torch.linalg.svd(coefficient, full_matrices=False)
    return ((left[:, :rank] * singular[:rank]) @ right[:rank]).contiguous()


def materialize_token_rows(
    *, token_embeddings: torch.Tensor, covered_token_ids: torch.Tensor,
    covered_rows: torch.Tensor, coefficient: torch.Tensor,
) -> torch.Tensor:
    """Covered lookup plus learned uncovered prediction, in exact token-ID order."""

    if token_embeddings.ndim != 2 or covered_rows.ndim != 2 or coefficient.ndim != 2 or (
        token_embeddings.shape[1] != covered_rows.shape[1]
    ) or coefficient.shape != (token_embeddings.shape[1], covered_rows.shape[1]) or (
        covered_token_ids.dtype != torch.long or covered_token_ids.ndim != 1
    ) or len(covered_token_ids) != len(covered_rows) or not bool(
        (covered_token_ids[1:] > covered_token_ids[:-1]).all()
    ):
        raise ValueError("token-row materialization inputs are malformed")
    vocabulary, width = token_embeddings.shape
    if int(covered_token_ids[0]) < 0 or int(covered_token_ids[-1]) >= vocabulary:
        raise ValueError("covered token ID is outside the embedding vocabulary")
    dense = (token_embeddings.double() @ coefficient).float().contiguous()
    dense[covered_token_ids] = covered_rows
    return dense


def output_nearest_indices(
    *, covered_probabilities: torch.Tensor, all_probabilities: torch.Tensor,
    covered_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Output-distribution nearest covered token, with exact covered self-indices."""

    if covered_probabilities.ndim != 2 or all_probabilities.ndim != 2 or (
        covered_probabilities.shape[1] != all_probabilities.shape[1]
    ) or len(covered_probabilities) != len(covered_token_ids) or (
        covered_token_ids.dtype != torch.long
    ):
        raise ValueError("output-nearest inputs are malformed")
    covered = covered_probabilities / covered_probabilities.norm(
        dim=-1, keepdim=True,
    ).clamp_min(1e-9)
    all_rows = all_probabilities / all_probabilities.norm(
        dim=-1, keepdim=True,
    ).clamp_min(1e-9)
    nearest = (all_rows @ covered.T).argmax(dim=-1).to(torch.long)
    nearest[covered_token_ids] = torch.arange(
        len(covered_token_ids), device=nearest.device,
    )
    return nearest.contiguous()


class SharedProgram:
    """Private immutable executable bank and its content-derived manifest."""

    __slots__ = (
        "_expected_sha256", "_fit_wave_receipt", "_manifest", "_metadata", "_sealed",
        "covered_token_ids", "dense_rows", "dimensions", "output_nearest_covered_index",
        "table_sha256s",
    )

    def __init__(
        self, *, dimensions: ProgramDimensions, fit_wave_receipt: dict[str, Any],
        covered_token_ids: torch.Tensor, output_nearest_covered_index: torch.Tensor,
        dense_rows: Sequence[tuple[measurement.cut.Site, torch.Tensor]],
        model_realization_sha256: str, program_source_sha256: str,
    ) -> None:
        object.__setattr__(self, "_sealed", False)
        if not isinstance(dimensions, ProgramDimensions) or not isinstance(
            fit_wave_receipt, dict
        ) or not _sha256_text(model_realization_sha256) or not _sha256_text(
            program_source_sha256
        ) or covered_token_ids.dtype != torch.long or covered_token_ids.ndim != 1 or (
            len(covered_token_ids) != dimensions.expected_covered_token_count
        ) or output_nearest_covered_index.dtype != torch.long or tuple(
            output_nearest_covered_index.shape
        ) != (dimensions.tokenizer_vocab,) or tuple(site for site, _ in dense_rows) != (
            adapter.ALL_NATIVE_SITES
        ):
            raise ValueError("shared program realization schema is malformed")
        self.dimensions = dimensions
        self._fit_wave_receipt = json.loads(json.dumps(fit_wave_receipt, allow_nan=False))
        self.covered_token_ids = covered_token_ids.detach().contiguous()
        self.output_nearest_covered_index = output_nearest_covered_index.detach().contiguous()
        self.dense_rows = tuple((site, value.detach().contiguous()) for site, value in dense_rows)
        if any(
            tuple(value.shape) != (dimensions.tokenizer_vocab, dimensions.model_width)
            or value.dtype != torch.float32 or value.requires_grad
            for _, value in self.dense_rows
        ):
            raise ValueError("shared program dense token rows are malformed")
        self._metadata = {
            "covered_token_ids": _tensor_metadata(self.covered_token_ids),
            "output_nearest_covered_index": _tensor_metadata(
                self.output_nearest_covered_index
            ),
            **{
                f"{site[0]}{site[1]}": _tensor_metadata(value)
                for site, value in self.dense_rows
            },
        }
        self.table_sha256s = tuple(
            (site, tensor_content_sha256(value)) for site, value in self.dense_rows
        )
        self._manifest = {
            "program_family": PROGRAM_FAMILY,
            "fallback_control": FALLBACK_CONTROL,
            "executed_uncovered_path": EXECUTED_UNCOVERED_PATH,
            "gain_policy": adapter.GAIN_POLICY,
            "dimensions": asdict(dimensions),
            "fit_wave_receipt": self._fit_wave_receipt,
            "covered_token_ids_sha256": tensor_content_sha256(self.covered_token_ids),
            "output_nearest_covered_index_sha256": tensor_content_sha256(
                self.output_nearest_covered_index
            ),
            "dense_row_sha256s": [
                {"site": list(site), "sha256": digest}
                for site, digest in self.table_sha256s
            ],
            "model_realization_sha256": model_realization_sha256,
            "program_source_sha256": program_source_sha256,
        }
        self._expected_sha256 = _logical_sha256(self._manifest)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("shared compiled program is sealed")
        object.__setattr__(self, name, value)

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._manifest, allow_nan=False))

    @property
    def sha256(self) -> str:
        if _logical_sha256(self._manifest) != self._expected_sha256:
            raise RuntimeError("shared compiled-program manifest changed")
        return self._expected_sha256

    def rows_for(self, site: measurement.cut.Site) -> torch.Tensor:
        for observed, value in self.dense_rows:
            if observed == site:
                return value
        raise RuntimeError("compiled site is absent from the shared program")

    def verify_metadata(self) -> str:
        observed = {
            "covered_token_ids": _tensor_metadata(self.covered_token_ids),
            "output_nearest_covered_index": _tensor_metadata(
                self.output_nearest_covered_index
            ),
            **{
                f"{site[0]}{site[1]}": _tensor_metadata(value)
                for site, value in self.dense_rows
            },
        }
        if observed != self._metadata:
            raise RuntimeError("shared compiled-program tensors changed")
        return self.sha256

    def verify_exact(self) -> str:
        self.verify_metadata()
        if tuple(
            (site, tensor_content_sha256(value)) for site, value in self.dense_rows
        ) != self.table_sha256s or tensor_content_sha256(
            self.covered_token_ids
        ) != self._manifest["covered_token_ids_sha256"] or tensor_content_sha256(
            self.output_nearest_covered_index
        ) != self._manifest["output_nearest_covered_index_sha256"]:
            raise RuntimeError("shared compiled-program content changed")
        return self.sha256


class Bilin18CutRankBackend:
    """Stateful production capability consumed only by the thin transaction adapter."""

    source_paths = SOURCE_PATHS

    def __init__(
        self, *, dimensions: ProgramDimensions = PRODUCTION_DIMENSIONS,
        device: str | torch.device = "cuda", fit_row_receipt: Path = FIT_ROW_RECEIPT,
        fit_row_role: str = FIT_ROW_ROLE,
        model_loader: Callable[[], tuple[torch.nn.Module, adapter.ModelBinding]] | None = None,
        fit_wave_loader: Callable[[], adapter.RowWave] | None = None,
        program_builder: Callable[..., SharedProgram] | None = None,
    ) -> None:
        self.dimensions = dimensions
        self.device = torch.device(device)
        self.fit_row_receipt = Path(fit_row_receipt)
        self.fit_row_role = fit_row_role
        self.batch_size = dimensions.eval_batch_size
        self._model_loader = model_loader
        self._fit_wave_loader = fit_wave_loader
        self._program_builder = program_builder
        self._model: torch.nn.Module | None = None
        self._model_binding: adapter.ModelBinding | None = None
        self._model_guard: ModelTreeGuard | None = None
        self._program: SharedProgram | None = None
        self._prepared_bank: adapter.PreparedProgramBank | None = None
        self._evaluation_rows_sha256: str | None = None
        self._active_handles: list[Any] = []
        self._closed = False

    def _production_model_loader(self) -> tuple[torch.nn.Module, adapter.ModelBinding]:
        model, receipt = facade.load_bilin18(
            device=self.device, dtype=torch.float32, verify_weights_sha256=True,
        )
        implementation = _logical_sha256({
            path: adapter.file_sha256(adapter.REPO / path) for path in SOURCE_PATHS
        })
        realization = _logical_sha256({
            "config_sha256": receipt.config_sha256,
            "weights_sha256": receipt.weights_sha256,
            "implementation_sha256": implementation,
        })
        component = model_tree_sha256(model)
        return model, adapter.ModelBinding(
            config_sha256=receipt.config_sha256,
            weights_sha256=receipt.weights_sha256,
            implementation_sha256=implementation,
            model_realization_sha256=realization,
            component_tree_sha256=component,
        )

    def _load_fit_wave(self) -> adapter.RowWave:
        if self._fit_wave_loader is not None:
            return self._fit_wave_loader()
        return adapter.load_row_wave(self.fit_row_receipt, self.fit_row_role)

    def _module(self, site: measurement.cut.Site) -> torch.nn.Module:
        if self._model is None:
            raise RuntimeError("bilin18 backend model is not prepared")
        block = self._model.transformer.h[site[1]]
        return block.attn if site[0] == "attn" else block.mlp

    @torch.no_grad()
    def _forward_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("bilin18 backend model is not prepared")
        width = self.dimensions.model_width
        x = F.rms_norm(self._model.transformer.wte(tokens), (width,))
        x0 = x
        first_value = None
        for block in self._model.transformer.h:
            x, first_value = block(x, first_value, x0)
        return 30.0 * torch.tanh(
            self._model.lm_head(F.rms_norm(x, (width,))) / 30.0
        ).float()

    @torch.no_grad()
    def _build_program(self, fit_wave: adapter.RowWave, source_sha256: str) -> SharedProgram:
        dimensions = self.dimensions
        if self._model is None or self._model_binding is None:
            raise RuntimeError("model must be loaded before program construction")
        fit = fit_wave.clone_rows()
        seen = torch.zeros(dimensions.tokenizer_vocab, dtype=torch.bool)
        seen[fit[:, :adapter.INPUT_STOP].reshape(-1)] = True
        covered_cpu = seen.nonzero(as_tuple=True)[0].contiguous()
        if len(covered_cpu) != dimensions.expected_covered_token_count:
            raise RuntimeError("fit-row covered-token census differs from section 1786")
        uncovered_cpu = (~seen).nonzero(as_tuple=True)[0].contiguous()
        covered = covered_cpu.to(self.device)
        uncovered = uncovered_cpu.to(self.device)

        # Frozen output-distribution nearest-neighbour realization.  It is the
        # registered fallback control; the learned rank-64 map executes below.
        covered_probability = torch.empty(
            (len(covered), dimensions.logit_vocab), device=self.device,
            dtype=torch.float16,
        )
        for start in range(0, len(covered), dimensions.build_batch_size):
            token = covered[start:start + dimensions.build_batch_size].unsqueeze(1)
            logits = self._forward_logits(token)[:, 0]
            probability = torch.softmax(torch.log_softmax(logits.float(), -1), -1)
            covered_probability[start:start + len(token)] = (
                probability / probability.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            ).half()
        nearest = torch.empty(
            dimensions.tokenizer_vocab, device=self.device, dtype=torch.long,
        )
        nearest[covered] = torch.arange(len(covered), device=self.device)
        for start in range(0, len(uncovered), 512):
            token = uncovered[start:start + 512]
            logits = self._forward_logits(token.unsqueeze(1))[:, 0]
            probability = torch.softmax(torch.log_softmax(logits.float(), -1), -1)
            probability = probability / probability.norm(
                dim=-1, keepdim=True,
            ).clamp_min(1e-9)
            nearest[token] = (probability.half() @ covered_probability.T).float().argmax(-1)
        del covered_probability

        tables = {
            site: torch.empty(
                (len(covered), dimensions.model_width), device=self.device,
                dtype=torch.float32,
            )
            for site in adapter.ALL_NATIVE_SITES
        }
        captured: dict[measurement.cut.Site, torch.Tensor] = {}

        def capture(site: measurement.cut.Site):
            def hook(_module, _arguments, output):
                value = output[0] if isinstance(output, tuple) else output
                captured[site] = value[:, 0].detach().float()
                return None
            return hook

        handles = [
            self._module(site).register_forward_hook(capture(site))
            for site in adapter.ALL_NATIVE_SITES
        ]
        try:
            for start in range(0, len(covered), dimensions.build_batch_size):
                token = covered[start:start + dimensions.build_batch_size].unsqueeze(1)
                self._forward_logits(token)
                if set(captured) != set(adapter.ALL_NATIVE_SITES):
                    raise RuntimeError("context-free capture missed a native site")
                for site in adapter.ALL_NATIVE_SITES:
                    tables[site][start:start + len(token)] = captured[site]
                captured.clear()
        finally:
            for handle in handles:
                handle.remove()

        embedding = self._model.transformer.wte.weight.detach().float()
        covered_embedding = embedding[covered]
        dense_rows = []
        for site in adapter.ALL_NATIVE_SITES:
            truncated = centered_rank_truncate(tables.pop(site), dimensions.table_rank)
            coefficient = rank_truncated_ridge_map(
                covered_embedding, truncated, rank=dimensions.map_rank,
                ridge=dimensions.ridge,
            )
            dense = materialize_token_rows(
                token_embeddings=embedding, covered_token_ids=covered,
                covered_rows=truncated, coefficient=coefficient,
            )
            dense_rows.append((site, dense))
            del truncated, coefficient
        return SharedProgram(
            dimensions=dimensions,
            fit_wave_receipt=fit_wave.receipt(),
            covered_token_ids=covered,
            output_nearest_covered_index=nearest,
            dense_rows=tuple(dense_rows),
            model_realization_sha256=self._model_binding.model_realization_sha256,
            program_source_sha256=source_sha256,
        )

    def prepare(
        self, rows: torch.Tensor,
        requests: tuple[measurement.MeasurementRequest, ...],
    ) -> adapter.PreparedProgramBank:
        if self._closed or self._model is not None or self._program is not None or (
            tuple(requests) != measurement.REQUESTS
        ) or not torch.is_tensor(rows) or rows.dtype != torch.long or rows.device.type != (
            "cpu"
        ) or rows.ndim != 2 or rows.shape[1] != adapter.TARGET_STOP or not (
            rows.is_contiguous()
        ):
            raise RuntimeError("bilin18 backend prepare is non-pristine or malformed")
        loader = self._model_loader or self._production_model_loader
        model, binding = loader()
        if not isinstance(model, torch.nn.Module) or not isinstance(
            binding, adapter.ModelBinding
        ):
            raise RuntimeError("bilin18 model loader returned an untyped realization")
        self._model = model
        self._model_binding = binding
        self._model_guard = ModelTreeGuard(
            model, initial_sha256=binding.component_tree_sha256,
        )
        if self._model_guard.sha256 != binding.component_tree_sha256:
            raise RuntimeError("model binding component hash differs from loaded model")
        fit_wave = self._load_fit_wave()
        if not isinstance(fit_wave, adapter.RowWave):
            raise RuntimeError("fit-row loader returned an untyped wave")
        program_source_sha256 = adapter.file_sha256(Path(__file__).resolve())
        builder = self._program_builder or self._build_program
        program = builder(fit_wave, program_source_sha256)
        if not isinstance(program, SharedProgram) or program.dimensions != self.dimensions or (
            program.manifest["gain_policy"] != adapter.GAIN_POLICY
        ) or program.manifest["model_realization_sha256"] != binding.model_realization_sha256:
            raise RuntimeError("program builder returned a mismatched shared realization")
        self._program = program
        self._evaluation_rows_sha256 = measurement.tensor_sha256(rows)
        descriptors = tuple(
            adapter.ProgramDescriptor(
                ordinal=request.ordinal,
                request_sha256=request.sha256,
                installed_compiled_sites=adapter._canonical_sites((
                    *request.always_compiled_sites, *request.additional_sites,
                )),
                live_attention_gain_sites=(),
                shared_program_state_sha256=program.sha256,
                cell_program_state_sha256=_logical_sha256({
                    "shared_program_state_sha256": program.sha256,
                    "request_sha256": request.sha256,
                    "installed_compiled_sites": [list(site) for site in adapter._canonical_sites((
                        *request.always_compiled_sites, *request.additional_sites,
                    ))],
                    "gain_policy": adapter.GAIN_POLICY,
                }),
                program_source_sha256=program_source_sha256,
            )
            for request in requests
        )
        self._prepared_bank = adapter.PreparedProgramBank(
            model=binding, programs=descriptors,
        )
        return self._prepared_bank

    @torch.no_grad()
    def execute_cell(
        self, request: measurement.MeasurementRequest, rows: torch.Tensor,
        program: adapter.ProgramDescriptor,
    ) -> adapter.BackendCellResult:
        if self._closed or self._model is None or self._model_guard is None or (
            self._program is None or self._prepared_bank is None
        ) or not isinstance(request, measurement.MeasurementRequest) or not isinstance(
            program, adapter.ProgramDescriptor
        ) or program != self._prepared_bank.programs[request.ordinal] or (
            program.shared_program_state_sha256 != self._program.sha256
        ) or not torch.is_tensor(rows) or rows.dtype != torch.long or rows.device.type != (
            "cpu"
        ) or not rows.is_contiguous() or measurement.tensor_sha256(rows) != (
            self._evaluation_rows_sha256
        ) or self._active_handles:
            raise RuntimeError("bilin18 cell request/program/rows/state differs from prepare")
        before = self._model_guard.verify_metadata()
        self._program.verify_metadata()
        native_counts = {site: 0 for site in adapter.ALL_NATIVE_SITES}
        substitution_counts = {site: 0 for site in program.installed_compiled_sites}
        current_tokens: torch.Tensor | None = None

        def count_native(site: measurement.cut.Site):
            def hook(_module, _arguments, _output):
                native_counts[site] += 1
                return None
            return hook

        def substitute(site: measurement.cut.Site):
            table = self._program.rows_for(site)

            def hook(_module, _arguments, output):
                nonlocal current_tokens
                if current_tokens is None:
                    raise RuntimeError("substitution hook fired outside a bound batch")
                value = output[0] if isinstance(output, tuple) else output
                replacement = table[current_tokens.reshape(-1)].reshape(value.shape).to(
                    value.dtype
                )
                substitution_counts[site] += 1
                if isinstance(output, tuple):
                    return (replacement, *output[1:])
                return replacement
            return hook

        handles = [
            self._module(site).register_forward_hook(count_native(site))
            for site in adapter.ALL_NATIVE_SITES
        ]
        handles.extend(
            self._module(site).register_forward_hook(substitute(site))
            for site in program.installed_compiled_sites
        )
        self._active_handles = handles
        top1 = torch.empty(len(rows), dtype=torch.long)
        ce = torch.empty(len(rows), dtype=torch.float64)
        returned = 0
        try:
            for start in range(0, len(rows), self.batch_size):
                batch = rows[start:start + self.batch_size]
                current_tokens = batch[:, :adapter.INPUT_STOP].to(
                    self.device
                ).contiguous()
                target = batch[:, adapter.TARGET_START:adapter.TARGET_STOP].to(self.device)
                logits = self._forward_logits(current_tokens)[:, adapter.SCORE_START:adapter.SCORE_STOP]
                if tuple(logits.shape[:2]) != tuple(target.shape):
                    raise RuntimeError("bilin18 scored logit/target support changed")
                prediction = logits.argmax(dim=-1)
                top1[start:start + len(batch)] = (prediction == target).sum(
                    dim=1
                ).to(device="cpu", dtype=torch.long)
                losses = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1),
                    reduction="none",
                ).reshape(target.shape).double()
                ce[start:start + len(batch)] = losses.sum(dim=1).cpu()
                returned += 1
                del logits, prediction, losses, target
                current_tokens = None
        finally:
            current_tokens = None
            for handle in reversed(handles):
                handle.remove()
            self._active_handles.clear()
        batch_count = (len(rows) + self.batch_size - 1) // self.batch_size
        if returned != batch_count or native_counts != {
            site: batch_count for site in adapter.ALL_NATIVE_SITES
        } or substitution_counts != {
            site: batch_count for site in program.installed_compiled_sites
        }:
            raise RuntimeError("bilin18 physical hook call census changed")
        after = self._model_guard.verify_metadata()
        self._program.verify_metadata()
        token_count = torch.full(
            (len(rows),), adapter.SCORE_STOP - adapter.SCORE_START,
            dtype=torch.long,
        )
        statistics = measurement.RowCellSufficientStatistics(
            top1_correct=top1.contiguous(), ce_sum=ce.contiguous(),
            row_token_count=token_count,
        )
        ledger = adapter.CellCallLedger(
            ordinal=request.ordinal,
            request_sha256=request.sha256,
            program_realization_sha256=program.sha256,
            execution_mode=adapter.EXECUTION_MODE,
            row_count=len(rows),
            scored_token_count=len(rows) * (adapter.SCORE_STOP - adapter.SCORE_START),
            batch_count=batch_count,
            outer_forward_count=batch_count,
            outer_returned_count=returned,
            native_module_calls=tuple(
                (site, native_counts[site]) for site in adapter.ALL_NATIVE_SITES
            ),
            substitution_calls=tuple(
                (site, substitution_counts[site]) for site in program.installed_compiled_sites
            ),
            live_attention_gain_calls=(),
            fitter_calls=0,
            retained_logits=0,
        )
        return adapter.BackendCellResult(
            statistics=statistics, call_ledger=ledger,
            component_tree_before_sha256=before,
            component_tree_after_sha256=after,
        )

    def close(self) -> str:
        if self._closed:
            # The transaction adapter may make one best-effort cleanup call after
            # a failed validating close.  Empty is never an admissible successful
            # close receipt, but lets the original failure remain the causal error.
            return ""
        self._closed = True
        if self._active_handles:
            for handle in reversed(self._active_handles):
                handle.remove()
            self._active_handles.clear()
            raise RuntimeError("bilin18 backend closed with active hooks")
        if self._model_guard is None:
            # Cleanup after a pre-model failure has no component identity to return.
            return ""
        component = self._model_guard.verify_exact()
        if self._program is not None:
            self._program.verify_exact()
        self._prepared_bank = None
        self._program = None
        self._model_guard = None
        self._model_binding = None
        self._model = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return component


def create_backend() -> Bilin18CutRankBackend:
    """Return a lazy production capability; no external state is touched here."""

    return Bilin18CutRankBackend()
