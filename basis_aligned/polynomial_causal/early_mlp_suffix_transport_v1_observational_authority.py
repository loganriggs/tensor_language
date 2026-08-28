"""Reduced fit-frequency loader and one-shot final observational factory.

The fit cache is opened only behind the frozen row receipt and immediately reduced
to a length-50257 target-count vector.  No fit row can escape.  Final rows are never
loaded here: production must receive the single tensor deserialized by
``final_execution.execute_final`` and may spend it once to construct the existing
source-closed observational executor.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import torch

import bilin18_observed_adapter as observed
import early_mlp_suffix_transport_v1_capabilities as capabilities
import early_mlp_suffix_transport_v1_fit as fit
import early_mlp_suffix_transport_v1_inherited as inherited
import early_mlp_suffix_transport_v1_lifecycle as lifecycle
import early_mlp_suffix_transport_v1_observational_execution as execution
import early_mlp_suffix_transport_v1_runtime as runtime


FIT_ROLE = lifecycle.ROLE_NAMES[0]
FIT_ROW_COUNT = capabilities.FIT_ROW_COUNT
FIT_ROW_WIDTH = 513
_MINT_TOKEN = object()


def _sha256(name: str, value: Any) -> str:
    if not runtime._sha256_text(value):
        raise ValueError(f"{name} must be a SHA-256 identity")
    return value


@dataclass(frozen=True, slots=True)
class FitTokenCountAuthorityReceipt:
    rows_receipt_sha256: str
    rows_manifest_sha256: str
    fit_cache_file_sha256: str
    fit_role_tensor_sha256: str
    fit_token_counts_sha256: str
    fit_target_count: int
    rule_sha256: str
    authority_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "rows_receipt_sha256", "rows_manifest_sha256",
            "fit_cache_file_sha256", "fit_role_tensor_sha256",
            "fit_token_counts_sha256", "rule_sha256", "authority_sha256",
        ):
            _sha256(name, getattr(self, name))
        if self.fit_target_count != FIT_ROW_COUNT * (
            runtime.SCORE_STOP - runtime.SCORE_START
        ):
            raise ValueError("fit token-count authority support changed")
        body = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__ if name != "authority_sha256"
        }
        if runtime.logical_identity_sha256(body) != self.authority_sha256:
            raise ValueError("fit token-count authority identity changed")

    @property
    def sha256(self) -> str:
        return self.authority_sha256


class LoadedFitTokenCountAuthority:
    """Nonserializable, one-use reduced capability; raw fit rows are absent."""

    __slots__ = ("__counts", "__receipt", "__spent", "__failed")

    def __init__(
        self, *, _token: object, counts: torch.Tensor,
        receipt: FitTokenCountAuthorityReceipt,
    ) -> None:
        if _token is not _MINT_TOKEN or type(receipt) is not FitTokenCountAuthorityReceipt:
            raise TypeError("fit token-count authority must be minted by its loader")
        if not torch.is_tensor(counts) or counts.dtype != torch.long or tuple(
            counts.shape
        ) != (execution.TOKEN_VOCAB,) or counts.device.type != "cpu" or (
            counts.requires_grad
        ) or bool((counts < 0).any()) or int(counts.sum()) != receipt.fit_target_count or (
            runtime.tensor_identity_sha256(counts) != receipt.fit_token_counts_sha256
        ):
            raise ValueError("loaded fit token-count reduction changed")
        self.__counts = counts.detach().clone().contiguous()
        self.__receipt = receipt
        self.__spent = False
        self.__failed = False

    def __copy__(self):
        raise RuntimeError("fit token-count authorities cannot be copied")

    def __deepcopy__(self, memo):
        raise RuntimeError("fit token-count authorities cannot be copied")

    def __reduce__(self):
        raise RuntimeError("fit token-count authorities cannot be serialized")

    @property
    def receipt(self) -> FitTokenCountAuthorityReceipt:
        return self.__receipt

    @property
    def spent(self) -> bool:
        return self.__spent

    def make_final_plan(
        self, *, final_rows: torch.Tensor,
        final_context: capabilities.FinalRunContext,
    ) -> execution.FinalFrequencyPlan:
        if self.__spent or self.__failed:
            raise RuntimeError("fit token-count authority is already closed")
        self.__spent = True
        try:
            plan = execution.FinalFrequencyPlan(
                fit_token_counts=self.__counts,
                fit_token_counts_sha256=self.__receipt.fit_token_counts_sha256,
                source_authority_sha256=self.__receipt.sha256,
                final_rows=final_rows,
                final_role_tensor_sha256=final_context.final_role_tensor_sha256,
            )
        except BaseException:
            self.__failed = True
            self.__counts.zero_()
            raise
        self.__counts.zero_()
        return plan


def load_fit_token_count_authority(
    *, paths: lifecycle.ArtifactPaths = lifecycle.PATHS,
) -> LoadedFitTokenCountAuthority:
    """Validate the frozen fit cache and release only its target-count reduction."""

    if lifecycle._FINAL_ROLE_LOADS != 0:
        raise RuntimeError("fit frequency authority must load before the final role")
    if not paths.rows_receipt.is_file() or not paths.rows_manifest.is_file():
        raise RuntimeError("fit frequency row authority is absent")
    receipt = json.loads(paths.rows_receipt.read_text())
    lifecycle._validate_rows_receipt(receipt, paths)
    entry = receipt.get("entries", {}).get(FIT_ROLE)
    if not isinstance(entry, Mapping) or entry.get("shape_full") != [
        FIT_ROW_COUNT, FIT_ROW_WIDTH
    ]:
        raise RuntimeError("fit frequency role entry changed")
    cache_path = Path(entry.get("cache_path", ""))
    if not cache_path.is_file() or lifecycle.file_sha256(cache_path) != entry.get(
        "cache_file_sha256"
    ):
        raise RuntimeError("fit frequency cache binding changed")
    value = torch.load(cache_path, map_location="cpu", weights_only=True)
    rows = value.get("rows") if isinstance(value, Mapping) else value
    if not torch.is_tensor(rows) or rows.dtype != torch.long or tuple(rows.shape) != (
        FIT_ROW_COUNT, FIT_ROW_WIDTH
    ) or rows.device.type != "cpu" or not rows.is_contiguous() or (
        lifecycle.tensor_sha256(rows) != entry.get("tensor_full_raw_sha256")
    ):
        raise RuntimeError("fit frequency role tensor changed")
    targets = rows[:, runtime.SCORE_START + 1:runtime.SCORE_STOP + 1].contiguous()
    if bool((targets < 0).any()) or bool((targets >= execution.TOKEN_VOCAB).any()):
        raise RuntimeError("fit frequency target is outside the GPT-2 vocabulary")
    counts = torch.bincount(
        targets.flatten(), minlength=execution.TOKEN_VOCAB,
    ).long().contiguous()
    rows = targets = value = None
    rule_sha256 = runtime.logical_identity_sha256({
        "source_role": FIT_ROLE,
        "fit_target_columns": [runtime.SCORE_START + 1, runtime.SCORE_STOP + 1],
        "vocabulary_size": execution.TOKEN_VOCAB,
        "boundaries": list(execution.TOKEN_FREQUENCY_BOUNDARIES),
        "bucketize_right": True,
    })
    body = {
        "rows_receipt_sha256": lifecycle.file_sha256(paths.rows_receipt),
        "rows_manifest_sha256": lifecycle.file_sha256(paths.rows_manifest),
        "fit_cache_file_sha256": entry["cache_file_sha256"],
        "fit_role_tensor_sha256": entry["tensor_full_raw_sha256"],
        "fit_token_counts_sha256": runtime.tensor_identity_sha256(counts),
        "fit_target_count": int(counts.sum()),
        "rule_sha256": rule_sha256,
    }
    reduced_receipt = FitTokenCountAuthorityReceipt(
        **body, authority_sha256=runtime.logical_identity_sha256(body),
    )
    return LoadedFitTokenCountAuthority(
        _token=_MINT_TOKEN, counts=counts, receipt=reduced_receipt,
    )


class FinalObservationalExecutorFactory:
    """Spend one reduced fit authority on the lifecycle-delivered final tensor."""

    def __init__(
        self, *, final_context: capabilities.FinalRunContext,
        inherited_initialization: inherited.LoadedInitialization,
        denominator_pass: fit.DenominatorPass,
        frequency_authority: LoadedFitTokenCountAuthority,
    ) -> None:
        if not isinstance(final_context, capabilities.FinalRunContext) or not isinstance(
            inherited_initialization, inherited.LoadedInitialization
        ) or not isinstance(denominator_pass, fit.DenominatorPass) or not isinstance(
            frequency_authority, LoadedFitTokenCountAuthority
        ):
            raise TypeError("observational executor factory requires typed authorities")
        self._context = final_context
        self._inherited = inherited_initialization
        self._denominator = denominator_pass
        self._frequency = frequency_authority
        self._spent = False
        self._failed = False

    def build(
        self, *, adapter: observed.ObservedBilin18Adapter,
        final_rows: torch.Tensor, validated_program_bank: Mapping[str, Any],
    ) -> execution.FinalObservationalBatchExecutor:
        if self._spent or self._failed:
            raise RuntimeError("observational executor factory is already closed")
        if lifecycle._FINAL_ROLE_LOADS != 1:
            self._failed = True
            raise RuntimeError(
                "observational executor requires the one licensed final-role load"
            )
        if not isinstance(adapter, observed.ObservedBilin18Adapter):
            self._failed = True
            raise TypeError("observational executor factory requires the real adapter")
        self._spent = True
        try:
            plan = self._frequency.make_final_plan(
                final_rows=final_rows, final_context=self._context,
            )
            result = execution.FinalObservationalBatchExecutor(
                adapter=adapter, validated_program_bank=validated_program_bank,
                inherited_initialization=self._inherited,
                final_context=self._context, final_rows=final_rows,
                denominator_pass=self._denominator, frequency_plan=plan,
            )
        except BaseException:
            self._failed = True
            raise
        self._inherited = self._denominator = self._frequency = None
        return result
