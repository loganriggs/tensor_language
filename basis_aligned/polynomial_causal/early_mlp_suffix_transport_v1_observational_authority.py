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
DENOMINATOR_LEDGER_KEY = "initial_denominator_pass"
DENOMINATOR_AUTHORITY_KEY = "initial_denominator_pass_authority"
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


def denominator_pass_payload(value: fit.DenominatorPass) -> dict[str, Any]:
    """Canonical weights-only representation for the protected fit ledger.

    This owns only the denominator namespace.  The fit-stage owner may add other
    ledger entries, but the protected receipt must bind this exact child payload.
    """

    if not isinstance(value, fit.DenominatorPass):
        raise TypeError("denominator payload requires a typed pass")
    identity = value.sha256
    return {
        "schema_version": 1,
        "kind": "early_mlp_suffix_transport_v1_denominator_pass",
        "site_records": tuple({
            key: (
                item.detach().cpu().contiguous().clone()
                if torch.is_tensor(item) else item
            ) for key, item in record.items()
        } for record in value.site_records),
        "transaction_history_sha256": value.transaction_history_sha256,
        "completed_steps": value.completed_steps,
        "denominator_pass_sha256": identity,
    }


def _restore_denominator_pass(value: Any) -> fit.DenominatorPass:
    keys = {
        "schema_version", "kind", "site_records", "transaction_history_sha256",
        "completed_steps", "denominator_pass_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != keys or value.get(
        "schema_version"
    ) != 1 or value.get("kind") != (
        "early_mlp_suffix_transport_v1_denominator_pass"
    ) or not runtime._sha256_text(value.get("denominator_pass_sha256")) or not isinstance(
        value.get("site_records"), (tuple, list)
    ) or len(value["site_records"]) != 2:
        raise RuntimeError("protected denominator payload schema changed")
    result = fit.DenominatorPass(
        site_records=tuple(dict(record) for record in value["site_records"]),
        transaction_history_sha256=value["transaction_history_sha256"],
        completed_steps=value["completed_steps"],
    )
    if result.sha256 != value["denominator_pass_sha256"]:
        raise RuntimeError("protected denominator pass identity changed")
    return result


def denominator_pass_authority_payload(
    value: fit.DenominatorPass, *, paths: lifecycle.ArtifactPaths,
) -> dict[str, Any]:
    """Fit-owner child receipt to publish after its ledger and manifest exist."""

    if not paths.fit_ledger.is_file() or not paths.fit_manifest.is_file():
        raise RuntimeError("denominator authority binding targets are absent")
    return {
        "schema_version": 1,
        "kind": "early_mlp_suffix_transport_v1_denominator_pass_authority",
        "ledger_key": DENOMINATOR_LEDGER_KEY,
        "fit_ledger": lifecycle.artifact_binding(paths.fit_ledger),
        "fit_manifest": lifecycle.artifact_binding(paths.fit_manifest),
        "denominator_pass_sha256": value.sha256,
    }


def _protected_binding(
    protected: Mapping[str, Any], path: Path,
) -> Mapping[str, Any]:
    key = str(path.resolve())
    binding = protected.get(key)
    expected = lifecycle.artifact_binding(path)
    if not isinstance(binding, Mapping) or dict(binding) != expected:
        raise RuntimeError(f"protected fit authority omitted or changed {path.name}")
    return binding


def load_protected_denominator_pass(
    *, paths: lifecycle.ArtifactPaths = lifecycle.PATHS,
) -> fit.DenominatorPass:
    """Reconstruct the denominator pass from the program-unlock snapshot."""

    if lifecycle._FINAL_ROLE_LOADS != 0:
        raise RuntimeError("protected denominators must load before the final role")
    unlock = lifecycle.load_programs_unlock(paths)
    protected = unlock.get("protected_before")
    if not isinstance(protected, Mapping):
        raise RuntimeError("program unlock lacks its protected fit snapshot")
    for path in (paths.fit_ledger, paths.fit_manifest, paths.fit_receipt):
        _protected_binding(protected, path)
    lifecycle.require_protected_snapshot(
        tuple(Path(key) for key in protected), protected,
    )
    receipt = json.loads(paths.fit_receipt.read_text())
    child = receipt.get(DENOMINATOR_AUTHORITY_KEY) if isinstance(receipt, Mapping) else None
    required = {
        "schema_version": 1,
        "kind": "early_mlp_suffix_transport_v1_denominator_pass_authority",
        "ledger_key": DENOMINATOR_LEDGER_KEY,
        "fit_ledger": lifecycle.artifact_binding(paths.fit_ledger),
        "fit_manifest": lifecycle.artifact_binding(paths.fit_manifest),
    }
    if not isinstance(child, Mapping) or set(child) != set(required) | {
        "denominator_pass_sha256"
    } or any(child.get(key) != expected for key, expected in required.items()) or not (
        runtime._sha256_text(child.get("denominator_pass_sha256"))
    ):
        raise RuntimeError("fit receipt denominator authority changed")
    ledger = torch.load(paths.fit_ledger, map_location="cpu", weights_only=True)
    if not isinstance(ledger, Mapping) or DENOMINATOR_LEDGER_KEY not in ledger:
        raise RuntimeError("fit ledger omits the protected denominator pass")
    result = _restore_denominator_pass(ledger[DENOMINATOR_LEDGER_KEY])
    if result.sha256 != child["denominator_pass_sha256"]:
        raise RuntimeError("fit ledger and receipt denominator identities differ")
    return result


def identity_teacher_mapping_sha256(rows_receipt_sha256: str) -> str:
    """Identity-map semantic nonce shared by fit/validation/final contexts."""

    _sha256("rows receipt", rows_receipt_sha256)
    return runtime.logical_identity_sha256({
        "kind": "early_mlp_suffix_transport_v1_identity_teacher_mapping",
        "rows_receipt_sha256": rows_receipt_sha256,
        "mapping": "same ordered source rows and targets",
    })


def reconstruct_final_run_context(
    *, bindings: Mapping[str, Any], attempt: Mapping[str, Any],
    inherited_initialization: inherited.LoadedInitialization,
) -> capabilities.FinalRunContext:
    """Derive the final broker context from terminal and inherited authorities."""

    if not isinstance(bindings, Mapping) or not isinstance(attempt, Mapping) or not isinstance(
        inherited_initialization, inherited.LoadedInitialization
    ):
        raise TypeError("final context reconstruction requires typed authorities")
    rows_binding = bindings.get("rows_receipt")
    attempt_rows = attempt.get("rows_receipt")
    final_cache = attempt.get("final_cache")
    source_commit = bindings.get("source_commit")
    if not isinstance(rows_binding, Mapping) or dict(rows_binding) != attempt_rows or not isinstance(
        final_cache, Mapping
    ) or final_cache.get("shape_full") != [capabilities.FINAL_ROW_COUNT, FIT_ROW_WIDTH] or (
        attempt.get("source_commit") != source_commit
    ) or not isinstance(source_commit, str) or len(source_commit) != 40 or any(
        char not in "0123456789abcdef" for char in source_commit
    ):
        raise RuntimeError("terminal final-context authorities disagree")
    rows_receipt_sha256 = rows_binding.get("sha256")
    final_role_sha256 = final_cache.get("tensor_full_raw_sha256")
    _sha256("terminal rows receipt", rows_receipt_sha256)
    _sha256("terminal final role", final_role_sha256)
    snapshot_sha256 = inherited_initialization.authority.snapshot_sha256
    _sha256("inherited snapshot", snapshot_sha256)
    return capabilities.FinalRunContext(
        source_commit=source_commit,
        inherited_snapshot_sha256=snapshot_sha256,
        rows_receipt_sha256=rows_receipt_sha256,
        final_role_tensor_sha256=final_role_sha256,
        identity_teacher_mapping_sha256=identity_teacher_mapping_sha256(
            rows_receipt_sha256
        ),
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
