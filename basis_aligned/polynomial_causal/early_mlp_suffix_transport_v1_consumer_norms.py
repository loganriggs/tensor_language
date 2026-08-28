"""Typed live-attention-consumer norm boundary for suffix final execution.

The implementation is deliberately separated from model/data loading.  An observed
owner wraps an already licensed forward in :class:`AttentionConsumerOutputCapture`,
then pairs its four-row scalar magnitudes with an O/O capture from the same
background.  Raw writes are reduced inside the output hooks and never escape.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch

import early_mlp_suffix_transport_v1_final_capability as final_capability
import early_mlp_suffix_transport_v1_runtime as runtime


MODEL_LAYER_COUNT = 18
MODEL_WIDTH = 1152
SEQUENCE_LENGTH = 256
SCORE_START = 64
SCORE_STOP = 256
BATCH_SIZE = 4
BATCH_COUNT = 48
DENOMINATOR_FLOOR = 1e-12
NATIVE_RATIO_TOLERANCE = 2e-6
_MINT_TOKEN = object()
_PAIR_TOKEN = object()


def _sha256(name: str, value: Any) -> str:
    if not runtime._sha256_text(value):
        raise ValueError(f"{name} must be a SHA-256 identity")
    return value


def _batch_vector(name: str, value: Any) -> torch.Tensor:
    if not torch.is_tensor(value) or tuple(value.shape) != (BATCH_SIZE,) or (
        value.dtype != torch.float64 or value.device.type != "cpu"
    ) or value.requires_grad or not bool(torch.isfinite(value).all()) or bool(
        (value < 0).any()
    ):
        raise ValueError(f"{name} is not an allowed consumer row scalar")
    return value.detach().clone().contiguous()


def _model_components(model: Any) -> tuple[torch.nn.Module, ...]:
    try:
        blocks = tuple(model.transformer.h)
        components = tuple(block.attn.c_proj for block in blocks)
    except (AttributeError, TypeError) as error:
        raise TypeError("consumer capture requires transformer.h[*].attn.c_proj") from error
    if len(components) != MODEL_LAYER_COUNT or any(
        not isinstance(component, torch.nn.Module) for component in components
    ):
        raise RuntimeError("consumer capture requires exactly 18 c_proj modules")
    return components


def _component_identity(
    components: tuple[torch.nn.Module, ...], model_identity_sha256: str,
) -> str:
    return runtime.logical_identity_sha256({
        "model_identity_sha256": model_identity_sha256,
        "consumers": [{
            "layer": layer,
            "qualified_name": f"transformer.h.{layer}.attn.c_proj",
            "module_type": (
                f"{type(component).__module__}.{type(component).__qualname__}"
            ),
            "capture": "forward_output_including_native_bias",
        } for layer, component in enumerate(components)],
    })


@dataclass(frozen=True, slots=True)
class ConsumerCaptureReceipt:
    action_key: str
    action_sha256: str
    action_identity_sha256: str
    model_identity_sha256: str
    component_identity_sha256: str
    common_support_sha256: str
    batch_ordinal: int
    magnitude_sha256: str
    hook_calls: tuple[tuple[int, int], ...]
    hooks_removed: bool
    hooks_inert: bool

    def __post_init__(self) -> None:
        for name in (
            "action_sha256", "action_identity_sha256", "model_identity_sha256",
            "component_identity_sha256", "common_support_sha256", "magnitude_sha256",
        ):
            _sha256(name, getattr(self, name))
        if self.action_key not in final_capability.CANONICAL_ACTION_KEYS or type(
            self.batch_ordinal
        ) is not int or not 0 <= self.batch_ordinal < BATCH_COUNT or self.hook_calls != tuple(
            (layer, 1) for layer in range(MODEL_LAYER_COUNT)
        ) or self.hooks_removed is not True or self.hooks_inert is not True:
            raise ValueError("consumer capture did not close exactly")

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256(asdict(self))


class _CapturedConsumerMagnitudes:
    """Private scalar-only bridge from output hooks to the paired reducer."""

    __slots__ = ("__action", "__magnitudes", "__receipt", "__spent")

    def __init__(
        self, *, token: object, action: final_capability.FinalAction,
        magnitudes: torch.Tensor, receipt: ConsumerCaptureReceipt,
    ) -> None:
        if token is not _MINT_TOKEN or type(action) is not final_capability.FinalAction or (
            type(receipt) is not ConsumerCaptureReceipt
        ) or receipt.action_key != action.key or receipt.action_sha256 != action.sha256 or (
            not torch.is_tensor(magnitudes)
        ) or tuple(magnitudes.shape) != (MODEL_LAYER_COUNT, BATCH_SIZE) or (
            magnitudes.dtype != torch.float64 or magnitudes.device.type != "cpu"
        ) or magnitudes.requires_grad or not bool(torch.isfinite(magnitudes).all()) or bool(
            (magnitudes < 0).any()
        ) or runtime.tensor_identity_sha256(magnitudes) != receipt.magnitude_sha256:
            raise RuntimeError("consumer magnitude capability was not validly minted")
        self.__action = action
        self.__magnitudes = magnitudes.detach().clone().contiguous()
        self.__receipt = receipt
        self.__spent = False

    def __copy__(self):
        raise RuntimeError("consumer magnitudes cannot be copied")

    def __deepcopy__(self, memo):
        raise RuntimeError("consumer magnitudes cannot be copied")

    def __reduce__(self):
        raise RuntimeError("consumer magnitudes cannot be serialized")

    @property
    def action(self) -> final_capability.FinalAction:
        return self.__action

    @property
    def receipt(self) -> ConsumerCaptureReceipt:
        return self.__receipt

    def _take_for_pair(self, token: object) -> torch.Tensor:
        native_denominator = self.__action.arm == "o_o"
        if token is not _PAIR_TOKEN or (
            self.__spent and not native_denominator
        ) or runtime.tensor_identity_sha256(
            self.__magnitudes
        ) != self.__receipt.magnitude_sha256:
            self.__spent = True
            raise RuntimeError("consumer magnitudes were replayed or mutated")
        if native_denominator:
            return self.__magnitudes.clone()
        self.__spent = True
        value = self.__magnitudes
        self.__magnitudes = torch.empty(0, dtype=torch.float64)
        return value


class AttentionConsumerOutputCapture:
    """One-shot context that reduces all 18 live ``c_proj`` outputs in-hook.

    The surrounding observed adapter retains ownership of the forward result.  This
    object retains only a 18-by-4 matrix of mean output norms and exposes it solely
    through the private paired-reduction capability.
    """

    def __init__(
        self, *, model: Any, action: final_capability.FinalAction,
        batch_ordinal: int, common_support_sha256: str,
        expected_action_identity_sha256: str,
        action_identity_reader: Callable[[], str],
        expected_model_identity_sha256: str,
        model_identity_reader: Callable[[], str],
    ) -> None:
        if type(action) is not final_capability.FinalAction or type(
            batch_ordinal
        ) is not int or not 0 <= batch_ordinal < BATCH_COUNT:
            raise ValueError("consumer capture action or batch changed")
        _sha256("consumer support", common_support_sha256)
        _sha256("consumer action identity", expected_action_identity_sha256)
        _sha256("consumer model identity", expected_model_identity_sha256)
        if not callable(action_identity_reader) or not callable(model_identity_reader):
            raise TypeError("consumer capture identity readers must be callable")
        self._model = model
        self._action = action
        self._batch = batch_ordinal
        self._support = common_support_sha256
        self._expected_action = expected_action_identity_sha256
        self._action_reader = action_identity_reader
        self._expected_model = expected_model_identity_sha256
        self._model_reader = model_identity_reader
        self._components: tuple[torch.nn.Module, ...] | None = None
        self._handles: list[Any] = []
        self._magnitudes: list[torch.Tensor | None] = [None] * MODEL_LAYER_COUNT
        self._calls = [0] * MODEL_LAYER_COUNT
        self._active = False
        self._entered = False
        self._closed = False
        self._failed = False
        self._capability: _CapturedConsumerMagnitudes | None = None

    def __enter__(self) -> "AttentionConsumerOutputCapture":
        if self._entered:
            raise RuntimeError("consumer capture cannot be entered twice")
        self._entered = True
        components = _model_components(self._model)
        if self._action_reader() != self._expected_action or (
            self._model_reader() != self._expected_model
        ):
            self._failed = True
            raise RuntimeError("consumer action/model identity changed before forward")
        self._components = components
        self._active = True

        def make_hook(layer: int, expected_component: torch.nn.Module):
            def hook(module: torch.nn.Module, _inputs: Any, output: Any) -> None:
                if not self._active or module is not expected_component:
                    self._failed = True
                    raise RuntimeError("consumer output hook is inactive or rebound")
                self._calls[layer] += 1
                if self._calls[layer] != 1 or not torch.is_tensor(output) or tuple(
                    output.shape
                ) != (BATCH_SIZE, SEQUENCE_LENGTH, MODEL_WIDTH) or not bool(
                    torch.isfinite(output.detach()).all()
                ):
                    self._failed = True
                    raise RuntimeError("consumer output capture is duplicated or malformed")
                # Match the historical layer_norms instrument exactly: the live
                # module output is detached and converted to float32 first.
                value = output.detach().float()[:, SCORE_START:SCORE_STOP]
                magnitude = value.norm(dim=-1).mean(dim=1).cpu().double().contiguous()
                self._magnitudes[layer] = magnitude
                return None
            return hook

        try:
            for layer, component in enumerate(components):
                self._handles.append(component.register_forward_hook(
                    make_hook(layer, component)
                ))
        except BaseException:
            self._active = False
            for handle in reversed(self._handles):
                handle.remove()
            self._failed = True
            raise
        return self

    def __exit__(self, error_type, error, traceback) -> bool:
        self._active = False
        handles = tuple(self._handles)
        components = self._components
        for handle in reversed(handles):
            handle.remove()
        self._handles.clear()
        self._closed = True
        hooks_removed = bool(components is not None) and all(
            handle.id not in component._forward_hooks
            for handle, component in zip(handles, components)
        )
        try:
            identities_stable = self._action_reader() == self._expected_action and (
                self._model_reader() == self._expected_model
            )
            components_stable = components is not None and _model_components(
                self._model
            ) == components
        except BaseException:
            identities_stable = components_stable = False
        exact_calls = tuple(self._calls) == (1,) * MODEL_LAYER_COUNT
        complete = all(value is not None for value in self._magnitudes)
        if error_type is not None or self._failed or not (
            hooks_removed and identities_stable and components_stable and exact_calls and complete
        ):
            self._failed = True
            self._magnitudes = [None] * MODEL_LAYER_COUNT
            if error_type is None:
                raise RuntimeError("consumer output capture did not close exactly")
            return False
        stacked = torch.stack([
            value for value in self._magnitudes if value is not None
        ]).contiguous()
        self._magnitudes = [None] * MODEL_LAYER_COUNT
        receipt = ConsumerCaptureReceipt(
            action_key=self._action.key, action_sha256=self._action.sha256,
            action_identity_sha256=self._expected_action,
            model_identity_sha256=self._expected_model,
            component_identity_sha256=_component_identity(
                components, self._expected_model,
            ),
            common_support_sha256=self._support, batch_ordinal=self._batch,
            magnitude_sha256=runtime.tensor_identity_sha256(stacked),
            hook_calls=tuple(enumerate(self._calls)),
            hooks_removed=hooks_removed, hooks_inert=not self._active,
        )
        self._capability = _CapturedConsumerMagnitudes(
            token=_MINT_TOKEN, action=self._action, magnitudes=stacked,
            receipt=receipt,
        )
        return False

    def finish(self) -> _CapturedConsumerMagnitudes:
        if not self._closed or self._failed or self._capability is None:
            raise RuntimeError("consumer capture is incomplete or failed")
        value = self._capability
        self._capability = None
        return value


@dataclass(frozen=True, slots=True)
class ConsumerNormBatchReceipt:
    action_key: str
    background: str
    batch_ordinal: int
    action_capture_sha256: str
    native_capture_sha256: str
    ratio_sha256s: tuple[str, ...]
    metric_role: str = "integrity_only"
    authorized_for_selection: bool = False

    def __post_init__(self) -> None:
        if self.action_key not in final_capability.CANONICAL_ACTION_KEYS or (
            self.background not in final_capability.BACKGROUNDS
        ) or type(self.batch_ordinal) is not int or not 0 <= self.batch_ordinal < BATCH_COUNT:
            raise ValueError("consumer norm receipt action changed")
        _sha256("consumer action capture", self.action_capture_sha256)
        _sha256("consumer native capture", self.native_capture_sha256)
        if len(self.ratio_sha256s) != MODEL_LAYER_COUNT or any(
            not runtime._sha256_text(value) for value in self.ratio_sha256s
        ) or self.metric_role != "integrity_only" or self.authorized_for_selection is not False:
            raise ValueError("consumer norm receipt is not diagnostic-only")

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class ConsumerNormBatchResult:
    action: final_capability.FinalAction
    batch_ordinal: int
    row_ratios: tuple[torch.Tensor, ...]
    receipt: ConsumerNormBatchReceipt

    def __post_init__(self) -> None:
        if type(self.action) is not final_capability.FinalAction or type(
            self.receipt
        ) is not ConsumerNormBatchReceipt or self.receipt.action_key != self.action.key or (
            self.receipt.batch_ordinal != self.batch_ordinal
        ) or not isinstance(self.row_ratios, tuple) or len(
            self.row_ratios
        ) != MODEL_LAYER_COUNT:
            raise ValueError("consumer norm batch is malformed")
        checked = tuple(
            _batch_vector(f"consumer layer {layer}", value)
            for layer, value in enumerate(self.row_ratios)
        )
        object.__setattr__(self, "row_ratios", checked)
        if tuple(runtime.tensor_identity_sha256(value) for value in checked) != (
            self.receipt.ratio_sha256s
        ):
            raise ValueError("consumer norm batch differs from its receipt")

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            "action_sha256": self.action.sha256,
            "batch_ordinal": self.batch_ordinal,
            "receipt_sha256": self.receipt.sha256,
            "ratio_sha256s": list(self.receipt.ratio_sha256s),
        })


def reduce_consumer_norm_batch(
    *, action_capture: _CapturedConsumerMagnitudes,
    native_capture: _CapturedConsumerMagnitudes,
) -> ConsumerNormBatchResult:
    """Consume one action/native pair and emit only four row ratios per layer."""

    if not isinstance(action_capture, _CapturedConsumerMagnitudes) or not isinstance(
        native_capture, _CapturedConsumerMagnitudes
    ):
        raise TypeError("consumer norm reduction requires observed captures")
    action_receipt = action_capture.receipt
    native_receipt = native_capture.receipt
    action = action_capture.action
    native = native_capture.action
    # Burn the action capability before semantic validation, so a mismatched attempt
    # is fail-closed.  O/O captures are immutable denominator authorities and may be
    # reused across actions on this exact background/batch/support.
    if action_capture is native_capture:
        if action.arm != "o_o":
            action_capture._take_for_pair(_PAIR_TOKEN)
            raise RuntimeError("only a native O/O capture may be self-denominated")
        numerator = action_capture._take_for_pair(_PAIR_TOKEN)
        denominator = numerator
    else:
        numerator = action_capture._take_for_pair(_PAIR_TOKEN)
        denominator = native_capture._take_for_pair(_PAIR_TOKEN)
    expected_native = final_capability.FinalAction(arm="o_o", background=action.background)
    if native != expected_native or action.background != native.background or (
        action_receipt.batch_ordinal != native_receipt.batch_ordinal
    ) or action_receipt.common_support_sha256 != native_receipt.common_support_sha256 or (
        action_receipt.model_identity_sha256 != native_receipt.model_identity_sha256
    ) or action_receipt.component_identity_sha256 != native_receipt.component_identity_sha256:
        raise RuntimeError("consumer denominator is not native on identical support")
    if bool((denominator <= DENOMINATOR_FLOOR).any()):
        raise RuntimeError("consumer native denominator is zero or numerically empty")
    ratio = (numerator / denominator).contiguous()
    if not bool(torch.isfinite(ratio).all()):
        raise RuntimeError("consumer norm ratio is nonfinite")
    if action.arm == "o_o" and not torch.allclose(
        ratio, torch.ones_like(ratio), atol=NATIVE_RATIO_TOLERANCE,
        rtol=NATIVE_RATIO_TOLERANCE,
    ):
        raise RuntimeError("native consumer norm ratio is not one")
    row_ratios = tuple(ratio[layer].clone().contiguous() for layer in range(
        MODEL_LAYER_COUNT
    ))
    receipt = ConsumerNormBatchReceipt(
        action_key=action.key, background=action.background,
        batch_ordinal=action_receipt.batch_ordinal,
        action_capture_sha256=action_receipt.sha256,
        native_capture_sha256=native_receipt.sha256,
        ratio_sha256s=tuple(
            runtime.tensor_identity_sha256(value) for value in row_ratios
        ),
    )
    return ConsumerNormBatchResult(
        action=action, batch_ordinal=action_receipt.batch_ordinal,
        row_ratios=row_ratios, receipt=receipt,
    )


@dataclass(frozen=True, slots=True)
class ConsumerNormActionResult:
    action: final_capability.FinalAction
    reductions: tuple[final_capability.RowReduction, ...]
    batch_receipt_sha256s: tuple[str, ...]
    result_sha256: str

    def __post_init__(self) -> None:
        if type(self.action) is not final_capability.FinalAction or not isinstance(
            self.reductions, tuple
        ) or len(self.reductions) != MODEL_LAYER_COUNT or any(
            type(value) is not final_capability.RowReduction for value in self.reductions
        ) or len(self.batch_receipt_sha256s) != BATCH_COUNT or len(set(
            self.batch_receipt_sha256s
        )) != BATCH_COUNT or any(
            not runtime._sha256_text(value) for value in self.batch_receipt_sha256s
        ):
            raise ValueError("consumer action result is incomplete")
        body = {
            "action_sha256": self.action.sha256,
            "reduction_sha256s": [value.sha256 for value in self.reductions],
            "batch_receipt_sha256s": list(self.batch_receipt_sha256s),
            "metric_role": "integrity_only", "authorized_for_selection": False,
        }
        if not runtime._sha256_text(self.result_sha256) or runtime.logical_identity_sha256(
            body
        ) != self.result_sha256:
            raise ValueError("consumer action result identity changed")


def aggregate_consumer_norm_action(
    action: final_capability.FinalAction,
    batches: tuple[ConsumerNormBatchResult, ...],
) -> ConsumerNormActionResult:
    """Join the exact 48 batches into the 18 final ``RowReduction`` values."""

    if type(action) is not final_capability.FinalAction or not isinstance(
        batches, tuple
    ) or len(batches) != BATCH_COUNT:
        raise ValueError("consumer action aggregation requires all 48 batches")
    if any(type(batch) is not ConsumerNormBatchResult or batch.action != action or (
        batch.batch_ordinal != ordinal
    ) for ordinal, batch in enumerate(batches)):
        raise RuntimeError("consumer action batches are skipped, reordered, or mixed")
    receipt_sha256s = tuple(batch.receipt.sha256 for batch in batches)
    if len(set(receipt_sha256s)) != BATCH_COUNT:
        raise RuntimeError("consumer action batch receipt was duplicated")
    reductions = tuple(final_capability.RowReduction(
        row_sum=torch.cat(tuple(batch.row_ratios[layer] for batch in batches)),
        row_count=torch.ones(final_capability.FINAL_ROW_COUNT, dtype=torch.long),
    ) for layer in range(MODEL_LAYER_COUNT))
    body = {
        "action_sha256": action.sha256,
        "reduction_sha256s": [value.sha256 for value in reductions],
        "batch_receipt_sha256s": list(receipt_sha256s),
        "metric_role": "integrity_only", "authorized_for_selection": False,
    }
    return ConsumerNormActionResult(
        action=action, reductions=reductions,
        batch_receipt_sha256s=receipt_sha256s,
        result_sha256=runtime.logical_identity_sha256(body),
    )
