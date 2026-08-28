"""Graph-safe physical-gate interventions for the owned bilin18 tensor program.

The response leaf has one gate scale per context and is shared over token positions.
This makes one backward pass return separate trajectory-complete gate responses for
every context without exporting a full residual-write VJP.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

import tensor_bilin18_tangent_collector as tangent
from tensor_bilin18_program import LAYERS, TensorBilin18Program


SOURCE_SITE = 1
PRODUCTION_BATCH = 4
PRODUCTION_SEQUENCE = 256
PRODUCTION_WIDTH = 1152
PRODUCTION_HIDDEN = 4608
PRODUCTION_TOKEN_VOCAB = 50_257
PRODUCTION_LOGIT_VOCAB = 50_304
RANK640_STORED_VALUES = 516_707_766


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _forward_with_gate_scale(
    program: TensorBilin18Program, tokens: torch.Tensor, gate_scale: torch.Tensor, *,
    source_site: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Execute the complete program with one context-specific, position-shared scale."""
    if not isinstance(program, TensorBilin18Program):
        raise TypeError("gate intervention requires an owned TensorBilin18Program")
    if type(source_site) is not int or not 0 <= source_site < LAYERS:
        raise ValueError("source site is outside the MLP bank")
    program.validate_tokens(tokens)
    hidden = program.mlp_bank.programs[source_site].hidden
    if (
        not torch.is_tensor(gate_scale)
        or tuple(gate_scale.shape) != (tokens.shape[0], hidden)
        or gate_scale.device != tokens.device
        or not gate_scale.is_floating_point()
        or not bool(torch.isfinite(gate_scale.detach()).all())
    ):
        raise ValueError("gate scale must be finite [context, hidden] on program device")

    state = F.embedding(tokens, program.token_embedding)
    state = F.rms_norm(state, (program.width,))
    initial = state
    first_value = None
    attention_calls: list[int] = []
    mlp_calls: list[int] = []
    scaled_calls: list[int] = []
    for site in range(LAYERS):
        lambdas = program.residual_lambdas[site].to(state.dtype)
        state = lambdas[0] * state + lambdas[1] * initial
        attention_state = F.rms_norm(state, (program.width,))
        attention_write, first_value = program.attention_bank.programs[site](
            attention_state, first_value,
        )
        attention_calls.append(site)
        state = state + attention_write
        mlp_state = F.rms_norm(state, (program.width,))
        mlp = program.mlp_bank.programs[site]
        if site == source_site:
            product = mlp.left(mlp_state) * mlp.right(mlp_state)
            write = mlp.down(product * gate_scale[:, None, :].to(product.dtype))
            write = write + mlp.down_bias.to(write.dtype)
            scaled_calls.append(site)
        else:
            write = mlp(mlp_state)
        mlp_calls.append(site)
        state = state + write
    final_state = F.rms_norm(state, (program.width,))
    logits = F.linear(final_state, program.unembedding.to(final_state.dtype))
    logits = (30.0 * torch.tanh(logits / 30.0)).float()
    if attention_calls != list(range(LAYERS)) or mlp_calls != list(range(LAYERS)) or (
        scaled_calls != [source_site]
    ):
        raise RuntimeError("gate intervention did not execute the complete tensor program")
    return logits, {
        "attention_calls": tuple(attention_calls),
        "mlp_calls": tuple(mlp_calls),
        "source_site": source_site,
        "scale_shared_across_positions": True,
        "context_scales_independent": True,
    }


def forward_with_global_gate_scale_leaf(
    program: TensorBilin18Program, tokens: torch.Tensor, *, source_site: int = SOURCE_SITE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return logits and a leaf ``alpha[context, gate]`` initialized to all ones."""
    if not isinstance(program, TensorBilin18Program):
        raise TypeError("gate intervention requires an owned TensorBilin18Program")
    if type(source_site) is not int or not 0 <= source_site < LAYERS:
        raise ValueError("source site is outside the MLP bank")
    program.validate_tokens(tokens)
    hidden = program.mlp_bank.programs[source_site].hidden
    alpha = torch.ones(
        tokens.shape[0], hidden, device=tokens.device,
        dtype=program.token_embedding.dtype, requires_grad=True,
    )
    logits, receipt = _forward_with_gate_scale(
        program, tokens, alpha, source_site=source_site,
    )
    if not alpha.is_leaf or alpha.grad_fn is not None or not logits.requires_grad:
        raise RuntimeError("global gate scale graph leaf is malformed")
    return logits, alpha, receipt


@dataclass(frozen=True)
class PairedGateResponse:
    first: torch.Tensor
    second: torch.Tensor
    receipt: Mapping[str, Any]


class GlobalGateResponseTransaction:
    """One-use categorical-Fisher transaction returning only CPU gate responses."""

    def __init__(
        self, *, program: TensorBilin18Program, tokens: torch.Tensor,
        row_ids: Sequence[str], first_probe_seeds: Sequence[int],
        second_probe_seeds: Sequence[int], score_start: int, score_stop: int,
        source_site: int = SOURCE_SITE, production: bool = True,
    ) -> None:
        rows = tuple(row_ids)
        first, second = tuple(first_probe_seeds), tuple(second_probe_seeds)
        if not isinstance(program, TensorBilin18Program):
            raise TypeError("gate response requires an owned TensorBilin18Program")
        if not rows or len(rows) != tokens.shape[0] or len(set(rows)) != len(rows):
            raise ValueError("row identities must align uniquely with contexts")
        if (
            not first or len(first) != len(second) or len(set(first)) != len(first)
            or len(set(second)) != len(second) or bool(set(first) & set(second))
        ):
            raise ValueError("probe halves must be equal, unique, and disjoint")
        if not 0 <= score_start < score_stop <= tokens.shape[1]:
            raise ValueError("score support is outside the token trajectory")
        program.validate_tokens(tokens)
        if production:
            cost = program.cost_receipt()
            mlp = program.mlp_bank.programs[source_site]
            if (
                source_site != SOURCE_SITE
                or tuple(tokens.shape) != (PRODUCTION_BATCH, PRODUCTION_SEQUENCE)
                or program.width != PRODUCTION_WIDTH or mlp.hidden != PRODUCTION_HIDDEN
                or program.vocab_size != PRODUCTION_LOGIT_VOCAB
                or program.logit_vocab != PRODUCTION_LOGIT_VOCAB
                or int(tokens.min()) < 0 or int(tokens.max()) >= PRODUCTION_TOKEN_VOCAB
                or int(cost["total_stored_values"]) != RANK640_STORED_VALUES
                or int(cost["native_calls_per_forward"]) != 0
                or not bool(cost["total_input_support"])
            ):
                raise ValueError("production global-gate response contract changed")
        self.__program: TensorBilin18Program | None = program
        self.__tokens: torch.Tensor | None = tokens.contiguous().clone()
        self.__tokens_sha256: str | None = tensor_sha256(self.__tokens)
        self.__row_ids: tuple[str, ...] | None = rows
        self.__first_seeds: tuple[int, ...] | None = first
        self.__second_seeds: tuple[int, ...] | None = second
        self.__score_start: int | None = score_start
        self.__score_stop: int | None = score_stop
        self.__source_site: int | None = source_site
        self.__closed = False

    @property
    def closed(self) -> bool:
        return self.__closed

    @property
    def aliases_revoked(self) -> bool:
        return self.__closed and all(getattr(self, name) is None for name in (
            "_GlobalGateResponseTransaction__program",
            "_GlobalGateResponseTransaction__tokens",
            "_GlobalGateResponseTransaction__tokens_sha256",
            "_GlobalGateResponseTransaction__row_ids",
            "_GlobalGateResponseTransaction__first_seeds",
            "_GlobalGateResponseTransaction__second_seeds",
            "_GlobalGateResponseTransaction__score_start",
            "_GlobalGateResponseTransaction__score_stop",
            "_GlobalGateResponseTransaction__source_site",
        ))

    def _revoke(self) -> None:
        self.__program = None
        self.__tokens = None
        self.__tokens_sha256 = None
        self.__row_ids = None
        self.__first_seeds = None
        self.__second_seeds = None
        self.__score_start = None
        self.__score_stop = None
        self.__source_site = None
        self.__closed = True

    def consume(self) -> PairedGateResponse:
        if self.__closed:
            raise RuntimeError("global-gate response transaction is spent")
        program, tokens, rows = self.__program, self.__tokens, self.__row_ids
        first, second = self.__first_seeds, self.__second_seeds
        score_start, score_stop = self.__score_start, self.__score_stop
        source_site, token_hash = self.__source_site, self.__tokens_sha256
        logits = alpha = targets = responses = None
        try:
            assert program is not None and tokens is not None and rows is not None
            assert first is not None and second is not None
            assert score_start is not None and score_stop is not None
            assert source_site is not None and token_hash is not None
            if tensor_sha256(tokens) != token_hash:
                raise RuntimeError("owned gate-response tokens changed")
            if tuple(program.parameters()) or any(
                value.requires_grad or value.grad is not None for value in program.buffers()
            ):
                raise RuntimeError("gate-response program gradient state changed")
            logits, alpha, forward_receipt = forward_with_global_gate_scale_leaf(
                program, tokens, source_site=source_site,
            )
            seeds = first + second
            targets = tangent.stateless_categorical_fisher_targets(
                logits, rows, seeds, score_start=score_start, score_stop=score_stop,
            )
            log_probabilities = F.log_softmax(
                logits[:, score_start:score_stop].float(), dim=-1,
            )
            target_device = targets.to(logits.device)
            responses = torch.empty(
                len(seeds), len(rows), alpha.shape[1], dtype=torch.float64,
            )
            for probe in range(len(seeds)):
                selected = torch.gather(
                    log_probabilities, -1, target_device[probe].unsqueeze(-1),
                ).squeeze(-1)
                gradient = torch.autograd.grad(
                    selected.sum(), alpha, retain_graph=probe + 1 < len(seeds),
                    create_graph=False, allow_unused=False,
                )[0].detach().cpu().double()
                if tuple(gradient.shape) != (len(rows), alpha.shape[1]) or not bool(
                    torch.isfinite(gradient).all()
                ):
                    raise RuntimeError("trajectory-complete gate response is malformed")
                responses[probe] = gradient
            split = len(first)
            first_block = responses[:split].permute(1, 0, 2).contiguous()
            second_block = responses[split:].permute(1, 0, 2).contiguous()
            receipt = {
                "status": "complete",
                "row_ids": list(rows),
                "first_probe_seeds": list(first),
                "second_probe_seeds": list(second),
                "probe_halves_disjoint": not bool(set(first) & set(second)),
                "tokens_sha256": token_hash,
                "first_target_ids_sha256": tensor_sha256(targets[:split]),
                "second_target_ids_sha256": tensor_sha256(targets[split:]),
                "first_response_sha256": tensor_sha256(first_block),
                "second_response_sha256": tensor_sha256(second_block),
                "response_shape_per_half": list(first_block.shape),
                "source_site": source_site,
                "score_support": [score_start, score_stop],
                "forward": forward_receipt,
                "all_token_positions_share_each_gate_scale": True,
                "contexts_have_independent_gate_scale_leaves": True,
                "raw_logits_returned": False,
                "raw_targets_returned": False,
                "raw_residual_vjps_returned": False,
            }
        finally:
            logits = alpha = targets = responses = None
            self._revoke()
        receipt["graph_aliases_revoked"] = self.aliases_revoked
        return PairedGateResponse(first_block, second_block, receipt)
