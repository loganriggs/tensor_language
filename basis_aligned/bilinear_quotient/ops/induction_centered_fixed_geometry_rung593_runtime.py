#!/usr/bin/env python3
"""Lazy model-backed call executor for R593.

Importing this file does not import torch or load a checkpoint.  Construction
of ``R593ModelExecutor`` is the only boundary that does so; owner tests use a
fake executor instead.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import sys
from typing import Mapping, Sequence

import numpy as np


CHECKPOINT_SHA256 = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"


class R593ModelExecutor:
    """Execute one frozen manifest call against the real observed-model facade."""

    def __init__(self, producer, r585):
        self.p = producer
        torch, functional, facade, induction, runtime_r585 = r585.__r593_runtime_loader__()
        self.r585 = runtime_r585
        self.torch = torch
        self.functional = functional
        self.facade = facade
        self.induction = induction
        self.model, checkpoint = facade.load_bilin18(
            device="cuda", dtype=torch.float32, verify_weights_sha256=True
        )
        facade.validate_production_model(self.model)
        if checkpoint.weights_sha256 != CHECKPOINT_SHA256:
            raise RuntimeError("checkpoint hash changed")
        self.checkpoint_sha256 = checkpoint.weights_sha256

    def _independent_full_attention_write(self, event, specs):
        """Reconstruct all heads and independent float64 equality partitions."""
        torch, functional = self.torch, self.functional
        state, attention = event.state, event.block.attn
        batch, length, width = state.shape
        q = functional.linear(state, attention.c_q.weight).view(batch, length, 9, 128)
        k = functional.linear(state, attention.c_k.weight).view(batch, length, 9, 128)
        q2 = functional.linear(state, attention.c_q2.weight).view(batch, length, 9, 128)
        k2 = functional.linear(state, attention.c_k2.weight).view(batch, length, 9, 128)
        raw_value = functional.linear(state, attention.c_v.weight).view(batch, length, 9, 128)
        value = (1 - attention.lamb) * raw_value + attention.lamb * event.first_value.view_as(raw_value)
        cos, sin = attention.rotary(q)
        attention_module = sys.modules[type(attention).__module__]
        q = attention_module.apply_rotary_emb(functional.rms_norm(q, (128,)), cos, sin)
        k = attention_module.apply_rotary_emb(functional.rms_norm(k, (128,)), cos, sin)
        q2 = attention_module.apply_rotary_emb(functional.rms_norm(q2, (128,)), cos, sin)
        k2 = attention_module.apply_rotary_emb(functional.rms_norm(k2, (128,)), cos, sin)
        score1 = torch.einsum("bqhd,bkhd->bhqk", q, k) / 128
        score2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2) / 128
        pattern = score1 * score2
        causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=state.device))
        pattern = pattern.masked_fill(~causal, 0)
        heads = torch.einsum("bhqk,bkhd->bhqd", pattern, value)
        flattened = heads.transpose(1, 2).contiguous().view(batch, length, width)
        reconstructed = functional.linear(flattened, attention.c_proj.weight)
        support = self.induction.induction_fetch_mask(event.tokens)
        audit_terms = [{site: None for site in self.p.SITES} for _ in specs]
        for local, spec in enumerate(specs):
            query = int(spec["final_position"])
            for site in self.p.SITES:
                layer, head = int(site[1]), int(site[3])
                if layer != event.site:
                    continue
                pattern64 = pattern[local, head, query].double()
                value64 = value[local, :, head].double()
                weight64 = attention.c_proj.weight[:, head * 128:(head + 1) * 128].double()
                mask = support[local, query]
                equality_head = torch.einsum("k,kd->d", pattern64 * mask, value64)
                remainder_head = torch.einsum("k,kd->d", pattern64 * ~mask, value64)
                full_head = torch.einsum("k,kd->d", pattern64, value64)
                audit_terms[local][site] = {
                    "canonical": functional.linear(equality_head, weight64).detach().cpu().contiguous(),
                    "remainder": functional.linear(remainder_head, weight64).detach().cpu().contiguous(),
                    "head_output": functional.linear(full_head, weight64).detach().cpu().contiguous(),
                }
        return reconstructed, audit_terms

    def _capture(self, tokens, specs):
        torch, functional = self.torch, self.functional
        device = next(self.model.parameters()).device
        token_tensor = torch.as_tensor(tokens, dtype=torch.long, device=device)
        per_row = [{site: None for site in self.p.SITES} for _ in specs]
        full_native = [{site: None for site in self.p.SITES} for _ in specs]
        full_reconstructed = [{site: None for site in self.p.SITES} for _ in specs]
        full_errors = []

        def attention(event):
            if event.site not in {5, 7, 8}:
                return event.block.attn(event.state, event.first_value)
            write, next_value, terms, full_error = self.r585.factorize_attention_event(
                event, specs, torch=torch, functional=functional, induction=self.induction
            )
            reconstructed, audit_terms = self._independent_full_attention_write(event, specs)
            independent_error = float((reconstructed.float() - write.float()).abs().max().detach().cpu())
            full_errors.extend((float(full_error), independent_error))
            for local, row in enumerate(terms):
                for site, value in row.items():
                    per_row[local][site] = value
                    per_row[local][site]["audit64"] = audit_terms[local][site]
                    query = int(specs[local]["final_position"])
                    full_native[local][site] = write[local, query].float().detach().cpu().contiguous()
                    full_reconstructed[local][site] = reconstructed[local, query].float().detach().cpu().contiguous()
            return write, next_value

        def mlp(event):
            return event.block.mlp(event.state)

        with torch.inference_mode():
            logits = self.facade.forward_with_dispatch(
                self.model, token_tensor, attention, mlp, require_production=False
            )
        b = len(specs)
        arrays = {
            "tokens.npy": np.ascontiguousarray(tokens, dtype="<i8"),
            "logits.npy": np.empty((b, self.p.VOCAB), dtype="<f4"),
            "factor_e.npy": np.empty((b, 4, 2), dtype="<f4"),
            "factor_u.npy": np.empty((b, 4, 2, self.p.RESIDUAL), dtype="<f4"),
            "support.npy": np.empty((b, 4, 2), dtype=np.bool_),
        }
        for name in (
            "factorized_equality_term.npy",
            "native_full_attention_write.npy",
            "independent_full_native_write.npy",
        ):
            arrays[name] = np.empty((b, 4, self.p.RESIDUAL), dtype="<f4")
        for name in (
            "native_equality_term.npy", "native_non_equality_remainder.npy", "native_head_write.npy",
        ):
            arrays[name] = np.empty((b, 4, self.p.RESIDUAL), dtype="<f8")
        for local, spec in enumerate(specs):
            arrays["logits.npy"][local] = logits[local, int(spec["final_position"])].float().detach().cpu().numpy()
            for site_index, site in enumerate(self.p.SITES):
                term = per_row[local][site]
                if term is None:
                    raise RuntimeError(f"missing captured site {site}")
                arrays["factor_e.npy"][local, site_index] = np.asarray(term["e"], dtype="<f4")
                arrays["factor_u.npy"][local, site_index] = np.stack(
                    [value.float().numpy() for value in term["u"]]
                ).astype("<f4", copy=False)
                query = int(spec["final_position"])
                for role_index, payload in enumerate(spec["payload_positions"]):
                    arrays["support.npy"][local, site_index, role_index] = (
                        int(tokens[local, int(payload) - 1]) == int(tokens[local, query])
                    )
                audit64 = term["audit64"]
                if audit64 is None:
                    raise RuntimeError(f"missing high-precision audit term {site}")
                arrays["native_equality_term.npy"][local, site_index] = audit64["canonical"].double().numpy()
                arrays["factorized_equality_term.npy"][local, site_index] = term["term"].float().numpy()
                arrays["native_non_equality_remainder.npy"][local, site_index] = audit64["remainder"].double().numpy()
                arrays["native_head_write.npy"][local, site_index] = audit64["head_output"].double().numpy()
                arrays["native_full_attention_write.npy"][local, site_index] = full_native[local][site].numpy()
                arrays["independent_full_native_write.npy"][local, site_index] = full_reconstructed[local][site].numpy()
        return arrays, max(full_errors, default=0.0)

    def _intervene(self, tokens, specs, planned):
        torch = self.torch
        device = next(self.model.parameters()).device
        token_tensor = torch.as_tensor(tokens, dtype=torch.long, device=device)
        planned_gpu = torch.as_tensor(planned, dtype=torch.float32, device=device)
        actual = torch.zeros_like(planned_gpu)

        def attention(event):
            write, next_value = event.block.attn(event.state, event.first_value)
            indices = [i for i, name in enumerate(self.p.SITES) if int(name[1]) == event.site]
            if not indices:
                return write, next_value
            modified = write.clone()
            for local, spec in enumerate(specs):
                query = int(spec["final_position"])
                before = modified[local, query].float().clone()
                total = planned_gpu[local, indices].sum(dim=0)
                modified[local, query] += total.to(dtype=modified.dtype)
                observed = modified[local, query].float() - before
                # Same-layer sites are applied in this one transaction.  Store
                # each planned component; the producer also checks their sum.
                for index in indices:
                    actual[local, index] = planned_gpu[local, index]
                if float((observed - total).abs().max().detach().cpu()) > self.p.TOLERANCE:
                    # Preserve returned evidence; producer owns predicate order.
                    actual[local, indices[0], 0] += 2 * self.p.TOLERANCE
            return modified, next_value

        def mlp(event):
            return event.block.mlp(event.state)

        with torch.inference_mode():
            logits = self.facade.forward_with_dispatch(
                self.model, token_tensor, attention, mlp, require_production=False
            )
        output = np.empty((len(specs), self.p.VOCAB), dtype="<f4")
        for local, spec in enumerate(specs):
            output[local] = logits[local, int(spec["final_position"])].float().detach().cpu().numpy()
        return {
            "tokens.npy": np.ascontiguousarray(tokens, dtype="<i8"),
            "logits.npy": output,
            "hook_deltas.npy": actual.detach().cpu().numpy().astype("<f4", copy=False),
            "planned_hook_deltas.npy": np.ascontiguousarray(planned, dtype="<f4"),
        }

    def execute(
        self, call: Mapping[str, object], tokens: np.ndarray,
        specs: Sequence[Mapping[str, object]], planned: np.ndarray | None,
    ) -> Mapping[str, object]:
        kind = str(call["call_kind"])
        if kind in ("endpoint", "native"):
            arrays, full_error = self._capture(tokens, specs)
            if kind == "native":
                arrays = {
                    **{key: arrays[key] for key in ("tokens.npy", "logits.npy")},
                    "live_e.npy": arrays["factor_e.npy"],
                    "live_u.npy": arrays["factor_u.npy"],
                    **{key: arrays[key] for key in (
                        "native_equality_term.npy", "factorized_equality_term.npy",
                        "native_non_equality_remainder.npy", "native_head_write.npy",
                        "native_full_attention_write.npy",
                        "independent_full_native_write.npy",
                    )},
                }
            return {"arrays": arrays, "native_full_write_reconstruction_max_abs": full_error}
        if planned is None:
            raise RuntimeError("scientific arm missing planned centered delta")
        return {"arrays": self._intervene(tokens, specs, planned)}
