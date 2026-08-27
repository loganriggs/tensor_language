"""Typed live runtime for state-complete compiler-v2 programs.

The runtime has no fitting logic.  It supports the registered A--E program
grammars, captures p/c/mo labels only when explicitly configured, and keeps all
N/Q arms independent of the original MLP implementation.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Mapping

import torch


D_MODEL = 1152
COEFFICIENT_DIM = 64
CAPTURE_SLICE = slice(64, None, 3)


def runtime_projected_output(z: torch.Tensor, state: Mapping[str, Any]) -> torch.Tensor:
    """Evaluate the serialized coefficient program in float32."""

    flat = z.float().reshape(-1, D_MODEL)
    grammar = state.get("grammar")
    if grammar == "affine":
        normalized = (flat - state["mean"]) / state["scale"]
        return (normalized @ state["left"]) @ state["right"] + state["bias"]
    if grammar == "native":
        products = (flat @ state["left"].T) * (flat @ state["right"].T)
        return products @ state["projected_decoder"] + state["beta"]
    if grammar == "constant":
        return state["bias"].expand(flat.shape[0], -1)
    raise ValueError(f"unknown serialized compiler grammar: {grammar}")


def runtime_coefficients(
    z: torch.Tensor,
    mo: torch.Tensor,
    basis: torch.Tensor,
    state: Mapping[str, Any],
) -> torch.Tensor:
    """Evaluate either the A anchor or legal B--E live-state interface."""

    predicted = runtime_projected_output(z, state)
    interface = state.get("interface")
    if interface == "z_only_c":
        return predicted
    if interface == "state_complete_p":
        live_mo = mo.float().reshape(-1, D_MODEL) @ basis
        return predicted - live_mo
    raise ValueError(f"unknown serialized compiler interface: {interface}")


@dataclass
class Capture:
    z: list[torch.Tensor]
    p: list[torch.Tensor]
    mo: list[torch.Tensor]
    c: list[torch.Tensor]
    adjoint: list[torch.Tensor]


class StateCompleteCorrectionHook:
    """One isolated hook for capture, compiled, projected-oracle, and exact arms."""

    def __init__(
        self,
        bases: Mapping[int, torch.Tensor],
        programs: Mapping[str, Mapping[int, Mapping[str, Any]]] | None = None,
    ) -> None:
        if set(bases) != {0, 1}:
            raise ValueError("compiler-v2 hook requires exactly the MLP0/1 bases")
        self.bases = {site: basis.float() for site, basis in bases.items()}
        for site, basis in self.bases.items():
            if tuple(basis.shape) != (D_MODEL, COEFFICIENT_DIM):
                raise ValueError(f"invalid MLP{site} basis shape: {basis.shape}")
        self.programs = {} if programs is None else programs
        self.site_states: dict[int, str] = {}
        self.program_name = "selected"
        self.capture_site: int | None = None
        self.capture = Capture([], [], [], [], [])
        self.capture_adjoint = False
        self.pending_adjoint_leaf: torch.Tensor | None = None
        self.calls = {0: 0, 1: 0, 2: 0}

    def clear(self) -> None:
        self.site_states = {}
        self.program_name = "selected"
        self.capture_site = None
        self.capture = Capture([], [], [], [], [])
        self.capture_adjoint = False
        self.pending_adjoint_leaf = None
        self.calls = {0: 0, 1: 0, 2: 0}

    def configure(
        self,
        states: Mapping[int, str],
        *,
        program_name: str = "selected",
        capture_site: int | None = None,
        capture_adjoint: bool = False,
    ) -> None:
        allowed = {0: {"N", "Q", "O"}, 1: {"N", "Q", "O"}, 2: {"N", "E"}}
        if any(site not in allowed or state not in allowed[site]
               for site, state in states.items()):
            raise ValueError(f"invalid compiler-v2 hook states: {states}")
        if capture_site not in (None, 0, 1):
            raise ValueError("only MLP0/1 may be captured")
        if capture_site is not None and states.get(capture_site, "N") != "N":
            raise ValueError("capture site must remain in deployed state")
        if capture_adjoint and capture_site is None:
            raise ValueError("adjoint capture requires a capture site")
        if any(state == "Q" for state in states.values()):
            if program_name not in self.programs:
                raise ValueError(f"unknown compiler-v2 program: {program_name}")
            needed = {site for site, state in states.items() if state == "Q"}
            if not needed.issubset(self.programs[program_name]):
                raise ValueError(f"program {program_name} lacks sites {sorted(needed)}")
        self.site_states = dict(states)
        self.program_name = program_name
        self.capture_site = capture_site
        self.capture = Capture([], [], [], [], [])
        self.capture_adjoint = bool(capture_adjoint)
        self.pending_adjoint_leaf = None
        self.calls = {0: 0, 1: 0, 2: 0}

    def captured(self) -> dict[str, torch.Tensor]:
        if self.capture_site is None or not self.capture.z:
            raise RuntimeError("no compiler-v2 capture is available")
        output = {
            "z": torch.cat(self.capture.z),
            "p": torch.cat(self.capture.p),
            "mo": torch.cat(self.capture.mo),
            "c": torch.cat(self.capture.c),
        }
        if self.capture_adjoint:
            if len(self.capture.adjoint) != len(self.capture.z):
                raise RuntimeError("not every teacher batch has a collected adjoint")
            output["adjoint"] = torch.cat(self.capture.adjoint)
        return output

    def collect_pending_adjoint(self) -> None:
        """Collect the coefficient adjoint after the caller runs ``loss.backward``."""

        leaf = self.pending_adjoint_leaf
        if not self.capture_adjoint or leaf is None:
            raise RuntimeError("no pending compiler-v2 adjoint leaf")
        if leaf.grad is None:
            raise RuntimeError("teacher coefficient leaf has no gradient")
        sampled = leaf.grad[:, CAPTURE_SLICE].float().reshape(-1, COEFFICIENT_DIM)
        self.capture.adjoint.append(sampled.detach().cpu().contiguous())
        self.pending_adjoint_leaf = None

    def __call__(self, site: int, block: Any, z: torch.Tensor, mo: torch.Tensor) -> torch.Tensor:
        self.calls[site] = self.calls.get(site, 0) + 1
        state_name = self.site_states.get(site, "N")
        needs_label = self.capture_site == site
        if not needs_label and state_name == "N":
            return mo

        if needs_label:
            original = block.mlp(z).float()
            basis = self.bases[site].to(original.device)
            p_full = original.reshape(-1, D_MODEL) @ basis
            mo_full = mo.float().reshape(-1, D_MODEL) @ basis
            p_sequence = p_full.view(*original.shape[:-1], COEFFICIENT_DIM)
            mo_sequence = mo_full.view(*mo.shape[:-1], COEFFICIENT_DIM)
            c_sequence = p_sequence - mo_sequence
            sampled_z = z[:, CAPTURE_SLICE].float().reshape(-1, D_MODEL)
            p = p_sequence[:, CAPTURE_SLICE].reshape(-1, COEFFICIENT_DIM)
            mo_coeff = mo_sequence[:, CAPTURE_SLICE].reshape(-1, COEFFICIENT_DIM)
            c = c_sequence[:, CAPTURE_SLICE].reshape(-1, COEFFICIENT_DIM)
            self.capture.z.append(sampled_z.detach().cpu().contiguous())
            self.capture.p.append(p.detach().cpu().contiguous())
            self.capture.mo.append(mo_coeff.detach().cpu().contiguous())
            self.capture.c.append(c.detach().cpu().contiguous())
            if self.capture_adjoint:
                if self.pending_adjoint_leaf is not None:
                    raise RuntimeError("previous teacher adjoint was not collected")
                leaf = c_sequence.detach().requires_grad_(True)
                self.pending_adjoint_leaf = leaf
                delta = (leaf.reshape(-1, COEFFICIENT_DIM) @ basis.T).view_as(mo)
                return mo.detach() + delta.to(mo.dtype)
            return mo

        if state_name == "Q":
            basis = self.bases[site].to(z.device)
            program = self.programs[self.program_name][site]
            coefficients = runtime_coefficients(z, mo, basis, program)
            delta = (coefficients @ basis.T).view_as(mo)
            return mo + delta.to(mo.dtype)

        if state_name in ("O", "E"):
            residual = block.mlp(z).float() - mo.float()
            if state_name == "E":
                delta = residual
            else:
                basis = self.bases[site].to(residual.device)
                flat = residual.reshape(-1, D_MODEL)
                delta = ((flat @ basis) @ basis.T).view_as(residual)
            return mo + delta.to(mo.dtype)

        raise RuntimeError(f"unhandled compiler-v2 state at site {site}: {state_name}")


class OriginalMLPCallGuard(AbstractContextManager["OriginalMLPCallGuard"]):
    """Poison every non-allowlisted original MLP0/1/2 call and restore exactly."""

    def __init__(self, blocks: Any, allowed_sites: set[int] | frozenset[int]) -> None:
        self.blocks = blocks
        self.allowed_sites = frozenset(allowed_sites)
        if not self.allowed_sites.issubset({0, 1, 2}):
            raise ValueError("original-call allowlist is outside MLP0/1/2")
        self.originals: dict[int, Any] = {}
        self.counts = {0: 0, 1: 0, 2: 0}

    def __enter__(self) -> "OriginalMLPCallGuard":
        if self.originals:
            raise RuntimeError("original-call guard is already installed")
        for site in (0, 1, 2):
            module = self.blocks[site].mlp
            original = module.forward
            self.originals[site] = original

            def guarded(*args: Any, _site: int = site, _original: Any = original,
                        **kwargs: Any) -> Any:
                self.counts[_site] += 1
                if _site not in self.allowed_sites:
                    raise RuntimeError(f"poisoned original MLP{_site} was called")
                return _original(*args, **kwargs)

            module.forward = guarded
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        restore_error = None
        for site, original in self.originals.items():
            try:
                self.blocks[site].mlp.forward = original
            except BaseException as error:  # pragma: no cover
                restore_error = error
        self.originals = {}
        if restore_error is not None and exc_value is None:
            raise restore_error

    def assert_contract(self, *, require_allowed_calls: bool) -> None:
        forbidden = set(self.counts).difference(self.allowed_sites)
        if any(self.counts[site] != 0 for site in forbidden):
            raise RuntimeError(f"forbidden original calls occurred: {self.counts}")
        if require_allowed_calls and any(self.counts[site] <= 0 for site in self.allowed_sites):
            raise RuntimeError(f"allowlisted original was not exercised: {self.counts}")
