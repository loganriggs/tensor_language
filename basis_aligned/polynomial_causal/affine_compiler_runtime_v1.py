"""Isolated live correction hook and original-call guard for compiler v1."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Mapping

import torch


D_MODEL = 1152
CAPTURE_SLICE = slice(64, None, 3)


def runtime_predict(x: torch.Tensor, state: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Float32 inference path serialized into the executable compiler artifact."""

    flat = x.float().reshape(-1, D_MODEL)
    normalized = (flat - state["mean"]) / state["scale"]
    return (normalized @ state["left"]) @ state["right"] + state["bias"]


@dataclass
class Capture:
    inputs: list[torch.Tensor]
    coefficients: list[torch.Tensor]


class CompilerCorrectionHook:
    """One isolated hook for capture, predicted, oracle, and inert arms."""

    def __init__(
        self,
        bases: Mapping[int, torch.Tensor],
        programs: Mapping[str, Mapping[int, Mapping[str, torch.Tensor]]] | None = None,
    ) -> None:
        if set(bases) != {0, 1}:
            raise ValueError("compiler hook requires exactly the MLP0/1 bases")
        self.bases = {site: basis.float() for site, basis in bases.items()}
        self.programs = {} if programs is None else programs
        self.site_states: dict[int, str] = {}
        self.program_name = "main"
        self.capture_site: int | None = None
        self.capture = Capture([], [])
        self.calls = {0: 0, 1: 0, 2: 0}

    def clear(self) -> None:
        self.site_states = {}
        self.program_name = "main"
        self.capture_site = None
        self.capture = Capture([], [])
        self.calls = {0: 0, 1: 0, 2: 0}

    def configure(
        self,
        states: Mapping[int, str],
        *,
        program_name: str = "main",
        capture_site: int | None = None,
    ) -> None:
        allowed = {0: {"N", "Q", "O"}, 1: {"N", "Q", "O"}, 2: {"N", "E"}}
        if any(site not in allowed or state not in allowed[site]
               for site, state in states.items()):
            raise ValueError(f"invalid compiler hook states: {states}")
        if capture_site not in (None, 0, 1):
            raise ValueError("only MLP0/1 may be captured")
        if capture_site is not None and states.get(capture_site, "N") != "N":
            raise ValueError("capture site must remain in deployed state")
        if any(state == "Q" for state in states.values()) and program_name not in self.programs:
            raise ValueError(f"unknown compiler program: {program_name}")
        self.site_states = dict(states)
        self.program_name = program_name
        self.capture_site = capture_site
        self.capture = Capture([], [])
        self.calls = {0: 0, 1: 0, 2: 0}

    def captured(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.capture_site is None or not self.capture.inputs:
            raise RuntimeError("no compiler capture is available")
        return torch.cat(self.capture.inputs), torch.cat(self.capture.coefficients)

    def __call__(self, site: int, block: Any, z: torch.Tensor, mo: torch.Tensor) -> torch.Tensor:
        self.calls[site] = self.calls.get(site, 0) + 1
        state = self.site_states.get(site, "N")
        needs_label = self.capture_site == site
        if not needs_label and state == "N":
            return mo

        if needs_label:
            original = block.mlp(z).float()
            residual = original - mo.float()
            sampled_z = z[:, CAPTURE_SLICE].float().reshape(-1, D_MODEL)
            sampled_residual = residual[:, CAPTURE_SLICE].reshape(-1, D_MODEL)
            basis = self.bases[site].to(sampled_residual.device)
            self.capture.inputs.append(sampled_z.detach().cpu().contiguous())
            self.capture.coefficients.append(
                (sampled_residual @ basis).detach().cpu().contiguous()
            )
            return mo

        if state == "Q":
            predictor = self.programs[self.program_name][site]
            coefficients = runtime_predict(z, predictor)
            basis = self.bases[site].to(coefficients.device)
            delta = (coefficients @ basis.T).view_as(mo)
            return mo + delta.to(mo.dtype)

        if state in ("O", "E"):
            residual = block.mlp(z).float() - mo.float()
            if state == "E":
                delta = residual
            else:
                basis = self.bases[site].to(residual.device)
                flat = residual.reshape(-1, D_MODEL)
                delta = ((flat @ basis) @ basis.T).view_as(residual)
            return mo + delta.to(mo.dtype)

        raise RuntimeError(f"unhandled compiler state at site {site}: {state}")


class OriginalMLPCallGuard(AbstractContextManager["OriginalMLPCallGuard"]):
    """Raise on every non-allowlisted original MLP0/1/2 call and count all calls."""

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
            except BaseException as error:  # pragma: no cover - defensive restore path
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
