#!/usr/bin/env python3
# BQGATE: LIBRARY -- shared code for the aspectual definition-of-done battery; not an experiment.
"""Definition-of-done battery for the aspectual has/had path: donor-free component removal.

WHY. `basis_aligned/better_circuits.md` §1 defines a circuit as *selective* only when removing
the component changes the target while at least three unrelated readers move less than a gate,
on fresh rows, against the null of removing an equal-norm RANDOM direction at the same site.
The aspectual line (program releases v1-v12) has paired base/donor interchange evidence at every
component but no donor-free removal, no unrelated-reader battery and no random-direction null.
This library supplies the three missing instruments on one shared exact forward.

Removal semantics. A component write is the pre-`c_proj` head slice (128-d) at a declared
position, or an MLP output (1152-d) at a declared position. "Remove" sets it to zero, i.e.
subtracts the write `w`. The null subtracts a random vector `r` with `|r| = |w|` at the same
slice and position, drawn per row from a registered seed, so the null is matched in norm and
site and differs only in direction.

Readers. Every reader is a two-token logit contrast at the final position of the same row:
target `has - had`, and unrelated `was - were` (number), `who - which` (animacy),
`night - day` (the corpus's canonical control vocabulary). The signed target margin is oriented
toward the native answer, so positive damage means the removal pushes away from the answer.

Instrument control (standing lesson: control the NEW code path). `forward` with no removal must
reproduce `producer.Bilin18TorchBackend.native` answer/foil logits to 1e-4 before any arm is
interpreted; the runner scores that as its first prediction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Mapping, Sequence

import circuit_fast_screen_candidate_aspectual as parent
import circuit_fast_screen_candidates as lex

ENCODING = lex.ENCODING
N_LAYERS, N_HEADS, N_EMBD = 18, 9, 1152
HEAD_DIM = N_EMBD // N_HEADS

# Fresh lexicon: disjoint from the discovery lexicon (parent), the lexical holdout v5 and the
# fresh-construction v2 panels. Every item must be a single GPT-2 token with a leading space.
AGENTS = ("banker", "monk", "scout", "miner", "tailor", "butcher", "sheriff", "priest",
          "poet", "mayor", "bishop", "dancer", "singer", "soldier", "surgeon", "merchant")
PERIODS = ("siege", "drought", "famine", "flood", "storm", "strike", "holiday", "century",
           "crusade", "mission", "journey", "pilgrimage", "summit", "wedding", "funeral", "plague")
_PRIOR_AGENTS = set(parent._AGENTS) | {
    "analyst", "baker", "dentist", "clerk", "historian", "inspector", "doctor", "keeper",
    "worker", "mechanic", "operator", "painter", "architect", "coach", "courier", "sailor"}
_PRIOR_PERIODS = set(getattr(parent, "_PERIODS", ())) | {
    "decade", "semester", "campaign", "expedition", "assignment", "project", "investigation",
    "renovation", "construction", "deployment", "migration", "recovery", "transition",
    "experiment", "tournament", "conference"}

READERS = {
    "target_has_had": (" has", " had"),
    "number_was_were": (" was", " were"),
    "animacy_who_which": (" who", " which"),
    "canonical_night_day": (" night", " day"),
}
UNRELATED = ("number_was_were", "animacy_who_which", "canonical_night_day")

CONSTRUCTIONS = {
    "fronted": lambda cue, period, agent: f"{cue.capitalize()} last {period} the {agent}",
    "report": lambda cue, period, agent: f"The record shows that {cue} last {period} the {agent}",
}


class RowError(ValueError):
    pass


def _single(token_text: str) -> int:
    ids = ENCODING.encode(token_text)
    if len(ids) != 1:
        raise RowError(f"not a single token: {token_text!r} -> {ids}")
    return ids[0]


def check_lexicon() -> dict[str, object]:
    if set(AGENTS) & _PRIOR_AGENTS or set(PERIODS) & _PRIOR_PERIODS:
        raise RowError("fresh lexicon overlaps a prior panel")
    if len(set(AGENTS)) != 16 or len(set(PERIODS)) != 16:
        raise RowError("lexicon tables must hold 16 distinct items each")
    for word in AGENTS + PERIODS:
        _single(" " + word)
    return {"agents": AGENTS, "periods": PERIODS, "single_token": True}


@dataclass(frozen=True)
class Row:
    row_id: str
    construction: str
    group: int
    present: bool
    text: str
    ids: tuple[int, ...]
    answer: str
    foil: str
    answer_id: int
    foil_id: int
    final: int                 # position of the last input token (the agent)
    source_positions: tuple[int, int, int]   # `last`, period noun, `the`
    reader_ids: Mapping[str, tuple[int, int]] = field(default_factory=dict)


def build_rows() -> list[Row]:
    """64 rows: 2 constructions x 16 lexical groups x {since, by}. Deterministic, no model."""
    check_lexicon()
    rows: list[Row] = []
    reader_ids = {name: (_single(a), _single(b)) for name, (a, b) in READERS.items()}
    last_id, the_id = _single(" last"), _single(" the")
    for construction, make in CONSTRUCTIONS.items():
        for group in range(16):
            agent, period = AGENTS[group], PERIODS[group]
            for present in (True, False):
                cue = "since" if present else "by"
                text = make(cue, period, agent)
                ids = ENCODING.encode(text)
                answer, foil = (" has", " had") if present else (" had", " has")
                a_id, f_id = _single(answer), _single(foil)
                if ENCODING.encode(text + answer) != ids + [a_id] or \
                        ENCODING.encode(text + foil) != ids + [f_id]:
                    raise RowError(f"joint tokenization changed for {text!r}")
                period_id = _single(" " + period)
                # the source bank is the LAST occurrence of `last`, the period noun, and `the`
                pos_last = len(ids) - 1 - ids[::-1].index(last_id)
                pos_period = len(ids) - 1 - ids[::-1].index(period_id)
                pos_the = len(ids) - 1 - ids[::-1].index(the_id)
                if not (pos_last + 1 == pos_period and pos_period + 1 == pos_the
                        and pos_the + 1 == len(ids) - 1):
                    raise RowError(f"source bank is not contiguous before the agent in {text!r}")
                row_id = hashlib.sha256(json.dumps(
                    [construction, group, present, text]).encode()).hexdigest()[:24]
                rows.append(Row(row_id, construction, group, present, text, tuple(ids), answer,
                                foil, a_id, f_id, len(ids) - 1, (pos_last, pos_period, pos_the),
                                reader_ids))
    if len(rows) != 64:
        raise RowError("expected 64 rows")
    return rows


def rows_sha256(rows: Sequence[Row]) -> str:
    payload = [[r.row_id, r.construction, r.group, r.present, r.text, list(r.ids), r.answer,
                r.foil, r.final, list(r.source_positions)] for r in rows]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


# ----------------------------------------------------------------------------- components

@dataclass(frozen=True)
class Component:
    """A named set of writes: (layer, kind, heads) at either the final query or the source bank."""
    name: str
    layer: int
    kind: str                      # "attn" (head slices) or "mlp" (full output)
    heads: tuple[int, ...]         # empty for mlp
    where: str                     # "final" or "source"


COMPONENTS = (
    Component("mlp4_source_bank", 4, "mlp", (), "source"),
    Component("attn5_h7_h1_h6_h8_final", 5, "attn", (7, 1, 6, 8), "final"),
    Component("attn9_h1_h4_final", 9, "attn", (1, 4), "final"),
    Component("attn11_h3_final", 11, "attn", (3,), "final"),
    Component("attn15_h5_final", 15, "attn", (5,), "final"),
)


def positions_of(row: Row, where: str) -> tuple[int, ...]:
    return (row.final,) if where == "final" else row.source_positions


# ----------------------------------------------------------------------------- forward

class ManualForward:
    """Mirror of `producer.Bilin18TorchBackend._forward` with donor-free slice edits."""

    def __init__(self, backend):
        self.backend = backend
        self.torch = backend.torch
        self.F = backend.F
        self.model = backend.model
        self.device = backend.device

    def _tokens(self, rows: Sequence[Row]):
        maximum = max(len(r.ids) for r in rows)
        padded = [list(r.ids) + [0] * (maximum - len(r.ids)) for r in rows]
        return self.torch.tensor(padded, dtype=self.torch.long, device=self.device)

    def _edit(self, value, rows, component: Component, mode: str, seed: int, layer_kind: str):
        """Apply the removal / null to `value` in place-of-copy and return it.

        Modes: "zero" removes the whole write; "random" subtracts an equal-norm random vector;
        "midpoint" subtracts half the exact paired (row minus partner) delta of the write, i.e.
        moves the write to the since/by midpoint (needs `self.deltas`); "midpoint_random"
        subtracts a random vector whose norm equals that half-delta's norm.
        """
        if component.kind != layer_kind:
            return value
        changed = value.clone()
        gen = None
        if mode in ("random", "midpoint_random", "project_random", "keep_only_random"):
            gen = self.torch.Generator(device="cpu").manual_seed(seed)
        self.use_subtract = mode in ("subtract", "replace")
        self.use_project = mode in ("project", "project_random", "keep_only", "keep_only_random")
        for index, row in enumerate(rows):
            for position in positions_of(row, component.where):
                if component.kind == "attn":
                    for head in component.heads:
                        s, e = head * HEAD_DIM, (head + 1) * HEAD_DIM
                        w = changed[index, position, s:e]
                        d = self._delta(row, component, position, head)
                        changed[index, position, s:e] = self._replacement(w, mode, gen, d)
                else:
                    w = changed[index, position]
                    d = self._delta(row, component, position, None)
                    changed[index, position] = self._replacement(w, mode, gen, d)
        return changed

    def _delta(self, row, component, position, head):
        """Paired delta for midpoint modes, or the explicit vector for `subtract` mode."""
        if getattr(self, "use_project", False):
            return getattr(self, "directions", {}).get((component.name, head))
        table = getattr(self, "subtract", None) if getattr(self, "use_subtract", False) else getattr(self, "deltas", None)
        if table is None:
            return None
        return table.get((row.row_id, component.name, position, head))

    def _replacement(self, w, mode: str, gen, d=None):
        if mode == "zero":
            return self.torch.zeros_like(w)
        if mode == "random":
            r = self.torch.randn(w.shape, generator=gen, dtype=self.torch.float32)
            r = r / r.norm() * float(w.float().norm())
            return w - r.to(device=w.device, dtype=w.dtype)
        if mode in ("project", "project_random", "keep_only", "keep_only_random"):
            # d here is the fixed weight-derived unit direction for this slice (see `directions`)
            if d is None:
                raise ValueError("project modes need a direction per slice")
            v = d.to(device=w.device, dtype=self.torch.float32)
            v = v / v.norm()
            coefficient = float((w.float() * v).sum())
            if mode == "project":
                return w - (coefficient * v).to(dtype=w.dtype)
            if mode == "keep_only":
                return (coefficient * v).to(dtype=w.dtype)
            if mode == "keep_only_random":
                r = self.torch.randn(w.shape, generator=gen, dtype=self.torch.float32)
                r = (r / r.norm()).to(device=w.device)
                return (float((w.float() * r).sum()) * r).to(dtype=w.dtype)
            r = self.torch.randn(w.shape, generator=gen, dtype=self.torch.float32)
            r = r / r.norm() * abs(coefficient)
            return w - r.to(device=w.device, dtype=w.dtype)
        if mode == "subtract":
            if d is None:
                raise ValueError("subtract mode needs an explicit vector per slice")
            return w - d.to(device=w.device, dtype=w.dtype)
        if mode == "replace":
            if d is None:
                raise ValueError("replace mode needs an explicit vector per slice")
            return d.to(device=w.device, dtype=w.dtype)
        if mode in ("midpoint", "midpoint_random"):
            if d is None:
                raise ValueError("midpoint modes need captured paired deltas")
            half = 0.5 * d.to(device=w.device, dtype=w.dtype)
            if mode == "midpoint":
                return w - half
            r = self.torch.randn(w.shape, generator=gen, dtype=self.torch.float32)
            r = r / r.norm() * float(half.float().norm())
            return w - r.to(device=w.device, dtype=w.dtype)
        raise ValueError(f"unknown mode {mode!r}")

    def capture(self, rows: Sequence[Row], components: Sequence[Component]):
        """Return native writes {(row_id, component, position, head|None): tensor} (no edits)."""
        torch, F, model = self.torch, self.F, self.model
        tokens = self._tokens(rows)
        store = {}
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
            x0, v1 = x, None
            for layer, block in enumerate(model.transformer.h):
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                attn_here = [c for c in components if c.kind == "attn" and c.layer == layer]
                mlp_here = [c for c in components if c.kind == "mlp" and c.layer == layer]
                handle = None
                if attn_here:
                    def c_proj_pre(_module, arguments, comps=attn_here):
                        value = arguments[0]
                        for c in comps:
                            for index, row in enumerate(rows):
                                for position in positions_of(row, c.where):
                                    for head in c.heads:
                                        s, e = head * HEAD_DIM, (head + 1) * HEAD_DIM
                                        store[(row.row_id, c.name, position, head)] = \
                                            value[index, position, s:e].detach().clone()
                        return None
                    handle = block.attn.c_proj.register_forward_pre_hook(c_proj_pre)
                try:
                    attention, v1 = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1)
                finally:
                    if handle is not None:
                        handle.remove()
                x = live + attention
                mlp = block.mlp(F.rms_norm(x, (model.config.n_embd,)))
                for c in mlp_here:
                    for index, row in enumerate(rows):
                        for position in positions_of(row, c.where):
                            store[(row.row_id, c.name, position, None)] = mlp[index, position].detach().clone()
                x = x + mlp
        return store

    def forward(self, rows: Sequence[Row], *, components: Sequence[Component] = (),
                mode: str = "zero", seed: int = 0):
        """One exact forward; every listed component is removed (mode zero) or nulled (random)."""
        torch, F, model = self.torch, self.F, self.model
        tokens = self._tokens(rows)
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
            x0, v1 = x, None
            for layer, block in enumerate(model.transformer.h):
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                attn_here = [c for c in components if c.kind == "attn" and c.layer == layer]
                mlp_here = [c for c in components if c.kind == "mlp" and c.layer == layer]
                handle = None
                if attn_here:
                    def c_proj_pre(_module, arguments, edits=attn_here):
                        value = arguments[0]
                        for k, c in enumerate(edits):
                            value = self._edit(value, rows, c, mode, seed + 1000 * k, "attn")
                        return (value,) + tuple(arguments[1:])
                    handle = block.attn.c_proj.register_forward_pre_hook(c_proj_pre)
                try:
                    attention, v1 = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1)
                finally:
                    if handle is not None:
                        handle.remove()
                x = live + attention
                mlp = block.mlp(F.rms_norm(x, (model.config.n_embd,)))
                for k, c in enumerate(mlp_here):
                    mlp = self._edit(mlp, rows, c, mode, seed + 1000 * k, "mlp")
                x = x + mlp
            logits = 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30.0)
            out = []
            for index, row in enumerate(rows):
                vec = logits[index, row.final].float()
                entry = {"answer": float(vec[row.answer_id]), "foil": float(vec[row.foil_id])}
                for name, (a, b) in row.reader_ids.items():
                    entry[name] = float(vec[a] - vec[b])
                out.append(entry)
        return out


def oriented_target(row: Row, entry: Mapping[str, float]) -> float:
    """Signed has/had margin oriented toward the row's native answer (positive = correct)."""
    return entry["answer"] - entry["foil"]


def summarize(rows: Sequence[Row], native: Sequence[Mapping[str, float]],
              arm: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Per-arm damage on the target and absolute movement of every reader."""
    target_native = [oriented_target(r, n) for r, n in zip(rows, native)]
    target_arm = [oriented_target(r, a) for r, a in zip(rows, arm)]
    damage = [n - a for n, a in zip(target_native, target_arm)]
    mean_native = sum(target_native) / len(rows)
    out = {
        "target_native_mean_margin": mean_native,
        "target_damage_mean": sum(damage) / len(rows),
        "target_damage_fraction": (sum(damage) / len(rows)) / mean_native if mean_native else float("nan"),
        "target_damage_positive_fraction": sum(1 for d in damage if d > 0) / len(rows),
        "flip_fraction": sum(1 for n, a in zip(target_native, target_arm) if n > 0 >= a) / len(rows),
    }
    for name in UNRELATED:
        moves = [abs(a[name] - n[name]) for n, a in zip(native, arm)]
        out[f"{name}_abs_move_mean"] = sum(moves) / len(rows)
        out[f"{name}_native_abs_mean"] = sum(abs(n[name]) for n in native) / len(rows)
    return out


def partner_of(rows: Sequence[Row]) -> dict[str, Row]:
    """Map each row to the row with the same construction/group and the opposite cue."""
    by_key = {(r.construction, r.group, r.present): r for r in rows}
    return {r.row_id: by_key[(r.construction, r.group, not r.present)] for r in rows}


def paired_deltas(store, rows: Sequence[Row], components: Sequence[Component]):
    """Exact per-row deltas: write(row) - write(partner) at every captured slice."""
    partner = partner_of(rows)
    deltas = {}
    for (row_id, name, position, head), w in store.items():
        row = next(r for r in rows if r.row_id == row_id)
        other = partner[row_id]
        # positions are construction-aligned, so the partner's slice sits at the same offset
        w2 = store[(other.row_id, name, position, head)]
        deltas[(row_id, name, position, head)] = w - w2
    return deltas


def random_coordinate_split(deltas, seed: int, pieces: int = 5):
    """Split every half-delta into `pieces` disjoint coordinate masks; the pieces sum to d/2.

    This is the composition null of better_circuits §1: the same total removal, cut into random
    parts that respect no module boundary. Deterministic in `seed`; keys visited in sorted order.
    """
    import torch
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = [dict() for _ in range(pieces)]
    for key in sorted(deltas, key=lambda k: (k[0], k[1], k[2], -1 if k[3] is None else k[3])):
        half = 0.5 * deltas[key]
        assignment = torch.randint(0, pieces, (half.numel(),), generator=gen)
        for k in range(pieces):
            mask = (assignment == k).to(device=half.device, dtype=half.dtype)
            out[k][key] = half * mask
    return out


def readout_directions(model, components: Sequence[Component], token_a: int, token_b: int):
    """Weight-only per-head directions `O_h^T (u_a - u_b)` in the 128-d pre-c_proj slice.

    `u` is the unembedding contrast; `O_h` is head h's block of `attn.c_proj.weight` (shape
    [n_embd, n_embd], columns h*128:(h+1)*128). No activation enters; this is a fold object.
    """
    u = (model.lm_head.weight[token_a] - model.lm_head.weight[token_b]).detach().float()
    out = {}
    for c in components:
        if c.kind != "attn":
            continue
        weight = model.transformer.h[c.layer].attn.c_proj.weight.detach().float()
        for head in c.heads:
            block = weight[:, head * HEAD_DIM:(head + 1) * HEAD_DIM]
            out[(c.name, head)] = block.T @ u
    return out


def mean_oriented_delta(deltas, rows: Sequence[Row], component: Component, head):
    """Mean over rows of (since-write minus by-write) at the final query for one head slice."""
    import torch
    acc, n = None, 0
    for row in rows:
        d = deltas.get((row.row_id, component.name, row.final, head))
        if d is None:
            continue
        d = d.float() if row.present else -d.float()
        acc = d.clone() if acc is None else acc + d
        n += 1
    return acc / n if n else None


TEMPLATE_VARYING = {
    # cue at the start, no `last`, an article before the period noun
    "ever_since_by_end": (lambda period, agent: f"Ever since the {period} the {agent}",
                          lambda period, agent: f"By the end of the {period} the {agent}"),
    # agent first, cue in a parenthetical, final input token is a comma
    "agent_first_comma": (lambda period, agent: f"The {agent}, ever since the {period},",
                          lambda period, agent: f"The {agent}, by the end of the {period},"),
    # long report frame with an extra lexical tense cue (began / ended)
    "clear_that_began_ended": (lambda period, agent: f"It is clear that since the {period} began the {agent}",
                               lambda period, agent: f"It is clear that by the time the {period} ended the {agent}"),
}


def build_template_rows(constructions: Mapping[str, tuple] = TEMPLATE_VARYING) -> list[Row]:
    """Rows for template-varying constructions: (present_builder, past_builder) per name.

    Same fresh lexicon as `build_rows`; the final input token is whatever the construction
    ends on, and `source_positions` is empty (no MLP4 source bank is declared here).
    """
    check_lexicon()
    rows: list[Row] = []
    reader_ids = {name: (_single(a), _single(b)) for name, (a, b) in READERS.items()}
    for construction, (present_make, past_make) in constructions.items():
        for group in range(16):
            agent, period = AGENTS[group], PERIODS[group]
            for present in (True, False):
                text = (present_make if present else past_make)(period, agent)
                ids = ENCODING.encode(text)
                answer, foil = (" has", " had") if present else (" had", " has")
                a_id, f_id = _single(answer), _single(foil)
                if ENCODING.encode(text + answer) != ids + [a_id] or \
                        ENCODING.encode(text + foil) != ids + [f_id]:
                    raise RowError(f"joint tokenization changed for {text!r}")
                row_id = hashlib.sha256(json.dumps(
                    ["template", construction, group, present, text]).encode()).hexdigest()[:24]
                rows.append(Row(row_id, construction, group, present, text, tuple(ids), answer,
                                foil, a_id, f_id, len(ids) - 1, (), reader_ids))
    return rows


def forward_trace(fw: "ManualForward", rows: Sequence[Row], *, components: Sequence[Component] = (),
                  mode: str = "project", seed: int = 0, from_layer: int = 0):
    """Same forward as `ManualForward.forward`, additionally returning per-row module outputs
    at the final position for every block >= from_layer, the final residual, and the block
    lambdas, so an edit's effect can be decomposed exactly through the residual recurrence
    x_{l+1} = lambda0_l x_l + lambda1_l x_0 + attn_l + mlp_l."""
    torch, F, model = fw.torch, fw.F, fw.model
    tokens = fw._tokens(rows)
    trace = [dict() for _ in rows]
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
        x0, v1 = x, None
        for layer, block in enumerate(model.transformer.h):
            live = block.lambdas[0] * x + block.lambdas[1] * x0
            attn_here = [c for c in components if c.kind == "attn" and c.layer == layer]
            mlp_here = [c for c in components if c.kind == "mlp" and c.layer == layer]
            handle = None
            if attn_here:
                def c_proj_pre(_module, arguments, edits=attn_here):
                    value = arguments[0]
                    for k, c in enumerate(edits):
                        value = fw._edit(value, rows, c, mode, seed + 1000 * k, "attn")
                    return (value,) + tuple(arguments[1:])
                handle = block.attn.c_proj.register_forward_pre_hook(c_proj_pre)
            try:
                attention, v1 = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1)
            finally:
                if handle is not None:
                    handle.remove()
            x = live + attention
            mlp = block.mlp(F.rms_norm(x, (model.config.n_embd,)))
            for k, c in enumerate(mlp_here):
                mlp = fw._edit(mlp, rows, c, mode, seed + 1000 * k, "mlp")
            x = x + mlp
            if layer >= from_layer:
                for i, row in enumerate(rows):
                    trace[i][f"attn:{layer:02d}"] = attention[i, row.final].detach().float().clone()
                    trace[i][f"mlp:{layer:02d}"] = mlp[i, row.final].detach().float().clone()
                    trace[i][f"lambda0:{layer:02d}"] = float(block.lambdas[0])
        for i, row in enumerate(rows):
            trace[i]["resid18"] = x[i, row.final].detach().float().clone()
    return trace


def final_margin(model, F, x, answer_id: int, foil_id: int) -> float:
    """The scored has/had contrast at one final-residual vector, exactly as the producer does."""
    normed = F.rms_norm(x, (x.shape[-1],))
    l = 30.0 * (model.lm_head(normed) / 30.0).tanh()
    return float(l[answer_id] - l[foil_id])


def head_source_terms(fw: "ManualForward", rows: Sequence[Row], component: Component, direction_by_head: Mapping):
    """FOLD: per row and head, the coefficient c_h = v_hat . z_h(final) split by source position and by
    the value branch: z_h(t) = sum_s p_h(t,s) [(1-lamb) V_h x_s + lamb v1_h(s)], where v1 is the
    block-0 value (a pure function of the source token). Returns, per row, per head, the exact
    reconstruction check and arrays term_current[s], term_inherited[s]. No intervention."""
    torch, F, model = fw.torch, fw.F, fw.model
    import jacclust.tt_model as TT
    tokens = fw._tokens(rows)
    block = model.transformer.h[component.layer]
    attn = block.attn
    captured = {}

    def pre_hook(_module, args):
        captured["x"] = args[0].detach().clone()
        captured["v1"] = None if len(args) < 2 or args[1] is None else args[1].detach().clone()
        return None

    handle = attn.register_forward_pre_hook(pre_hook)
    try:
        fw.forward(rows)
    finally:
        handle.remove()
    x, v1 = captured["x"], captured["v1"]
    B, T, C = x.shape
    H, D = attn.n_head, attn.head_dim
    with torch.no_grad():
        q = attn.c_q(x).view(B, T, H, D); k = attn.c_k(x).view(B, T, H, D)
        q2 = attn.c_q2(x).view(B, T, H, D); k2 = attn.c_k2(x).view(B, T, H, D)
        v_cur = attn.c_v(x).view(B, T, H, D)
        v1v = v_cur if v1 is None else v1.view_as(v_cur)
        lamb = float(attn.lamb)
        cos, sin = attn.rotary(q)
        q, k = F.rms_norm(q, (D,)), F.rms_norm(k, (D,))
        q, k = TT.apply_rotary_emb(q, cos, sin), TT.apply_rotary_emb(k, cos, sin)
        q2, k2 = F.rms_norm(q2, (D,)), F.rms_norm(k2, (D,))
        q2, k2 = TT.apply_rotary_emb(q2, cos, sin), TT.apply_rotary_emb(k2, cos, sin)
        out = []
        for i, row in enumerate(rows):
            t = row.final
            entry = {}
            for head in component.heads:
                vh = direction_by_head[(component.name, head)].to(device=x.device, dtype=torch.float32)
                vh = vh / vh.norm()
                s1 = (q[i, t, head].float() @ k[i, :t + 1, head].float().T) / D
                s2 = (q2[i, t, head].float() @ k2[i, :t + 1, head].float().T) / D
                p = s1 * s2                                   # (t+1,)
                cur = (1 - lamb) * (v_cur[i, :t + 1, head].float() @ vh)   # (t+1,)
                inh = lamb * (v1v[i, :t + 1, head].float() @ vh)
                term_cur, term_inh = p * cur, p * inh
                total = float(term_cur.sum() + term_inh.sum())
                entry[head] = {"pattern": p.tolist(), "term_current": term_cur.tolist(),
                               "term_inherited": term_inh.tolist(), "coefficient": total}
            out.append(entry)
    return out, lamb


def source_restricted_slices(fw: "ManualForward", rows: Sequence[Row], component: Component,
                             keep_source, branches=("current", "inherited"), pattern_override=None):
    """Recompute head slices z_h(final) keeping only sources s with keep_source(row, s) True and only
    the named value branches. Returns {(row_id, component.name, final, head): 128-d tensor}."""
    torch, F, model = fw.torch, fw.F, fw.model
    import jacclust.tt_model as TT
    attn = model.transformer.h[component.layer].attn
    captured = {}

    def pre_hook(_module, args):
        captured["x"] = args[0].detach().clone()
        captured["v1"] = None if len(args) < 2 or args[1] is None else args[1].detach().clone()
        return None

    handle = attn.register_forward_pre_hook(pre_hook)
    try:
        fw.forward(rows)
    finally:
        handle.remove()
    x, v1 = captured["x"], captured["v1"]
    B, T, C = x.shape
    H, D = attn.n_head, attn.head_dim
    out = {}
    with torch.no_grad():
        q = attn.c_q(x).view(B, T, H, D); k = attn.c_k(x).view(B, T, H, D)
        q2 = attn.c_q2(x).view(B, T, H, D); k2 = attn.c_k2(x).view(B, T, H, D)
        v_cur = attn.c_v(x).view(B, T, H, D)
        v1v = v_cur if v1 is None else v1.view_as(v_cur)
        lamb = float(attn.lamb)
        cos, sin = attn.rotary(q)
        q, k = F.rms_norm(q, (D,)), F.rms_norm(k, (D,))
        q, k = TT.apply_rotary_emb(q, cos, sin), TT.apply_rotary_emb(k, cos, sin)
        q2, k2 = F.rms_norm(q2, (D,)), F.rms_norm(k2, (D,))
        q2, k2 = TT.apply_rotary_emb(q2, cos, sin), TT.apply_rotary_emb(k2, cos, sin)
        for i, row in enumerate(rows):
            t = row.final
            for head in component.heads:
                s1 = (q[i, t, head].float() @ k[i, :t + 1, head].float().T) / D
                s2 = (q2[i, t, head].float() @ k2[i, :t + 1, head].float().T) / D
                p = s1 * s2
                if pattern_override is not None:
                    p = p.clone()
                    for s in range(t + 1):
                        value = pattern_override(row, s)
                        if value is not None:
                            p[s] = float(value)
                mask = torch.tensor([1.0 if keep_source(row, s) else 0.0 for s in range(t + 1)], device=p.device)
                value = torch.zeros(t + 1, D, device=p.device)
                if "current" in branches:
                    value = value + (1 - lamb) * v_cur[i, :t + 1, head].float()
                if "inherited" in branches:
                    value = value + lamb * v1v[i, :t + 1, head].float()
                out[(row.row_id, component.name, t, head)] = ((p * mask) @ value).detach()
    return out
