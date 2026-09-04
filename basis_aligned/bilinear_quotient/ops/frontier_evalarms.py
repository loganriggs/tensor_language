"""frontier_evalarms -- fit the SS312 pipeline once, evaluate many arms. (ops lane, additive, opt-in.)

Written 2026-09-04 10:06Z after measuring where the hour's 3,481 GPU-seconds went.

THE MEASUREMENT. Twelve frontier receipts, 37 full pipeline runs, 94.1 GPU-seconds each. Three of those rungs
(`frontier_error_decomposition`, `frontier_mlp_side_error_share`, `frontier_attn5_error_share`) are
**eval-only**: their knob `EVALMODE` is read at ONE place, the L2 evaluation site at lines ~673-677, which is
AFTER every fit in the file -- `fit_tableres` (289), `fit_attnd` (302), `build_arm` (493), the tail refit loop
(632). So for those rungs the entire fitting phase is byte-identical across arms and only the final `evalM`
differs, yet each arm re-runs the whole pipeline. Measured cost: **11 pipeline runs, 1,036.8 GPU-seconds, where
3 fits plus 11 evaluations would do.**

THE PATTERN. Instead of calling `main()` once per arm, call it once with a list of arms and evaluate each
against the single fitted stack. The replacement is local to the eval site and the driver:

    # module level
    ARMS = []          # list of (name, spec); empty = single-arm behaviour, unchanged

    # at the L2 evaluation site, replacing the two evalM lines
    _out = {}
    for _name, _spec in (ARMS or [(None, None)]):
        _ml, _o2 = resolve(_spec, ML, order2)      # the rung's own knob logic, per arm
        _c = evalM(FW[R0:R1], R1-R0, _o2, _ml) - baseC
        _f = evalM(FR, 120, _o2, _ml) - baseF
        _out[_name] = {'L1_F': L1F, 'L2_C': _c, 'L2_F': _f, 'increment': _f - L1F}
    return _out if ARMS else _out[None]

    # driver
    ARMS = [('baseline', None), ('motif_off', ...), ('tail_off', ...)]
    res = main()                                   # ONE call, all arms

WHAT THIS DOES NOT TOUCH. No existing rung is modified: every landed script's bytes are cited by a ledger
section and its receipt must stay reproducible. `ops/frontier_fisher8.py` (SS2125 rung 30) is not touched
either. This file is a template for the NEXT eval-only derivation, and `check_eval_only()` below is the guard
that says whether a given rung qualifies -- so the 4x refit is never paid again without noticing.

CAUTION, and the reason this is opt-in rather than automatic: a rung whose knob is read DURING fitting
(`frontier_a5_constant_collapse`, `frontier_tail_refit_split`, the collapse family) genuinely needs one full
run per arm, and applying this pattern there would silently evaluate every arm against the baseline's fits.
`check_eval_only()` exists to make that mistake impossible to make by hand.
"""
from __future__ import annotations

import re
from pathlib import Path

# The boundary is the last statement that STORES a fitted entry, not the last line that mentions a fitter.
# Using the refit loop's HEADER as the boundary mislabelled `frontier_tail_refit_split`, whose knob is read
# inside that loop's body -- caught by test_refitting_rungs_are_NOT_mislabelled before this shipped.
FIT_ASSIGNMENTS = ("S[f'a{li}L']=", "S[f'a{li}']=fit_attnd", "=fit_tableres(", "=fit_res(", "cfgF=build_arm(")


def check_eval_only(path, knob="EVALMODE"):
    """True iff every read of `knob` inside main() occurs AFTER the last fit call.

    Returns (is_eval_only, last_fit_line, knob_lines). A rung that is eval-only can share one fitted stack
    across arms; one that is not must re-run the pipeline per arm.
    """
    src = Path(path).read_text().splitlines()
    try:
        start = next(i for i, l in enumerate(src) if l.startswith("def main("))
    except StopIteration:
        return False, None, []
    end = next((i for i in range(start + 1, len(src)) if src[i].startswith("if __name__")), len(src))
    body = list(enumerate(src[start:end], start=start + 1))
    last_fit = max((n for n, l in body if any(m in l.replace(" ", "") for m in
                    [m.replace(" ", "") for m in FIT_ASSIGNMENTS])), default=None)
    knob_lines = [n for n, l in body if re.search(rf"\b{re.escape(knob)}\b", l)]
    if last_fit is None or not knob_lines:
        return False, last_fit, knob_lines
    return min(knob_lines) > last_fit, last_fit, knob_lines


def savings(n_arms, seconds_per_run, eval_fraction=0.07):
    """GPU-seconds saved by fitting once instead of per arm, given a measured per-run cost."""
    per_run = seconds_per_run
    naive = n_arms * per_run
    shared = per_run + (n_arms - 1) * per_run * eval_fraction
    return round(naive - shared, 1)


# ---------------------------------------------------------------------------
# Arm builders, added 2026-09-04 12:06Z from that hour's measurement.
#
# THE MEASUREMENT. GPU utilisation was 32% -- 1,253.8 GPU-seconds busy in a 66-minute hour -- with two idle
# gaps of 19 and 21 minutes, both with the queue at zero. The GPU is no longer the constraint; MY AUTHORING
# LATENCY is. And the economics of arms make that worse than it needs to be: a fit costs ~90 s and an arm
# ~3.5 s, so
#
#     8 arms in a rung -> 14.8 s per arm      40 arms -> 5.8 s per arm      80 arms -> 4.6 s per arm
#
# This hour ran 79 arms across EIGHT rungs at 15.9 s per arm. The same 79 arms as two rungs of 40 would have
# cost 460 s instead of 1,253.8 -- a 63% saving -- and, more importantly, needed SIX FEWER AUTHORING CYCLES.
# Bigger rungs are strictly better on both axes, so the useful tool is one that makes a big arm set cheap to
# WRITE. Hand-enumerating a factorial is exactly the step that keeps rungs small.
#
# These builders are opt-in and change no existing rung.

def factorial_arms(knobs, baseline=True, prefix=""):
    """Full factorial over {knob_name: [values]} as [(arm_name, spec), ...].

    A value of None means "leave this knob alone", so grids that include the unmodified setting are natural.
    Arm names are deterministic and collision-free, so a receipt keyed by them is stable across runs.
    """
    import itertools
    names = sorted(knobs)
    arms = [("baseline", {})] if baseline else []
    for combo in itertools.product(*(knobs[k] for k in names)):
        spec = {k: v for k, v in zip(names, combo) if v is not None}
        if not spec:
            continue
        tag = "_".join(f"{k[:4]}{_fmt(v)}" for k, v in sorted(spec.items()))
        arms.append((prefix + tag, spec))
    return arms


def subset_arms(blocks, baseline=True):
    """Every non-empty subset of {label: spec_fragment}, the 2^k pattern used for interaction structure.

    `blocks` maps a single-letter label to the spec fragment that switches that block on; the arm for a
    subset merges its fragments, and is named by concatenating the labels in sorted order (T, C, TC, ...).
    """
    import itertools
    labels = sorted(blocks)
    arms = [("baseline", {})] if baseline else []
    for r in range(1, len(labels) + 1):
        for combo in itertools.combinations(labels, r):
            spec = {}
            for lab in combo:
                spec.update(blocks[lab])
            arms.append(("".join(combo), spec))
    return arms


def _fmt(v):
    """Compact, filename-safe rendering of a knob value for an arm name."""
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, (int, float)):
        return f"{int(round(float(v) * 100)):03d}"
    return str(v)
