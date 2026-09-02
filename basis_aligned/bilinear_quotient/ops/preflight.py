#!/usr/bin/env python3
"""Optional static preflight for new rung scripts (ops lane, advisory only).

Targets the three instrument-clause failure classes that cost reruns on
2026-09-01 (rungs 419 and 424: 5 extra GPU runs + audit cycles):

  1. HASH CLASS   -- comparing a raw-byte digest against a rows receipt's
                     semantic `tensor_sha256` field (never equal).
  2. RETRACTION   -- optimizing bases with Adam then gating pred_a on
                     `_orth_error(...) <= tol` without a QR retraction.
  3. ABS-VS-REL   -- absolute `*_max_abs` tolerance bars on float32
                     replay quantities (prefer relative-squared bars).
  6. CHAIN HOP    -- multi-hop module attribute chains like
                     `parent.parent.path_parent.parent.helper(...)` (rung
                     467: one hop wrong in a 6-deep inheritance chain =
                     one abort).  Import the owning module directly.
  5. ORDER CHECK  -- tuple(...) equality against a frozen ID sequence
                     where the operand comes from JSON/dict iteration
                     (rung 456: spec reserialized in a different order,
                     zero content change, one abort).  Prefer set
                     equality + explicit IDS-indexed lookup.
  4. CONTROL WIN  -- absolute numeric windows on shuffle/permutation
                     control statistics (rungs 428/429/432: three window
                     mis-derivations; prefer matched-control EXCESS).

Usage: python ops/preflight.py ops/<script>.py   (warnings only; exit 0)
Never a gate: registration bars are the registrant's; this only warns.
"""
import re
import sys
from pathlib import Path


def check(path: Path) -> list[str]:
    text = path.read_text()
    warnings = []
    raw_digest = re.search(r"numpy\(\)\.tobytes\(\)|_digest\(", text)
    semantic = re.search(r"['\"]tensor_sha256['\"]", text)
    uses_semantic_fn = "rows_life.base.tensor_sha256" in text
    if raw_digest and semantic and not uses_semantic_fn:
        warnings.append(
            "HASH CLASS: file compares against receipt 'tensor_sha256' but "
            "computes raw-byte digests; use rows_life.base.tensor_sha256 "
            "(rung 424 cost 2 reruns on this).")
    optimizes = re.search(r"Adam|_optimize\(", text)
    orth_gate = re.search(r"_orth_error\([^)]*\)\s*<=", text)
    retracts = re.search(r"linalg\.qr|retraction", text)
    if optimizes and orth_gate and not retracts:
        warnings.append(
            "RETRACTION: bases are optimized and pred gates on _orth_error "
            "but no QR retraction found (rung 424 cost 1 rerun on this).")
    control_window = any(
        re.search(r"control", line)
        and re.search(r"(<=?|>=?)\s*\.\d|\.\d+\s*<", line)
        for line in text.splitlines())
    excess = re.search(r"excess[_a-z]*\s*=|control_gap|minus_control", text)
    if control_window and not excess:
        warnings.append(
            "CONTROL WINDOW: absolute numeric bounds on a shuffle/control "
            "statistic; three window mis-derivations on 2026-09-01 "
            "(428/429/432) -- prefer matched-control excess statistics.")
    abs_replay = any(
        re.search(r"replay|logit_difference", line)
        and re.search(r"(<=|<)\s*[0-9.]*e-[0-9]+", line)
        and not re.search(r"relative", line)
        for line in text.splitlines())
    rel_available = False  # suppression is line-scoped above
    if abs_replay and not rel_available:
        warnings.append(
            "ABS-VS-REL: absolute max-abs bar on a replay quantity with no "
            "relative-squared companion; float32 magnitudes broke this bar "
            "on rung 419 (1 rerun).")
    func_ad = re.search(r"torch\.func\.(jvp|vjp|grad|jacfwd|jacrev)", text)
    bf16 = re.search(r"bfloat16|bf16", text)
    if func_ad and bf16:
        warnings.append(
            "FUNC-AD DTYPE: torch.func autodiff alongside a bfloat16 model; "
            "torch.func.jvp promoted BF16 duals to float32 and crashed rung "
            "483 pre-outcome (11:29). Prefer torch.autograd.functional.* or "
            "pin dual dtypes explicitly.")
    batch_consts = {m.group(1): int(m.group(2)) for m in re.finditer(
        r"^([A-Z_]*BATCH[A-Z_]*)\s*=\s*(\d+)\s*$", text, re.M)}
    bound_consts = {m.group(1): int(m.group(3)) for m in re.finditer(
        r"^([A-Z_]*(HALF|BOUNDARY|SPLIT|STOP)[A-Z_]*)\s*=\s*(\d+)\s*$", text, re.M)}
    batch_granular = re.search(r"int\(\bstart\b\s*>=|=\s*\bstart\b\s*>=", text)
    per_row = re.search(r"\brows?\b\s*>=", text)
    if batch_consts and bound_consts and batch_granular and not per_row:
        for bn, bv in batch_consts.items():
            for cn, cv in bound_consts.items():
                if bv > 1 and cv % bv != 0:
                    warnings.append(
                        f"BATCH-BOUNDARY STRADDLE: {cn}={cv} is not a multiple of "
                        f"{bn}={bv} and a group label is derived from the batch "
                        "start; the batch crossing the boundary gets wholly "
                        "assigned to one side (rung 477 data defect, cost a "
                        "repair rung). Use per-row masks (rows >= boundary).")
    loop_consts = {m.group(1): int(m.group(2)) for m in re.finditer(
        r"^([A-Z_]+)\s*=\s*(\d+)\s*$", text, re.M)}
    for stride_name in [n for n in ("BATCH",) if n in loop_consts]:
        stride = loop_consts[stride_name]
        if stride <= 1:
            continue
        for name, value in loop_consts.items():
            if name == stride_name or value % stride == 0:
                continue
            if re.search(rf"range\([^)]*{name}[^)]*,\s*{stride_name}\)", text):
                warnings.append(
                    f"BOUNDARY LOOP: range(...{name}..., {stride_name}) with "
                    f"{name}={value} % {stride_name}={stride} != 0 -- misaligned "
                    "batches; (i//BATCH)*BATCH cache keys break (v2 crash 18:32, "
                    "Codex 502 caught same class pre-outcome 19:02).")
    return warnings



def _chain_check(text):
    import re as _re
    hits = [line.strip() for line in text.splitlines()
            if _re.search(r"\w+(\.(parent|path_parent)){2,}\.", line)]
    if hits:
        return ("CHAIN HOP: multi-hop module attribute chain (rung 467 "
                "class); import the helper's owning module directly. "
                "Lines: " + " | ".join(h[:60] for h in hits[:2]))
    return None


def _order_check(text):
    import re as _re
    loads_json = _re.search(r"json\.loads?\(", text)
    hits = [line.strip() for line in text.splitlines()
            if _re.search(r"tuple\([^)]*\)\s*[!=]=", line)]
    if loads_json and hits:
        return ("ORDER CHECK: tuple(...) equality likely freezes dict/JSON "
                "iteration order (rung 456 class); prefer set equality + "
                "IDS-indexed lookup. Lines: " + " | ".join(h[:70] for h in hits[:2]))
    return None


def main() -> None:
    bad = 0
    for arg in sys.argv[1:]:
        _text = Path(arg).read_text()
        extras = [w for w in (_order_check(_text), _chain_check(_text)) if w]
        warnings_all = check(Path(arg)) + extras
        for warning in warnings_all:
            bad += 1
            print(f"{arg}: WARN {warning}")
    if not bad:
        print("preflight: no findings")


if __name__ == "__main__":
    main()
