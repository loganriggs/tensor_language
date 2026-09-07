#!/usr/bin/env python3
"""Durable terminal-evidence table for the removal-greedy sets (hub / hub+3 / hub+8 / C-penalised), from receipts only.

Sources: v66, v67 (removal curves, cross), v68 (extraction), v69 (rows 3-5 on odd rows), v70 (C-penalised sets).
All numbers are ODD-row evaluations with directions fit on EVEN rows unless the column says otherwise.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
F = ROOT / "circuits/followups"
J = lambda n: json.loads((F / n).read_text())
v66, v67, v68, v69, v70 = (J(f"unit_{n}_result.json") for n in ("verb_greedy_saturation_v66", "four_sets_greedy_saturation_v67", "greedy_sets_extraction_v68", "greedy_sets_terminal_rows_v69", "c_penalised_greedy_v70"))
v75 = J("unit_six_sets_constrained_das_v75_result.json")
v76 = J("unit_constrained_das_extraction_v76_result.json")
v80 = J("unit_six_sets_cross_inert_v80_result.json")
import os as _os
v81 = J("unit_cross_inert_extraction_v81_result.json") if _os.path.exists(str(F / "unit_cross_inert_extraction_v81_result.json")) else None
HUB16 = {"voice_frame": ("unit_voice_greedy_continuation_v83_result.json", "unit_voice_hub16_full_specificity_v85_result.json")}
curves = {**v66["sets"], **v67["sets"]}
ORDER = ["quantifier_number", "verb_preposition", "polarity_licensing", "dative", "verb_complementizer", "voice_frame"]
r3 = lambda x: f"{x:.3f}"
mark = lambda b: "✓" if b else "✗"
lines = [f"# Terminal evidence — removal-greedy head sets (generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC by ops/terminal_table_greedy_sets.py)", "",
         "Rows evaluated on the ODD half of each family (directions = per-block diff-in-means fit on the EVEN half); rubric rows: 2 extraction ≥0.80 (LB ≥0.60), 3 removal LB>0 with own-C specificity LB>0, 4 own-C CE UB ≤0.01, 5 A1-fit direction on A2 LB>0 and ≥0.50× A1. Receipts: v66/v67 (curves, cross), v68 (extraction), v69 (rows 3–5), v70 (C-penalised), v75 (constrained DAS: direction = rank-1-per-block DAS on pooled A1 + v15 verb-variant EVEN rows, complement term, C-removal-inertness regularizer λ=30 on C EVEN rows; A2 and the odd rows never fitted), v76 (extraction of those directions on A1 and A2 odd rows).", "",
         "| behaviour | set | n | removal A1 (LB) | extraction (LB) | own C (UB) | A2 a1-fit | A2 a2-fit | random | max cross | rows 2/3/4/5 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for n in ORDER:
    c = curves[n]; e = v68["sets"][n]["extraction"]; t = v69["sets"][n]; rm = v69["rows_met"][n]
    hub_n, fin = len(c["hub"]), c["final"]
    cross = max(t["cross"].values())
    # hub row (removal from the curve base; extraction from v68; rows 3-5 not re-measured at hub size here)
    lines.append(f"| {n} | hub | {hub_n} | {r3(c['base_odd'])} | {r3(e['hub']['point'])} ({r3(e['hub']['lb95'])}) | — | — | — | — | — | {mark(e['hub']['point'] >= 0.8 and e['hub']['lb95'] >= 0.6)}/·/·/· |")
    k3 = c["curve"][2]
    lines.append(f"| {n} | hub+3 | {hub_n + 3} | {r3(k3['odd_damage'])} ({r3(k3['odd_lb'])}) | {r3(e['hub3']['point'])} ({r3(e['hub3']['lb95'])}) | — | — | — | — | — | {mark(e['hub3']['point'] >= 0.8 and e['hub3']['lb95'] >= 0.6)}/·/·/· |")
    lines.append(f"| {n} | hub+8 | {len(fin)} | {r3(t['A1']['ce_damage'])} ({r3(t['A1']['ce_lb975'])}) | {r3(e['hub8']['point'])} ({r3(e['hub8']['lb95'])}) | {r3(t['C']['ce_damage'])} ({r3(t['C']['ce_ub975'])}) | {r3(t['A2_a1fit']['ce_damage'])} | {r3(t['A2_a2fit']['ce_damage'])} | {r3(t['random_A1']['ce_damage'])} | {r3(cross)} | {mark(e['hub8']['point'] >= 0.8 and e['hub8']['lb95'] >= 0.6)}/{mark(rm['row3'])}/{mark(rm['row4'])}/{mark(rm['row5'])} |")
    if n in v70["sets"]:
        p = v70["sets"][n]; ev = p["eval_odd"]; ex = p["extraction_odd"]
        row4 = ev["C"]["ce_ub975"] <= 0.01
        row5 = ev["A2"]["ce_lb975"] > 0 and ev["A2"]["ce_damage"] >= 0.5 * ev["A1"]["ce_damage"]
        lines.append(f"| {n} | hub+8 C-pen (λ=2) | {len(p['final'])} | {r3(ev['A1']['ce_damage'])} ({r3(ev['A1']['ce_lb975'])}) | {r3(ex['point'])} ({r3(ex['lb95'])}) | {r3(ev['C']['ce_damage'])} ({r3(ev['C']['ce_ub975'])}) | {r3(ev['A2']['ce_damage'])} | — | — | — | {mark(ex['point'] >= 0.8 and ex['lb95'] >= 0.6)}/{mark(ev['A1']['ce_lb975'] > 0 and ev['A1']['ce_lb975'] - ev['C']['ce_ub975'] > 0)}/{mark(row4)}/{mark(row5)} |")
    if n in v75["sets"]:
        P = v75["sets"][n]["damage"]["30.0"]; Z = v75["sets"][n]["damage"]["0.0"]; rn = v75["sets"][n]["damage"]["random"]["A1"]
        crossp = max(P["cross"].values())
        row3 = P["A1"]["ce_lb975"] > 0 and P["A1"]["ce_lb975"] - P["C"]["ce_ub975"] > 0
        row4 = P["C"]["ce_ub975"] <= 0.01
        row5 = P["A2"]["ce_lb975"] > 0 and P["A2"]["ce_damage"] >= 0.5 * P["A1"]["ce_damage"]
        x1 = v76["sets"][n]["extraction"]["A1"]["cdas"]; x2 = v76["sets"][n]["extraction"]["A2"]["cdas"]
        row2 = x1["point"] >= 0.8 and x1["lb95"] >= 0.6
        lines.append(f"| {n} | hub+8 constrained DAS (pooled, λ=30) | {len(v75['sets'][n]['units'])} | {r3(P['A1']['ce_damage'])} ({r3(P['A1']['ce_lb975'])}) | {r3(x1['point'])} ({r3(x1['lb95'])}); A2 {r3(x2['point'])} | {r3(P['C']['ce_damage'])} ({r3(P['C']['ce_ub975'])}); λ=0 {r3(Z['C']['ce_damage'])} | {r3(P['A2']['ce_damage'])} | — | {r3(rn['ce_damage'])} | {r3(crossp)} | {mark(row2)}/{mark(row3)}/{mark(row4)}/{mark(row5)} |")
    if n in v80["sets"]:
        X = v80["sets"][n]["damage"]["xctl"]; rn = v80["sets"][n]["damage"]["random"]["A1"]
        crossx = max(abs(v) for v in X["cross"].values())
        row3 = X["A1"]["ce_lb975"] > 0 and X["A1"]["ce_lb975"] - X["C"]["ce_ub975"] > 0
        row4 = X["C"]["ce_ub975"] <= 0.01
        row5 = X["A2"]["ce_lb975"] > 0 and X["A2"]["ce_damage"] >= 0.5 * X["A1"]["ce_damage"]
        if v81 is not None:
            x1 = v81["sets"][n]["extraction"]["A1"]["xdas"]; x2 = v81["sets"][n]["extraction"]["A2"]["xdas"]
            ext = f"{r3(x1['point'])} ({r3(x1['lb95'])}); A2 {r3(x2['point'])}"; row2 = mark(x1["point"] >= 0.8 and x1["lb95"] >= 0.6)
        else:
            ext, row2 = "pending (v81)", "?"
        unseen = r3(X["unseen"]["ce_damage"]) if "unseen" in X else "—"
        lines.append(f"| {n} | hub+8 full-specificity DAS (own C + 5 other A1 as inertness controls, 30 each) | {len(v80['sets'][n]['units'])} | {r3(X['A1']['ce_damage'])} ({r3(X['A1']['ce_lb975'])}) | {ext} | {r3(X['C']['ce_damage'])} ({r3(X['C']['ce_ub975'])}) | {r3(X['A2']['ce_damage'])} | unseen pair {unseen} | {r3(rn['ce_damage'])} | max abs {r3(crossx)} | {row2}/{mark(row3)}/{mark(row4)}/{mark(row5)} |")
    if n in HUB16 and all(_os.path.exists(str(F / f)) for f in HUB16[n]):
        g16, d16 = (J(f) for f in HUB16[n])
        S16 = g16["sets"]["hub16"]; e16 = S16["extraction"]; t16 = S16["removal"]
        lines.append(f"| {n} | hub+16 (v83 continuation, dim) | {len(g16['hub16'])} | {r3(t16['A1']['ce_damage'])} ({r3(t16['A1']['ce_lb975'])}) | {r3(e16['A1']['point'])} ({r3(e16['A1']['lb95'])}); A2 {r3(e16['A2']['point'])} | {r3(t16['C']['ce_damage'])} ({r3(t16['C']['ce_ub975'])}) | {r3(t16['A2']['ce_damage'])} | — | — | max abs {r3(max(abs(v) for v in t16['cross'].values()))} | {mark(e16['A1']['point'] >= 0.8 and e16['A1']['lb95'] >= 0.6)}/{mark(t16['A1']['ce_lb975'] > 0 and t16['A1']['ce_lb975'] - t16['C']['ce_ub975'] > 0)}/{mark(t16['C']['ce_ub975'] <= 0.01)}/{mark(t16['A2']['ce_lb975'] > 0 and t16['A2']['ce_damage'] >= 0.5 * t16['A1']['ce_damage'])} |")
        D = d16["sets"][n]; X = D["removal"]["xdas"]; x1 = D["extraction"]["A1"]["xdas"]; x2 = D["extraction"]["A2"]["xdas"]; rn = D["extraction"]["A1"]["random"]
        lines.append(f"| {n} | hub+16 full-specificity DAS (v85) | {len(D['units'])} | {r3(X['A1']['ce_damage'])} ({r3(X['A1']['ce_lb975'])}) | {r3(x1['point'])} ({r3(x1['lb95'])}); A2 {r3(x2['point'])} | {r3(X['C']['ce_damage'])} ({r3(X['C']['ce_ub975'])}) | {r3(X['A2']['ce_damage'])} | — | ext {r3(rn['point'])} | max abs {r3(max(abs(v) for v in X['cross'].values()))} | {mark(x1['point'] >= 0.8 and x1['lb95'] >= 0.6)}/{mark(X['A1']['ce_lb975'] > 0 and X['A1']['ce_lb975'] - X['C']['ce_ub975'] > 0)}/{mark(X['C']['ce_ub975'] <= 0.01)}/{mark(X['A2']['ce_lb975'] > 0 and X['A2']['ce_damage'] >= 0.5 * X['A1']['ce_damage'])} |")
lines += ["", "## Sets", ""]
for n in ORDER:
    c = curves[n]
    lines.append(f"- **{n}** hub {c['hub']} → additions in order {[u[5:] for u in c['final'][len(c['hub']):]]}" + (f"; C-penalised additions {[u[5:] for u in v70['sets'][n]['final'][len(c['hub']):]]}" if n in v70["sets"] else ""))
lines += ["", "## Notes", "",
          "- verb_complementizer C: the hub alone damages own C (\"The leader noted/replied quickly → that\", foil whether) by 0.31 (v70 curve k=0) on the that/whether margin (v69: margin −2.1, KL 0.03). Row 4 as written fails at every set size. v71: the direction is NOT a verb-class axis — it transfers only 0.26–0.41× to three unseen verb pairs (per-pair refits 1.7–3.1× stronger, block |cos| 0.43–0.50); single-pair directions are pair-keyed. v72: a POOLED direction (three pairs, 48 docs) transfers to the unseen pair at 0.56× (pooled diff-in-means 0.619), 0.61× (DAS+inertness 0.679), 0.67× (DAS 0.740) of its refit 1.110 while keeping the fitted pairs at ≥0.86×: a shared rank-1 axis exists and the single-pair fits were noisy samples. Pooling RAISES own-C damage (0.45–0.73). The reading posted at 00:42 ('C shares the that/whether output axis, row 4 unmeasurable') was REFUTED by v75: the C-inertness regularizer drives own C to 0.029 (UB 0.046) while keeping A1 at 1.025 (0.92× of the unconstrained 1.112) — the C-damaging component was separable and the C control is sound. Row 4 by the UB ≤0.01 bar is still not met for this set (residual 0.03, LB 0.011). v57 tested the verb sets against polarity's C (borrowed control) and is superseded.",
          "- Row 4 on 16-document halves: bootstrap half-width ≈0.025, so UB ≤0.01 is unreachable at zero mean (polarity: point 0.006, UB 0.030). Read polarity's row 4 from full rows (v51) or as point+width.",
          "- dative's A2 deficit is direction-keyed (A2-fit 0.63 vs A1-fit 0.22 at hub+8; v61/v62), unchanged by enlargement. v73/v74: a DAS direction pooled over A1 + two verb variants with a C-removal-inertness regularizer (λ=30, C even rows; `g.fit_block_subspace_constrained`) meets row 5 on odd rows (A2 0.313 = 0.55× A1 0.565, LB 0.279), transfers to an unseen verb pair at 0.87× its refit, keeps cross-collateral ≤0.042, and holds own C at 0.006 (UB 0.021 — misses the row-4 UB bar by bootstrap width). Without the regularizer the pooled direction damages C by 0.125.",
          "- Cross-collateral (A1-fit direction on the other five A1 families, odd rows): max 0.069 (dative→verb set), otherwise ≤0.043; quantifier's direction LOWERS dative/polarity CE by 0.11–0.13 (shared number axis, v54)."]
out = F / "TERMINAL_TABLE_GREEDY_2026-09-07.md"
out.write_text("\n".join(lines) + "\n")
print(out); print("\n".join(lines[4:12]))
