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
curves = {**v66["sets"], **v67["sets"]}
ORDER = ["quantifier_number", "verb_preposition", "polarity_licensing", "dative", "verb_complementizer", "voice_frame"]
r3 = lambda x: f"{x:.3f}"
mark = lambda b: "✓" if b else "✗"
lines = [f"# Terminal evidence — removal-greedy head sets (generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC by ops/terminal_table_greedy_sets.py)", "",
         "Rows evaluated on the ODD half of each family (directions = per-block diff-in-means fit on the EVEN half); rubric rows: 2 extraction ≥0.80 (LB ≥0.60), 3 removal LB>0 with own-C specificity LB>0, 4 own-C CE UB ≤0.01, 5 A1-fit direction on A2 LB>0 and ≥0.50× A1. Receipts: v66/v67 (curves, cross), v68 (extraction), v69 (rows 3–5), v70 (C-penalised).", "",
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
lines += ["", "## Sets", ""]
for n in ORDER:
    c = curves[n]
    lines.append(f"- **{n}** hub {c['hub']} → additions in order {[u[5:] for u in c['final'][len(c['hub']):]]}" + (f"; C-penalised additions {[u[5:] for u in v70['sets'][n]['final'][len(c['hub']):]]}" if n in v70["sets"] else ""))
lines += ["", "## Notes", "",
          "- verb_complementizer's C family (\"The leader noted/replied quickly → that\", foil whether) is the same that/whether prediction from that-taking verbs; the hub alone damages it by 0.31 (v70 curve k=0) on the that/whether margin (v69: margin −2.1, KL 0.03). Row 4 as written fails at every set size; the direction is a verb-class axis shared by all reporting verbs. v57's C for the verb sets was polarity's C (borrowed control) and is superseded.",
          "- Row 4 on 16-document halves: bootstrap half-width ≈0.025, so UB ≤0.01 is unreachable at zero mean (polarity: point 0.006, UB 0.030). Read polarity's row 4 from full rows (v51) or as point+width.",
          "- dative's A2 deficit is direction-keyed (A2-fit 0.63 vs A1-fit 0.22 at hub+8; v61/v62), unchanged by enlargement.",
          "- Cross-collateral (A1-fit direction on the other five A1 families, odd rows): max 0.069 (dative→verb set), otherwise ≤0.043; quantifier's direction LOWERS dative/polarity CE by 0.11–0.13 (shared number axis, v54)."]
out = F / "TERMINAL_TABLE_GREEDY_2026-09-07.md"
out.write_text("\n".join(lines) + "\n")
print(out); print("\n".join(lines[4:12]))
