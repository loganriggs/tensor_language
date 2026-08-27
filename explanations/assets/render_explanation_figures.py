#!/usr/bin/env python3
"""Render reader-facing figures from checked-in reverse-engineering evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["svg.hashsalt"] = "tensor-language-explanations-v1"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "basis_aligned" / "bilinear_quotient"

COLORS = {"navy": "#17324d", "blue": "#3478b8", "green": "#2d936c",
          "gold": "#e3a82b", "red": "#c94c4c", "gray": "#eef2f5",
          "ink": "#17202a"}


def save(fig, name):
    svg_path = HERE / name
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white",
                metadata={"Date": "2026-08-27"})
    # Matplotlib emits harmless trailing spaces in path definitions. Normalize them
    # so repeated generation is deterministic and passes repository whitespace gates.
    svg_path.write_text("\n".join(line.rstrip() for line in
                                  svg_path.read_text().splitlines())+"\n")
    fig.savefig(HERE / name.replace(".svg", ".png"), dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def flow_figure():
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.2); ax.axis("off")
    boxes = [
        (0.2, 2.3, 2.0, 1.1, "token $t$", "embedding / lexical identity"),
        (2.8, 2.3, 2.1, 1.1, "$x_0$, attention-0", "local context arrives"),
        (5.5, 2.3, 2.2, 1.1, "$z_0=\\mathrm{RMSNorm}(\u00b7)$", "angular residual state"),
        (8.3, 2.3, 2.4, 1.1, "$(Lz_0)\\odot(Rz_0)$", "bilinear class detectors"),
        (11.2, 2.3, 1.6, 1.1, "$m_0$ write", "residual update"),
    ]
    for x, y, w, h, title, subtitle in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=.08",
                               facecolor=COLORS["gray"], edgecolor=COLORS["navy"], lw=1.5)
        ax.add_patch(patch)
        ax.text(x+w/2, y+.70, title, ha="center", va="center", fontsize=12,
                color=COLORS["navy"])
        ax.text(x+w/2, y+.27, subtitle, ha="center", va="center", fontsize=8.5,
                color=COLORS["ink"])
    for left, right in zip(boxes, boxes[1:]):
        ax.add_patch(FancyArrowPatch((left[0]+left[2], 2.85), (right[0], 2.85),
                                    arrowstyle="-|>", mutation_scale=13,
                                    lw=1.4, color=COLORS["blue"]))
    surrogate = FancyBboxPatch((1.25, .35), 9.7, .85,
                               boxstyle="round,pad=0.05,rounding_size=.08",
                               facecolor="#eaf6f1", edgecolor=COLORS["green"], lw=1.5)
    ax.add_patch(surrogate)
    ax.text(6.1, .89, "Measured surrogate: token table  +  low-rank linear correction from [attention-0 output, $x_0$]",
            ha="center", va="center", fontsize=11, color=COLORS["ink"])
    ax.text(6.1, .52, "This predicts the write; it is not a claim that the native MLP literally contains a lookup table.",
            ha="center", va="center", fontsize=8.8, color=COLORS["red"])
    ax.add_patch(FancyArrowPatch((6.1, 1.2), (9.5, 2.28), arrowstyle="-|>",
                                mutation_scale=12, connectionstyle="arc3,rad=-.18",
                                lw=1.2, color=COLORS["green"]))
    ax.set_title("MLP0: native computation and the simpler program that reproduces most of it",
                 fontsize=15, color=COLORS["navy"], pad=8)
    save(fig, "mlp0_computation.svg")


def evidence_curve():
    held = json.loads((DATA / "mlp0_quantized_eval_results.json").read_text())
    comp = json.loads((DATA / "mlp0_attention_composite_curve_results.json").read_text())
    ranks = [row["ridge_rank"] for row in held["points"]]
    fidelity = [100*row["fidelity"] for row in held["points"]]
    composed = [row["composed_ce"] for row in comp["points"]]
    x = range(len(ranks))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(x, fidelity, marker="o", lw=2.4, color=COLORS["blue"], label="held-out fidelity")
    ax.set_xticks(list(x), [str(rank) for rank in ranks])
    ax.set_xlabel("context-correction rank (0 = token table only)")
    ax.set_ylabel("fraction of MLP0 ablation gap recovered (%)", color=COLORS["blue"])
    ax.tick_params(axis="y", colors=COLORS["blue"]); ax.set_ylim(80, 92)
    ax.grid(axis="y", alpha=.22)
    other = ax.twinx()
    other.plot(x, composed, marker="s", lw=2.2, color=COLORS["gold"],
               label="CE with attention replacement")
    other.set_ylabel("composed cross-entropy (lower is better)", color="#9a6a00")
    other.tick_params(axis="y", colors="#9a6a00"); other.set_ylim(3.18, 3.31)
    lines = ax.lines + other.lines
    ax.legend(lines, [line.get_label() for line in lines], loc="center right", frameon=False)
    ax.set_title("MLP0 evidence: lexical table carries most behavior; context correction helps monotonically")
    fig.tight_layout(); save(fig, "mlp0_fidelity_curve.svg")


def status_map():
    rows = [
        ("MLP0", .90, .62, .90, "strong operational program; partial semantics"),
        ("MLP1", .97, .42, .78, "high fidelity, but quadratic closure / meaning incomplete"),
        ("MLP2", .82, .25, .66, "rank-128 affine program; downstream reader issue"),
        ("MLP3", .81, .22, .72, "local-DAG affine program; semantics sparse"),
        ("MLP4", .72, .18, .35, "partial input model only; not operationally closed"),
        ("Attention QK", .96, .28, .88, "139-head decoded routing program; V/O still live"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    y = list(range(len(rows)))[::-1]
    width = .22
    for offset, index, color, label in [(-width, 1, COLORS["blue"], "computational reproduction"),
                                         (0, 2, COLORS["gold"], "semantic localization"),
                                         (width, 3, COLORS["green"], "operational closure")]:
        ax.barh([v+offset for v in y], [row[index] for row in rows], height=.19,
                color=color, label=label)
    ax.set_yticks(y, [row[0] for row in rows]); ax.set_xlim(0, 1.62)
    ax.set_xlabel("evidence maturity (editorial summary, not a measured score)")
    ax.grid(axis="x", alpha=.2); ax.legend(ncol=3, loc="lower center",
                                           bbox_to_anchor=(.5, 1.01), frameon=False)
    for yi, row in zip(y, rows):
        ax.text(1.02, yi, row[4], va="center", fontsize=8, color=COLORS["ink"])
    ax.set_title("What “understood” currently means — the dimensions are not interchangeable", pad=35)
    fig.tight_layout(); save(fig, "understanding_status.svg")


def mlp1_flow_figure():
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.8); ax.axis("off")
    native = [(0.2, 3.05, 1.8, "$m_0$ write"),
              (2.45, 3.05, 1.8, "attention-1"),
              (4.7, 3.05, 2.1, "$z_1=\\mathrm{RMSNorm}(\u00b7)$"),
              (7.25, 3.05, 2.2, "$(L_1z_1)\\odot(R_1z_1)$"),
              (9.9, 3.05, 1.8, "$D_1h+b_1$"),
              (12.1, 3.05, .7, "$m_1$")]
    for x, y, w, label in native:
        ax.add_patch(FancyBboxPatch((x, y), w, .85, boxstyle="round,pad=.04",
                                    facecolor=COLORS["gray"], edgecolor=COLORS["navy"], lw=1.4))
        ax.text(x+w/2, y+.43, label, ha="center", va="center", fontsize=11,
                color=COLORS["navy"])
    for left, right in zip(native, native[1:]):
        ax.add_patch(FancyArrowPatch((left[0]+left[2], 3.48), (right[0], 3.48),
                                    arrowstyle="-|>", mutation_scale=12,
                                    color=COLORS["blue"], lw=1.3))
    parts = [(1.05, .75, 2.6, "token table $T[t]$", "dominant lexical term"),
             (4.15, .75, 3.7, "affine $[a_1;m_0]W$", "broad early-context correction"),
             (8.35, .75, 3.55, "selected $\\sum_r c_r h_r(z_1)$", "exact native bilinear residual terms")]
    for x, y, w, title, subtitle in parts:
        ax.add_patch(FancyBboxPatch((x, y), w, 1.05, boxstyle="round,pad=.04",
                                    facecolor="#eaf6f1", edgecolor=COLORS["green"], lw=1.4))
        ax.text(x+w/2, y+.68, title, ha="center", va="center", fontsize=11)
        ax.text(x+w/2, y+.27, subtitle, ha="center", va="center", fontsize=8.5)
    for part in parts:
        ax.add_patch(FancyArrowPatch((part[0]+part[2]/2, 1.82), (10.8, 3.02),
                                    arrowstyle="-|>", mutation_scale=11,
                                    connectionstyle="arc3,rad=-.12", color=COLORS["green"], lw=1.1))
    ax.text(6.5, .2, "Recovered program is a sum of three terms; semantic labels for the affine and quadratic terms remain incomplete.",
            ha="center", fontsize=9, color=COLORS["red"])
    ax.set_title("MLP1: native bilinear integrator and its measured three-part surrogate",
                 fontsize=15, color=COLORS["navy"], pad=8)
    save(fig, "mlp1_computation.svg")


def mlp1_evidence_figure():
    linear = json.loads((DATA / "mlp1_quantized_eval_results.json").read_text())
    induction = json.loads((DATA / "mlp1_induction_localization_results.json").read_text())
    quadratic = json.loads((DATA / "mlp1_quadratic_residual_eval_results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    ranks = [row["ridge_rank"] for row in linear["points"]]
    axes[0].plot(range(len(ranks)), [100*row["fidelity"] for row in linear["points"]],
                 marker="o", lw=2.3, color=COLORS["blue"])
    axes[0].set_xticks(range(len(ranks)), ranks); axes[0].set_ylim(91, 98)
    axes[0].set_xlabel("affine correction rank"); axes[0].set_ylabel("held-out fidelity (%)")
    axes[0].set_title("Natural text: token table already high;\naffine context closes another 4.7 points")
    axes[0].grid(axis="y", alpha=.25)
    base = induction["points"]
    xlabels = [f"linear\nr{row['ridge_rank']}" for row in base]
    values = [100*row["advantage_retained_fraction"]["global_permuted"] for row in base]
    promoted = next(row for row in quadratic["points"] if row["quadratic_rank"] == 32)
    xlabels.append("+ quadratic\nr32")
    values.append(100*promoted["advantage_retained_fraction"]["global_permuted"])
    colors = [COLORS["blue"]]*len(base)+[COLORS["green"]]
    axes[1].bar(range(len(values)), values, color=colors)
    axes[1].set_xticks(range(len(values)), xlabels); axes[1].set_ylim(0, 80)
    axes[1].set_ylabel("synthetic induction advantage retained (%)")
    axes[1].set_title("Induction exposes the missing context;\nquadratic residual recovers part, not all")
    axes[1].grid(axis="y", alpha=.25)
    for i, value in enumerate(values):
        axes[1].text(i, value+1.3, f"{value:.1f}%", ha="center", fontsize=9)
    fig.tight_layout(); save(fig, "mlp1_evidence.svg")


def mlp2_flow_figure():
    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis("off")
    native = [(0.15, 3.25, 1.75, "$m_1$ write"),
              (2.25, 3.25, 1.75, "attention-2"),
              (4.35, 3.25, 2.15, "$z_2=\\mathrm{RMSNorm}(\\cdot)$"),
              (6.85, 3.25, 2.2, "$(L_2z_2)\\odot(R_2z_2)$"),
              (9.4, 3.25, 1.75, "$D_2h+b_2$"),
              (11.5, 3.25, 1.25, "$m_2$")]
    for x, y, w, label in native:
        ax.add_patch(FancyBboxPatch((x, y), w, .85, boxstyle="round,pad=.04",
                                    facecolor=COLORS["gray"], edgecolor=COLORS["navy"], lw=1.4))
        ax.text(x+w/2, y+.43, label, ha="center", va="center", fontsize=11,
                color=COLORS["navy"])
    for left, right in zip(native, native[1:]):
        ax.add_patch(FancyArrowPatch((left[0]+left[2], 3.68), (right[0], 3.68),
                                    arrowstyle="-|>", mutation_scale=12,
                                    color=COLORS["blue"], lw=1.3))
    ax.add_patch(FancyBboxPatch((1.0, 1.25), 4.1, 1.05, boxstyle="round,pad=.04",
                                facecolor="#eaf6f1", edgecolor=COLORS["green"], lw=1.4))
    ax.text(3.05, 1.93, "$\\widehat m_2=b+[a_2;m_1]A_{128}$", ha="center", fontsize=13)
    ax.text(3.05, 1.52, "measured low-rank affine translator", ha="center", fontsize=9)
    ax.add_patch(FancyArrowPatch((5.1, 1.78), (7.0, 1.78), arrowstyle="-|>",
                                mutation_scale=13, color=COLORS["green"], lw=1.4))
    ax.add_patch(FancyBboxPatch((7.0, 1.25), 4.8, 1.05, boxstyle="round,pad=.04",
                                facecolor="#fff4dc", edgecolor=COLORS["gold"], lw=1.4))
    ax.text(9.4, 1.93, "layer-5 head 7 product-routing reader", ha="center", fontsize=11)
    ax.text(9.4, 1.52, "amplifies interface errors; route $\\gg$ payload", ha="center", fontsize=9)
    ax.add_patch(FancyArrowPatch((3.05, 2.32), (10.3, 3.22), arrowstyle="-|>",
                                mutation_scale=12, connectionstyle="arc3,rad=-.12",
                                color=COLORS["green"], lw=1.2))
    ax.text(6.5, .48,
            "The affine map is a high-value writer approximation. The reader interface is typed but unpriced, so the whole program is not closed.",
            ha="center", fontsize=9.2, color=COLORS["red"])
    ax.set_title("MLP2: a compressed early-state translator with a multiplicative downstream reader",
                 fontsize=15, color=COLORS["navy"], pad=8)
    save(fig, "mlp2_computation.svg")


def mlp2_evidence_figure():
    inventory = json.loads((DATA / "mlp2_replacement_inventory.json").read_text())
    reader = json.loads((DATA / "mlp2_reader_closure_certificate.json").read_text())
    points = inventory["candidates"]
    ranks = [point["rank"] for point in points]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    x = range(len(ranks))
    axes[0].plot(x, [100*point["held_out"]["fidelity"] for point in points],
                 marker="o", lw=2.3, color=COLORS["blue"], label="held-out fidelity")
    axes[0].plot(x, [100*point["extraction"]["global_retained_fraction"] for point in points],
                 marker="s", lw=1.8, color=COLORS["green"], label="extraction retention")
    axes[0].set_xticks(list(x), ranks); axes[0].set_ylim(20, 96)
    axes[0].set_xlabel("affine rank"); axes[0].set_ylabel("effect recovered (%)")
    axes[0].set_title("Writer approximation improves smoothly;\nrank 128 is the frozen operational knee")
    axes[0].grid(axis="y", alpha=.25); axes[0].legend(frameon=False)

    labels = ["isolated\nMLP2", "after replaced\nMLP0+1"]
    deltas = [reader["composition_failure"]["isolated_mlp2_delta_ce"],
              reader["composition_failure"]["conditioned_on_mlp0_mlp1_delta_ce"]]
    axes[1].bar(labels, deltas, color=[COLORS["blue"], COLORS["red"]], width=.58)
    axes[1].set_ylim(0, .82); axes[1].set_ylabel("incremental CE from rank-128 MLP2")
    axes[1].set_title("The same writer error costs 5.46× more\nunder the composed upstream state")
    axes[1].grid(axis="y", alpha=.25)
    for i, value in enumerate(deltas):
        axes[1].text(i, value+.025, f"{value:.3f}", ha="center", fontsize=10)
    axes[1].text(.5, .47, "layer-5 head 7 removes\n52.7% of angular margin",
                 ha="center", va="center", fontsize=9, color=COLORS["navy"],
                 bbox={"boxstyle": "round,pad=.3", "facecolor": "white", "edgecolor": COLORS["gold"]})
    fig.tight_layout(); save(fig, "mlp2_evidence.svg")


def mlp3_flow_figure():
    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis("off")
    native = [(0.15, 3.25, 1.75, "$m_2$ write"),
              (2.25, 3.25, 1.75, "attention-3"),
              (4.35, 3.25, 2.15, "$z_3=\\mathrm{RMSNorm}(\\cdot)$"),
              (6.85, 3.25, 2.2, "$(L_3z_3)\\odot(R_3z_3)$"),
              (9.4, 3.25, 1.75, "$D_3h+b_3$"),
              (11.5, 3.25, 1.25, "$m_3$")]
    for x, y, w, label in native:
        ax.add_patch(FancyBboxPatch((x, y), w, .85, boxstyle="round,pad=.04",
                                    facecolor=COLORS["gray"], edgecolor=COLORS["navy"], lw=1.4))
        ax.text(x+w/2, y+.43, label, ha="center", va="center", fontsize=11,
                color=COLORS["navy"])
    for left, right in zip(native, native[1:]):
        ax.add_patch(FancyArrowPatch((left[0]+left[2], 3.68), (right[0], 3.68),
                                    arrowstyle="-|>", mutation_scale=12,
                                    color=COLORS["blue"], lw=1.3))
    parts = [(1.0, 1.15, 3.25, "$T_{64}[t]$ + 2,000 exceptions", "lexical baseline"),
             (4.85, 1.15, 4.15, "$[a_3;m_2]W_r+b$", "prediction-metric local-DAG correction"),
             (9.6, 1.15, 2.35, "$\\widehat m_3$", "decoded write")]
    for x, y, w, title, subtitle in parts:
        ax.add_patch(FancyBboxPatch((x, y), w, 1.05, boxstyle="round,pad=.04",
                                    facecolor="#eaf6f1", edgecolor=COLORS["green"], lw=1.4))
        ax.text(x+w/2, y+.68, title, ha="center", va="center", fontsize=11)
        ax.text(x+w/2, y+.27, subtitle, ha="center", va="center", fontsize=8.6)
    for left, right in zip(parts, parts[1:]):
        ax.add_patch(FancyArrowPatch((left[0]+left[2], 1.68), (right[0], 1.68),
                                    arrowstyle="-|>", mutation_scale=12,
                                    color=COLORS["green"], lw=1.3))
    ax.add_patch(FancyArrowPatch((7.0, 2.22), (10.3, 3.22), arrowstyle="-|>",
                                mutation_scale=12, connectionstyle="arc3,rad=-.12",
                                color=COLORS["green"], lw=1.2))
    ax.text(6.5, .43,
            "The program is independently executable; its affine axes are predictive coordinates, not yet named semantic variables.",
            ha="center", fontsize=9.2, color=COLORS["red"])
    ax.set_title("MLP3: native bilinear module and its token-plus-local-DAG replacement",
                 fontsize=15, color=COLORS["navy"], pad=8)
    save(fig, "mlp3_computation.svg")


def mlp3_evidence_figure():
    inventory = json.loads((DATA / "mlp3_replacement_inventory.json").read_text())
    points = inventory["candidates"]
    ranks = [point["ridge_rank"] for point in points]
    x = range(len(ranks))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].plot(x, [100*point["held_out"]["fidelity"] for point in points],
                 marker="o", lw=2.3, color=COLORS["blue"], label="held-out fidelity")
    axes[0].plot(x, [100*point["extraction"]["global_retained_fraction"] for point in points],
                 marker="s", lw=1.8, color=COLORS["green"], label="extraction retention")
    axes[0].set_xticks(list(x), ranks); axes[0].set_ylim(48, 90)
    axes[0].set_xlabel("local-DAG correction rank"); axes[0].set_ylabel("effect recovered (%)")
    axes[0].set_title("Token table starts at 52%;\nlocal context raises recovery monotonically")
    axes[0].grid(axis="y", alpha=.25); axes[0].legend(frameon=False)

    induction = [point["ood"]["per_member_delta_ce"]["synthetic_induction"]
                 for point in points]
    axes[1].plot(x, induction, marker="o", lw=2.3, color=COLORS["red"])
    axes[1].fill_between(x, induction, alpha=.12, color=COLORS["red"])
    axes[1].set_xticks(list(x), ranks); axes[1].set_ylim(0, 5.25)
    axes[1].set_xlabel("local-DAG correction rank")
    axes[1].set_ylabel("synthetic-induction excess CE (lower is better)")
    axes[1].set_title("The first 32 contextual directions\nremove 3.59 CE of induction damage")
    axes[1].grid(axis="y", alpha=.25)
    for i in (0, 1, len(induction)-1):
        axes[1].annotate(f"{induction[i]:.2f}", (i, induction[i]),
                         xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    fig.tight_layout(); save(fig, "mlp3_evidence.svg")


def mlp4_flow_figure():
    fig, ax = plt.subplots(figsize=(13, 5.1))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.1); ax.axis("off")
    sources = [(0.2, 3.55, 1.45, "$m_0$"), (1.9, 3.55, 1.45, "$m_2$"),
               (3.6, 3.55, 1.45, "$m_3$"), (5.3, 3.55, 1.45, "$a_4$"),
               (7.0, 3.55, 1.45, "other writes")]
    for x, y, w, label in sources:
        ax.add_patch(FancyBboxPatch((x, y), w, .72, boxstyle="round,pad=.04",
                                    facecolor=COLORS["gray"], edgecolor=COLORS["navy"], lw=1.3))
        ax.text(x+w/2, y+.36, label, ha="center", va="center", fontsize=11)
        ax.add_patch(FancyArrowPatch((x+w/2, y), (6.2, 2.92), arrowstyle="-|>",
                                    mutation_scale=10, color=COLORS["blue"], lw=1.0,
                                    connectionstyle="arc3,rad=.08"))
    ax.add_patch(FancyBboxPatch((5.1, 2.05), 2.25, .85, boxstyle="round,pad=.04",
                                facecolor=COLORS["gray"], edgecolor=COLORS["navy"], lw=1.4))
    ax.text(6.225, 2.48, "$z_4=\\mathrm{RMSNorm}(\\sum writes)$", ha="center", fontsize=10.5)
    ax.add_patch(FancyArrowPatch((7.36, 2.48), (8.0, 2.48), arrowstyle="-|>",
                                mutation_scale=12, color=COLORS["blue"], lw=1.3))
    ax.add_patch(FancyBboxPatch((8.0, 2.05), 3.25, .85, boxstyle="round,pad=.04",
                                facecolor=COLORS["gray"], edgecolor=COLORS["navy"], lw=1.4))
    ax.text(9.625, 2.48, "$D_4[(L_4z_4)\\odot(R_4z_4)]+b_4$", ha="center", fontsize=11)
    ax.add_patch(FancyArrowPatch((11.26, 2.48), (12.05, 2.48), arrowstyle="-|>",
                                mutation_scale=12, color=COLORS["blue"], lw=1.3))
    ax.text(12.45, 2.48, "$m_4$", ha="center", va="center", fontsize=12)
    ax.add_patch(FancyBboxPatch((1.25, .55), 8.9, .9, boxstyle="round,pad=.04",
                                facecolor="#fff4dc", edgecolor=COLORS["gold"], lw=1.4))
    ax.text(5.7, 1.08, "Best tested surrogate: broad linear parents + generic quadratic capacity",
            ha="center", fontsize=10.5)
    ax.text(5.7, .76, "67.9% linear; at most 74.3% tested quadratic recovery", ha="center", fontsize=9)
    ax.add_patch(FancyArrowPatch((8.5, 1.46), (9.5, 2.03), arrowstyle="-|>",
                                mutation_scale=11, color=COLORS["gold"], lw=1.2))
    ax.text(11.55, 1.05, "not decoded\nnot priced\nnot operationally closed", ha="center",
            va="center", fontsize=8.7, color=COLORS["red"],
            bbox={"boxstyle": "round,pad=.25", "facecolor": "white", "edgecolor": COLORS["red"]})
    ax.set_title("MLP4: causal source localization and a partial predictive model",
                 fontsize=15, color=COLORS["navy"], pad=8)
    save(fig, "mlp4_computation.svg")


def mlp4_evidence_figure():
    inputs = json.loads((DATA / "mlp4_from_inputs_results.json").read_text())
    quad = json.loads((DATA / "mlp4_quad_results.json").read_text())
    squares = json.loads((DATA / "mlp4_squares_results.json").read_text())
    bow = json.loads((DATA / "mlp4_bow_results.json").read_text())
    labels = ["MLP3\nlinear", "five-input\nlinear", "designed\nquadratic",
              "random-feature\nquadratic", "largest tested\nrandom pairs"]
    values = [100*inputs["recovery"]["lin3"], 100*quad["recovery"]["lin5"],
              100*quad["recovery"]["quad_cross"], 100*quad["recovery"]["quad_rand"],
              100*squares["recoveries"]["pair_r512_F8192"]]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9))
    colors = [COLORS["blue"], COLORS["blue"], COLORS["gold"], COLORS["red"], COLORS["red"]]
    axes[0].bar(range(len(values)), values, color=colors)
    axes[0].set_xticks(range(len(values)), labels); axes[0].set_ylim(55, 78)
    axes[0].set_ylabel("optimal-constant gap recovered (%)")
    axes[0].set_title("More generic capacity helps slightly;\ndesigned structure does not beat its null")
    axes[0].grid(axis="y", alpha=.25)
    for i, value in enumerate(values):
        axes[0].text(i, value+.45, f"{value:.1f}%", ha="center", fontsize=8.7)

    tests = ["topic over\ntoken", "designed quad\nover linear",
             "random quad\nover linear", "joint-diag\nover linear"]
    gains = [100*bow["topic_gain"]/bow["stake"], 100*quad["gains"]["cross"],
             100*quad["gains"]["rand"],
             100*(json.loads((DATA / "mlp4_jointdiag_results.json").read_text())
                  ["recovery"]["jd_squares"] - 0.6794)]
    axes[1].bar(range(len(gains)), gains,
                color=[COLORS["red"], COLORS["gold"], COLORS["red"], COLORS["gold"]])
    axes[1].axhline(0, color=COLORS["ink"], lw=.9)
    axes[1].set_xticks(range(len(gains)), tests)
    axes[1].set_ylabel("recovery gain over corresponding linear baseline (points)")
    axes[1].set_title("Semantic and structured hypotheses\nfailed to earn their specificity")
    axes[1].grid(axis="y", alpha=.25)
    fig.tight_layout(); save(fig, "mlp4_evidence.svg")


def attention_flow_figure():
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.2); ax.axis("off")
    boxes = [(0.2, 3.2, 1.65, "$u_\\ell$", "RMS-normalized input"),
             (2.15, 3.2, 2.25, "$Q/K \\to$ head RMS + RoPE", "two routing branches"),
             (4.7, 3.2, 2.2, "$(q^Tk)(q'^Tk')/128^2$", "signed causal kernel"),
             (7.2, 3.2, 2.15, "$v_\\ell$ mixed with $v_0$", "shared value bus"),
             (9.65, 3.2, 1.75, "$K V$", "routed payload"),
             (11.7, 3.2, 1.05, "$O_\\ell$", "write")]
    for x, y, w, title, subtitle in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, .95, boxstyle="round,pad=.04",
                                    facecolor=COLORS["gray"], edgecolor=COLORS["navy"], lw=1.4))
        ax.text(x+w/2, y+.61, title, ha="center", va="center", fontsize=10.5)
        ax.text(x+w/2, y+.24, subtitle, ha="center", va="center", fontsize=8.1)
    for left, right in zip(boxes, boxes[1:]):
        ax.add_patch(FancyArrowPatch((left[0]+left[2], 3.68), (right[0], 3.68),
                                    arrowstyle="-|>", mutation_scale=12,
                                    color=COLORS["blue"], lw=1.3))
    ax.add_patch(FancyBboxPatch((1.25, .75), 5.25, 1.1, boxstyle="round,pad=.04",
                                facecolor="#eaf6f1", edgecolor=COLORS["green"], lw=1.4))
    ax.text(3.875, 1.46, "decoded for 139/162 heads", ha="center", fontsize=11)
    ax.text(3.875, 1.05, "rank-32 $Q/K/Q'/K'$ routing maps; 23 heads exact/live",
            ha="center", fontsize=8.8)
    ax.add_patch(FancyArrowPatch((3.9, 1.86), (5.8, 3.18), arrowstyle="-|>",
                                mutation_scale=11, connectionstyle="arc3,rad=-.12",
                                color=COLORS["green"], lw=1.2))
    ax.add_patch(FancyBboxPatch((7.1, .75), 4.7, 1.1, boxstyle="round,pad=.04",
                                facecolor="#fff4dc", edgecolor=COLORS["gold"], lw=1.4))
    ax.text(9.45, 1.46, "still live and unpriced", ha="center", fontsize=11)
    ax.text(9.45, 1.05, "all V maps, shared $v_0$ bus, all O writers, all MLPs",
            ha="center", fontsize=8.8)
    ax.text(6.5, .28, "Routing recovery is not complete attention recovery.",
            ha="center", fontsize=9.5, color=COLORS["red"])
    ax.set_title("Bilin18 attention: multiplicative routing, shared payload, residual write",
                 fontsize=15, color=COLORS["navy"], pad=8)
    save(fig, "attention_computation.svg")


def attention_evidence_figure():
    decoded = json.loads((DATA / "attention_mixed_decoded_eval_results.json").read_text())
    ood = json.loads((DATA / "attention_mixed_ood_results.json").read_text())
    handles = json.loads((DATA / "attention_handle_curve_results.json").read_text())
    motifs = json.loads((DATA / "attn_motifs3_results.json").read_text())["census"]
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8))
    labels = ["decoded\nrouting", "exact/live\nrouting"]
    axes[0].bar(labels, [139, 23], color=[COLORS["green"], COLORS["gold"]], width=.62)
    axes[0].set_ylim(0, 162); axes[0].set_ylabel("heads")
    axes[0].set_title("Q/K routing roster")
    for i, value in enumerate((139, 23)):
        axes[0].text(i, value+4, str(value), ha="center", fontsize=10)

    regimes = ["ID", "code", "Pile", "synthetic\ninduction"]
    damage = [decoded["scores"]["decoded_mixed_ce"]-
              decoded["scores"]["clean_native_attention_ce"],
              ood["members"]["code"]["decoded_delta_from_clean_ce"],
              ood["members"]["pile"]["decoded_delta_from_clean_ce"],
              ood["members"]["synthetic_induction"]["decoded_delta_from_clean_ce"]]
    axes[1].bar(regimes, damage, color=[COLORS["blue"], COLORS["blue"], COLORS["blue"], COLORS["red"]])
    axes[1].set_ylabel("decoded-routing excess CE")
    axes[1].set_title("Transfer passes; induction\nabsolute adequacy fails")
    axes[1].grid(axis="y", alpha=.25)
    for i, value in enumerate(damage):
        axes[1].text(i, value+.08, f"{value:.2f}", ha="center", fontsize=8.5)

    order = ["self", "prev", "ind", "first", "diffuse"]
    names = ["self", "previous", "induction", "first", "diffuse"]
    axes[2].barh(names, [motifs[key] for key in order],
                 color=[COLORS["blue"], COLORS["blue"], COLORS["green"], COLORS["gold"], "#aab4bd"])
    axes[2].set_xlim(0, 82); axes[2].set_xlabel("descriptively classified heads")
    axes[2].set_title("Observed routing motifs\n(not semantic closure)")
    axes[2].grid(axis="x", alpha=.25)
    axes[2].text(2, 4.25, f"98.78% extraction retention\n{handles['removal']['global_aligned_minus_random_damage_ce']:.2f} CE selective excess",
                 fontsize=8.6, color=COLORS["navy"])
    fig.tight_layout(); save(fig, "attention_evidence.svg")


if __name__ == "__main__":
    flow_figure(); evidence_curve(); status_map(); mlp1_flow_figure(); mlp1_evidence_figure()
    mlp2_flow_figure(); mlp2_evidence_figure()
    mlp3_flow_figure(); mlp3_evidence_figure()
    mlp4_flow_figure(); mlp4_evidence_figure()
    attention_flow_figure(); attention_evidence_figure()
    print("rendered reader-facing explanation figures")
