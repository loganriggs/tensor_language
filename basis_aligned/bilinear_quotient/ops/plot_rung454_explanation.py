"""Render the user-facing rung-454 result and denominator diagnostic."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
BUNDLE = ROOT / "simplicity_vocabulary_complete_candidate_consequences_bundle.pt"
OUT = ROOT.parent / "polynomial_causal/explanations/assets/rung454_vocabulary_consequences.png"


def norm(value: torch.Tensor) -> float:
    return float(value.double().norm())


bundle = torch.load(BUNDLE, map_location="cpu", weights_only=False)
native = bundle["native_ce"]
partner = bundle["partner_ce"]
ranks = [0, 128, 256, 512]
names = [f"vocab_r300_independent_{rank}" for rank in ranks]

interaction_norm = []
moving_denominator = []
registered_ratio = []
for name in names:
    arm = bundle["arms"][name]
    candidate = arm["candidate_ce"]
    joint = arm["candidate_partner_ce"]
    interaction = joint - candidate - partner + native
    additive = (candidate - native) + (partner - native)
    interaction_norm.append(norm(interaction))
    moving_denominator.append(norm(additive))
    registered_ratio.append(norm(interaction) / norm(additive))


def percent_of_first(values: list[float]) -> list[float]:
    return [100.0 * value / values[0] for value in values]


fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
ax = axes[0]
labels = ["Full 96 docs", "Wave 1\n48 docs", "Wave 2\n48 docs"]
x = range(3)
width = 0.34
ax.bar([i - width / 2 for i in x], [100, 100, 100], width, label="Removal score", color="#2878b5")
ax.bar([i + width / 2 for i in x], [35.7, 21.4, 42.9], width, label="Composition score", color="#d95f02")
ax.axhline(85, color="#333333", linestyle="--", linewidth=1, label="Full-data pass bar (85%)")
ax.axhline(70, color="#777777", linestyle=":", linewidth=1, label="Per-wave pass bar (70%)")
ax.set_xticks(list(x), labels)
ax.set_ylim(0, 108)
ax.set_ylabel("Pre-registered rank directions correct (%)")
ax.set_title("A. The removal result passed; composition failed")
ax.legend(fontsize=8, loc="lower left")
ax.grid(axis="y", alpha=0.2)

ax = axes[1]
ax.plot(ranks, percent_of_first(interaction_norm), "o-", linewidth=2.2,
        label="Interaction size (also fixed-scale score)", color="#2ca02c")
ax.plot(ranks, percent_of_first(moving_denominator), "s-", linewidth=2.2,
        label="Candidate-dependent denominator", color="#9467bd")
ax.plot(ranks, percent_of_first(registered_ratio), "^-", linewidth=2.2,
        label="Registered ratio", color="#d62728")
ax.axhline(100, color="#999999", linewidth=0.8)
ax.set_xticks(ranks)
ax.set_xlabel("Independent vocabulary-program rank")
ax.set_ylabel("Value relative to rank 0 (%)")
ax.set_title("B. A shrinking denominator reverses the conclusion")
ax.legend(fontsize=8)
ax.grid(alpha=0.2)

fig.suptitle("Rung 454: exact vocabulary-program consequence test", fontsize=14)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=180, bbox_inches="tight")
print(OUT)
