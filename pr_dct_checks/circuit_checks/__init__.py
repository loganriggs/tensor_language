"""Validation checks for participation-ratio-regularized DCT circuits.

See README.md for the experiment plan and BACKGROUND.md for the reasoning.
"""
from .core import (
    mixed_hessian,
    participation_ratio,
    FreezeSpec,
    apply_freeze,
    finite_mixed_difference,
)
from .checks import (
    pathway_completeness,
    top_units,
    random_units,
    neuron_readoff_candidates,
    factor_alignment_to_units,
    ablation_effect,
    match_factors,
    random_match_null,
    jaccard,
)
