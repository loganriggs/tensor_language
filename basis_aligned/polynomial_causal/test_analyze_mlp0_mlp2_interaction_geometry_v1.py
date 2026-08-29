from __future__ import annotations

import torch

import analyze_mlp0_mlp2_interaction_geometry_v1 as analysis


def arm(dce: torch.Tensor, difficulty: torch.Tensor) -> torch.Tensor:
    value = torch.zeros(192, 9, dtype=torch.float64)
    value[:, 0] = difficulty; value[:, 1] = difficulty + dce
    value[:, 2] = dce.abs(); value[:, 4] = 1; value[:, 8] = 1
    return value


def synthetic_arms(interactions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    x = torch.linspace(-1, 1, 192, dtype=torch.float64)
    difficulty = 4 + x
    c = 0.01 + 0.002 * x
    result = {"NATIVE": arm(torch.zeros_like(x), difficulty),
              "C512": arm(c, difficulty)}
    for index, program in enumerate(analysis.PROGRAMS):
        standalone = 0.04 + 0.003 * index + 0.001 * x
        result[program] = arm(standalone, difficulty)
        result[f"C512_{program}"] = arm(
            c + standalone + interactions[program], difficulty,
        )
    return result


def test_identical_diffuse_interactions_form_shared_mode() -> None:
    x = torch.linspace(-0.01, 0.03, 192, dtype=torch.float64)
    interactions = {program: x + 0.001 * index
                    for index, program in enumerate(analysis.PROGRAMS)}
    result = analysis.analyze(synthetic_arms(interactions))
    assert result["diagnostic_rules"]["diffuse_all_programs"]
    assert result["diagnostic_rules"]["shared_document_mode"]
    assert not result["diagnostic_rules"]["sparse_gate_candidate"]
    assert result["pairwise_ce_interaction_correlations"]


def test_sparse_interaction_is_detected() -> None:
    sparse = torch.zeros(192, dtype=torch.float64); sparse[:10] = 1.0
    interactions = {program: sparse * (index + 1)
                    for index, program in enumerate(analysis.PROGRAMS)}
    result = analysis.analyze(synthetic_arms(interactions))
    assert result["diagnostic_rules"]["sparse_gate_candidate"]
    assert not result["diagnostic_rules"]["diffuse_all_programs"]


def test_vector_summary_concentration_and_bootstrap_are_deterministic() -> None:
    x = torch.arange(1, 193, dtype=torch.float64)
    first = analysis.vector_summary(x); second = analysis.vector_summary(x)
    assert first == second
    assert first["effective_participation_documents"] > 100
    assert 0 < first["absolute_gini"] < 1
