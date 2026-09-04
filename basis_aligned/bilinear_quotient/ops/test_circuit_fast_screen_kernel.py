from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import inspect
import math
from pathlib import Path
import unittest

import circuit_fast_screen_kernel as kernel


class FakeModelAdapter:
    """No-torch stand-in for residual/module/head evidence producers."""

    def __init__(self, site: kernel.SiteRef) -> None:
        self.site = site

    def evidence(
        self,
        *,
        a1: tuple[float, ...] = (0.8, 0.7),
        a2: tuple[float, ...] = (0.7, 0.6),
        p: tuple[float, ...] = (0.05, 0.10),
        c: tuple[float, ...] = (0.05, -0.10),
    ) -> tuple[kernel.ScalarInterventionEvidence, ...]:
        records = []
        for family, recoveries in (("A1", a1), ("A2", a2), ("C", c)):
            for index, recovery in enumerate(recoveries):
                records.append(kernel.ScalarInterventionEvidence(
                    record_id=f"{family}:{index}", pair_id=f"pair:{family}:{index}",
                    family=family, evidence_kind=self.site.evidence_kind,
                    site_id=self.site.site_id, base_score=-1.0, donor_score=1.0,
                    intervened_score=-1.0 + 2.0 * recovery,
                ))
        for index, effect in enumerate(p):
            records.append(kernel.ScalarInterventionEvidence(
                record_id=f"P:{index}", pair_id=f"pair:P:{index}", family="P",
                evidence_kind=self.site.evidence_kind, site_id=self.site.site_id,
                base_score=2.0, donor_score=None,
                intervened_score=2.0 + effect, effect_scale=1.0,
            ))
        return tuple(records)


def capability_evidence(
    *, a1: int = 17, a2: int = 17, p: int = 17, c: int = 15,
    observed: int = 20, expected: int = 20, complete: bool = True,
) -> kernel.CapabilityEvidence:
    return kernel.CapabilityEvidence((
        kernel.FamilyCapabilityEvidence("A1", a1, observed, expected),
        kernel.FamilyCapabilityEvidence("A2", a2, observed, expected),
        kernel.FamilyCapabilityEvidence("P", p, observed, expected),
        kernel.FamilyCapabilityEvidence("C", c, observed, expected),
    ), complete=complete)


CAPABILITY = capability_evidence()


def score(
    adapter: FakeModelAdapter,
    records: tuple[kernel.ScalarInterventionEvidence, ...] | None = None,
    capability: kernel.CapabilityEvidence = CAPABILITY,
    *,
    c_answer_changes: bool = True,
) -> kernel.SiteScreenResult:
    evidence = adapter.evidence() if records is None else records
    return kernel.score_site(
        adapter.site,
        evidence=evidence,
        expected_record_ids=tuple(record.record_id for record in evidence),
        capability=capability,
        c_answer_changes=c_answer_changes,
    )


class FastScreenKernelTests(unittest.TestCase):
    def test_kernel_is_model_independent_and_does_not_import_battery(self) -> None:
        text = Path(kernel.__file__).read_text()
        self.assertNotIn("import torch", text)
        self.assertNotIn("circuit_battery", text)

    def test_signed_recovery_is_unclipped_and_rejects_unsafe_denominators(self) -> None:
        self.assertEqual(kernel.signed_pairwise_donor_recovery(-2.0, 2.0, 1.0), 0.75)
        self.assertEqual(kernel.signed_pairwise_donor_recovery(-1.0, 1.0, 2.0), 1.5)
        for base, donor in ((0.0, -1.0), (0.0, 0.0), (0.0, 1.0e-6)):
            with self.subTest(base=base, donor=donor), self.assertRaises(
                kernel.InvalidEvidenceError
            ):
                kernel.signed_pairwise_donor_recovery(base, donor, 0.5)
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(kernel.InvalidEvidenceError):
                kernel.signed_pairwise_donor_recovery(0.0, 1.0, value)
        with self.assertRaises(kernel.InvalidEvidenceError):
            kernel.signed_pairwise_donor_recovery(-1.0e308, 1.0e308, 0.0)

    def test_fixed_bars_are_immutable_and_not_score_arguments(self) -> None:
        self.assertEqual(kernel.FIXED_BARS.minimum_target_family_recovery, 0.50)
        self.assertEqual(kernel.FIXED_BARS.maximum_p_invariance_effect, 0.20)
        self.assertEqual(kernel.FIXED_BARS.maximum_c_absolute_recovery, 0.35)
        self.assertNotIn("bars", inspect.signature(kernel.score_site).parameters)
        with self.assertRaises(FrozenInstanceError):
            kernel.FIXED_BARS.minimum_target_family_recovery = 0.0  # type: ignore[misc]

    def test_fake_residual_module_and_head_adapters_can_screen(self) -> None:
        for kind in ("residual", "module", "head"):
            adapter = FakeModelAdapter(kernel.SiteRef(kind, f"site:{kind}"))
            result = score(adapter)
            self.assertEqual(result.terminal, "screen")
            self.assertEqual(
                tuple(cell.accuracy for cell in result.capability or ()),
                (0.85, 0.85, 0.85, 0.75),
            )
            self.assertEqual(result.evidence_count, 8)
            self.assertAlmostEqual(result.target_recovery or 0.0, 0.70)
            self.assertAlmostEqual(result.p_invariance_effect or 0.0, 0.075)
            self.assertAlmostEqual(result.c_absolute_recovery or 0.0, 0.075)
            self.assertAlmostEqual(result.c_signed_recovery or 0.0, -0.025)

    def test_a1_a2_are_aggregated_with_equal_family_weight(self) -> None:
        adapter = FakeModelAdapter(kernel.SiteRef("module", "unequal-cells"))
        records = adapter.evidence(a1=(0.9,), a2=(0.5,) * 9)
        result = score(adapter, records)
        self.assertEqual(result.terminal, "screen")
        self.assertAlmostEqual(result.target_recovery or 0.0, 0.70)
        pooled = (0.9 + 9 * 0.5) / 10
        self.assertNotAlmostEqual(result.target_recovery or 0.0, pooled)

    def test_valid_gate_or_science_failures_are_nulls(self) -> None:
        adapter = FakeModelAdapter(kernel.SiteRef("residual", "nulls"))
        cases = {
            "A2/all_capability_below_fixed_bar": (
                adapter.evidence(), capability_evidence(a2=16)
            ),
            "A1_recovery_below_fixed_bar": (
                adapter.evidence(a1=(0.49, 0.49)), CAPABILITY
            ),
            "P_invariance_above_fixed_bar": (
                adapter.evidence(p=(0.21, 0.21)), CAPABILITY
            ),
            "C_absolute_recovery_above_fixed_bar": (
                adapter.evidence(c=(0.36, -0.36)), CAPABILITY
            ),
        }
        for reason, (records, case_capability) in cases.items():
            with self.subTest(reason=reason):
                result = score(adapter, records, case_capability)
                self.assertEqual(result.terminal, "null")
                self.assertIn(reason, result.reasons)

    def test_c_is_negative_control_and_uses_mean_absolute_recovery(self) -> None:
        adapter = FakeModelAdapter(kernel.SiteRef("module", "c-selectivity"))
        near_zero = score(adapter, adapter.evidence(c=(0.01, -0.01)))
        self.assertEqual(near_zero.terminal, "screen")
        self.assertAlmostEqual(near_zero.c_absolute_recovery or -1.0, 0.01)

        high_transfer = score(adapter, adapter.evidence(c=(0.60, -0.60)))
        self.assertEqual(high_transfer.terminal, "null")
        self.assertIn("C_absolute_recovery_above_fixed_bar", high_transfer.reasons)
        self.assertAlmostEqual(high_transfer.c_signed_recovery or 0.0, 0.0)
        self.assertAlmostEqual(high_transfer.c_absolute_recovery or 0.0, 0.60)
        self.assertAlmostEqual(high_transfer.c_direction_fraction or 0.0, 0.50)

    def test_c_can_be_a_same_answer_active_negative_control(self) -> None:
        adapter = FakeModelAdapter(kernel.SiteRef("module", "c-same-answer"))
        records = tuple(
            replace(
                record,
                donor_score=None,
                base_score=2.0,
                intervened_score=2.0 + effect,
                effect_scale=1.0,
            ) if record.family == "C" else record
            for record, effect in zip(
                adapter.evidence(),
                (0.0, 0.0, 0.0, 0.0, 0.05, -0.10, 0.0, 0.0),
            )
        )
        result = score(adapter, records, c_answer_changes=False)
        self.assertEqual(result.terminal, "screen")
        self.assertAlmostEqual(result.c_absolute_recovery or 0.0, 0.075)
        self.assertIsNone(result.c_direction_fraction)

        c_index = next(index for index, record in enumerate(records)
                       if record.family == "C")
        mismatched = records[:c_index] + (
            replace(records[c_index], donor_score=3.0, effect_scale=None),
        ) + records[c_index + 1:]
        invalid = score(adapter, mismatched, c_answer_changes=False)
        self.assertEqual(invalid.terminal, "invalid")
        self.assertIn("same-answer record", invalid.reasons[0])

    def test_capability_is_thresholded_per_family_not_pooled(self) -> None:
        adapter = FakeModelAdapter(kernel.SiteRef("head", "family-capability"))
        # Pooled accuracy is 95%, but A2 is only 80% and must independently fail.
        result = score(
            adapter,
            capability=capability_evidence(a1=20, a2=16, p=20, c=20),
        )
        self.assertEqual(result.terminal, "null")
        self.assertEqual(result.reasons, ("A2/all_capability_below_fixed_bar",))
        self.assertEqual(
            tuple(
                (cell.family, cell.cell_id, cell.accuracy)
                for cell in result.capability or ()
            ),
            (
                ("A1", "all", 1.0), ("A2", "all", 0.8),
                ("P", "all", 1.0), ("C", "all", 1.0),
            ),
        )

    def test_capability_is_thresholded_per_ordered_construction_cell(self) -> None:
        adapter = FakeModelAdapter(kernel.SiteRef("module", "ordered-cells"))
        capability = kernel.CapabilityEvidence((
            kernel.FamilyCapabilityEvidence(
                "A1", 18, 20, 20, cell_id="statement_to_question"
            ),
            kernel.FamilyCapabilityEvidence(
                "A1", 16, 20, 20, cell_id="question_to_statement"
            ),
            kernel.FamilyCapabilityEvidence(
                "A2", 18, 20, 20, cell_id="direct_to_indirect"
            ),
            kernel.FamilyCapabilityEvidence("P", 18, 20, 20),
            kernel.FamilyCapabilityEvidence("C", 16, 20, 20),
        ))
        result = score(adapter, capability=capability)
        self.assertEqual(result.terminal, "null")
        self.assertIn(
            "A1/question_to_statement_capability_below_fixed_bar", result.reasons,
        )
        self.assertEqual(
            tuple(
                (cell.family, cell.cell_id, cell.accuracy)
                for cell in result.capability or ()
            ),
            (
                ("A1", "question_to_statement", 0.8),
                ("A1", "statement_to_question", 0.9),
                ("A2", "direct_to_indirect", 0.9),
                ("P", "all", 0.9),
                ("C", "all", 0.8),
            ),
        )

    def test_direction_bars_prevent_mean_from_hiding_sign_failures(self) -> None:
        adapter = FakeModelAdapter(kernel.SiteRef("head", "mixed-sign"))
        a1 = (4.0, 4.0, 4.0, -0.1, -0.1)
        result = score(adapter, adapter.evidence(a1=a1))
        self.assertEqual(result.terminal, "null")
        self.assertGreater(result.a1.mean_effect if result.a1 else 0.0, 0.5)
        self.assertIn("A1_direction_below_fixed_bar", result.reasons)

    def test_missing_duplicate_incomplete_or_cross_site_evidence_is_invalid(self) -> None:
        adapter = FakeModelAdapter(kernel.SiteRef("module", "site-a"))
        records = adapter.evidence()
        attacks = {
            "missing": records[:-1],
            "duplicate": records + (records[0],),
            "duplicate-pair": (
                records[0], replace(records[1], pair_id=records[0].pair_id), *records[2:]
            ),
            "incomplete": (replace(records[0], complete=False),) + records[1:],
            "cross-site": (replace(records[0], site_id="site-b"),) + records[1:],
        }
        expected = tuple(record.record_id for record in records)
        for name, attacked in attacks.items():
            with self.subTest(name=name):
                result = kernel.score_site(
                    adapter.site, evidence=attacked,
                    expected_record_ids=expected, capability=CAPABILITY,
                )
                self.assertEqual(result.terminal, "invalid")
                self.assertIsNone(result.target_recovery)

    def test_nonfinite_and_weak_evidence_are_invalid_not_null(self) -> None:
        adapter = FakeModelAdapter(kernel.SiteRef("residual", "bad-scalars"))
        records = adapter.evidence()
        attacks = (
            replace(records[0], intervened_score=math.nan),
            replace(records[0], donor_score=-2.0),
            replace(records[-1], effect_scale=0.0),
        )
        for attacked in attacks:
            changed = (attacked,) + records[1:]
            result = score(adapter, changed, capability_evidence(a1=0))
            self.assertEqual(result.terminal, "invalid")
            self.assertIsNone(result.capability)
            self.assertIsNone(result.target_recovery)

    def test_incomplete_or_truthy_capability_is_invalid(self) -> None:
        adapter = FakeModelAdapter(kernel.SiteRef("head", "capability"))
        for case_capability in (
            capability_evidence(observed=19),
            capability_evidence(complete=False),
            kernel.CapabilityEvidence((
                kernel.FamilyCapabilityEvidence("A1", True, 20, 20),
                *capability_evidence().families[1:],
            )),
        ):
            with self.subTest(capability=case_capability):
                self.assertEqual(
                    score(adapter, capability=case_capability).terminal, "invalid"
                )

    def test_p_semantics_require_scale_and_forbid_donor_denominator(self) -> None:
        adapter = FakeModelAdapter(kernel.SiteRef("residual", "p-contract"))
        records = adapter.evidence()
        p_index = next(index for index, record in enumerate(records) if record.family == "P")
        for attacked in (
            replace(records[p_index], effect_scale=None),
            replace(records[p_index], donor_score=3.0),
        ):
            changed = records[:p_index] + (attacked,) + records[p_index + 1:]
            self.assertEqual(score(adapter, changed).terminal, "invalid")

    def test_ranking_is_deterministic_with_explicit_exact_tie_break(self) -> None:
        # Equal scientific values tie-break residual, module, head, then site ID.
        sites = (
            kernel.SiteRef("head", "a"),
            kernel.SiteRef("module", "a"),
            kernel.SiteRef("residual", "b"),
            kernel.SiteRef("residual", "a"),
        )
        results = [score(FakeModelAdapter(site)) for site in reversed(sites)]
        ranked = kernel.rank_sites(results)
        self.assertEqual([item.result.site for item in ranked], list(reversed(sites)))
        self.assertEqual([item.rank for item in ranked], [1, 2, 3, 4])

        # A scientifically stronger screen precedes an otherwise valid screen;
        # valid nulls precede invalid instruments.
        strong_adapter = FakeModelAdapter(kernel.SiteRef("head", "strong"))
        strong = score(strong_adapter, strong_adapter.evidence(a1=(0.95,), a2=(0.95,)))
        weak_adapter = FakeModelAdapter(kernel.SiteRef("head", "weak"))
        weak = score(weak_adapter)
        null = score(
            FakeModelAdapter(kernel.SiteRef("head", "null")),
            capability=capability_evidence(a1=0),
        )
        invalid = score(
            FakeModelAdapter(kernel.SiteRef("head", "invalid")),
            capability=capability_evidence(observed=19),
        )
        ordered = kernel.rank_sites((invalid, weak, null, strong))
        self.assertEqual(
            [item.result.site.site_id for item in ordered],
            ["strong", "weak", "null", "invalid"],
        )

    def test_ranking_rejects_duplicate_site_results(self) -> None:
        result = score(FakeModelAdapter(kernel.SiteRef("module", "duplicate")))
        with self.assertRaises(ValueError):
            kernel.rank_sites((result, result))


if __name__ == "__main__":
    unittest.main()
