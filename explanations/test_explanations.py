import json
import re
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class ExplanationsTest(unittest.TestCase):
    def test_component_chapters_exist(self):
        expected = ["00_WHAT_UNDERSTOOD_MEANS.md", "01_MLP0.md", "02_MLP1.md",
                    "03_MLP2.md", "04_MLP3.md", "05_MLP4.md", "06_ATTENTION.md"]
        for name in expected:
            self.assertTrue((HERE / name).is_file(), name)

    def test_every_local_markdown_link_resolves(self):
        missing = []
        for document in HERE.glob("*.md"):
            for target in re.findall(r"\[[^]]*\]\(([^)#]+)", document.read_text()):
                if "://" not in target and not (document.parent / target).resolve().exists():
                    missing.append((document.name, target))
        self.assertEqual(missing, [])

    def test_inline_figures_are_real_pngs_not_mermaid_placeholders(self):
        documents = "\n".join(path.read_text() for path in HERE.glob("*.md"))
        self.assertNotIn("```mermaid", documents)
        targets = re.findall(r"!\[[^]]*\]\(([^)]+\.png)\)", documents)
        self.assertGreaterEqual(len(targets), 3)
        for target in targets:
            data = (HERE / target).read_bytes()
            self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"), target)

    def test_mlp0_chapter_separates_support_from_counterevidence(self):
        text = (HERE / "01_MLP0.md").read_text()
        for heading in ("What the native module computes", "simpler recovered program",
                        "What it appears to mean semantically", "does **not** explain",
                        "Confidence ledger"):
            self.assertIn(heading, text)
        self.assertIn("not a literal assertion", text)
        self.assertIn("contradicted", text)

    def test_overview_does_not_claim_mlp0_to_4_are_fully_understood(self):
        text = (HERE / "00_WHAT_UNDERSTOOD_MEANS.md").read_text()
        self.assertIn("should **not** say", text)
        self.assertIn("MLP4", text)
        self.assertIn("not operationally closed", text)

    def test_mlp1_chapter_separates_reproduction_from_semantics(self):
        text = (HERE / "02_MLP1.md").read_text()
        for phrase in ("recovered three-part program", "Context is small on average",
                       "Counterevidence", "Confidence ledger", "Next decisive experiment"):
            self.assertIn(phrase, text)
        self.assertIn("cannot yet name", text)
        self.assertIn("parallel (independent)", text)

    def test_mlp2_chapter_separates_writer_from_reader_closure(self):
        text = (HERE / "03_MLP2.md").read_text()
        for phrase in ("recovered writer program", "isolated fidelity overstates",
                       "layer-5 product attention", "Quotient-aware simplicity",
                       "Confidence ledger", "Next decisive experiment"):
            self.assertIn(phrase, text)
        self.assertIn("5.46", text)
        self.assertIn("whole-program price", text)
        self.assertIn("not a compact dictionary", text)

    def test_mlp2_reader_numbers_match_machine_readable_evidence(self):
        data = HERE.parent / "basis_aligned" / "bilinear_quotient"
        inventory = json.loads((data / "mlp2_replacement_inventory.json").read_text())
        certificate = json.loads((data / "mlp2_reader_closure_certificate.json").read_text())
        rank128 = next(row for row in inventory["candidates"] if row["rank"] == 128)
        self.assertAlmostEqual(rank128["held_out"]["fidelity"], 0.8160634135)
        self.assertAlmostEqual(rank128["extraction"]["global_retained_fraction"],
                               0.8160728222)
        self.assertAlmostEqual(certificate["composition_failure"]
                               ["conditioned_to_isolated_ratio"], 5.4574407546)
        self.assertAlmostEqual(certificate["reader"]
                               ["head7_fraction_of_unpatched_margin_removed"],
                               0.5266953636)
        self.assertFalse(certificate["simplicity_accounting"]
                         ["eligible_for_whole_program_mdl"])

    def test_mlp3_chapter_separates_program_from_semantic_naming(self):
        text = (HERE / "04_MLP3.md").read_text()
        for phrase in ("independently executable replacement",
                       "not ordinary SVD ranks", "Semantic split",
                       "What remains unexplained", "Confidence ledger",
                       "Next decisive experiment"):
            self.assertIn(phrase, text)
        self.assertIn("not an eight-variable decomposition", text)
        self.assertIn("functionally compressed", text)

    def test_mlp3_frontier_numbers_match_machine_readable_evidence(self):
        data = HERE.parent / "basis_aligned" / "bilinear_quotient"
        inventory = json.loads((data / "mlp3_replacement_inventory.json").read_text())
        rank0 = next(row for row in inventory["candidates"] if row["ridge_rank"] == 0)
        rank32 = next(row for row in inventory["candidates"] if row["ridge_rank"] == 32)
        rank128 = next(row for row in inventory["candidates"] if row["ridge_rank"] == 128)
        self.assertAlmostEqual(rank0["held_out"]["fidelity"], 0.5233942695)
        self.assertAlmostEqual(rank32["held_out"]["fidelity"], 0.7063798443)
        self.assertAlmostEqual(rank128["held_out"]["fidelity"], 0.8117474837)
        induction_gain = (rank0["ood"]["per_member_delta_ce"]["synthetic_induction"] -
                          rank32["ood"]["per_member_delta_ce"]["synthetic_induction"])
        self.assertAlmostEqual(induction_gain, 3.5900480272)
        self.assertEqual(inventory["selection_summary"]
                         ["marginal_efficiency_knee_rank"], 32)

    def test_mlp4_chapter_is_explicitly_partial_and_null_controlled(self):
        text = (HERE / "05_MLP4.md").read_text()
        for phrase in ("Causal ancestry", "Predictive models",
                       "Semantic hypothesis that failed", "no simplicity price",
                       "Confidence ledger", "Next decisive experiment"):
            self.assertIn(phrase, text)
        self.assertIn("random-feature null performs at least as well", text)
        self.assertIn("ineligible for whole-program MDL", text)
        self.assertIn("complete semantic understanding | absent", text)

    def test_mlp4_numbers_match_machine_readable_evidence(self):
        data = HERE.parent / "basis_aligned" / "bilinear_quotient"
        inputs = json.loads((data / "mlp4_from_inputs_results.json").read_text())
        quad = json.loads((data / "mlp4_quad_results.json").read_text())
        reads = json.loads((data / "mlp4_reads_results.json").read_text())
        weights = json.loads((data / "mlp4_weight_tensor_results.json").read_text())
        self.assertAlmostEqual(inputs["stake"], 0.1017)
        self.assertAlmostEqual(inputs["recovery"]["lin3"], 0.6116)
        self.assertAlmostEqual(quad["recovery"]["lin5"], 0.6794)
        self.assertGreater(quad["recovery"]["quad_rand"],
                           quad["recovery"]["quad_cross"])
        self.assertEqual(reads["ranked"][:3], ["mlp0", "mlp3", "mlp2"])
        self.assertAlmostEqual(weights["top5pct_mass"], 0.2338)

    def test_attention_chapter_separates_route_payload_and_writer(self):
        text = (HERE / "06_ATTENTION.md").read_text()
        for phrase in ("headwise RMSNorm", "route:", "payload:", "writer:",
                       "shared value/output gauge", "complete attention reverse engineering"):
            self.assertIn(phrase, text)
        self.assertIn("not globally a quartic polynomial", text)
        self.assertIn("remain live", text)

    def test_attention_numbers_match_machine_readable_evidence(self):
        data = HERE.parent / "basis_aligned" / "bilinear_quotient"
        decoded = json.loads((data / "attention_mixed_decoded_eval_results.json").read_text())
        handles = json.loads((data / "attention_handle_curve_results.json").read_text())
        ood = json.loads((data / "attention_mixed_ood_results.json").read_text())
        motifs = json.loads((data / "attn_motifs3_results.json").read_text())
        quotient = json.loads((data / "shared_value_output_quotient_contract.json").read_text())
        self.assertEqual(decoded["quotient_bits"], 290859424)
        self.assertAlmostEqual(handles["extraction"]["global_retained_fraction"],
                               0.9878032701)
        self.assertAlmostEqual(handles["removal"]
                               ["global_aligned_minus_random_damage_ce"], 9.3056913800)
        self.assertAlmostEqual(ood["members"]["synthetic_induction"]
                               ["decoded_delta_from_clean_ce"], 5.1409420857)
        self.assertEqual(motifs["census"],
                         {"diffuse": 77, "prev": 27, "self": 47, "ind": 9, "first": 2})
        self.assertEqual(quotient["generic_dimension"]["independent_layer_undercount"],
                         2506752)
        self.assertIsNone(quotient.get("quotient_bits"))


if __name__ == "__main__":
    unittest.main()
