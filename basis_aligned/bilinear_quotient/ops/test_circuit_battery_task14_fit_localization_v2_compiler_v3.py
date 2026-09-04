#!/usr/bin/env python3
# BQLANE: cpu
"""Focused CPU tests for the stagewise task14 localization compiler v3."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "circuit_battery_task14_fit_localization_v2_compiler_v3.py"
MANIFEST = HERE / "circuit_battery_task14_fit_localization_v2_call_manifest_v3.json"
INDEX = HERE / "circuit_battery_task14_fit_localization_v2_call_index_v3.bin"
DRYRUN = HERE / "circuit_battery_task14_fit_localization_v2_compiler_v3_dryrun.json"
EXPECTED_SOURCE_SHA256 = "UNFROZEN"
EXPECTED_MANIFEST_SHA256 = "UNFROZEN"
EXPECTED_INDEX_SHA256 = "UNFROZEN"
EXPECTED_DRYRUN_SHA256 = "UNFROZEN"
EVIDENCE = "ab" * 32


def load_module():
    spec = importlib.util.spec_from_file_location("task14_v3_compiler_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Task14LocalizationV3CompilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = load_module()
        cls.authority, cls.partition, cls.donors = cls.m.load_inputs()
        cls.rows, cls.endpoints = cls.m._row_maps(cls.authority)
        cls.records, cls.records_by_id = cls.m._record_maps(cls.donors)
        cls.manifest_raw = MANIFEST.read_bytes() if MANIFEST.exists() else None
        cls.manifest = json.loads(cls.manifest_raw) if cls.manifest_raw else None
        cls.index_raw = INDEX.read_bytes() if INDEX.exists() else None

    def setUp(self) -> None:
        self.m._GLOBAL_TOKENS.clear()
        self.m._GLOBAL_CONTEXTS.clear()
        self.m._STAGE_CAPABILITIES.clear()
        self.m._STAGE_REPLAYS.clear()
        self.m._STAGE_COMPLETIONS.clear()
        self.m._SCIENTIFIC_TERMINALS.clear()
        self.m._TIMING_CAPABILITIES.clear()
        self.m._TIMING_AUTHORIZATIONS.clear()
        self.m._DEADLINE_CAPABILITIES.clear()

    def _fake_token(self):
        source = SOURCE.read_bytes()
        manifest = {
            "contract_sha256": "11" * 32,
            "compiler_source_sha256": hashlib.sha256(source).hexdigest(),
        }
        core = {
            "manifest_contract_sha256": manifest["contract_sha256"],
            "call_chunks_root_sha256": self.m.CANONICAL_CALL_CHUNKS_ROOT_SHA256,
            "call_index_sha256": self.m.CANONICAL_CALL_INDEX_SHA256,
            "call_count": self.m.CANONICAL_CALL_COUNT,
            "compiler_sha256": manifest["compiler_source_sha256"],
        }
        token_id, seal = self.m.canonical_sha256(core), object()
        token = self.m.GlobalPreflightToken(**core, token_id=token_id, _seal=seal)
        input_bytes = self.m.capture_input_bytes()
        self.m._GLOBAL_TOKENS[token_id] = {"seal": seal, "started": False}
        self.m._GLOBAL_CONTEXTS[token_id] = self.m._PreflightContext(
            self.m.canonical_bytes(manifest, newline=True), b"", input_bytes, source,
        )
        return token

    def _fake_replay(self, capability):
        record = self.m._STAGE_CAPABILITIES[capability.capability_id]
        self.assertIn(capability.next_stage, self.m.CALL_STAGES)
        record["attempted"] = True
        child_core = {
            "chunk_id": "test-chunk", "stage": capability.next_stage,
            "activation_guard": "test_only", "guard_evaluated": True,
            "guard_state_sha256": "21" * 32, "status": "active_completed",
            "call_index_offset": 0, "template_call_count": 1, "executed_call_count": 1,
            "call_index_slice_sha256": "20" * 32, "call_root_sha256": "23" * 32,
            "forward_calls": 1, "backward_calls": 0, "backward_graph_batches": 0,
            "optimizer_updates": 0, "example_evaluations": 1, "token_evaluations": 1,
        }
        child = self.m.ChunkReplayReceipt(
            **child_core, receipt_id=self.m.canonical_sha256(child_core),
        )
        child_rows = [{**child_core, "receipt_id": child.receipt_id}]
        core = {
            "stage": capability.next_stage,
            "capability_id": capability.capability_id,
            "active_path_root_sha256": self.m.canonical_sha256([
                {"chunk_receipt_id": child.receipt_id, "status": child.status},
            ]),
            "chunk_receipts_root_sha256": self.m.canonical_sha256(child_rows),
            "chunk_receipts": child_rows,
            "template_call_count": 1, "executed_call_count": 1, "forward_calls": 1,
            "backward_calls": 0, "backward_graph_batches": 0, "optimizer_updates": 0,
            "example_evaluations": 1, "token_evaluations": 1,
        }
        receipt_id, seal = self.m.canonical_sha256(core), object()
        receipt = self.m.StageReplayReceipt(
            capability.next_stage, capability.capability_id,
            core["active_path_root_sha256"], core["chunk_receipts_root_sha256"], (child,),
            1, 1, 1, 0, 0, 0, 1, 1, receipt_id, seal,
        )
        self.m._STAGE_REPLAYS[receipt_id] = {"seal": seal, "core": core}
        record["replay_id"] = receipt_id
        return receipt

    def _advance(self, capability, payload):
        receipt = self._fake_replay(capability) if capability.next_stage in self.m.CALL_STAGES else None
        return self.m.complete_stage(capability, payload, receipt)

    def _top3_sha(self, ranked, retained):
        return self.m.canonical_sha256({
            "schema": "task14_v3_top_three_h_evidence_v1",
            "eligible_h_count": len(ranked),
            "eligible_h_ranked_scores": [list(item) for item in ranked],
            "retained_h": list(retained),
        })

    def _ceiling(self, *, h_ranked=None, q=(-1, 0, 1)):
        ranked = h_ranked or ((0, 3.0), (-1, 2.0), (1, 1.0))
        retained = tuple(sorted(site for site, _score in ranked[:3]))
        return self.m.CeilingState(
            True, True, True, 0, len(ranked), ranked, retained, len(q), q,
            self._top3_sha(ranked, retained), EVIDENCE,
        )

    def _selection(self, *, ceiling=None, h_scores=None, q_scores=None):
        ceiling = ceiling or self._ceiling()
        hs = h_scores or ((-1, 0.2), (0, 0.9), (1, 0.4))
        qs = q_scores or ((-1, 0.1), (0, 0.2), (1, 1.0))
        h = min(hs, key=lambda item: (-item[1], item[0]))[0]
        q_max = max(score for _site, score in qs)
        table = dict(qs)
        onset = next((site for site in self.m.BOUNDARIES if site in table and site - 1 in table
                      and table[site] >= .9 * q_max and table[site - 1] < .5 * q_max), None)
        q = onset if onset is not None else min(qs, key=lambda item: (-item[1], item[0]))[0]
        top2 = tuple(sorted(site for site, _score in sorted(qs, key=lambda item: (-item[1], item[0]))[:2])) \
            if len(qs) >= 2 else None
        core = {
            "schema": "task14_v3_discovery_selection_evidence_v1",
            "selected_h": h, "selected_q": q,
            "top_two_q": None if top2 is None else list(top2),
            "h_objective_scores": [list(item) for item in hs],
            "q_t_scores": [list(item) for item in qs],
            "reader_selection_eligible": onset is not None,
        }
        return self.m.SelectionState(h, q, top2, hs, qs, onset is not None,
                                     self.m.canonical_sha256(core))

    def _prefix_to_validation(self, *, ceiling=None, selection=None):
        token = self._fake_token()
        cap = self.m.start_stagewise_execution(token)
        cap = self._advance(cap, self.m.NativeState(True, EVIDENCE))
        cap = self._advance(cap, self.m.GradientState(True, True, True, 0, EVIDENCE))
        ceiling = ceiling or self._ceiling()
        cap = self._advance(cap, ceiling)
        cap = self._advance(cap, self.m.JointFitState(True, True, EVIDENCE))
        cap = self._advance(cap, self.m.SpectralState(True, EVIDENCE))
        selection = selection or self._selection(ceiling=ceiling)
        cap = self._advance(cap, selection)
        cap = self._advance(cap, self.m.SelectedFitState(True, True, EVIDENCE))
        cap = self._advance(cap, self.m.ValidationCeilingState(True, EVIDENCE))
        return cap

    def test_source_has_no_forbidden_legacy_surface_or_model_import(self) -> None:
        text = SOURCE.read_text()
        for symbol in (
            "ActivePlanState", "BranchState", "project_terminal", "active_path_receipts",
            "def replay_active_path(", "derive_active_chunk_ids", "evaluate_chunk_guard",
        ):
            self.assertNotIn(symbol, text)
        tree = ast.parse(text)
        imports = {
            alias.name.split(".")[0] for node in ast.walk(tree)
            if isinstance(node, ast.Import) for alias in node.names
        } | {
            (node.module or "").split(".")[0] for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertTrue(imports.isdisjoint({"torch", "jax", "fastload", "tt_model"}))

    def test_exact_frozen_v2_initializer_known_answers(self) -> None:
        cases = (
            ((14001, 1, "H:-1", "joint", 0, 0), 1),
            ((14001, 1, "H:-1", "joint", 0, 1), -1),
            ((14005, 4, "Q:17", "A2_only", 1151, 3), 1),
            ((14003, 2, "Q:7", "A1_only", 511, 1), -1),
        )
        for (seed, rank, site, objective, d, j), expected in cases:
            self.assertEqual(self.m.initialization_entry_sign(
                seed=seed, rank=rank, site=site, objective_name=objective, d=d, j=j,
            ), expected)
        contract = self.m._initialization_contract()
        self.assertIn("task14-localization-v2-init", contract["entry_text"])
        self.assertIsNone(contract["counter_encoding"])

    def test_frozen_closure_and_identity(self) -> None:
        for _role, (relative, digest) in self.m.FROZEN.items():
            self.assertEqual(hashlib.sha256((self.m.REPO_ROOT / relative).read_bytes()).hexdigest(), digest)
        self.assertEqual(self.m.BLOCK_REVIEW_COMMIT, "60892e3994250b7f58330f4b2a84f8ed4126c928")
        self.assertEqual(self.m.BLOCK_REVIEW_SHA256, "3131fffd0b6c8cd18789b69e4909b0002ca3e90f2c965391c07444f56b63756a")

    def test_dag_is_path_sensitive_and_v2_order_fails(self) -> None:
        dag = self.m._dag()
        self.m._validate_dag(dag)
        nodes = [item["node"] for item in dag]
        self.assertLess(nodes.index("joint_rank1_fits"), nodes.index("discovery_selection"))
        self.assertLess(nodes.index("discovery_selection"), nodes.index("selected_family_and_rank_fits"))
        moved = copy.deepcopy(dag)
        selected = moved.pop(nodes.index("selected_family_and_rank_fits"))
        moved.insert(nodes.index("joint_rank1_fits"), selected)
        with self.assertRaises(self.m.CompileError):
            self.m._validate_dag(moved)
        wrong_branch = copy.deepcopy(dag)
        terminal = next(item for item in wrong_branch if item["node"] == "terminal_projection")
        field = "ordered_reader.reader_pass"
        terminal["optional_reads"].remove(field)
        terminal["required_reads"].append(field)
        terminal.pop("node_contract_sha256")
        terminal["node_contract_sha256"] = self.m.canonical_sha256(terminal)
        with self.assertRaises(self.m.CompileError):
            self.m._validate_dag(wrong_branch)

    def test_dag_writes_match_payload_fields_and_spectral_is_nonselective(self) -> None:
        dag = {item["node"]: item for item in self.m._dag()}
        for stage, payload_type in self.m._PAYLOAD_TYPES.items():
            expected = {f"{stage}.{field}" for field in payload_type.__dataclass_fields__}
            self.assertEqual(set(dag[stage]["writes"]), expected)
        spectral_reads = {
            item for item in dag["discovery_selection"]["required_reads"]
            if item.startswith("spectral_finite_diagnostic.")
        }
        self.assertEqual(spectral_reads, {
            "spectral_finite_diagnostic.diagnostic_complete",
            "spectral_finite_diagnostic.evidence_sha256",
        })

    def test_longest_path_uses_one_compatible_branch(self) -> None:
        weights = {item["node"]: 1.0 for item in self.m._dag()}
        result = self.m.longest_compatible_stage_path(weights)
        self.assertEqual(result["seconds"], 14.0)
        self.assertEqual(result["witness"][0], "preflight")
        with self.assertRaises(self.m.CompileError):
            self.m.longest_compatible_stage_path({**weights, "extra": 1.0})

    def test_call_stage_and_operation_shape_are_explicit(self) -> None:
        chunks = self.m._compile_stage_chunks("native_cache", (self.authority, self.partition, self.donors))
        self.assertEqual({chunk["stage"] for chunk in chunks}, {"native_cache"})
        endpoint = next(iter(self.endpoints))
        call = self.m._call(
            stage="native_cache", call_kind="native_cache_full_forward", branch="native:test",
            item_kind="endpoint", item_ids=[endpoint], position=None, boundary=None,
            endpoints=self.endpoints, records_by_id=self.records_by_id, retained=True,
        )
        shape = self.m.physical_call_shape(call)
        for field in ("cache_reads", "cache_writes", "array_contracts", "state_array_contracts",
                      "item_use_role_histogram", "logical_slot_histogram"):
            self.assertIn(field, shape)
        changed = copy.deepcopy(call)
        changed["cache_reads"] = ["different"]
        self.assertNotEqual(shape["call_shape_sha256"], self.m.physical_call_shape(changed)["call_shape_sha256"])

    def test_stage_capabilities_are_linear_and_first_start_works(self) -> None:
        token = self._fake_token()
        cap = self.m.start_stagewise_execution(token)
        self.assertEqual(cap.next_stage, "native_cache")
        with self.assertRaises(self.m.CompileError):
            self.m.start_stagewise_execution(token)
        receipt = self._fake_replay(cap)
        with self.assertRaises(self.m.CompileError):
            self.m.replay_stage(token, cap, lambda *_: None)
        next_cap = self.m.complete_stage(cap, self.m.NativeState(True, EVIDENCE), receipt)
        self.assertEqual(next_cap.next_stage, "discovery_gradients")
        self.assertIsNotNone(next_cap.predecessor_completion_id)
        completion = self.m._STAGE_COMPLETIONS[next_cap.predecessor_completion_id]
        self.assertEqual(completion["core"]["replay_receipt_id"], receipt.receipt_id)
        self.assertEqual(
            completion["core"]["replay_active_path_root_sha256"],
            receipt.active_path_root_sha256,
        )
        with self.assertRaises(self.m.CompileError):
            self.m.complete_stage(cap, self.m.NativeState(True, EVIDENCE), receipt)
        with self.assertRaises(self.m.CompileError):
            self.m.abort_stage(cap, "late")
        cap = self.m.start_stagewise_execution(self._fake_token())
        with self.assertRaises(self.m.CompileError):
            self.m.complete_stage(cap, self.m.NativeState(True, EVIDENCE))
        self.m.abort_stage(cap, "missing replay")

    def test_failed_completion_cannot_retry_but_can_abort(self) -> None:
        cap = self.m.start_stagewise_execution(self._fake_token())
        receipt = self._fake_replay(cap)
        with self.assertRaises(self.m.CompileError):
            self.m.complete_stage(cap, self.m.NativeState(1, EVIDENCE), receipt)
        with self.assertRaises(self.m.CompileError):
            self.m.complete_stage(cap, self.m.NativeState(True, EVIDENCE), receipt)
        abort = self.m.abort_stage(cap, "invalid evidence type")
        self.assertFalse(abort.package_allowed)
        self.assertIsNone(abort.scientific_terminal)

    def test_wrong_replay_receipt_consumes_completion_attempt(self) -> None:
        cap = self.m.start_stagewise_execution(self._fake_token())
        good = self._fake_replay(cap)
        forged = self.m.StageReplayReceipt(
            "discovery_gradients", good.capability_id, good.active_path_root_sha256,
            good.chunk_receipts_root_sha256, good.chunk_receipts, good.template_call_count,
            good.executed_call_count, good.forward_calls, good.backward_calls,
            good.backward_graph_batches, good.optimizer_updates, good.example_evaluations,
            good.token_evaluations, good.receipt_id, good._seal,
        )
        with self.assertRaises(self.m.CompileError):
            self.m.complete_stage(cap, self.m.NativeState(True, EVIDENCE), forged)
        with self.assertRaises(self.m.CompileError):
            self.m.complete_stage(cap, self.m.NativeState(True, EVIDENCE), good)

    def test_finite_invalid_screen_denominators_are_first_scientific_terminal(self) -> None:
        # Completed finite gradient denominator <= 1e-12.
        cap = self.m.start_stagewise_execution(self._fake_token())
        cap = self._advance(cap, self.m.NativeState(True, EVIDENCE))
        cap = self._advance(cap, self.m.GradientState(True, True, False, 1, EVIDENCE))
        terminal = self.m.project_stagewise_terminal(cap)
        self.assertEqual(terminal.terminal, "instrument_invalid")
        self.assertEqual(dict(terminal.node_statuses)["discovery_gradients"], "completed_health_invalid")
        self.m.validate_scientific_terminal(terminal)

        # Completed finite natural-margin denominator <= 1e-6 nat.
        cap = self.m.start_stagewise_execution(self._fake_token())
        cap = self._advance(cap, self.m.NativeState(True, EVIDENCE))
        cap = self._advance(cap, self.m.GradientState(True, True, True, 0, EVIDENCE))
        ordinary = self._ceiling()
        invalid = self.m.CeilingState(
            True, True, False, 1, ordinary.eligible_h_count,
            ordinary.eligible_h_ranked_scores, ordinary.retained_h,
            ordinary.eligible_q_count, ordinary.retained_q,
            ordinary.top_three_h_evidence_sha256, EVIDENCE,
        )
        cap = self._advance(cap, invalid)
        terminal = self.m.project_stagewise_terminal(cap)
        self.assertEqual(terminal.terminal, "instrument_invalid")
        self.assertEqual(dict(terminal.node_statuses)["discovery_full_ceilings"], "completed_health_invalid")

    def test_q_eligibility_must_be_subset_of_h_eligibility(self) -> None:
        ranked = ((-1, 2.0), (0, 1.0))
        retained = (-1, 0)
        bad = self.m.CeilingState(
            True, True, True, 0, 2, ranked, retained, 2, (-1, 1),
            self._top3_sha(ranked, retained), EVIDENCE,
        )
        with self.assertRaisesRegex(self.m.CompileError, "subset"):
            self.m._validate_stage_payload("discovery_full_ceilings", bad, {
                "native_cache": self.m.NativeState(True, EVIDENCE),
                "discovery_gradients": self.m.GradientState(True, True, True, 0, EVIDENCE),
            })

    def test_operational_abort_every_prefix_never_packages(self) -> None:
        pre = self.m.abort_preflight("source mismatch")
        self.assertEqual(dict(pre.node_statuses)["preflight"], "failed")
        self.assertFalse(pre.package_allowed)
        cap = self.m.start_stagewise_execution(self._fake_token())
        for payload in (
            self.m.NativeState(True, EVIDENCE),
            self.m.GradientState(True, True, True, 0, EVIDENCE), self._ceiling(),
            self.m.JointFitState(True, True, EVIDENCE), self.m.SpectralState(True, EVIDENCE),
            self._selection(), self.m.SelectedFitState(True, True, EVIDENCE),
            self.m.ValidationCeilingState(True, EVIDENCE), self.m.ValidationState(False, True, EVIDENCE),
            self.m.NecessityState(True, EVIDENCE), self.m.ReaderState(True, EVIDENCE),
        ):
            probe = self.m.abort_stage(cap, f"fault at {cap.next_stage}")
            self.assertFalse(probe.package_allowed)
            self.assertIsNone(probe.scientific_terminal)
            self.assertEqual(dict(probe.node_statuses)[probe.failed_stage], "failed")
            # Rebuild the same prefix in a fresh process-local session for the next stage.
            cap = self._rebuild_prefix_and_advance(payload, cap.next_stage)
        terminal_abort = self.m.abort_stage(cap, "terminal projection fault")
        self.assertEqual(terminal_abort.failed_stage, "terminal_projection")
        self.assertFalse(terminal_abort.package_allowed)

    def _rebuild_prefix_and_advance(self, final_payload, final_stage):
        sequence = [
            ("native_cache", self.m.NativeState(True, EVIDENCE)),
            ("discovery_gradients", self.m.GradientState(True, True, True, 0, EVIDENCE)),
            ("discovery_full_ceilings", self._ceiling()),
            ("joint_rank1_fits", self.m.JointFitState(True, True, EVIDENCE)),
            ("spectral_finite_diagnostic", self.m.SpectralState(True, EVIDENCE)),
            ("discovery_selection", self._selection()),
            ("selected_family_and_rank_fits", self.m.SelectedFitState(True, True, EVIDENCE)),
            ("validation_full_ceilings", self.m.ValidationCeilingState(True, EVIDENCE)),
            ("locked_validation", self.m.ValidationState(False, True, EVIDENCE)),
            ("single_necessity", self.m.NecessityState(True, EVIDENCE)),
            ("ordered_reader", self.m.ReaderState(True, EVIDENCE)),
        ]
        cap = self.m.start_stagewise_execution(self._fake_token())
        for stage, payload in sequence:
            use = final_payload if stage == final_stage else payload
            cap = self._advance(cap, use)
            if stage == final_stage:
                return cap
        return cap

    def test_ceiling_retains_exact_top_three_even_more_eligible(self) -> None:
        ranked = ((3, 5.0), (-1, 4.0), (2, 3.0), (0, 2.0), (1, 1.0))
        ceiling = self._ceiling(h_ranked=ranked)
        token = self._fake_token()
        cap = self.m.start_stagewise_execution(token)
        cap = self._advance(cap, self.m.NativeState(True, EVIDENCE))
        cap = self._advance(cap, self.m.GradientState(True, True, True, 0, EVIDENCE))
        next_cap = self._advance(cap, ceiling)
        self.assertEqual(next_cap.next_stage, "joint_rank1_fits")
        self.assertEqual(ceiling.retained_h, (-1, 2, 3))

    def test_selection_onset_need_not_be_top_two_and_fallback_blocks_reader(self) -> None:
        q = (-1, 0, 1, 2, 3)
        ceiling = self._ceiling(q=q)
        qs = ((-1, .1), (0, .91), (1, .95), (2, .96), (3, 1.0))
        state = self._selection(
            ceiling=ceiling,
            h_scores=((-1, .2), (0, .9), (1, .4)), q_scores=qs,
        )
        self.assertEqual(state.selected_q, 0)
        self.assertNotIn(state.selected_q, state.top_two_q)
        self.m._validate_stage_payload("discovery_selection", state, {
            "discovery_full_ceilings": ceiling,
            "joint_rank1_fits": self.m.JointFitState(True, True, EVIDENCE),
            "spectral_finite_diagnostic": self.m.SpectralState(True, EVIDENCE),
        })
        fallback = self._selection(
            ceiling=ceiling,
            h_scores=((-1, .2), (0, .9), (1, .4)),
            q_scores=((-1, .6), (0, .7), (1, .8), (2, .9), (3, 1.0)),
        )
        self.assertFalse(fallback.reader_selection_eligible)

    def test_higher_rank_requires_semantic_null_and_routes_terminal(self) -> None:
        cap = self._prefix_to_validation()
        cap = self._advance(cap, self.m.ValidationState(True, None, EVIDENCE))
        result = self.m.project_stagewise_terminal(cap)
        self.assertEqual(result.terminal, "fit_binary_state_rejected_higher_rank_needed_or_better")
        cap = self._prefix_to_validation()
        receipt = self._fake_replay(cap)
        with self.assertRaises(self.m.CompileError):
            self.m.complete_stage(cap, self.m.ValidationState(True, False, EVIDENCE), receipt)

    def test_health_invalid_only_after_complete_finite_fit_stage(self) -> None:
        token = self._fake_token()
        cap = self.m.start_stagewise_execution(token)
        cap = self._advance(cap, self.m.NativeState(True, EVIDENCE))
        cap = self._advance(cap, self.m.GradientState(True, True, True, 0, EVIDENCE))
        cap = self._advance(cap, self._ceiling())
        receipt = self._fake_replay(cap)
        with self.assertRaises(self.m.CompileError):
            self.m.complete_stage(cap, self.m.JointFitState(False, False, EVIDENCE), receipt)
        abort = self.m.abort_stage(cap, "incomplete optimizer schedule")
        self.assertFalse(abort.package_allowed)
        cap = self.m.start_stagewise_execution(self._fake_token())
        cap = self._advance(cap, self.m.NativeState(True, EVIDENCE))
        cap = self._advance(cap, self.m.GradientState(True, True, True, 0, EVIDENCE))
        cap = self._advance(cap, self._ceiling())
        cap = self._advance(cap, self.m.JointFitState(True, False, EVIDENCE))
        terminal = self.m.project_stagewise_terminal(cap)
        self.assertEqual(terminal.terminal, "instrument_invalid")
        self.assertEqual(dict(terminal.node_statuses)["joint_rank1_fits"], "completed_health_invalid")

    def test_terminal_routes_have_explicit_skips(self) -> None:
        cap = self._prefix_to_validation()
        cap = self._advance(cap, self.m.ValidationState(False, True, EVIDENCE))
        cap = self._advance(cap, self.m.NecessityState(True, EVIDENCE))
        cap = self._advance(cap, self.m.ReaderState(False, EVIDENCE))
        terminal = self.m.project_stagewise_terminal(cap)
        statuses = dict(terminal.node_statuses)
        self.assertEqual(statuses["two_site_redundancy"], "skipped")
        self.assertEqual(terminal.terminal, "fit_rank1_state_supported_reader_unresolved")

    def test_terminal_identity_binds_evidence_and_replay_chain(self) -> None:
        def make(evidence):
            cap = self.m.start_stagewise_execution(self._fake_token())
            cap = self._advance(cap, self.m.NativeState(True, evidence))
            cap = self._advance(cap, self.m.GradientState(True, True, False, 1, evidence))
            return self.m.project_stagewise_terminal(cap)

        first = make("ab" * 32)
        second = make("cd" * 32)
        self.assertEqual(first.terminal, second.terminal)
        self.assertNotEqual(first.terminal_id, second.terminal_id)
        forged = self.m.ScientificTerminalState(
            first.completed_stages, first.terminal, first.terminal_id, first.node_statuses,
            first.root_token_id, "ee" * 32, first.predecessor_completion_id,
            True, first._seal,
        )
        with self.assertRaises(self.m.CompileError):
            self.m.validate_scientific_terminal(forged)

    def test_terminal_projector_exhaustively_reaches_exact_registered_set(self) -> None:
        observed = set()

        token = self._fake_token()
        cap = self.m.start_stagewise_execution(token)
        cap = self._advance(cap, self.m.NativeState(True, EVIDENCE))
        cap = self._advance(cap, self.m.GradientState(True, True, True, 0, EVIDENCE))
        empty_ranked = ()
        empty = self.m.CeilingState(
            True, True, True, 0, 0, empty_ranked, (), 0, (),
            self._top3_sha(empty_ranked, ()), EVIDENCE,
        )
        observed.add(self.m.project_stagewise_terminal(self._advance(cap, empty)).terminal)

        token = self._fake_token()
        cap = self.m.start_stagewise_execution(token)
        cap = self._advance(cap, self.m.NativeState(True, EVIDENCE))
        cap = self._advance(cap, self.m.GradientState(True, True, True, 0, EVIDENCE))
        cap = self._advance(cap, self._ceiling())
        observed.add(self.m.project_stagewise_terminal(
            self._advance(cap, self.m.JointFitState(True, False, EVIDENCE))
        ).terminal)

        # Higher-rank rescue and semantic failure.
        for validation in (
            self.m.ValidationState(True, None, EVIDENCE),
            self.m.ValidationState(False, False, EVIDENCE),
        ):
            cap = self._prefix_to_validation()
            observed.add(self.m.project_stagewise_terminal(self._advance(cap, validation)).terminal)

        # Single-site reader pass and unresolved.
        for reader in (True, False):
            cap = self._prefix_to_validation()
            cap = self._advance(cap, self.m.ValidationState(False, True, EVIDENCE))
            cap = self._advance(cap, self.m.NecessityState(True, EVIDENCE))
            cap = self._advance(cap, self.m.ReaderState(reader, EVIDENCE))
            observed.add(self.m.project_stagewise_terminal(cap).terminal)

        # Redundancy fail, then redundancy pass with reader pass/fail.
        cap = self._prefix_to_validation()
        cap = self._advance(cap, self.m.ValidationState(False, True, EVIDENCE))
        cap = self._advance(cap, self.m.NecessityState(False, EVIDENCE))
        cap = self._advance(cap, self.m.RedundancyState(False, EVIDENCE))
        observed.add(self.m.project_stagewise_terminal(cap).terminal)
        for reader in (True, False):
            cap = self._prefix_to_validation()
            cap = self._advance(cap, self.m.ValidationState(False, True, EVIDENCE))
            cap = self._advance(cap, self.m.NecessityState(False, EVIDENCE))
            cap = self._advance(cap, self.m.RedundancyState(True, EVIDENCE))
            cap = self._advance(cap, self.m.ReaderState(reader, EVIDENCE))
            observed.add(self.m.project_stagewise_terminal(cap).terminal)

        self.assertEqual(observed, set(self.m.TERMINALS))

    def test_stage_replay_uses_captured_inputs_without_filesystem_reopen(self) -> None:
        self.assertEqual(
            list(inspect.signature(self.m.replay_stage).parameters),
            ["token", "capability", "visitor"],
        )
        input_bytes = self.m.capture_input_bytes()
        parsed = self.m.parse_captured_inputs(input_bytes)
        calls = []
        chunks = self.m._compiler_visit_stage_call_descriptors(
            "native_cache", lambda chunk, call: calls.append((chunk, call)), parsed,
        )
        raw = b"".join(bytes.fromhex(call["call_id"]) for _chunk, call in calls)
        offset = 0
        for chunk in chunks:
            ids = [call["call_id"] for owner, call in calls if owner == chunk["chunk_id"]]
            encoded = b"".join(bytes.fromhex(item) for item in ids)
            chunk["call_index_offset"] = offset
            chunk["call_index_count"] = len(ids)
            chunk["call_index_slice_sha256"] = hashlib.sha256(encoded).hexdigest()
            offset += len(ids)
        source = SOURCE.read_bytes()
        manifest = {
            "contract_sha256": "77" * 32,
            "compiler_source_sha256": hashlib.sha256(source).hexdigest(),
            "call_chunks": chunks,
        }
        core = {
            "manifest_contract_sha256": manifest["contract_sha256"],
            "call_chunks_root_sha256": self.m.CANONICAL_CALL_CHUNKS_ROOT_SHA256,
            "call_index_sha256": self.m.CANONICAL_CALL_INDEX_SHA256,
            "call_count": self.m.CANONICAL_CALL_COUNT,
            "compiler_sha256": manifest["compiler_source_sha256"],
        }
        token_id, seal = self.m.canonical_sha256(core), object()
        token = self.m.GlobalPreflightToken(**core, token_id=token_id, _seal=seal)
        self.m._GLOBAL_TOKENS[token_id] = {"seal": seal, "started": False}
        self.m._GLOBAL_CONTEXTS[token_id] = self.m._PreflightContext(
            self.m.canonical_bytes(manifest, newline=True), raw, input_bytes, source,
        )
        cap = self.m.start_stagewise_execution(token)
        visited = []
        with mock.patch.object(self.m, "safe_read", side_effect=AssertionError("reopened filesystem")):
            receipt = self.m.replay_stage(token, cap, lambda chunk, call: visited.append((chunk, call)))
        self.assertEqual(receipt.executed_call_count, 8)
        self.assertEqual(len(visited), 8)
        self.assertEqual(receipt.chunk_receipts[0].template_call_count, 8)
        self.assertEqual(receipt.chunk_receipts[0].executed_call_count, 8)

        # A callback failure preserves the exact attempted position and charges
        # the possibly-incurred call conservatively, while completion stays one
        # call behind the attempt.
        failed_manifest = {**manifest, "contract_sha256": "78" * 32}
        failed_core = {**core, "manifest_contract_sha256": failed_manifest["contract_sha256"]}
        failed_id, failed_seal = self.m.canonical_sha256(failed_core), object()
        failed_token = self.m.GlobalPreflightToken(
            **failed_core, token_id=failed_id, _seal=failed_seal,
        )
        self.m._GLOBAL_TOKENS[failed_id] = {"seal": failed_seal, "started": False}
        self.m._GLOBAL_CONTEXTS[failed_id] = self.m._PreflightContext(
            self.m.canonical_bytes(failed_manifest, newline=True), raw, input_bytes, source,
        )
        failed_cap = self.m.start_stagewise_execution(failed_token)
        calls_seen = 0

        def fail_second(_chunk, _call):
            nonlocal calls_seen
            calls_seen += 1
            if calls_seen == 2:
                raise RuntimeError("planted callback failure")

        with self.assertRaises(self.m.OperationalAbort):
            self.m.replay_stage(failed_token, failed_cap, fail_second)
        aborted = self.m.abort_stage(failed_cap, "callback failure")
        self.assertEqual(aborted.attempted_call_count, 2)
        self.assertEqual(aborted.completed_call_count, 1)
        self.assertEqual(aborted.forward_calls, 2)
        self.assertEqual(aborted.completed_slice_count, 0)
        self.assertEqual(aborted.active_chunk_call_offset, 1)
        self.assertFalse(aborted.package_allowed)

    def test_preflight_hashes_captured_source_bytes_itself(self) -> None:
        source = SOURCE.read_bytes()
        call_id = "88" * 32
        raw = bytes.fromhex(call_id)
        value = {
            "compiler_source_sha256": hashlib.sha256(source).hexdigest(),
            "contract_sha256": "99" * 32,
            "call_chunks": [{"chunk_id": "00_native_cache", "call_index_count": 1}],
            "call_index": {"call_count": 1},
        }
        with mock.patch.object(self.m, "validate_manifest"), \
             mock.patch.object(self.m, "validate_call_index"), \
             mock.patch.object(self.m, "_compiler_iter_call_descriptors", return_value=iter([
                 ("00_native_cache", {"call_id": call_id}),
             ])):
            token = self.m.preflight_global_call_index(
                value, raw,
            )
            self.assertEqual(token.compiler_sha256, hashlib.sha256(source).hexdigest())
        self.m._GLOBAL_TOKENS.clear()
        with mock.patch.object(self.m, "validate_manifest"), \
             mock.patch.object(self.m, "validate_call_index"), \
             mock.patch.object(self.m, "safe_read", return_value=source + b"mutation"):
            with self.assertRaises(self.m.CompileError):
                self.m.preflight_global_call_index(value, raw)

    def test_mutated_import_cannot_present_old_frozen_source(self) -> None:
        original = SOURCE.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            changed_path = Path(directory) / "a" / "b" / "c" / SOURCE.name
            changed_path.parent.mkdir(parents=True)
            changed_path.write_bytes(original + b"\n# adversarial post-review mutation\n")
            spec = importlib.util.spec_from_file_location("task14_v3_mutated_import", changed_path)
            changed = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            sys.modules[spec.name] = changed
            spec.loader.exec_module(changed)
            value = {
                "compiler_source_sha256": hashlib.sha256(original).hexdigest(),
                "contract_sha256": "99" * 32, "call_chunks": [], "call_index": {"call_count": 0},
            }
            with mock.patch.object(changed, "validate_manifest"), \
                 mock.patch.object(changed, "validate_call_index"):
                with self.assertRaises(changed.CompileError):
                    changed.preflight_global_call_index(value, b"")

    def test_strict_runtime_types_reject_truthiness_and_site_floats(self) -> None:
        cap = self.m.start_stagewise_execution(self._fake_token())
        receipt = self._fake_replay(cap)
        with self.assertRaises(self.m.CompileError):
            self.m.complete_stage(cap, self.m.NativeState(1, EVIDENCE), receipt)
        ceiling = self._ceiling()
        bad = self.m.CeilingState(
            True, True, True, 0, True, ceiling.eligible_h_ranked_scores, ceiling.retained_h,
            ceiling.eligible_q_count, ceiling.retained_q,
            ceiling.top_three_h_evidence_sha256, EVIDENCE,
        )
        with self.assertRaises(self.m.CompileError):
            self.m._validate_stage_payload("discovery_full_ceilings", bad, {})
        with self.assertRaises(self.m.CompileError):
            self.m._exact_site_tuple((-1, 0.0), "sites", maximum=3)

    def test_timing_is_stateful_exact_limit_and_schema_only(self) -> None:
        receipt = {
            "schema": "task14_v3_physical_call_shape_timing_v1",
            "stage": "native_cache", "call_shape_sha256": "33" * 32,
            "p99_seconds": 2.0, "independent_review_sha256": "44" * 32,
        }
        receipt["receipt_sha256"] = self.m.canonical_sha256(receipt)
        timing = self.m.validate_timing_receipt_schema(receipt)
        self.assertIn("schema_only", self.m._TIMING_CAPABILITIES[timing.token_id]["status"])
        forged = self.m.TimingAuthorization((timing.token_id,), "55" * 32, object())
        with self.assertRaises(self.m.OperationalAbort):
            self.m.start_deadline(lambda: 0.0, (timing,), authorization=forged)
        seal = object()
        authorization_id = "66" * 32
        authorization = self.m.TimingAuthorization((timing.token_id,), authorization_id, seal)
        self.m._TIMING_AUTHORIZATIONS[authorization_id] = {
            "seal": seal, "timing_token_ids": (timing.token_id,), "test_only": True,
        }
        ticks = iter([0.0, 1.0, 2.0, 1.5])
        deadline = self.m.start_deadline(lambda: next(ticks), (timing,), authorization=authorization)
        self.m.deadline_check(
            lambda: next(ticks), deadline=deadline, timing=timing,
            stage="native_cache", call_shape_sha256="33" * 32,
        )
        self.m.deadline_check_after(lambda: next(ticks), deadline=deadline, stage="native_cache")
        with self.assertRaises(self.m.OperationalAbort):
            self.m.deadline_check(
                lambda: next(ticks), deadline=deadline, timing=timing,
                stage="native_cache", call_shape_sha256="33" * 32,
            )
        with self.assertRaises(self.m.OperationalAbort):
            self.m.start_deadline(
                lambda: 0.0, (timing,), authorization=authorization,
                hard_limit_seconds=28801,
            )

    def test_namespace_has_no_public_override_and_rejects_entries(self) -> None:
        self.assertEqual(list(inspect.signature(self.m.preflight_namespace_absent).parameters), [])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(self.m, "_NAMESPACE_ROOT", root):
                receipt = self.m.preflight_namespace_absent()
                self.assertEqual(receipt["status"], "all_absent")
                for relative in self.m.RESERVED_NAMESPACE_PATHS:
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.symlink_to(root / "missing")
                    with self.assertRaises(self.m.OperationalAbort):
                        self.m.preflight_namespace_absent()
                    path.unlink()

    def test_boundary_x0_v1_and_composed_trajectory(self) -> None:
        minus1 = self.m.apply_boundary_edit(
            boundary=-1, target_residual=[1.0, 2.0], donor_residual=[7.0, 8.0],
            coordinate=0, derive_v1=lambda x0: [10.0 * x0[0], 10.0 * x0[1]],
        )
        self.assertEqual(minus1["x0"], (7.0, 2.0))
        self.assertEqual(minus1["v1"], (70.0, 20.0))
        composed = self.m.apply_composed_boundary_edits(
            first_boundary=-1, second_boundary=0,
            initial_state={"residual": [1.0, 2.0], "x0": None, "v1": None},
            first_donor=[7.0, 8.0], second_donor=[9.0, 11.0], coordinate=0,
            derive_v1=lambda x0: [2.0 * item for item in x0],
            propagate_to_second=lambda first: {
                "residual": [first["residual"][0] + first["v1"][0], first["residual"][1]],
                "x0": first["x0"], "v1": first["v1"],
            },
        )
        self.assertEqual(composed["live_at_second"]["residual"][0], 21.0)
        self.assertEqual(composed["second"]["x0"], (7.0, 2.0))
        with self.assertRaises(self.m.CompileError):
            self.m.apply_boundary_edit(
                boundary=True, target_residual=[1.0], donor_residual=[2.0], coordinate=0,
                target_x0=[1.0], target_v1=[1.0],
            )

    @unittest.skipUnless(MANIFEST.exists() and INDEX.exists() and DRYRUN.exists(), "materialize after source freeze")
    def test_materialized_hashes_and_exact_manifest(self) -> None:
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), EXPECTED_SOURCE_SHA256)
        self.assertEqual(hashlib.sha256(self.manifest_raw).hexdigest(), EXPECTED_MANIFEST_SHA256)
        self.assertEqual(hashlib.sha256(self.index_raw).hexdigest(), EXPECTED_INDEX_SHA256)
        self.assertEqual(hashlib.sha256(DRYRUN.read_bytes()).hexdigest(), EXPECTED_DRYRUN_SHA256)
        self.m.validate_manifest(self.manifest)
        self.m.validate_call_index(self.manifest, self.index_raw)
        self.assertEqual(self.manifest["compiler_source_sha256"], EXPECTED_SOURCE_SHA256)

    @unittest.skipUnless(MANIFEST.exists() and INDEX.exists(), "materialize after source freeze")
    def test_coherent_manifest_and_index_attacks_reject(self) -> None:
        value = copy.deepcopy(self.manifest)
        removed = value["call_chunks"].pop(len(value["call_chunks"]) // 2)
        count = removed["call_index_count"]
        for chunk in value["call_chunks"]:
            if chunk["call_index_offset"] > removed["call_index_offset"]:
                chunk["call_index_offset"] -= count
        value["call_chunk_count"] -= 1
        value["call_chunks_root_sha256"] = self.m.canonical_sha256(value["call_chunks"])
        value["call_index"]["call_count"] -= count
        value["call_index"]["byte_count"] -= 32 * count
        value["call_index"]["sha256"] = "55" * 32
        value["conditional_price"] = self.m._price_contract(value["call_chunks"])
        value["stage_ranges"] = self.m._stage_ranges(value["call_chunks"])
        value.pop("contract_sha256")
        value["contract_sha256"] = self.m.canonical_sha256(value)
        with self.assertRaises(self.m.CompileError):
            self.m.validate_manifest(value)
        changed = bytearray(self.index_raw)
        changed[len(changed) // 2] ^= 1
        with self.assertRaises(self.m.CompileError):
            self.m.validate_call_index(self.manifest, bytes(changed))

    @unittest.skipUnless(MANIFEST.exists(), "materialize after source freeze")
    def test_all_static_contract_mutations_reject(self) -> None:
        paths = [
            ("science", "decision_contract", "validation_thresholds", "higher_rank_improvement_strict_gt"),
            ("science", "terminal_precedence"), ("initialization", "seeds", 0),
            ("model_contract", "boundary_semantics", "17"),
            ("physical_batching", "logical_relations_per_update"),
            ("runtime_and_publication", "arithmetic", "training_objective"),
            ("runtime_and_publication", "interventions", "two_site_order"),
            ("retained_arrays", 0, "dtype"),
            ("retained_byte_contract", "maximum_raw_numeric_bytes"), ("fit_only", "phase"),
            ("dag", 0, "physical_call_stage"), ("stage_state_contract", "schema"),
            ("stage_replay_contract", "schema"),
            ("conditional_price", "maximum_active_upper_bound", "forward_calls"),
        ]
        for path in paths:
            value = copy.deepcopy(self.manifest)
            target = value
            for key in path[:-1]:
                target = target[key]
            key = path[-1]
            old = target[key]
            target[key] = old + 1 if type(old) in {int, float} else "changed"
            value.pop("contract_sha256")
            value["contract_sha256"] = self.m.canonical_sha256(value)
            with self.subTest(path=path), self.assertRaises(self.m.CompileError):
                self.m.validate_manifest(value)


if __name__ == "__main__":
    unittest.main()
