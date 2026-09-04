#!/usr/bin/env python3
# BQLANE: cpu
"""Adversarial CPU tests for the repaired task14 FIT localization-v2 compiler."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import itertools
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "circuit_battery_task14_fit_localization_v2_compiler_v2.py"
MANIFEST = HERE / "circuit_battery_task14_fit_localization_v2_call_manifest_v2.json"
INDEX = HERE / "circuit_battery_task14_fit_localization_v2_call_index_v2.bin"
DRYRUN = HERE / "circuit_battery_task14_fit_localization_v2_compiler_v2_dryrun.json"
EXPECTED_SOURCE_SHA256 = "6024009bc045200bc3525765dc1dd66261f84f9ccee0dbf9da7b2ddff3101415"
EXPECTED_MANIFEST_SHA256 = "5f870a292e9e2db0830156d09d17af10d6d2c8201cb134c80aee12d9261f1b2e"
EXPECTED_INDEX_SHA256 = "ae399e393d03af9b6232b7fc5339dd892b418ec7c88943735f8b72fc064c8ad9"
EXPECTED_DRYRUN_SHA256 = "6cae7b207e372d82c061d189c67bff05bd4772da366d41ebf03a5cdd0c58c0dd"


def load_module():
    spec = importlib.util.spec_from_file_location("task14_v2_compiler_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Task14LocalizationV2CompilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = load_module()
        cls.authority, cls.partition, cls.donors = cls.m.load_inputs()
        cls.rows, cls.endpoints = cls.m._row_maps(cls.authority)
        cls.records, cls.records_by_id = cls.m._record_maps(cls.donors)
        cls.manifest_raw = MANIFEST.read_bytes()
        cls.manifest = json.loads(cls.manifest_raw)
        cls.index_raw = INDEX.read_bytes()

    def test_frozen_file_hashes(self) -> None:
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), EXPECTED_SOURCE_SHA256)
        self.assertEqual(hashlib.sha256(self.manifest_raw).hexdigest(), EXPECTED_MANIFEST_SHA256)
        self.assertEqual(hashlib.sha256(self.index_raw).hexdigest(), EXPECTED_INDEX_SHA256)
        self.assertEqual(hashlib.sha256(DRYRUN.read_bytes()).hexdigest(), EXPECTED_DRYRUN_SHA256)

    def test_manifest_and_entire_call_index_validate(self) -> None:
        self.m.validate_manifest(self.manifest)
        self.m.validate_call_index(self.manifest, self.index_raw)
        self.assertEqual(
            self.manifest["call_chunk_count"],
            len({chunk["chunk_id"] for chunk in self.manifest["call_chunks"]}),
        )

    def test_every_conditional_guard_has_replayable_identity(self) -> None:
        for node in self.manifest["dag"]:
            core = dict(node)
            guard_id = core.pop("guard_id")
            self.assertEqual(guard_id, self.m.canonical_sha256(core))

    def test_call_index_interior_mutation_deletion_and_reorder_reject(self) -> None:
        midpoint = len(self.index_raw) // 2
        changed = bytearray(self.index_raw)
        changed[midpoint] ^= 1
        cases = [bytes(changed), self.index_raw[:-32]]
        reordered = bytearray(self.index_raw)
        reordered[midpoint:midpoint + 32], reordered[midpoint + 32:midpoint + 64] = (
            reordered[midpoint + 32:midpoint + 64], reordered[midpoint:midpoint + 32]
        )
        cases.append(bytes(reordered))
        for raw in cases:
            with self.subTest(size=len(raw)), self.assertRaises(self.m.CompileError):
                self.m.validate_call_index(self.manifest, raw)

    def test_frozen_source_mutation_and_symlink_reject(self) -> None:
        original = self.m._load_frozen("fit_authority")
        with mock.patch.object(self.m, "safe_read", return_value=original + b" "):
            with self.assertRaises(self.m.CompileError):
                self.m._load_frozen("fit_authority")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target"
            link = Path(directory) / "link"
            target.write_bytes(b"x")
            link.symlink_to(target)
            with self.assertRaises(OSError):
                self.m.safe_read(link)

    def test_foil_ids_derive_from_exact_answer_vocabulary(self) -> None:
        for endpoint in self.endpoints.values():
            self.assertIn(endpoint["answer_id"], {318, 389})
            self.assertIn(endpoint["foil_id"], {318, 389})
            self.assertNotEqual(endpoint["answer_id"], endpoint["foil_id"])

    def test_reader_binding_contains_distinct_H_and_Q_coordinates(self) -> None:
        record = next(
            record for record in self.records
            if record["arm"] == "cross_syntax"
            and self.endpoints[record["target_endpoint_id"]]["Q_position"]
            != self.endpoints[record["donor_endpoint_id"]]["Q_position"]
        )
        call = self.m._call(
            stage="test", call_kind="ordered_H_Q_reader_forward", branch="test",
            item_kind="record", item_ids=[record["record_id"]], position="H", boundary=0,
            endpoints=self.endpoints, records_by_id=self.records_by_id,
            extra_positions=("Q",),
        )
        self.assertEqual(call["extra_positions"], ["Q"])
        altered = copy.deepcopy(self.endpoints)
        altered[record["donor_endpoint_id"]]["Q_position"] += 1
        changed = self.m._call(
            stage="test", call_kind="ordered_H_Q_reader_forward", branch="test",
            item_kind="record", item_ids=[record["record_id"]], position="H", boundary=0,
            endpoints=altered, records_by_id=self.records_by_id, extra_positions=("Q",),
        )
        self.assertNotEqual(call["call_id"], changed["call_id"])

    def test_A_C_slots_and_normalizer_roles_are_hash_bound(self) -> None:
        step, digest, slots, normalizers = next(self.m._fit_step_stream(
            position="Q", boundary=-1, objective="joint", rank=1, seed=14001,
            records=self.records, endpoints=self.endpoints,
        ))
        self.assertEqual(step, 0)
        self.assertEqual(len(slots), 32)
        self.assertEqual(sum(slot["aggregate"] == "A_C" for slot in slots), 4)
        for slot in slots:
            if slot["aggregate"] == "A_C":
                self.assertIn(slot["item_id"], self.endpoints)
        changed_slots = copy.deepcopy(slots)
        changed_slots.pop(next(i for i, slot in enumerate(changed_slots) if slot["aggregate"] == "A_C"))
        changed_core = {
            "normalizer_cells": normalizers, "objective": "joint", "rank": 1,
            "seed": 14001, "site": "Q:-1", "slots": changed_slots, "step": 0,
        }
        self.assertNotEqual(digest, self.m.canonical_sha256(changed_core))

    def test_role_reassignment_changes_call_identity(self) -> None:
        record = self.records[0]
        common = dict(
            stage="test", call_kind="projector_intervention_train_forward", branch="test",
            item_kind="record", item_ids=[record["record_id"]], position="Q", boundary=-1,
            endpoints=self.endpoints, records_by_id=self.records_by_id,
        )
        left = self.m._call(**common, uses={record["record_id"]: ["train:A1:cell"]})
        right = self.m._call(**common, uses={record["record_id"]: ["normalizer:A1"]})
        self.assertNotEqual(left["call_id"], right["call_id"])

    def test_family_only_training_excludes_cross_syntax_and_validation(self) -> None:
        pools = self.m._fit_pools(
            position="Q", objective="A1_only", records=self.records, endpoints=self.endpoints,
        )
        ids = {item for cells in pools.values() for values in cells.values() for item in values}
        donor_records = [self.records_by_id[item] for item in ids if item in self.records_by_id]
        self.assertTrue(donor_records)
        self.assertTrue(all(record["partition"] == "DISCOVERY" for record in donor_records))
        self.assertFalse(any(record["arm"] == "cross_syntax" for record in donor_records))

    def test_exact_step_count_slots_and_single_backward(self) -> None:
        stream = list(self.m._fit_step_stream(
            position="H", boundary=0, objective="joint", rank=1, seed=14001,
            records=self.records, endpoints=self.endpoints,
        ))
        self.assertEqual([stream[0][0], stream[-1][0]], [0, 399])
        self.assertTrue(all(len(item[2]) == 32 for item in stream))
        observed = []
        self.m._CALL_VISITOR = lambda _chunk, call: observed.append(call)
        try:
            chunk = self.m._fit_chunk(
                position="H", boundary=0, objective="joint", rank=1, seed=14001,
                records=self.records, endpoints=self.endpoints,
                records_by_id=self.records_by_id, activation="test",
            )
        finally:
            self.m._CALL_VISITOR = None
        self.assertEqual(chunk["optimizer_updates"], 400)
        self.assertEqual(chunk["backward_calls"], 400)
        self.assertGreater(chunk["backward_graph_batches"], 400)
        self.assertEqual(sum(call["logical_backward_after_this_call"] for call in observed), 400)
        self.assertTrue(all(call["item_count"] <= 192 for call in observed))
        self.assertTrue(all(len(call["item_ids"]) == call["item_count"] for call in observed))

    def test_boundary_and_full_forward_cache_semantics(self) -> None:
        semantics = self.manifest["model_contract"]["boundary_semantics"]
        self.assertIn("before block 0", semantics["-1"])
        self.assertIn("before x0", self.manifest["model_contract"]["auxiliary_state_semantics"]["boundary_-1"])
        self.assertIn("residual after complete block 17", semantics["17"])
        runtime = self.manifest["runtime_and_publication"]
        self.assertIn("no full-sequence boundary or suffix cache exists", runtime["intervention_call_semantics"])
        cache_names = json.dumps(self.manifest["call_chunks"])
        self.assertNotIn("native_full_sequence_boundary_cache", cache_names)
        self.assertNotIn("native_x0_and_first_value_cache", cache_names)

    def test_price_separates_updates_forwards_backwards_and_sequences(self) -> None:
        price = self.manifest["conditional_price"]
        maximum = price["maximum_active_upper_bound"]
        self.assertEqual(maximum["optimizer_updates"], 60000)
        self.assertEqual(maximum["backward_calls"], 60004)
        self.assertGreater(maximum["forward_calls"], maximum["optimizer_updates"])
        self.assertGreater(maximum["example_evaluations"], maximum["forward_calls"])
        self.assertGreater(maximum["token_evaluations"], maximum["example_evaluations"])

    def test_retained_array_dtypes_and_exact_byte_extrema(self) -> None:
        arrays = self.manifest["retained_arrays"]
        self.assertTrue(all(item["dtype"] == "float32" for item in arrays))
        contract = self.manifest["retained_byte_contract"]
        self.assertEqual(contract["minimum_valid_no_ceiling_raw_numeric_bytes"], 61694592)
        self.assertGreater(contract["maximum_raw_numeric_bytes"], contract["fixed_raw_numeric_bytes"])

    def test_runtime_guards_and_fit_only_closure(self) -> None:
        runtime = self.manifest["runtime_and_publication"]
        self.assertEqual(runtime["deadline"]["hard_gpu_seconds"], 28800)
        self.assertTrue(all(runtime["dead_intervention_tripwires"].values()))
        self.assertTrue(runtime["namespace"]["create_only"])
        self.assertIn("noreplace", runtime["namespace"]["publication"])
        self.assertEqual(self.manifest["fit_only"]["forbidden_phases"], ["SELECT", "TEST", "OOD"])
        self.assertFalse(any(
            token in item["path"].lower()
            for item in self.manifest["artifact_closure"]
            for token in ("select_authority", "test_authority", "ood_authority", "results.json", "evidence")
        ))
        peak = runtime["temporary_peak_storage"]
        self.assertTrue(peak["preauthorization_peak_receipt_required"])
        self.assertEqual(peak["largest_registered_batch"], {"sequences": 192, "sequence_tokens": 8})
        self.assertIn("1.25", peak["required_free_device_bytes_before_model_load"])

    def test_spectral_diagnostic_cannot_select_or_pass(self) -> None:
        spectral = self.manifest["science"]["spectral"]
        self.assertFalse(spectral["success_predicate"])
        self.assertFalse(spectral["validation_selector"])
        self.assertFalse(spectral["registered_DAS_initialization_changed"])
        self.assertIn("DISCOVERY", spectral["uses"])

    def test_terminal_precedence_and_rank_falsifier_are_exact(self) -> None:
        decision = self.manifest["science"]["decision_contract"]
        self.assertEqual(decision["fit"]["all_five_seeds_must_be_healthy"], True)
        self.assertEqual(decision["validation_thresholds"]["higher_rank_improvement_strict_gt"], 0.10)
        self.assertEqual(decision["terminal_precedence"][:3], [
            "instrument_invalid", "no_intervention_ceiling",
            "fit_binary_state_rejected_higher_rank_needed_or_better",
        ])
        self.assertEqual(len(decision["terminal_precedence"]), 9)

    def test_intervention_order_and_reader_baselines_are_frozen(self) -> None:
        interventions = self.manifest["runtime_and_publication"]["interventions"]
        self.assertIn("ascending boundary", interventions["two_site_order"])
        self.assertIn("native-target", interventions["reader_reset"])
        self.assertIn("natural-donor", interventions["reader_rescue"])
        failures = self.manifest["runtime_and_publication"]["failure_semantics"]
        self.assertEqual(failures["deadline_or_incomplete_call_index"], "hard_abort_without_scientific_terminal")

    def test_coherent_critical_manifest_mutations_reject(self) -> None:
        paths = [
            ("science", "decision_contract", "validation_thresholds", "ordinary_paired_and_cross_noun", "recovery"),
            ("science", "decision_contract", "discovery", "selected_H"),
            ("science", "decision_contract", "validation_thresholds", "higher_rank_improvement_strict_gt"),
            ("science", "terminal_precedence"),
            ("initialization", "seeds", 0),
            ("physical_batching", "logical_relations_per_update"),
            ("model_contract", "boundary_semantics", "17"),
            ("conditional_price", "maximum_active_upper_bound", "forward_calls"),
            ("science", "decision_contract", "fit", "steps"),
            ("runtime_and_publication", "arithmetic", "training_objective"),
            ("runtime_and_publication", "interventions", "two_site_order"),
            ("science", "spectral", "validation_selector"),
            ("dag", 0, "condition"),
            ("retained_arrays", 0, "dtype"),
            ("retained_byte_contract", "maximum_raw_numeric_bytes"),
            ("fit_only", "phase"),
        ]
        for path in paths:
            value = copy.deepcopy(self.manifest)
            target = value
            for key in path[:-1]:
                target = target[key]
            key = path[-1]
            if isinstance(target[key], bool):
                target[key] = not target[key]
            elif isinstance(target[key], (int, float)):
                target[key] = target[key] + 1
            elif isinstance(target[key], list):
                target[key] = list(reversed(target[key]))
            else:
                target[key] = "changed"
            value.pop("contract_sha256")
            value["contract_sha256"] = self.m.canonical_sha256(value)
            with self.subTest(path=path), self.assertRaises(self.m.CompileError):
                self.m.validate_manifest(value)

    def test_manifest_rejects_extra_self_hashed_field(self) -> None:
        value = copy.deepcopy(self.manifest)
        value["unreviewed_extension"] = True
        value.pop("contract_sha256")
        value["contract_sha256"] = self.m.canonical_sha256(value)
        with self.assertRaises(self.m.CompileError):
            self.m.validate_manifest(value)

    def test_coherent_conditional_chunk_deletion_rejects(self) -> None:
        value = copy.deepcopy(self.manifest)
        removed = value["call_chunks"].pop(len(value["call_chunks"]) // 2)
        removed_count = int(removed["call_index_count"])
        for chunk in value["call_chunks"]:
            if int(chunk["call_index_offset"]) > int(removed["call_index_offset"]):
                chunk["call_index_offset"] -= removed_count
        value["call_chunk_count"] -= 1
        value["call_chunks_root_sha256"] = self.m.canonical_sha256(value["call_chunks"])
        value["call_index"]["call_count"] -= removed_count
        value["call_index"]["byte_count"] -= 32 * removed_count
        value["call_index"]["sha256"] = "11" * 32
        value["conditional_price"] = self.m._price_contract(value["call_chunks"])
        value.pop("contract_sha256")
        value["contract_sha256"] = self.m.canonical_sha256(value)
        with self.assertRaises(self.m.CompileError):
            self.m.validate_manifest(value)

    def test_terminal_projector_exhaustive_routes_and_skips(self) -> None:
        common = dict(
            eligible_h_count=3, eligible_q_count=4,
            fit_health_stage="selected_family_rank_passed", higher_rank_rescue=False,
            semantic_gates_pass=True,
        )
        cases = [
            (self.m.BranchState(operational_fault=True), None),
            (self.m.BranchState(eligible_h_count=0, eligible_q_count=4), "no_intervention_ceiling"),
            (self.m.BranchState(
                eligible_h_count=3, eligible_q_count=4, fit_health_stage="joint_rank1_failed",
            ), "instrument_invalid"),
            (self.m.BranchState(
                eligible_h_count=3, eligible_q_count=4, fit_health_stage="selected_family_rank_failed",
            ), "instrument_invalid"),
            (self.m.BranchState(
                eligible_h_count=3, eligible_q_count=4,
                fit_health_stage="selected_family_rank_passed", higher_rank_rescue=True,
            ),
             "fit_binary_state_rejected_higher_rank_needed_or_better"),
            (self.m.BranchState(**{**common, "semantic_gates_pass": False}),
             "fit_rank1_complete_subject_state_not_identified"),
            (self.m.BranchState(**{**common, "single_necessity_pass": False,
                                   "redundancy_available": False}),
             "fit_rank1_state_sufficiency_only"),
            (self.m.BranchState(**{**common, "single_necessity_pass": False,
                                   "redundancy_available": True, "redundancy_pass": False}),
             "fit_rank1_state_sufficiency_only"),
            (self.m.BranchState(**{**common, "single_necessity_pass": True,
                                   "h_before_q": True, "reader_pass": True}),
             "fit_rank1_state_and_ordered_reader_supported"),
            (self.m.BranchState(**{**common, "single_necessity_pass": True,
                                   "h_before_q": True, "reader_pass": False}),
             "fit_rank1_state_supported_reader_unresolved"),
            (self.m.BranchState(**{**common, "single_necessity_pass": True,
                                   "h_before_q": False}),
             "fit_rank1_state_supported_reader_unresolved"),
            (self.m.BranchState(**{**common, "single_necessity_pass": False,
                                   "redundancy_available": True, "redundancy_pass": True,
                                   "h_before_q": True, "reader_pass": True}),
             "fit_rank1_redundant_state_and_ordered_reader_supported"),
            (self.m.BranchState(**{**common, "single_necessity_pass": False,
                                   "redundancy_available": True, "redundancy_pass": True,
                                   "h_before_q": False}),
             "fit_rank1_two_site_redundant_state_reader_unresolved"),
        ]
        for state, terminal in cases:
            with self.subTest(terminal=terminal):
                result = self.m.project_terminal(state)
                self.assertEqual(result["scientific_terminal"], terminal)
                self.assertEqual(set(result["node_statuses"]), set(self.m._DAG_NODES))
                self.assertEqual(
                    result["package_allowed"], result["disposition"] == "scientific_terminal",
                )
        single = self.m.project_terminal(cases[8][0])
        self.assertEqual(single["node_statuses"]["two_site_redundancy"], "skipped")
        unavailable = self.m.project_terminal(cases[6][0])
        self.assertEqual(unavailable["node_statuses"]["two_site_redundancy"], "skipped")
        failed = self.m.project_terminal(cases[7][0])
        self.assertEqual(failed["node_statuses"]["two_site_redundancy"], "failed")
        unordered = self.m.project_terminal(cases[10][0])
        self.assertEqual(unordered["node_statuses"]["ordered_reader"], "skipped")
        joint_bad = self.m.project_terminal(cases[2][0])
        self.assertEqual(joint_bad["node_statuses"]["joint_rank1_fits"], "failed_health")
        self.assertEqual(joint_bad["node_statuses"]["discovery_selection"], "skipped")
        selected_bad = self.m.project_terminal(cases[3][0])
        self.assertEqual(selected_bad["node_statuses"]["selected_family_and_rank_fits"], "failed_health")

    def test_terminal_projector_rejects_incomplete_and_simultaneous_routes(self) -> None:
        with self.assertRaises(self.m.CompileError):
            self.m.project_terminal(self.m.BranchState(
                eligible_h_count=3, eligible_q_count=4, fit_health_stage=None,
            ))
        with self.assertRaises(self.m.CompileError):
            self.m.project_terminal(self.m.BranchState(
                eligible_h_count=3, eligible_q_count=4,
                fit_health_stage="selected_family_rank_passed", higher_rank_rescue=False,
                semantic_gates_pass=True, single_necessity_pass=True,
                redundancy_available=True, redundancy_pass=True,
                h_before_q=True, reader_pass=True,
            ))
        abort = self.m.project_terminal(self.m.BranchState(operational_fault=True))
        self.assertIsNone(abort["scientific_terminal"])
        self.assertFalse(abort["package_allowed"])

    def test_deadline_and_namespace_guards_never_create_scientific_terminal(self) -> None:
        ticks = iter([9.5])
        with self.assertRaises(self.m.OperationalAbort):
            self.m.deadline_check(
                lambda: next(ticks), start=0.0, limit_seconds=10.0,
                reviewed_p99_seconds=1.0, phase="call",
            )
        ticks = iter([10.1])
        with self.assertRaises(self.m.OperationalAbort):
            self.m.deadline_check_after(
                lambda: next(ticks), start=0.0, limit_seconds=10.0, phase="call",
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            absent = root / "absent"
            receipt = self.m.preflight_namespace_absent([absent])
            self.assertFalse(absent.exists())
            self.assertEqual(receipt["status"], "all_absent")
            occupied = root / "occupied"
            occupied.write_bytes(b"x")
            with self.assertRaises(self.m.OperationalAbort):
                self.m.preflight_namespace_absent([occupied])
            dangling = root / "dangling"
            dangling.symlink_to(root / "missing")
            with self.assertRaises(self.m.OperationalAbort):
                self.m.preflight_namespace_absent([dangling])
        clock = iter([9.5])
        projection = self.m.guarded_terminal_projection(
            self.m.BranchState(eligible_h_count=0, eligible_q_count=0),
            action=lambda: None, clock=lambda: next(clock), start=0.0,
            limit_seconds=10.0, reviewed_p99_seconds=1.0,
        )
        self.assertEqual(projection["disposition"], "operational_abort")
        self.assertIsNone(projection["scientific_terminal"])
        self.assertFalse(projection["package_allowed"])

    def test_boundary_x0_v1_and_composed_trajectory_tripwires(self) -> None:
        minus1 = self.m.apply_boundary_edit(
            boundary=-1, target_residual=[1.0, 2.0], donor_residual=[7.0, 8.0],
            coordinate=0, derive_v1=lambda x0: [10.0 * x0[0], 10.0 * x0[1]],
        )
        self.assertEqual(minus1["x0"], (7.0, 2.0))
        self.assertEqual(minus1["v1"], (70.0, 20.0))
        later = self.m.apply_boundary_edit(
            boundary=0, target_residual=[1.0, 2.0], donor_residual=[7.0, 8.0],
            coordinate=1, target_x0=[3.0, 4.0], target_v1=[5.0, 6.0],
        )
        self.assertEqual(later["x0"], (3.0, 4.0))
        self.assertEqual(later["v1"], (5.0, 6.0))
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

    def test_global_preflight_compares_each_descriptor_in_order(self) -> None:
        descriptors = list(itertools.islice(self.m.iter_call_descriptors(), 12))
        chunks = copy.deepcopy(self.manifest["call_chunks"][:2])
        raw = b"".join(bytes.fromhex(call["call_id"]) for _chunk, call in descriptors)
        mini = {
            "call_chunks": chunks,
            "call_index": {
                "byte_count": len(raw), "call_count": len(descriptors),
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
        }
        with mock.patch.object(self.m, "iter_call_descriptors", return_value=iter(descriptors)):
            receipt = self.m.preflight_global_call_index(mini, raw)
        self.assertEqual(receipt["call_count"], 12)
        broken = copy.deepcopy(descriptors)
        broken[5][1]["call_id"] = "00" * 32
        with mock.patch.object(self.m, "iter_call_descriptors", return_value=iter(broken)):
            with self.assertRaises(self.m.CompileError):
                self.m.preflight_global_call_index(mini, raw)
        active_state = self.m.ActivePlanState(
            preflight_pass=True, native_cache_complete=True, gradient_cache_complete=True,
            retained_h=(), retained_q=(), selected_h=None, selected_q=None,
            top_two_q=None, fit_health_stage=None,
            semantic_and_falsifier_gates_pass=None, single_necessity_pass=None,
            redundancy_pass=None,
        )
        branch_state = self.m.BranchState(eligible_h_count=0, eligible_q_count=0)
        visited = []
        with mock.patch.object(self.m, "iter_call_descriptors", return_value=iter(descriptors)):
            replay = self.m.replay_active_path(
                mini, raw, active_state, branch_state,
                lambda chunk_id, call: visited.append((chunk_id, call["call_id"])),
            )
        self.assertEqual(replay["executed_descriptor_count"], 12)
        self.assertEqual(len(visited), 12)

    def _successful_states(self):
        active = self.m.ActivePlanState(
            preflight_pass=True, native_cache_complete=True, gradient_cache_complete=True,
            retained_h=(-1, 0, 1), retained_q=(-1, 0, 1, 2),
            selected_h=0, selected_q=1, top_two_q=(-1, 2),
            fit_health_stage="selected_family_rank_passed",
            semantic_and_falsifier_gates_pass=True, single_necessity_pass=False,
            redundancy_pass=True,
        )
        branch = self.m.BranchState(
            eligible_h_count=3, eligible_q_count=4,
            fit_health_stage="selected_family_rank_passed", higher_rank_rescue=False,
            semantic_gates_pass=True, single_necessity_pass=False,
            redundancy_available=True, redundancy_pass=True, h_before_q=True,
            reader_pass=True,
        )
        return active, branch

    def test_active_plan_is_derived_with_independent_top_two_q(self) -> None:
        active, branch = self._successful_states()
        result = self.m.active_path_receipts(self.manifest, self.index_raw, active, branch)
        active_ids = set(result["active_chunk_ids"])
        self.assertIn("redundancy:Q:-1:2:seed14001", active_ids)
        self.assertIn("necessity:Q:1:seed14001", active_ids)
        self.assertIn("reader:H:0:Q:1:seed14001", active_ids)
        self.assertNotIn("fit:H:1:A1_only:rank1:seed14001", active_ids)
        self.assertEqual(result["active_count"] + result["inactive_count"], 3821)
        skipped = next(
            item for item in result["receipts"]
            if item["chunk_id"] == "fit:H:1:A1_only:rank1:seed14001"
        )
        self.assertEqual(skipped["status"], "inactive_skip_zero_calls")

    def test_terminal_and_active_path_mismatch_rejects(self) -> None:
        active, branch = self._successful_states()
        inactive_reader = self.m.ActivePlanState(**{
            **self.m.asdict(active), "selected_h": 2,
        })
        with self.assertRaises(self.m.CompileError):
            self.m.active_path_receipts(self.manifest, self.index_raw, inactive_reader, branch)
        early_bad = self.m.BranchState(
            eligible_h_count=3, eligible_q_count=4, fit_health_stage="joint_rank1_failed",
        )
        with self.assertRaises(self.m.CompileError):
            self.m.active_path_receipts(self.manifest, self.index_raw, active, early_bad)

    def test_active_guards_stop_at_exact_health_or_empty_stage(self) -> None:
        no_q = self.m.ActivePlanState(
            preflight_pass=True, native_cache_complete=True, gradient_cache_complete=True,
            retained_h=(-1, 0, 1), retained_q=(), selected_h=None, selected_q=None,
            top_two_q=None, fit_health_stage=None,
            semantic_and_falsifier_gates_pass=None, single_necessity_pass=None,
            redundancy_pass=None,
        )
        ids = self.m.derive_active_chunk_ids(self.manifest, no_q)
        self.assertFalse(any(item.startswith(("fit:", "spectral:")) for item in ids))
        joint_bad = self.m.ActivePlanState(
            preflight_pass=True, native_cache_complete=True, gradient_cache_complete=True,
            retained_h=(-1, 0, 1), retained_q=(-1, 0), selected_h=None, selected_q=None,
            top_two_q=None, fit_health_stage="joint_rank1_failed",
            semantic_and_falsifier_gates_pass=None, single_necessity_pass=None,
            redundancy_pass=None,
        )
        ids = self.m.derive_active_chunk_ids(self.manifest, joint_bad)
        self.assertTrue(any(item.startswith("fit:H:-1:joint:rank1") for item in ids))
        self.assertFalse(any(item.startswith("spectral:") for item in ids))
        selected_bad = self.m.ActivePlanState(
            preflight_pass=True, native_cache_complete=True, gradient_cache_complete=True,
            retained_h=(-1, 0, 1), retained_q=(-1, 0), selected_h=0, selected_q=0,
            top_two_q=(-1, 0), fit_health_stage="selected_family_rank_failed",
            semantic_and_falsifier_gates_pass=None, single_necessity_pass=None,
            redundancy_pass=None,
        )
        ids = self.m.derive_active_chunk_ids(self.manifest, selected_bad)
        self.assertIn("fit:H:0:A1_only:rank1:seed14001", ids)
        self.assertNotIn("eval:VALIDATION:H:0:A1_only:rank1:seed14001", ids)

    def test_source_has_no_model_or_GPU_import(self) -> None:
        tree = ast.parse(SOURCE.read_text())
        imports = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports |= {
            (node.module or "").split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertTrue(imports.isdisjoint({"torch", "jax", "fastload", "tt_model"}))


if __name__ == "__main__":
    unittest.main()
