#!/usr/bin/env python3
# BQLANE: cpu
"""Adversarial CPU tests for the task14 FIT localization-v2 compiler."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "circuit_battery_task14_fit_localization_v2_compiler.py"
MANIFEST = HERE / "circuit_battery_task14_fit_localization_v2_call_manifest.json"
INDEX = HERE / "circuit_battery_task14_fit_localization_v2_call_index.bin"
DRYRUN = HERE / "circuit_battery_task14_fit_localization_v2_compiler_dryrun.json"
EXPECTED_SOURCE_SHA256 = "ffa56273f6fee686e193fa53cb8021f782536e79fbb629d30020a78cce065e6b"
EXPECTED_MANIFEST_SHA256 = "f264ef64c03a2053f2c5344588d0adc8eb03ef3a8cb257d7d02c04f3a478568d"
EXPECTED_INDEX_SHA256 = "ae399e393d03af9b6232b7fc5339dd892b418ec7c88943735f8b72fc064c8ad9"
EXPECTED_DRYRUN_SHA256 = "c9c113dcd1b99fcd51a11046b984cde50d29d31be200aa778242eab079ab13a7"


def load_module():
    spec = importlib.util.spec_from_file_location("task14_v2_compiler_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
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
        self.assertEqual(semantics["-1"], "normalized embedding input before block 0")
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
            ("dag", 0, "condition"),
            ("runtime_and_publication", "deadline", "hard_gpu_seconds"),
            ("runtime_and_publication", "dead_intervention_tripwires", "live_exact_single_position_delta_check_every_call"),
            ("science", "spectral", "validation_selector"),
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
            target[key] = (not target[key]) if isinstance(target[key], bool) else "changed"
            value.pop("contract_sha256")
            value["contract_sha256"] = self.m.canonical_sha256(value)
            with self.subTest(path=path), self.assertRaises(self.m.CompileError):
                self.m.validate_manifest(value)

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
