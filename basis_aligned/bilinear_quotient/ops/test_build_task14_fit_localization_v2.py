from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest


OPS = Path(__file__).resolve().parent
sys.path.insert(0, str(OPS))
import build_task14_fit_localization_v2 as B  # noqa: E402


EXPECTED_PARTITION_FILE_SHA256 = "1f43b767fb39082d7872629d1a8b700e90e055c9529d9d319fe483f77d91fad3"
EXPECTED_DONOR_FILE_SHA256 = "ff702f2936e2445a247c6fca3a55d177e80974b2a5e14fb6de0a5fe2761db50a"
EXPECTED_PARTITION_RECORDS_SHA256 = "285092178ef25e5aee923a2b02ec791c6b2df83e7c47f185626cd5cfa507d08c"
EXPECTED_DONOR_RECORDS_SHA256 = "6e1fc1fef2715e0c87f0e494646057957bad284f7b69b1e52dcc4ec0f3e6f905"


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def built():
    authority = B.load_frozen_authority()
    partition = B.build_partition(authority)
    donors = B.build_donors(authority, partition)
    return authority, partition, donors


def test_materialized_artifacts_are_exact_and_hash_frozen(built):
    authority, partition, donors = built
    B.check_artifacts()
    assert B.safe_read_regular(B.PARTITION_PATH) == B.canonical_bytes(partition, newline=True)
    assert B.safe_read_regular(B.DONORS_PATH) == B.canonical_bytes(donors, newline=True)
    assert file_sha(B.PARTITION_PATH) == EXPECTED_PARTITION_FILE_SHA256
    assert file_sha(B.DONORS_PATH) == EXPECTED_DONOR_FILE_SHA256
    assert partition["records_sha256"] == EXPECTED_PARTITION_RECORDS_SHA256
    assert donors["records_sha256"] == EXPECTED_DONOR_RECORDS_SHA256
    assert authority["split"] == "FIT"


def test_v1_preregistration_is_unchanged():
    path = OPS.parents[1] / "polynomial_causal" / "TASK14_SUBJECT_VERB_AGREEMENT_FIT_LOCALIZATION_PREREGISTRATION_2026-09-04.md"
    assert file_sha(path) == B.FROZEN_V1_PREREG_SHA256


def test_partition_is_pair_coherent_balanced_and_prompt_disjoint(built):
    authority, partition, _ = built
    group_to_partition = {r["group_number"]: r["partition"] for r in partition["records"]}
    assert tuple(sorted(g for g, p in group_to_partition.items() if p == "DISCOVERY")) == B.DISCOVERY_GROUPS
    assert tuple(sorted(g for g, p in group_to_partition.items() if p == "VALIDATION")) == B.VALIDATION_GROUPS
    for g in range(16):
        assert group_to_partition[g] == group_to_partition[g + 16]
    for part in B.PARTITION_ORDER:
        groups = {g for g, p in group_to_partition.items() if p == part}
        a1 = [r for r in authority["rows"] if r["transform_id"] == "A1" and r["group_number"] in groups]
        assert Counter((r["base_head_plural"], r["base_attractor_plural"]) for r in a1) == Counter(
            {(False, False): 4, (False, True): 4, (True, False): 4, (True, True): 4}
        )
        prompts = [r[f"{side}_text"] for r in authority["rows"] if r["group_number"] in groups for side in ("base", "donor")]
        assert len(prompts) == len(set(prompts)) == 128
    discovery_prompts = {
        r[f"{side}_text"]
        for r in authority["rows"]
        if group_to_partition[r["group_number"]] == "DISCOVERY"
        for side in ("base", "donor")
    }
    validation_prompts = {
        r[f"{side}_text"]
        for r in authority["rows"]
        if group_to_partition[r["group_number"]] == "VALIDATION"
        for side in ("base", "donor")
    }
    assert discovery_prompts.isdisjoint(validation_prompts)


def test_pair_coherence_is_not_claimed_to_prevent_exact_mirror_prompts(built):
    authority, _, _ = built
    rows = {(r["group_number"], r["transform_id"]): r for r in authority["rows"]}
    for g in range(16):
        left = {rows[(g, f)][f"{s}_text"] for f in ("A1", "A2", "P", "C") for s in ("base", "donor")}
        right = {rows[(g + 16, f)][f"{s}_text"] for f in ("A1", "A2", "P", "C") for s in ("base", "donor")}
        assert left.isdisjoint(right)


def test_partition_discloses_shared_templates_and_cross_role_nouns(built):
    authority, partition, _ = built
    group_to_partition = {r["group_number"]: r["partition"] for r in partition["records"]}
    templates = {}
    head_pairs = {}
    all_role_pairs = {}
    for part in B.PARTITION_ORDER:
        rows = [r for r in authority["rows"] if group_to_partition[r["group_number"]] == part]
        templates[part] = {r["base_template_id"] for r in rows}
        head_pairs[part] = {tuple(r["head_pair"]) for r in rows}
        all_role_pairs[part] = {
            tuple(r[field])
            for r in rows
            for field in ("head_pair", "attractor_pair", "surface_attractor_pair", "second_head_pair")
        }
    assert templates["DISCOVERY"] == templates["VALIDATION"]
    assert head_pairs["DISCOVERY"].isdisjoint(head_pairs["VALIDATION"])
    assert head_pairs["DISCOVERY"] & all_role_pairs["VALIDATION"]


def test_original_704_relations_are_preserved_exactly(built):
    _, _, donors = built
    original = [r for r in donors["records"] if r["source_contract"] == "v1_original_704"]
    assert len(original) == 704
    assert donors["original_704_core_sha256"] == B.ORIGINAL_704_CORE_SHA256
    assert Counter((r["partition"], r["arm"]) for r in original) == Counter(
        {
            (part, arm): count
            for part in B.PARTITION_ORDER
            for arm, count in {
                "answer_change": 192,
                "cross_syntax": 64,
                "P_positive_transfer": 64,
                "P_zero_coordinate_control": 16,
                "C_zero_coordinate_control": 16,
            }.items()
        }
    )


def test_complete_subject_arm_census_and_q_only(built):
    _, _, donors = built
    new = [r for r in donors["records"] if r["source_contract"] == "v2_complete_subject_Q"]
    assert len(new) == 384
    expected = {
        "C_to_ordinary_singular": 64,
        "ordinary_singular_to_C": 32,
        "C_to_ordinary_plural_control": 64,
        "ordinary_plural_to_C_control": 32,
    }
    for part in B.PARTITION_ORDER:
        assert Counter(r["arm"] for r in new if r["partition"] == part) == Counter(expected)
    assert all(r["q_only"] for r in new)


def test_complete_subject_relations_are_bidirectional_and_semantic(built):
    _, _, donors = built
    endpoints = {e["endpoint_id"]: e for e in donors["endpoints"]}
    new = [r for r in donors["records"] if r["source_contract"] == "v2_complete_subject_Q"]
    for record in new:
        target, donor = endpoints[record["target_endpoint_id"]], endpoints[record["donor_endpoint_id"]]
        assert target["attractor_plural"] == donor["attractor_plural"]
        assert target["head_pair"] != donor["head_pair"]
        if record["arm"] == "C_to_ordinary_singular":
            assert target["family"] == "C" and target["subject_state"] == 1 and donor["subject_state"] == -1
        elif record["arm"] == "ordinary_singular_to_C":
            assert target["family"] in B.ORDINARY_FAMILIES and target["subject_state"] == -1 and donor["family"] == "C" and donor["subject_state"] == 1
        elif record["arm"] == "C_to_ordinary_plural_control":
            assert target["family"] == "C" and target["subject_state"] == donor["subject_state"] == 1
        elif record["arm"] == "ordinary_plural_to_C_control":
            assert target["subject_state"] == donor["subject_state"] == 1 and donor["family"] == "C"
        else:
            raise AssertionError(record["arm"])


def test_every_c_endpoint_has_both_answer_change_and_same_state_donors(built):
    _, _, donors = built
    endpoints = {e["endpoint_id"]: e for e in donors["endpoints"]}
    new = [r for r in donors["records"] if r["source_contract"] == "v2_complete_subject_Q"]
    for part in B.PARTITION_ORDER:
        c_ids = {
            eid
            for eid, endpoint in endpoints.items()
            if endpoint["family"] == "C"
            and any(r["target_endpoint_id"] == eid and r["partition"] == part for r in new)
        }
        assert len(c_ids) == 32
        for eid in c_ids:
            arms = Counter(r["arm"] for r in new if r["partition"] == part and r["target_endpoint_id"] == eid)
            assert arms == Counter({"C_to_ordinary_singular": 2, "C_to_ordinary_plural_control": 2})


def test_records_have_exact_ids_ordinals_and_order(built):
    _, _, donors = built
    assert [r["ordinal"] for r in donors["records"]] == list(range(1088))
    assert len({r["record_id"] for r in donors["records"]}) == 1088
    for record in donors["records"]:
        identity = {
            key: record[key]
            for key in (
                "arm",
                "donor_endpoint_id",
                "expected_relation",
                "family",
                "matching",
                "partition",
                "q_only",
                "source_contract",
                "target_endpoint_id",
            )
        }
        assert record["record_id"] == B.canonical_sha256(identity)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x.update(schema="wrong"),
        lambda x: x["records"].pop(),
        lambda x: x["records"][0].update(partition="VALIDATION"),
        lambda x: x["records"][0]["mirror_group_numbers"].__setitem__(1, 17),
        lambda x: x.update(source_review_sha256="0" * 64),
    ],
)
def test_partition_coherent_mutations_reject(built, mutation):
    authority, partition, _ = built
    bad = copy.deepcopy(partition)
    mutation(bad)
    bad["records_sha256"] = B.canonical_sha256(bad["records"])
    with pytest.raises(ValueError):
        B.validate_partition(bad, authority)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x.update(schema="wrong"),
        lambda x: x["records"].pop(),
        lambda x: x["records"][0].update(q_only=not x["records"][0]["q_only"]),
        lambda x: x["records"][352].update(arm="answer_change"),
        lambda x: x["records"][352].update(partition="VALIDATION"),
        lambda x: x["records"][352].update(record_id="0" * 64),
        lambda x: x["endpoints"][0].update(subject_state=-x["endpoints"][0]["subject_state"]),
        lambda x: x.update(source_review_sha256="0" * 64),
    ],
)
def test_donor_coherent_mutations_reject(built, mutation):
    authority, partition, donors = built
    bad = copy.deepcopy(donors)
    mutation(bad)
    bad["records_sha256"] = B.canonical_sha256(bad["records"])
    bad["endpoints_sha256"] = B.canonical_sha256(bad["endpoints"])
    with pytest.raises(ValueError):
        B.validate_donors(bad, authority, partition)


def test_authority_mutation_and_symlink_reject(tmp_path):
    value = json.loads(B.AUTHORITY_PATH.read_text())
    value["rows"][0]["base_text"] += " changed"
    bad = tmp_path / "authority.json"
    bad.write_text(json.dumps(value, sort_keys=True))
    with pytest.raises(ValueError, match="hash mismatch"):
        B.load_frozen_authority(bad)
    link = tmp_path / "authority-link.json"
    link.symlink_to(B.AUTHORITY_PATH)
    with pytest.raises(OSError):
        B.load_frozen_authority(link)


def test_deterministic_bytes_across_python_hash_seeds(tmp_path):
    outputs = []
    script = str(B.Path(B.__file__).resolve())
    for seed in ("0", "1", "999"):
        out = tmp_path / seed
        env = dict(os.environ, PYTHONHASHSEED=seed)
        subprocess.run([sys.executable, script, "--write-dir", str(out)], check=True, env=env)
        outputs.append((out / B.PARTITION_PATH.name).read_bytes() + (out / B.DONORS_PATH.name).read_bytes())
    assert outputs[0] == outputs[1] == outputs[2]


def test_create_only_writer_refuses_existing_destination(tmp_path):
    B.write_artifacts(tmp_path)
    with pytest.raises(FileExistsError):
        B.write_artifacts(tmp_path)


def test_builder_imports_only_cpu_standard_library_and_one_fit_authority():
    source = B.Path(B.__file__).read_text()
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert imports <= {"__future__", "argparse", "hashlib", "json", "os", "stat", "pathlib", "typing"}
    assert B.AUTHORITY_PATH.name == "circuit_battery_task14_agreement_fit_authority.json"
    assert not any(name in source for name in ("torch.", "cuda.", "fastload", "queue.txt", "SELECT_authority", "TEST_authority", "OOD_authority"))
