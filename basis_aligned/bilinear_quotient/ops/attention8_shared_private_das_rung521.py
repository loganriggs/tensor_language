#!/usr/bin/env python3
"""Rung 521, Stage A: power gate for shared/private attention8 DAS.

The registered scientific question is whether one rank-4 projector on the
1152-dimensional attention8 output carries a causal computation reused by
three fitted circuits and a held-out fourth circuit, while three orthogonal
rank-4 projectors carry circuit-specific residual computations.  The future
predictions are: (B) a shared projector transfers to a left-out fitted target,
(C) it transfers to r.2.0.1 without generic attention8 damage, (D) private
projectors selectively improve their owners, and (E) the four projectors
compose under swaps, mean removal, and an attention6 background change.  The
registered null after a valid Stage A is failure of those fixed-capacity,
held-out causal tests; it is not permission to increase rank until a result
appears.

This entrypoint intentionally implements *only Stage A*.  It verifies the
frozen inputs and masks, deterministically constructs matched controls and two
ensembles of four disjoint donor derangements, captures the native
attention8 write once, and measures whole-attention8 swaps on FIT.  It then
applies the preregistered materiality, selectivity, bootstrap, donor-transfer,
32-circuit fingerprint, and overlap-preserving permutation gates.  It writes a
Stage-A result and exits before creating an optimizer or any learned tensor.

Price of this implementation: for R FIT rows and batch size B, capture/native,
independent native replay, and self-donor checks cost 3*ceil(R/B) forward calls.
The two ensembles x four maps x two swap directions cost another
16*ceil(R/B), for 19*ceil(568/4)=2,698 inference-only calls.  There are zero
backward calls and zero learned values.  Control construction, bootstraps, and
200 label permutations are CPU-only.  A future optimization stage is outside
this file's executed path.
"""

# BQGATE: EXPERIMENT
# pred_a: exact, live, selective, donor-stable whole-attention8 instrument
# pred_b: future shared rank-4 projector transfers to a left-out fitted circuit
# pred_c: future shared projector reuses on frozen r.2.0.1 without generic damage
# pred_d: future orthogonal private rank-4 projectors split owner residuals
# pred_e: future shared/private projectors compose under independent physical actions

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


REGISTERED_PREDICTIONS = {
    "pred_a": "exact, live, selective, donor-stable whole-attention8 instrument",
    "pred_b": "future shared rank-4 projector transfers to a left-out fitted circuit",
    "pred_c": "future shared projector reuses on frozen r.2.0.1 without generic damage",
    "pred_d": "future orthogonal private rank-4 projectors split owner residuals",
    "pred_e": "future shared/private projectors compose under independent physical actions",
}


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OPS = ROOT / "ops"
POLY = ROOT.parent / "polynomial_causal"
PREREG = POLY / "ATTENTION8_SHARED_PRIVATE_DAS_RUNG521_PREREGISTRATION.md"
STATE = ROOT / "census_state_diverse.pt"
CURATED = ROOT / "curated_rows.pt"
BATTERY = ROOT / "circuits/BATTERY.json"
HISTORICAL_DAS = ROOT / "circuit_das.py"
A8_GROUPING = ROOT / "circuits/A8_GROUPING.json"
A8_GROUPING_LEARNED = ROOT / "circuits/A8_GROUPING_LEARNED.json"
DAS_RESULT = ROOT / "circuits/DAS.json"
DEFAULT_OUT = ROOT / "attention8_shared_private_das_rung521_stage_a_results.json"
DEFAULT_PREFLIGHT_OUT = ROOT / "attention8_shared_private_das_rung521_preflight.json"

DEPENDENCY_HASHES = {
    PREREG: "e40ca9654485d8fcc04dd09e0b86628fa633e98d97c0b444c6661f56f73461de",
    STATE: "c785f3d938091253535aa4f613ab2b4107bf297c8d615da4f7eab4f8282f5e0b",
    CURATED: "faaf89f38ddf1471234a1d30d978213367a566a9927bb3c73b274ab32afaa9dd",
    BATTERY: "86d7ac72eeb95f9ec80a3e92ef65e28c0df66a36b9291d2d1d2d01f7bb6c5030",
    HISTORICAL_DAS: "b2e6670a223a01c7115487eb886e91fd0aa1ca9d746d60f5fa5a57c43ebeffe7",
    A8_GROUPING: "08fd57c286e4908323afe2568c53d0193e8e002c4603976340582b32ac98a755",
    A8_GROUPING_LEARNED: "3def72ab041683c3923fa487799af9a34b44d6cd3d8abf3bf96e1e1f709bf45f",
    DAS_RESULT: "91bd4cb80f8077cd62af4b5a402c06845af447796c63c873a7ab677a3a4310de",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_dependency_bytes() -> dict[str, str]:
    observed = {}
    for path, expected in DEPENDENCY_HASHES.items():
        if not path.is_file():
            raise RuntimeError(f"frozen dependency is absent: {path}")
        value = _sha256_file(path)
        observed[str(path.relative_to(ROOT.parent.parent))] = value
        if value != expected:
            raise RuntimeError(f"frozen dependency hash changed: {path}")
    return observed


# The managed enqueue helper executes this branch with BQLIB_NO_MODEL=1.  It
# must run before torch, census_lib, TT.GPT, or the local model facade is
# imported, because those imports can otherwise construct or load the model.
if os.environ.get("BQLIB_DRYRUN") == "1":
    hashes = _validate_dependency_bytes()
    if len(hashes) != 8:
        raise SystemExit("DRYRUN FAIL: dependency census changed")
    print(
        "DRYRUN OK: rung521 Stage A only; 8 frozen dependencies; "
        "2 ensembles x 4 donors x 2 directions; 2,698 forwards; 0 backwards; "
        "main requires a smoke-frozen --edit-rms-floor",
        flush=True,
    )
    raise SystemExit(0)


import torch  # noqa: E402  (deliberately after BQLIB_DRYRUN)
import torch.nn.functional as F  # noqa: E402

for _path in (ROOT, OPS, POLY, Path("/workspace/tensor_language")):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import bilin18_observed_model_facade as facade  # noqa: E402
import das_shared_private_lib as daslib  # noqa: E402


D = 1152
TOKENS = 256
BATCH = 4
A8_SITE = 8
CONTROL_SEED = 52100
DONOR_SEED = 52100
BOOTSTRAPS = 2000
PERMUTATIONS = 200
CONTROL_STAGES = (
    ("token_id", "position_bin", "ce_decile"),
    ("token_id", "ce_decile"),
    ("token_class", "position_bin", "ce_decile"),
    ("token_class", "ce_decile"),
    # Pre-outcome feasibility correction.  The four registered stages leave
    # rare token classes with no legal same-half control in eight quartet
    # cells.  These deterministic terminal stages progressively drop CE,
    # class, and position constraints while keeping the parent slice, data
    # split, own-member exclusion, and quartet exclusion fixed.  Exact counts
    # and hashes are frozen in the preflight addendum before CUDA is allowed.
    ("token_class", "position_bin"),
    ("position_bin", "ce_decile"),
    ("token_class",),
    ("position_bin",),
    ("ce_decile",),
    ("all",),
)
FITTED_TAGS = ("r.2.0.2", "r.2.1.1", "r.2.2.1")
REUSE_TAG = "r.2.0.1"
QUARTET_TAGS = (REUSE_TAG,) + FITTED_TAGS
SPLITS = {
    "fit": (0, 6),
    "validation": (6, 8),
    "test": (8, 10),
}
CONTROL_CELLS = {
    "fit_half0": (0, 3),
    "fit_half1": (3, 6),
    "validation": (6, 8),
    "test": (8, 10),
}

# This is the frozen discovery half from the R506--R520 circuit fingerprints.
# It is written literally so Stage A does not import an old experiment script
# whose module body would load a model or run work.
FINGERPRINT_TAGS = (
    "r.0.0", "r.0.0.0", "r.0.0.1", "r.0.3.0", "r.18.2.0",
    "r.2.0", "r.2.0.0", "r.2.0.1", "r.2.0.2", "r.2.1",
    "r.2.1.0", "r.2.1.1", "r.2.2", "r.2.2.0", "r.2.2.1",
    "r.2.2.2", "r.2.3", "r.4.1.0", "r.4.1.1", "r.6.0.0",
    "r.6.0.1", "r.6.0.2", "r.6.0.3", "r.6.1.0", "r.6.1.1",
    "r.6.2.0", "r.6.2.1", "r.6.2.2", "r.6.2.3", "r.6.3.0",
    "r.6.3.1", "r.8.1.0",
)

FOLD_HASH = "305f944328a83406a873a41cb3982288dff5c6bd0c5a3282540c8cd86815aa60"
SPLIT_HASHES = {
    "fit": "cf68c6efb50399b07b4de99c6777b00176dd4cabe730451b3bb69dd199dc3128",
    "validation": "0a0df35d1db9df41cc717c9da28737f0ac7f7dbbc4678c6056bce4f1afc35c62",
    "test": "f141d3362442f4a74849446993c6cf4c271172f1e8b34447b8fdc341bc377dc4",
}
SPLIT_ROW_COUNTS = {"fit": 568, "validation": 216, "test": 216}
MASK_HASHES = {
    "r.2.0.1": (
        "fad6c5613776c0d069e9326d0991d46bb2c2337c430dfb5addc748b6a9e62299",
        "ac6fea9504bf6cc464edc7b8686ca8ed8ab921f65b804423b41d7597b6441c19",
    ),
    "r.2.0.2": (
        "174cb79448e2192771388b7a048e2bba4f71eb04120d7e905d398086ac2a3551",
        "846f4e2fa2aa5b40409ddda756fbf7ac547d1e778bb0a4dbe7339ff0fa182fe2",
    ),
    "r.2.1.1": (
        "ef0ebaff2022b1c9a1d0630de3229d3ffb0b8464e9a4a8eeaf908919dc7012e6",
        "ad63ffa0396271381f724a451bbb053c3704ec69a7cfb9c324d6f190f66dab4d",
    ),
    "r.2.2.1": (
        "01004d10ffe47b4fe8fbccac2cef87b54aca3cb3d2bc45e482c11a8f80eaf0f0",
        "d7105378481ce95d2e8bb1ccc4d36c5adc36d100a77545447ad7c6ea14473eb3",
    ),
}
EXPECTED_SPLIT_MASK_COUNTS = {
    "fit": ((494, 290), (480, 254), (461, 242), (502, 284)),
    "validation": ((195, 95), (187, 95), (208, 112), (189, 95)),
    "test": ((175, 103), (197, 111), (195, 114), (173, 91)),
}
EXPECTED_FIT_HALF_EXCLUSIVE = {
    "r.2.0.2": (123, 131),
    "r.2.1.1": (119, 123),
    "r.2.2.1": (145, 139),
}


def _tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _folds(docids: torch.Tensor) -> torch.Tensor:
    values = [
        int.from_bytes(
            hashlib.sha256(
                ("a8-shared-private-v1:" + str(int(document))).encode("utf-8")
            ).digest()[:8],
            "little",
        ) % 10
        for document in docids.tolist()
    ]
    return torch.tensor(values, dtype=torch.int64)


def _flat_row_mask(row_mask: torch.Tensor) -> torch.Tensor:
    return row_mask[:, None].expand(-1, TOKENS).reshape(-1)


def _load_cpu_inputs() -> dict:
    _validate_dependency_bytes()
    state = torch.load(STATE, map_location="cpu", weights_only=False)
    curated = torch.load(CURATED, map_location="cpu", weights_only=False)
    if set(state) != {"rows", "basev", "leaves"}:
        raise RuntimeError("census state schema changed")
    if not isinstance(curated, dict) or not {"rows", "docid"} <= set(curated):
        raise RuntimeError("curated row schema changed")
    rows = state["rows"].long().contiguous()
    if tuple(rows.shape) != (1000, 513) or not torch.equal(rows, curated["rows"]):
        raise RuntimeError("census and curated rows differ")
    base_ce = state["basev"].float().reshape(1000, TOKENS).contiguous()
    docids = curated["docid"].long().contiguous()
    if tuple(docids.shape) != (1000,):
        raise RuntimeError("document-id vector changed")
    leaves = {leaf["tag"]: leaf for leaf in state["leaves"]}
    required = set(FINGERPRINT_TAGS) | set(QUARTET_TAGS)
    if not required <= set(leaves):
        raise RuntimeError(f"census leaves absent: {sorted(required - set(leaves))}")

    folds = _folds(docids)
    if _tensor_sha256(folds) != FOLD_HASH:
        raise RuntimeError("frozen document-fold vector changed")
    row_masks = {}
    for name, (low, high) in SPLITS.items():
        mask = ((folds >= low) & (folds < high)).contiguous()
        if int(mask.sum()) != SPLIT_ROW_COUNTS[name] or _tensor_sha256(mask) != SPLIT_HASHES[name]:
            raise RuntimeError(f"frozen {name} row mask changed")
        row_masks[name] = mask

    full_masks = {}
    for tag in QUARTET_TAGS:
        mask = torch.zeros(1000 * TOKENS, dtype=torch.bool)
        member = leaves[tag]["member"].long()
        if member.unique().numel() != 864:
            raise RuntimeError(f"{tag} member support changed")
        mask[member] = True
        full_masks[tag] = mask
    quartet_union = torch.stack(tuple(full_masks.values())).any(0)
    exclusive_masks = {}
    identities = {}
    for tag in QUARTET_TAGS:
        others = torch.stack([full_masks[other] for other in QUARTET_TAGS if other != tag]).any(0)
        exclusive = full_masks[tag] & ~others
        expected_full, expected_exclusive = MASK_HASHES[tag]
        if _tensor_sha256(full_masks[tag]) != expected_full:
            raise RuntimeError(f"{tag} full-member mask changed")
        if _tensor_sha256(exclusive) != expected_exclusive:
            raise RuntimeError(f"{tag} exclusive-member mask changed")
        exclusive_masks[tag] = exclusive
        identities[tag] = {
            "full_sha256": expected_full,
            "exclusive_sha256": expected_exclusive,
            "full_count": int(full_masks[tag].sum()),
            "exclusive_count": int(exclusive.sum()),
        }
    pair_counts = tuple(
        int((full_masks[a] & full_masks[b]).sum())
        for a, b in ((QUARTET_TAGS[0], QUARTET_TAGS[1]),
                     (QUARTET_TAGS[0], QUARTET_TAGS[2]),
                     (QUARTET_TAGS[0], QUARTET_TAGS[3]))
    )
    # The preregistered 185/193/208 are the three fitted-target pairwise
    # intersections, not intersections against the reserved member.
    fitted_pair_counts = tuple(
        int((full_masks[a] & full_masks[b]).sum())
        for a, b in ((FITTED_TAGS[0], FITTED_TAGS[1]),
                     (FITTED_TAGS[0], FITTED_TAGS[2]),
                     (FITTED_TAGS[1], FITTED_TAGS[2]))
    )
    triple_count = int(torch.stack([full_masks[tag] for tag in FITTED_TAGS]).all(0).sum())
    if fitted_pair_counts != (185, 193, 208) or triple_count != 77:
        raise RuntimeError("frozen fitted-target overlap lattice changed")

    split_counts = {}
    for name, row_mask in row_masks.items():
        flat = _flat_row_mask(row_mask)
        got = tuple(
            (int((full_masks[tag] & flat).sum()), int((exclusive_masks[tag] & flat).sum()))
            for tag in QUARTET_TAGS
        )
        if got != EXPECTED_SPLIT_MASK_COUNTS[name]:
            raise RuntimeError(f"frozen {name} quartet support changed: {got}")
        split_counts[name] = {tag: {"full": pair[0], "exclusive": pair[1]}
                              for tag, pair in zip(QUARTET_TAGS, got)}
    for tag, expected in EXPECTED_FIT_HALF_EXCLUSIVE.items():
        got = tuple(
            int((exclusive_masks[tag] & _flat_row_mask((folds >= low) & (folds < high))).sum())
            for low, high in ((0, 3), (3, 6))
        )
        if got != expected:
            raise RuntimeError(f"frozen FIT-half support changed for {tag}: {got}")

    battery = json.loads(BATTERY.read_text())
    if any(battery["by_tag"][tag].get("best_mean") != "a8" for tag in QUARTET_TAGS):
        raise RuntimeError("a quartet circuit is no longer localized to attention8")
    return {
        "rows": rows,
        "base_ce": base_ce,
        "docids": docids,
        "folds": folds,
        "row_masks": row_masks,
        "leaves": leaves,
        "full_masks": full_masks,
        "exclusive_masks": exclusive_masks,
        "quartet_union": quartet_union,
        "mask_identities": identities,
        "split_counts": split_counts,
        "fitted_pair_counts": fitted_pair_counts,
        "unused_reserved_pair_counts": pair_counts,
        "fitted_triple_count": triple_count,
    }


def _token_classes(target_tokens: torch.Tensor) -> torch.Tensor:
    import tiktoken

    encoder = tiktoken.get_encoding("gpt2")
    unique = target_tokens.unique().tolist()
    classes = {}
    for token in unique:
        text = encoder.decode([int(token)])
        stripped = text.strip()
        if "\n" in text or "\r" in text:
            value = 0  # newline/control-line token
        elif stripped.isdigit() and stripped:
            value = 1  # number token
        elif stripped and not any(character.isalnum() for character in stripped):
            value = 2  # punctuation/symbol token
        elif text.startswith(" ") and stripped[:1].isupper():
            value = 3  # word-initial capitalized token
        elif text.startswith(" ") and stripped.isalpha():
            value = 4  # word-initial alphabetic token
        elif not text.startswith(" ") and stripped.isalpha():
            value = 5  # alphabetic continuation token
        else:
            value = 6  # mixed/other token
        classes[int(token)] = value
    flat = target_tokens.reshape(-1)
    return torch.tensor([classes[int(token)] for token in flat.tolist()], dtype=torch.int64)


def _rank_deciles(base_ce: torch.Tensor, row_masks: dict[str, torch.Tensor]) -> torch.Tensor:
    """Assign deterministic within-split empirical CE deciles.

    Sorting uses (CE, global position); therefore equal CE values do not make
    the matching design platform-dependent through an interpolation rule.
    """
    flat = base_ce.reshape(-1)
    result = torch.full((flat.numel(),), -1, dtype=torch.int64)
    for row_mask in row_masks.values():
        indices = _flat_row_mask(row_mask).nonzero().flatten()
        ordered = sorted(indices.tolist(), key=lambda index: (float(flat[index]), index))
        for rank, index in enumerate(ordered):
            result[index] = min(9, (10 * rank) // len(ordered))
    if bool((result < 0).any()):
        raise RuntimeError("CE deciles do not cover all positions")
    return result


def _descriptors(data: dict) -> dict[str, torch.Tensor]:
    target = data["rows"][:, 1:TOKENS + 1].reshape(-1).long()
    position_bin = torch.arange(TOKENS, dtype=torch.int64).repeat(1000) // 32
    token_class = _token_classes(target.view(1000, TOKENS))
    ce_decile = _rank_deciles(data["base_ce"], data["row_masks"])
    return {
        "token": target,
        "position_bin": position_bin,
        "token_class": token_class,
        "ce_decile": ce_decile,
        "all": torch.zeros_like(target),
    }


def _match_key(desc: dict[str, torch.Tensor], index: int, tier: int) -> tuple[int, ...]:
    if tier < 0 or tier >= len(CONTROL_STAGES):
        raise ValueError("matching tier outside the registered control hierarchy")
    names = tuple("token" if name == "token_id" else name for name in CONTROL_STAGES[tier])
    return tuple(int(desc[name][index]) for name in names)


def _construct_controls(
    positive: torch.Tensor,
    pool: torch.Tensor,
    desc: dict[str, torch.Tensor],
    *,
    tag: str,
    cell: str,
) -> dict:
    positives = sorted(positive.nonzero().flatten().tolist())
    candidates = pool.nonzero().flatten().tolist()
    strata = {
        "token_id": desc["token"],
        "position_bin": desc["position_bin"],
        "ce_decile": desc["ce_decile"],
        "token_class": desc["token_class"],
        "all": desc["all"],
    }
    try:
        matched = daslib.deterministic_stratified_match(
            positives,
            candidates,
            strata,
            stages=CONTROL_STAGES,
            seed=CONTROL_SEED,
            namespace=f"rung521:{cell}:{tag}",
        )
    except daslib.MatchingError as error:
        raise RuntimeError(f"control matching failed for {tag}/{cell}: {error}") from error
    members = matched.recipient_indices
    controls = matched.matched_indices
    tiers = matched.relaxation_levels
    if controls.unique().numel() != controls.numel() or bool((positive[controls]).any()):
        raise RuntimeError(f"matched controls are not negative and unique for {tag}/{cell}")
    for member, control, tier in zip(members.tolist(), controls.tolist(), tiers.tolist()):
        if _match_key(desc, member, tier) != _match_key(desc, control, tier):
            raise RuntimeError("matched control violates its recorded relaxation tier")
    return {
        "members": members,
        "controls": controls,
        "tiers": tiers,
        "member_sha256": _tensor_sha256(members),
        "control_sha256": _tensor_sha256(controls),
        "tier_sha256": _tensor_sha256(tiers),
        "tier_counts": {str(tier): count for tier, count in enumerate(matched.relaxation_counts)},
        "matching_sha256": matched.sha256,
        "count": len(positives),
    }


def _construct_donors_for_split(
    split: str,
    row_mask: torch.Tensor,
    base_ce: torch.Tensor,
    docids: torch.Tensor,
) -> dict:
    # Preflight clarification for the addendum: a whole-output interchange
    # needs one coherent donor *row*.  Matching each of 145,408 token positions
    # independently would splice 256 unrelated contexts into one attention
    # write and would also make an exact bipartite matcher needlessly huge.
    # Therefore all 256 positions use the same different-document donor row at
    # the same position.  The tested bipartite matcher prefers an exact
    # row-mean-native-CE decile and relaxes only that decile if a complete
    # permutation would otherwise be impossible.
    # This clarification is emitted in the preflight identity and must be
    # frozen in an addendum before a scientific Stage-A run.
    recipient_rows = row_mask.nonzero().flatten()
    row_mean_ce = base_ce.mean(1)
    ordered = sorted(recipient_rows.tolist(), key=lambda row: (float(row_mean_ce[row]), row))
    row_decile = torch.full((1000,), -1, dtype=torch.int64)
    for rank, row in enumerate(ordered):
        row_decile[row] = min(9, (10 * rank) // len(ordered))
    local_deciles = row_decile[recipient_rows]
    local_documents = docids[recipient_rows]
    d0 = daslib.deterministic_row_donor_maps(
        recipient_rows,
        local_documents,
        row_ce_deciles=local_deciles,
        count=4,
        seed=DONOR_SEED,
        namespace=f"rung521:{split}:D0",
    )
    d1 = daslib.deterministic_row_donor_maps(
        recipient_rows,
        local_documents,
        row_ce_deciles=local_deciles,
        count=4,
        seed=DONOR_SEED,
        namespace=f"rung521:{split}:D1",
        prior_maps=d0,
    )
    row_results = d0 + d1
    row_maps = []
    for row_result in row_results:
        row_map = torch.full((1000,), -1, dtype=torch.int64)
        row_map[recipient_rows] = row_result.matched_indices
        row_maps.append(row_map)

    positions = torch.arange(TOKENS, dtype=torch.int64)
    recipient = (recipient_rows[:, None] * TOKENS + positions[None, :]).reshape(-1)
    maps = []
    for row_map in row_maps:
        donor = (row_map[recipient_rows, None] * TOKENS + positions[None, :]).reshape(-1)
        donor_map = torch.full((1000 * TOKENS,), -1, dtype=torch.int64)
        donor_map[recipient] = donor
        maps.append(donor_map)
    recipient_docs = docids[recipient_rows]
    for map_index, (row_map, donor_map) in enumerate(zip(row_maps, maps)):
        donor_rows = row_map[recipient_rows]
        donor = donor_map[recipient]
        if bool((donor_rows < 0).any()) or donor_rows.unique().numel() != recipient_rows.numel():
            raise RuntimeError(f"{split} donor row map {map_index} is not a permutation")
        if not torch.equal(donor_rows.sort().values, recipient_rows):
            raise RuntimeError(f"{split} donor row map {map_index} leaves its data split")
        if bool((docids[donor_rows] == recipient_docs).any()):
            raise RuntimeError(f"{split} donor row map {map_index} retained a source document")
        if bool((donor % TOKENS != recipient % TOKENS).any()):
            raise RuntimeError(f"{split} donor map {map_index} changed token position")
    stacked_rows = torch.stack([row_map[recipient_rows] for row_map in row_maps])
    if any(stacked_rows[:, column].unique().numel() != 8 for column in range(stacked_rows.shape[1])):
        raise RuntimeError(f"{split} donor ensembles are not row-wise disjoint")

    inverses = []
    for donor_map in maps:
        inverse = torch.full_like(donor_map, -1)
        inverse[donor_map[recipient]] = recipient
        inverses.append(inverse)
    return {
        "recipient": recipient,
        "maps": maps,
        "inverse_maps": inverses,
        "identity": {
            "recipient_count": recipient.numel(),
            "recipient_sha256": _tensor_sha256(recipient),
            "recipient_row_count": recipient_rows.numel(),
            "recipient_row_sha256": _tensor_sha256(recipient_rows),
            "row_native_CE_decile_sha256": _tensor_sha256(row_decile[recipient_rows]),
            "row_native_CE_decile_counts": {
                str(decile): int((row_decile[recipient_rows] == decile).sum()) for decile in range(10)
            },
            "row_CE_decile_distance_by_map": [
                {
                    "mean_absolute_decile_distance": float(
                        (row_decile[recipient_rows] - row_decile[row_map[recipient_rows]])
                        .abs().double().mean()
                    ),
                    "exact_decile_count": int(
                        (row_decile[recipient_rows] == row_decile[row_map[recipient_rows]]).sum()
                    ),
                    "within_one_decile_count": int(
                        ((row_decile[recipient_rows] - row_decile[row_map[recipient_rows]]).abs() <= 1).sum()
                    ),
                    "relaxation_counts": {
                        "0_exact_decile": int(row_result.relaxation_counts[0]),
                        "1_split_only": int(row_result.relaxation_counts[1]),
                    },
                }
                for row_map, row_result in zip(row_maps, row_results)
            ],
            "row_matching_sha256": [result.sha256 for result in row_results],
            "row_map_sha256": [_tensor_sha256(row_map[recipient_rows]) for row_map in row_maps],
            "D0_sha256": [_tensor_sha256(maps[index][recipient]) for index in range(4)],
            "D1_sha256": [_tensor_sha256(maps[index][recipient]) for index in range(4, 8)],
            "reverse_D0_sha256": [_tensor_sha256(inverses[index][recipient]) for index in range(4)],
            "reverse_D1_sha256": [_tensor_sha256(inverses[index][recipient]) for index in range(4, 8)],
            "different_document": True,
            "bijective": True,
            "row_wise_eight_donors_disjoint": True,
            "same_token_position": True,
            "preflight_addendum_required": (
                "clarify coherent whole-row donors: same token position; deterministic different-document row permutations; exact row-mean-native-CE decile preferred, split-only relaxation"
            ),
        },
    }


def _build_design(data: dict) -> dict:
    desc = _descriptors(data)
    donors = {
        name: _construct_donors_for_split(
            name, data["row_masks"][name], data["base_ce"], data["docids"],
        )
        for name in SPLITS
    }
    cells = {}
    for name, (low, high) in CONTROL_CELLS.items():
        row_mask = (data["folds"] >= low) & (data["folds"] < high)
        flat_cell = _flat_row_mask(row_mask)
        cell = {"exclusive": {}, "fingerprint": {}}
        for tag in QUARTET_TAGS:
            positive = data["exclusive_masks"][tag] & flat_cell
            parent = torch.zeros(1000 * TOKENS, dtype=torch.bool)
            parent[data["leaves"][tag]["slice"].long()] = True
            pool = parent & flat_cell & ~data["quartet_union"] & ~data["full_masks"][tag]
            cell["exclusive"][tag] = _construct_controls(
                positive, pool, desc, tag=tag, cell=f"{name}:exclusive",
            )
        for tag in FINGERPRINT_TAGS:
            member = torch.zeros(1000 * TOKENS, dtype=torch.bool)
            member[data["leaves"][tag]["member"].long()] = True
            positive = member & flat_cell
            parent = torch.zeros_like(member)
            parent[data["leaves"][tag]["slice"].long()] = True
            pool = parent & flat_cell & ~data["quartet_union"] & ~member
            cell["fingerprint"][tag] = _construct_controls(
                positive, pool, desc, tag=tag, cell=f"{name}:fingerprint",
            )
        cells[name] = cell
    identity = {
        "descriptor_hashes": {name: _tensor_sha256(value) for name, value in desc.items()},
        "donors": {name: value["identity"] for name, value in donors.items()},
        "controls": {
            cell_name: {
                kind: {
                    tag: {key: value for key, value in match.items()
                          if key not in ("members", "controls", "tiers")}
                    for tag, match in group.items()
                }
                for kind, group in cell.items()
            }
            for cell_name, cell in cells.items()
        },
        "matching": {
            "seed": CONTROL_SEED,
            "tiers": [
                "next_token+position_bin_32+within_split_native_CE_decile",
                "next_token+within_split_native_CE_decile",
                "token_class+position_bin_32+within_split_native_CE_decile",
                "token_class+within_split_native_CE_decile",
                "token_class+position_bin_32",
                "position_bin_32+within_split_native_CE_decile",
                "token_class",
                "position_bin_32",
                "within_split_native_CE_decile",
                "same_parent_slice_and_same_data_cell_only",
            ],
            "token_classes": [
                "newline_or_control_line", "number", "punctuation_or_symbol",
                "space_initial_capitalized", "space_initial_alphabetic",
                "alphabetic_continuation", "mixed_or_other",
            ],
        },
    }
    return {"descriptors": desc, "donors": donors, "cells": cells, "identity": identity}


def preflight() -> tuple[dict, dict]:
    data = _load_cpu_inputs()
    design = _build_design(data)
    report = {
        "dependency_sha256": _validate_dependency_bytes(),
        "fold_sha256": _tensor_sha256(data["folds"]),
        "row_split_sha256": {name: _tensor_sha256(mask) for name, mask in data["row_masks"].items()},
        "row_split_counts": {name: int(mask.sum()) for name, mask in data["row_masks"].items()},
        "quartet_masks": data["mask_identities"],
        "quartet_split_counts": data["split_counts"],
        "fitted_pair_intersections": list(data["fitted_pair_counts"]),
        "fitted_triple_intersection": data["fitted_triple_count"],
        "design": design["identity"],
        "fingerprint_tags": list(FINGERPRINT_TAGS),
    }
    return data, design, report


def _direct_logits(model, tokens: torch.Tensor) -> tuple[torch.Tensor, int]:
    calls = 0

    def count_a8(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    handle = model.transformer.h[A8_SITE].attn.register_forward_hook(count_a8)
    try:
        x = F.rms_norm(model.transformer.wte(tokens), (D,))
        x0 = x
        v1 = None
        for block in model.transformer.h:
            x, v1 = block(x, v1, x0)
        logits = model.lm_head(F.rms_norm(x, (D,)))
        logits = (30.0 * torch.tanh(logits / 30.0)).float()
    finally:
        handle.remove()
    return logits, calls


def _dispatched_logits(
    model,
    tokens: torch.Tensor,
    *,
    replacement: torch.Tensor | None = None,
    capture: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
    calls = 0
    captured = None
    edit_rms = 0.0
    write_difference_max = 0.0

    def attention(event):
        nonlocal calls, captured, edit_rms, write_difference_max
        write, first_value = event.block.attn(event.state, event.first_value)
        if event.site == A8_SITE:
            calls += 1
            if capture:
                if captured is not None:
                    raise RuntimeError("attention8 capture was entered twice")
                captured = write.detach().float().cpu().clone()
            if replacement is not None:
                if replacement.shape != write.shape:
                    raise RuntimeError("attention8 replacement shape changed")
                changed = replacement.to(device=write.device, dtype=write.dtype)
                difference = changed.float() - write.float()
                edit_rms = float(difference.square().mean().sqrt())
                write_difference_max = float(difference.abs().max())
                write = changed
        return write, first_value

    def mlp(event):
        return event.block.mlp(event.state)

    logits = facade.forward_with_dispatch(model, tokens, attention, mlp)
    if calls != 1:
        raise RuntimeError(f"attention8 was called {calls} times in one dispatched execution")
    return logits, captured, {
        "attention8_calls": calls,
        "edit_rms": edit_rms,
        "write_difference_max": write_difference_max,
    }


def _per_token_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none",
    ).view_as(targets)


@torch.no_grad()
def _capture_native_fit(model, data: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
    fit_rows = data["row_masks"]["fit"].nonzero().flatten()
    row_to_local = torch.full((1000,), -1, dtype=torch.int64)
    row_to_local[fit_rows] = torch.arange(fit_rows.numel())
    captures = torch.empty((fit_rows.numel(), TOKENS, D), dtype=torch.float32)
    base_ce = torch.empty((fit_rows.numel(), TOKENS), dtype=torch.float32)
    diagnostics = {
        "forward_calls": 0,
        "native_replay_logits_exact": True,
        "native_replay_max_logit_error": 0.0,
        "attention8_once_every_execution": True,
        "self_donor_logits_exact": True,
        "self_donor_write_exact": True,
        "self_donor_max_logit_error": 0.0,
        "self_donor_max_write_error": 0.0,
    }
    device = next(model.parameters()).device
    for start in range(0, fit_rows.numel(), BATCH):
        selected = fit_rows[start:start + BATCH]
        batch = data["rows"][selected]
        tokens = batch[:, :TOKENS].to(device)
        targets = batch[:, 1:TOKENS + 1].to(device)
        native, captured, native_diag = _dispatched_logits(model, tokens, capture=True)
        diagnostics["forward_calls"] += 1
        if captured is None:
            raise RuntimeError("native attention8 capture is absent")
        direct, direct_calls = _direct_logits(model, tokens)
        diagnostics["forward_calls"] += 1
        if direct_calls != 1:
            diagnostics["attention8_once_every_execution"] = False
        replay_error = float((native - direct).abs().max())
        diagnostics["native_replay_max_logit_error"] = max(
            diagnostics["native_replay_max_logit_error"], replay_error,
        )
        diagnostics["native_replay_logits_exact"] &= bool(torch.equal(native, direct))
        self_logits, _, self_diag = _dispatched_logits(
            model, tokens, replacement=captured.to(device),
        )
        diagnostics["forward_calls"] += 1
        self_error = float((native - self_logits).abs().max())
        diagnostics["self_donor_max_logit_error"] = max(
            diagnostics["self_donor_max_logit_error"], self_error,
        )
        diagnostics["self_donor_logits_exact"] &= bool(torch.equal(native, self_logits))
        diagnostics["self_donor_max_write_error"] = max(
            diagnostics["self_donor_max_write_error"], self_diag["write_difference_max"],
        )
        diagnostics["self_donor_write_exact"] &= self_diag["write_difference_max"] == 0.0
        diagnostics["attention8_once_every_execution"] &= (
            native_diag["attention8_calls"] == 1 and self_diag["attention8_calls"] == 1
        )
        stop = start + selected.numel()
        captures[start:stop] = captured
        base_ce[start:stop] = _per_token_ce(native, targets).cpu()
        del native, direct, self_logits
    return captures, base_ce, {**diagnostics, "fit_rows": fit_rows, "row_to_local": row_to_local}


def _local_donor_write(
    captures: torch.Tensor,
    donor_map: torch.Tensor,
    recipient_global: torch.Tensor,
    row_to_local: torch.Tensor,
) -> torch.Tensor:
    donor = donor_map[recipient_global]
    donor_row_local = row_to_local[donor // TOKENS]
    if bool((donor_row_local < 0).any()):
        raise RuntimeError("FIT intervention requested a donor outside captured FIT")
    donor_position = donor % TOKENS
    return captures[donor_row_local, donor_position]


@torch.no_grad()
def _whole_a8_fit_swaps(
    model,
    data: dict,
    design: dict,
    captures: torch.Tensor,
    native_ce: torch.Tensor,
    capture_diag: dict,
    *,
    edit_rms_floor: float,
) -> tuple[dict, dict]:
    fit_rows = capture_diag["fit_rows"]
    row_to_local = capture_diag["row_to_local"]
    fit_donors = design["donors"]["fit"]
    device = next(model.parameters()).device
    sums = {
        direction: {
            ensemble: torch.zeros((fit_rows.numel(), TOKENS), dtype=torch.float64)
            for ensemble in ("D0", "D1")
        }
        for direction in ("forward", "reverse")
    }
    calls = 0
    minimum_edit_rms = math.inf
    edits = 0
    for direction, maps in (
        ("forward", fit_donors["maps"]),
        ("reverse", fit_donors["inverse_maps"]),
    ):
        for map_index, donor_map in enumerate(maps):
            ensemble = "D0" if map_index < 4 else "D1"
            for start in range(0, fit_rows.numel(), BATCH):
                selected = fit_rows[start:start + BATCH]
                count = selected.numel()
                batch = data["rows"][selected]
                tokens = batch[:, :TOKENS].to(device)
                targets = batch[:, 1:TOKENS + 1].to(device)
                recipient = (
                    selected[:, None] * TOKENS + torch.arange(TOKENS)[None, :]
                ).reshape(-1)
                donor_write = _local_donor_write(
                    captures, donor_map, recipient, row_to_local,
                ).view(count, TOKENS, D)
                logits, _, diag = _dispatched_logits(
                    model, tokens, replacement=donor_write.to(device),
                )
                calls += 1
                minimum_edit_rms = min(minimum_edit_rms, diag["edit_rms"])
                edits += recipient.numel()
                stop = start + count
                sums[direction][ensemble][start:stop] += (
                    _per_token_ce(logits, targets).cpu().double()
                    - native_ce[start:stop].double()
                )
                del logits
    deltas = {
        direction: {ensemble: value / 4.0 for ensemble, value in group.items()}
        for direction, group in sums.items()
    }
    diagnostics = {
        "forward_calls": calls,
        "real_edit_rms_min": minimum_edit_rms,
        "real_edit_rms_floor": edit_rms_floor,
        "real_edits_live": bool(minimum_edit_rms > edit_rms_floor),
        "edited_token_writes": edits,
    }
    return deltas, diagnostics


def _fit_local_indices(global_indices: torch.Tensor, row_to_local: torch.Tensor) -> torch.Tensor:
    local_row = row_to_local[global_indices // TOKENS]
    if bool((local_row < 0).any()):
        raise RuntimeError("a Stage-A analysis index is outside FIT")
    return local_row * TOKENS + global_indices % TOKENS


def _higher_quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = math.ceil(probability * (len(ordered) - 1))
    return float(ordered[index])


def _bootstrap_lower(
    member_values: torch.Tensor,
    control_values: torch.Tensor,
    members_global: torch.Tensor,
    tiers: torch.Tensor,
    docids: torch.Tensor,
    *,
    seed: int,
) -> float:
    """Matching-tier-stratified, member-document cluster bootstrap."""
    if member_values.numel() != control_values.numel() or member_values.numel() == 0:
        return float("nan")
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for pair_index, (global_index, tier) in enumerate(zip(members_global.tolist(), tiers.tolist())):
        groups[(int(tier), int(docids[global_index // TOKENS]))].append(pair_index)
    strata: dict[int, list[tuple[int, list[int]]]] = defaultdict(list)
    for (tier, document), positions in groups.items():
        strata[tier].append((document, positions))
    generator = torch.Generator().manual_seed(seed)
    samples = []
    difference = member_values.abs() - control_values.abs()
    for _ in range(BOOTSTRAPS):
        total = 0.0
        count = 0
        for tier in sorted(strata):
            clusters = sorted(strata[tier])
            draw = torch.randint(len(clusters), (len(clusters),), generator=generator).tolist()
            for selected in draw:
                positions = clusters[selected][1]
                total += float(difference[positions].sum())
                count += len(positions)
        samples.append(total / max(count, 1))
    return _higher_quantile(samples, 0.05)


def _cosine_residual_recovery(left: torch.Tensor, right: torch.Tensor) -> dict:
    left = left.double()
    right = right.double()
    dot = float(left @ right)
    left_norm2 = float(left @ left)
    right_norm2 = float(right @ right)
    cosine = dot / math.sqrt(max(left_norm2 * right_norm2, 1e-30))
    scale = dot / max(left_norm2, 1e-30)
    residual = float((scale * left - right).norm()) / math.sqrt(max(right_norm2, 1e-30))
    recovery = dot / max(right_norm2, 1e-30)
    return {
        "signed_cosine": cosine,
        "optimal_scale_D0_to_D1": scale,
        "relative_residual": residual,
        "aligned_recovery": recovery,
    }


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.double() - left.double().mean()
    right = right.double() - right.double().mean()
    denominator = float(left.norm() * right.norm())
    return float(left @ right) / denominator if denominator > 0 else float("nan")


def _fingerprint(delta_flat: torch.Tensor, matches: dict[str, dict], row_to_local: torch.Tensor) -> torch.Tensor:
    values = []
    for tag in FINGERPRINT_TAGS:
        match = matches[tag]
        member = _fit_local_indices(match["members"], row_to_local)
        control = _fit_local_indices(match["controls"], row_to_local)
        values.append(delta_flat[member].abs().mean() - delta_flat[control].abs().mean())
    return torch.stack(values).double()


def _permutation_groups(
    match: dict[str, dict],
    design: dict,
    row_to_local: torch.Tensor,
) -> list[torch.Tensor]:
    # The union of all member/control labels is moved by one common
    # permutation within token-class + position-bin + CE-decile strata.  A
    # common bijection preserves every intersection in the 32-label lattice.
    touched = torch.unique(torch.cat([
        value[key] for value in match.values() for key in ("members", "controls")
    ])).tolist()
    desc = design["descriptors"]
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for global_index in touched:
        key = (
            int(desc["token_class"][global_index]),
            int(desc["position_bin"][global_index]),
            int(desc["ce_decile"][global_index]),
        )
        groups[key].append(global_index)
    return [
        _fit_local_indices(torch.tensor(sorted(values), dtype=torch.int64), row_to_local)
        for _, values in sorted(groups.items())
    ]


def _permuted_effect(
    values: torch.Tensor,
    groups: list[torch.Tensor],
    *,
    seed: int,
) -> torch.Tensor:
    result = values.clone()
    generator = torch.Generator().manual_seed(seed)
    for indices in groups:
        if indices.numel() > 1:
            result[indices] = values[indices[torch.randperm(indices.numel(), generator=generator)]]
    return result


def _score_stage_a(data: dict, design: dict, deltas: dict, runtime_diag: dict) -> tuple[dict, bool]:
    row_to_local = runtime_diag["row_to_local"]
    flat_delta = {
        direction: {ensemble: value.reshape(-1) for ensemble, value in group.items()}
        for direction, group in deltas.items()
    }
    target_report = {}
    target_pass = True
    for direction in ("forward", "reverse"):
        direction_report = {}
        for tag_index, tag in enumerate(FITTED_TAGS):
            tag_report = {"halves": {}, "half_magnitude_ratio": {}}
            half_magnitudes = {ensemble: [] for ensemble in ("D0", "D1")}
            for half_index, cell_name in enumerate(("fit_half0", "fit_half1")):
                match = design["cells"][cell_name]["exclusive"][tag]
                member = _fit_local_indices(match["members"], row_to_local)
                control = _fit_local_indices(match["controls"], row_to_local)
                half_report = {"ensembles": {}}
                for ensemble_index, ensemble in enumerate(("D0", "D1")):
                    values = flat_delta[direction][ensemble]
                    member_values = values[member]
                    control_values = values[control]
                    member_abs = float(member_values.abs().mean())
                    control_abs = float(control_values.abs().mean())
                    concentration = member_abs / max(control_abs, 1e-30)
                    lower = _bootstrap_lower(
                        member_values,
                        control_values,
                        match["members"],
                        match["tiers"],
                        data["docids"],
                        seed=521000 + 1000 * tag_index + 100 * half_index
                        + 10 * ensemble_index + (0 if direction == "forward" else 1),
                    )
                    passed = member_abs >= 0.10 and concentration >= 3.0 and lower > 0.0
                    half_report["ensembles"][ensemble] = {
                        "exclusive_member_count": member.numel(),
                        "matched_control_count": control.numel(),
                        "mean_member_abs_delta_CE_nat": member_abs,
                        "mean_control_abs_delta_CE_nat": control_abs,
                        "member_control_concentration": concentration,
                        "bootstrap_lower_95_member_minus_control_abs_delta_CE_nat": lower,
                        "material_selective_bootstrap_pass": passed,
                    }
                    half_magnitudes[ensemble].append(member_abs)
                    target_pass &= passed
                d0 = flat_delta[direction]["D0"][member]
                d1 = flat_delta[direction]["D1"][member]
                transfer = _cosine_residual_recovery(d0, d1)
                transfer_pass = (
                    transfer["signed_cosine"] >= 0.70
                    and transfer["relative_residual"] <= 0.60
                    and transfer["aligned_recovery"] > 0.0
                )
                half_report["D0_D1_signed_response_transfer"] = {
                    **transfer, "pass": transfer_pass,
                }
                target_pass &= transfer_pass
                tag_report["halves"][cell_name] = half_report
            for ensemble in ("D0", "D1"):
                low, high = half_magnitudes[ensemble]
                ratio = low / max(high, 1e-30)
                passed = 0.5 <= ratio <= 2.0
                tag_report["half_magnitude_ratio"][ensemble] = {"half0_over_half1": ratio, "pass": passed}
                target_pass &= passed
            direction_report[tag] = tag_report
        target_report[direction] = direction_report

    fingerprint_report = {}
    fingerprint_pass = True
    for direction_index, direction in enumerate(("forward", "reverse")):
        direction_report = {}
        for ensemble_index, ensemble in enumerate(("D0", "D1")):
            values = flat_delta[direction][ensemble]
            half0_match = design["cells"]["fit_half0"]["fingerprint"]
            half1_match = design["cells"]["fit_half1"]["fingerprint"]
            half0 = _fingerprint(values, half0_match, row_to_local)
            half1 = _fingerprint(values, half1_match, row_to_local)
            observed = _pearson(half0, half1)
            groups = _permutation_groups(half1_match, design, row_to_local)
            null = []
            for permutation in range(PERMUTATIONS):
                shuffled = _permuted_effect(
                    values,
                    groups,
                    seed=523000 + 10000 * direction_index + 1000 * ensemble_index + permutation,
                )
                null.append(_pearson(half0, _fingerprint(shuffled, half1_match, row_to_local)))
            finite_null = [value for value in null if math.isfinite(value)]
            if len(finite_null) != PERMUTATIONS:
                raise RuntimeError("fingerprint permutation produced a nonfinite Pearson statistic")
            q95 = _higher_quantile(finite_null, 0.95)
            passed = math.isfinite(observed) and observed >= 0.50 and observed > q95
            direction_report[ensemble] = {
                "half0": half0.tolist(),
                "half1": half1.tolist(),
                "pearson": observed,
                "overlap_lattice_preserving_permutation_q95": q95,
                "permutations": PERMUTATIONS,
                "pass": passed,
            }
            fingerprint_pass &= passed
        fingerprint_report[direction] = direction_report

    exact_pass = (
        runtime_diag["native_replay_logits_exact"]
        and runtime_diag["attention8_once_every_execution"]
        and runtime_diag["self_donor_logits_exact"]
        and runtime_diag["self_donor_write_exact"]
        and runtime_diag["real_edits_live"]
    )
    prediction_a = exact_pass and target_pass and fingerprint_pass
    return {
        "exact_object_and_liveness": {"pass": exact_pass},
        "exclusive_target_power": target_report,
        "exclusive_target_power_pass": target_pass,
        "circuit_fingerprint": fingerprint_report,
        "circuit_fingerprint_pass": fingerprint_pass,
    }, prediction_a


def _public_runtime_diag(value: dict) -> dict:
    return {
        key: item for key, item in value.items()
        if key not in ("fit_rows", "row_to_local", "forward_calls")
    }


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Stage-A result: {path}")
    with temporary.open("x", encoding="utf-8") as sink:
        json.dump(value, sink, indent=2, sort_keys=True, allow_nan=False)
        sink.write("\n")
        sink.flush()
        os.fsync(sink.fileno())
    os.link(temporary, path)
    temporary.unlink()


@torch.no_grad()
def _gpu_smoke() -> dict:
    """Return only instrument diagnostics; never retain task/circuit outcomes."""
    data, design, preflight_report = preflight()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32)
    fit_rows = data["row_masks"]["fit"].nonzero().flatten()
    selected = fit_rows[:BATCH]
    batch = data["rows"][selected]
    tokens = batch[:, :TOKENS].cuda()
    native, captured, native_diag = _dispatched_logits(model, tokens, capture=True)
    direct, direct_calls = _direct_logits(model, tokens)
    self_logits, _, self_diag = _dispatched_logits(model, tokens, replacement=captured.cuda())

    donor_map = design["donors"]["fit"]["maps"][0]
    recipient = (selected[:, None] * TOKENS + torch.arange(TOKENS)[None, :]).reshape(-1)
    donors = donor_map[recipient]
    donor_rows = torch.unique(donors // TOKENS)
    donor_capture = {}
    for start in range(0, donor_rows.numel(), BATCH):
        donor_selected = donor_rows[start:start + BATCH]
        donor_tokens = data["rows"][donor_selected, :TOKENS].cuda()
        _, write, _ = _dispatched_logits(model, donor_tokens, capture=True)
        for local, row in enumerate(donor_selected.tolist()):
            donor_capture[row] = write[local]
    replacement = torch.stack([
        donor_capture[int(index // TOKENS)][int(index % TOKENS)] for index in donors.tolist()
    ]).view(selected.numel(), TOKENS, D)
    real_logits, _, real_diag = _dispatched_logits(model, tokens, replacement=replacement.cuda())
    result = {
        "namespace": "rung521_gpu_smoke_no_task_or_circuit_outcome",
        "checkpoint": checkpoint.__dict__,
        "dependency_sha256": preflight_report["dependency_sha256"],
        "native_replay_logits_exact": bool(torch.equal(native, direct)),
        "native_replay_max_logit_error": float((native - direct).abs().max()),
        "native_attention8_calls": native_diag["attention8_calls"],
        "direct_attention8_calls": direct_calls,
        "self_donor_logits_exact": bool(torch.equal(native, self_logits)),
        "self_donor_max_logit_error": float((native - self_logits).abs().max()),
        "self_donor_write_exact": self_diag["write_difference_max"] == 0.0,
        "real_donor_edit_rms": real_diag["edit_rms"],
        "real_donor_logits_changed": not bool(torch.equal(native, real_logits)),
        "suggested_frozen_floor": real_diag["edit_rms"] / 10.0,
        "scientific_metrics_retained": False,
    }
    del model, native, direct, self_logits, real_logits
    torch.cuda.empty_cache()
    return result


def _future_optimization_stage_is_not_implemented() -> None:
    """Deliberate kill-switch: no optimizer may be reached from this entrypoint."""
    raise RuntimeError(
        "rung521 entrypoint stops after Stage A; future gradients require a separately audited implementation"
    )


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="build and print the CPU-only frozen design")
    parser.add_argument(
        "--preflight-output",
        type=Path,
        default=None,
        help=f"create-only JSON receipt (recommended: {DEFAULT_PREFLIGHT_OUT})",
    )
    parser.add_argument("--edit-rms-floor", type=float, default=None,
                        help="positive liveness floor frozen from the managed smoke")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    started = time.time()
    data, design, preflight_report = preflight()
    if args.preflight:
        if args.preflight_output is not None:
            _atomic_json(args.preflight_output, preflight_report)
        print(json.dumps(preflight_report, indent=2, sort_keys=True), flush=True)
        return preflight_report
    if args.edit_rms_floor is None or not math.isfinite(args.edit_rms_floor) or args.edit_rms_floor <= 0:
        raise RuntimeError("science run requires a positive smoke-frozen --edit-rms-floor")
    if os.environ.get("BQLIB_NO_MODEL") == "1":
        raise RuntimeError("BQLIB_NO_MODEL forbids the Stage-A model run")

    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32)
    captures, native_ce, capture_diag = _capture_native_fit(model, data)
    deltas, swap_diag = _whole_a8_fit_swaps(
        model,
        data,
        design,
        captures,
        native_ce,
        capture_diag,
        edit_rms_floor=args.edit_rms_floor,
    )
    runtime_diag = {
        **capture_diag,
        **swap_diag,
        "capture_forward_calls": capture_diag["forward_calls"],
        "swap_forward_calls": swap_diag["forward_calls"],
    }
    analysis, prediction_a = _score_stage_a(data, design, deltas, runtime_diag)
    result = {
        "schema_version": 1,
        "rung": 521,
        "stage": "A_only",
        "status": "stage_a_passed_ready_for_separately_audited_future_optimization"
                  if prediction_a else "instrument_power_failure_stop_before_gradients",
        "claim_level": "whole-attention8 power and instrument validation only",
        "prediction_a": prediction_a,
        "predictions_b_through_e_opened": bool(prediction_a),
        "optimizer_created": False,
        "backward_calls": 0,
        "learned_values": 0,
        "checkpoint": checkpoint.__dict__,
        "preflight": preflight_report,
        "runtime_diagnostics": _public_runtime_diag(runtime_diag),
        "analysis": analysis,
        "execution_price": {
            "inference_forward_calls": capture_diag["forward_calls"] + swap_diag["forward_calls"],
            "registered_exact_call_formula": "19*ceil(568/4)=2698",
            "backward_calls": 0,
            "stored_learned_floating_values": 0,
            "runtime_seconds": time.time() - started,
        },
        "next_step": "stop_before_gradients_and_audit_future_stage"
                     if prediction_a else "increase_donors_or_documents_and_rebuild_masks",
    }
    _atomic_json(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "prediction_a": prediction_a,
        "output": str(args.output),
        "forward_calls": result["execution_price"]["inference_forward_calls"],
        "backward_calls": 0,
    }, indent=2), flush=True)
    # Deliberately return after Stage A.  Do not call the future-stage
    # kill-switch in normal execution: its presence documents that no hidden
    # fall-through is legal, while a valid Stage-A result remains usable.
    return result


if __name__ == "__main__":
    main()
