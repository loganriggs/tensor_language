"""RUNG454 -- COMPLETE VOCABULARY-PROGRAM TEACHING CONSEQUENCES."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

from receipt import dump
from simplicity_mlp0_complete_candidate_consequences import factored, metrics


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
OUT = ROOT / "simplicity_vocabulary_complete_candidate_consequences_results.json"
BUNDLE = ROOT / "simplicity_vocabulary_complete_candidate_consequences_bundle.pt"
ROWS = ROOT / ".rowcache_simplicity_consequence_v1/teaching_96.pt"
ROWS_RECEIPT = ROOT / "simplicity_consequence_v1_rows_receipt.json"
BANK = POLY / "prospective_consequence_candidate_bank_v1.json"
FIT96A = ROOT / ".rowcache/fineweb_n96_skip1200.pt"
FIT96B = ROOT / ".rowcache/fineweb_n96_skip80.pt"
FIT480 = ROOT / ".rowcache/fineweb_n480_skip80.pt"
FIT192_7K = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
FIT192_11K = ROOT / ".rowcache/fineweb_n192_skip11000.pt"
FACTOR = ROOT / "mlp16_rank2_quadratic_factored.pt"
D = 1152
W = 50304
V_REAL = 50257
TOKENS = 256
DOCUMENTS = 96
RIDGE_REL = 1e-6
SPARSE_K = 1129
RANDOM_SEED = 30420260901
UNIFORM_RANKS = (0, 128, 256, 512)
WEIGHTED_RANKS = (512, 640, 768)
WEIGHTS = ("sqrt_count_plus_1", "count_plus_1")
KINDS = ("shared", "independent")
IDS = (
    "vocab_r300_shared_0", "vocab_r300_independent_0",
    "vocab_r300_shared_128", "vocab_r300_independent_128",
    "vocab_r300_shared_256", "vocab_r300_independent_256",
    "vocab_r300_shared_512", "vocab_r300_independent_512",
    "vocab_r304_fisher_rare", "vocab_r304_norm_rare", "vocab_r304_random_rare",
    "vocab_r305_sqrt_count_plus_1_s512_shared", "vocab_r305_sqrt_count_plus_1_s512_independent",
    "vocab_r305_sqrt_count_plus_1_s640_shared", "vocab_r305_sqrt_count_plus_1_s640_independent",
    "vocab_r305_sqrt_count_plus_1_s768_shared", "vocab_r305_sqrt_count_plus_1_s768_independent",
    "vocab_r305_count_plus_1_s512_shared", "vocab_r305_count_plus_1_s512_independent",
    "vocab_r305_count_plus_1_s640_shared", "vocab_r305_count_plus_1_s640_independent",
    "vocab_r305_count_plus_1_s768_shared", "vocab_r305_count_plus_1_s768_independent",
)
PRICES = {
    "vocab_r300_shared_0": 59_277_312, "vocab_r300_independent_0": 59_236_608,
    "vocab_r300_shared_128": 65_863_680, "vocab_r300_independent_128": 65_822_976,
    "vocab_r300_shared_256": 72_450_048, "vocab_r300_independent_256": 72_409_344,
    "vocab_r300_shared_512": 85_622_784, "vocab_r300_independent_512": 85_582_080,
    "vocab_r304_fisher_rare": 86_924_521, "vocab_r304_norm_rare": 86_924_521,
    "vocab_r304_random_rare": 86_924_521,
    "vocab_r305_sqrt_count_plus_1_s512_shared": 85_622_784,
    "vocab_r305_sqrt_count_plus_1_s512_independent": 85_582_080,
    "vocab_r305_sqrt_count_plus_1_s640_shared": 92_209_152,
    "vocab_r305_sqrt_count_plus_1_s640_independent": 92_168_448,
    "vocab_r305_sqrt_count_plus_1_s768_shared": 98_795_520,
    "vocab_r305_sqrt_count_plus_1_s768_independent": 98_754_816,
    "vocab_r305_count_plus_1_s512_shared": 85_622_784,
    "vocab_r305_count_plus_1_s512_independent": 85_582_080,
    "vocab_r305_count_plus_1_s640_shared": 92_209_152,
    "vocab_r305_count_plus_1_s640_independent": 92_168_448,
    "vocab_r305_count_plus_1_s768_shared": 98_795_520,
    "vocab_r305_count_plus_1_s768_independent": 98_754_816,
}
HASHES = {
    BANK: "e35d5c0aa1dae34173b93ae4d81cafa8317539adfaf7c74bfe7decb068ac47be",
    ROWS_RECEIPT: "1611c5bd60491a6b600950874ae55cd5925afad12096a48de3426e88e9cfc5d8",
    ROWS: "b94fb82be422e17411ed8ebf1b3e94956848e074687dbdf593ae9285da837014",
    FIT96A: "21707551f35d13818c10ac59e12e9445ef076d0522371fe779691bfab719d34f",
    FIT96B: "94bc1fb3e3a6a061541e555295e0af8c50ae6068fdff84e95a69c25844091eda",
    FIT480: "2acf75382486988a1e124a1a575ef3230af43aa1b1507d80dee02eefc7bba496",
    FIT192_7K: "d66c1ee7807bc6b9bd7d0ddba5cdd7e3bc64926b00320a10675a2f817d67128c",
    FIT192_11K: "b1564bfd071418f401a816cb01e3d26b082a3e73ba858838f1c83c250db4d868",
    FACTOR: "b9870f738b528e988ff9a1e04cdc6e1096de8ab0dc5fa86bb76229812d9ffb6e",
    ROOT / "ops/joint_vocab_shared_code_screen.py":
        "8eb17993fda6e284b512ad86519f3729e3bd59d38bee251d4da195dc7dc0ed5b",
    ROOT / "ops/joint_vocab_frequency_weighted_followup.py":
        "371fec5deb9148b055e4a45b6ae965f149a9e3774ca1e7aa383de4e47f8bcba9",
    ROOT / "ops/joint_vocab_sparse_rare_residual.py":
        "18c38fdbc6849dd4aa33ff30b58471a02c0427dfcf6bf16d842b6cb9c9a83532",
    ROOT / "ops/joint_vocab_distributed_rank_frontier.py":
        "76a6525b67e6128d733c62343741d79f058072966b38a06af4ad2add6c4421de",
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_rows(path: Path, count: int) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    value = value["rows"] if isinstance(value, dict) else value
    return value[:count, :257].long().contiguous()


def fit_uniform(embedding: torch.Tensor, output: torch.Tensor) -> dict[str, torch.Tensor | float]:
    gram = embedding.T @ embedding
    ridge = RIDGE_REL * float(torch.trace(gram)) / D
    mapping = torch.linalg.solve(gram + ridge * torch.eye(D, device="cuda"), embedding.T @ output)
    residual = output - embedding @ mapping
    values, vectors = torch.linalg.eigh(residual.T @ residual)
    residual_vectors = vectors[:, torch.argsort(values, descending=True)]
    values, vectors = torch.linalg.eigh(output.T @ output)
    output_vectors = vectors[:, torch.argsort(values, descending=True)]
    return {"mapping": mapping, "residual": residual, "residual_vectors": residual_vectors,
            "output_vectors": output_vectors, "ridge": ridge}


def rank_edges() -> list[tuple[str, str]]:
    edges = []
    for kind in KINDS:
        for left, right in zip(UNIFORM_RANKS[:-1], UNIFORM_RANKS[1:]):
            edges.append((f"vocab_r300_{kind}_{left}", f"vocab_r300_{kind}_{right}"))
    for weight in WEIGHTS:
        for kind in KINDS:
            for left, right in zip(WEIGHTED_RANKS[:-1], WEIGHTED_RANKS[1:]):
                edges.append((f"vocab_r305_{weight}_s{left}_{kind}",
                              f"vocab_r305_{weight}_s{right}_{kind}"))
    if len(edges) != 14:
        raise RuntimeError("structured vocabulary edge count changed")
    return edges


def pearson(left: list[float], right: list[float]) -> float:
    a = torch.tensor(left, dtype=torch.float64); b = torch.tensor(right, dtype=torch.float64)
    a -= a.mean(); b -= b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))


def edge_accuracy(values: dict[str, float]) -> dict[str, Any]:
    rows = [{"lower_rank": left, "higher_rank": right,
             "lower_minus_higher": values[left] - values[right],
             "ordered": values[left] > values[right]} for left, right in rank_edges()]
    count = sum(row["ordered"] for row in rows)
    return {"correct": count, "total": len(rows), "accuracy": count / len(rows), "edges": rows}


@torch.no_grad()
def capture_hidden(model, rows: torch.Tensor, mode: str, mean_value: torch.Tensor | None,
                   factor: dict[str, torch.Tensor] | None) -> tuple[torch.Tensor, int]:
    handles = []; calls = {"count": 0}
    if mode == "knockout":
        def knockout(_module, _args, output):
            calls["count"] += 1
            return mean_value.expand_as(output[0]).to(output[0].dtype), output[1]
        handles.append(model.transformer.h[16].attn.register_forward_hook(knockout))
    elif mode == "partner":
        def partner(_module, args, output):
            calls["count"] += 1
            return factored(args[0], factor).to(output.dtype)
        handles.append(model.transformer.h[16].mlp.register_forward_hook(partner))
    elif mode != "native":
        raise ValueError(mode)
    values = []
    try:
        for start in range(0, len(rows), 4):
            index = rows[start:start + 4, :-1].cuda()
            x = F.rms_norm(model.transformer.wte(index), (D,)); x0 = x; value0 = None
            for block in model.transformer.h:
                x, value0 = block(x, value0, x0)
            values.append(F.rms_norm(x, (D,)).float().cpu().reshape(-1, D))
    finally:
        for handle in handles:
            handle.remove()
    return torch.cat(values), calls["count"]


@torch.no_grad()
def attention_mean(model, rows: torch.Tensor) -> tuple[torch.Tensor, int]:
    total = torch.zeros(D, device="cuda", dtype=torch.float64); count = 0
    def hook(_module, _args, output):
        nonlocal count
        value = output[0].detach().double().reshape(-1, D)
        total.add_(value.sum(0)); count += value.shape[0]
    handle = model.transformer.h[16].attn.register_forward_hook(hook)
    try:
        capture_hidden(model, rows, "native", None, None)
    finally:
        handle.remove()
    return (total / count).float(), count


def append_loss(store: dict[str, list[torch.Tensor]], name: str,
                logits: torch.Tensor, target: torch.Tensor) -> None:
    logits = 30.0 * torch.tanh(logits / 30.0)
    store[name].append(F.cross_entropy(logits.float(), target, reduction="none").cpu())


@torch.no_grad()
def score_native(hidden: torch.Tensor, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    values = []
    for start in range(0, len(hidden), 64):
        h = hidden[start:start + 64].cuda(); y = target[start:start + 64].cuda()
        logits = 30.0 * torch.tanh((h @ output.T) / 30.0)
        values.append(F.cross_entropy(logits.float(), y, reduction="none").cpu())
    return torch.cat(values).float()


@torch.no_grad()
def score_all(hidden: torch.Tensor, target: torch.Tensor, embedding: torch.Tensor,
              uniform: dict[str, torch.Tensor | float],
              weighted: dict[str, dict[str, torch.Tensor | float]],
              sparse: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    store = {name: [] for name in IDS}
    for start in range(0, len(hidden), 64):
        h = hidden[start:start + 64].cuda(); y = target[start:start + 64].cuda()

        base = (h @ uniform["mapping"].T) @ embedding.T
        cumulative = torch.zeros_like(base); previous = 0
        for rank in UNIFORM_RANKS:
            if rank > previous:
                cumulative.add_((h @ uniform["residual_vectors"][:, previous:rank])
                                @ uniform["residual_code"][:, previous:rank].T)
            append_loss(store, f"vocab_r300_shared_{rank}", base + cumulative, y)
            previous = rank
        cumulative = torch.zeros_like(base); previous = 0
        for label_rank in UNIFORM_RANKS:
            rank = label_rank + 25
            cumulative.add_((h @ uniform["output_vectors"][:, previous:rank])
                            @ uniform["output_code"][:, previous:rank].T)
            append_loss(store, f"vocab_r300_independent_{label_rank}", cumulative, y)
            previous = rank

        for weight_name in WEIGHTS:
            fit = weighted[weight_name]
            base = (h @ fit["mapping"].T) @ embedding.T
            cumulative = torch.zeros_like(base); previous = 0
            shared512 = None
            for rank in WEIGHTED_RANKS:
                cumulative.add_((h @ fit["residual_vectors"][:, previous:rank])
                                @ fit["residual_code"][:, previous:rank].T)
                logits = base + cumulative
                append_loss(store, f"vocab_r305_{weight_name}_s{rank}_shared", logits, y)
                if weight_name == "count_plus_1" and rank == 512:
                    shared512 = logits
                previous = rank
            cumulative = torch.zeros_like(base); previous = 0
            for label_rank in WEIGHTED_RANKS:
                rank = label_rank + 25
                cumulative.add_((h @ fit["output_vectors"][:, previous:rank])
                                @ fit["output_code"][:, previous:rank].T)
                append_loss(store, f"vocab_r305_{weight_name}_s{label_rank}_independent", cumulative, y)
                previous = rank
            if weight_name == "count_plus_1":
                if shared512 is None:
                    raise RuntimeError("count-weighted rank512 logits missing")
                for selection in ("fisher", "norm", "random"):
                    ids = sparse[f"{selection}_ids"]
                    corrected = shared512.clone()
                    corrected[:, ids] += h @ sparse[f"{selection}_rows"].T
                    append_loss(store, f"vocab_r304_{selection}_rare", corrected, y)

    result = {name: torch.cat(parts).float() for name, parts in store.items()}
    if tuple(result) != IDS or any(value.numel() != DOCUMENTS * TOKENS for value in result.values()):
        raise RuntimeError("vocabulary score identity changed")
    return result


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert all(path.exists() and sha256(path) == digest for path, digest in HASHES.items())
        bank = {row["candidate_id"]: row for row in json.loads(BANK.read_text())["rows"]}
        assert tuple(name for name in IDS if name in bank) == IDS
        assert all(bank[name]["price_scalars"] == PRICES[name] for name in IDS)
        assert len(rank_edges()) == 14
        print("RUNG454 VOCABULARY CONSEQUENCES | dry run: hashes,23 candidates,14 edges, bars valid")
        return
    started = time.time()
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung454 output namespace already exists")
    for path, digest in HASHES.items():
        if sha256(path) != digest:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    bank = {row["candidate_id"]: row for row in json.loads(BANK.read_text())["rows"]}
    if any(bank[name]["price_scalars"] != PRICES[name]
           or bank[name]["program_family"] != "vocabulary_factorization"
           or bank[name]["family_role"] != "teaching" for name in IDS):
        raise RuntimeError("vocabulary bank identity/price/role changed")

    sys.path.insert(0, str(POLY))
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    sys.path.insert(0, "/workspace/rspd")
    from bilin18_observed_model_facade import validate_production_model
    import joint_vocab_distributed_rank_frontier as distributed
    import joint_vocab_sparse_rare_residual as rare
    from tier2_model import load_elriggs

    model, config = load_elriggs("bilin18")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    validate_production_model(model)
    embedding = model.transformer.wte.weight.detach().float().contiguous()
    output = model.lm_head.weight.detach().float().contiguous()
    if tuple(embedding.shape) != (W, D) or tuple(output.shape) != (W, D) or config["n_embd"] != D:
        raise RuntimeError("native vocabulary matrix identity changed")

    rows = load_rows(ROWS, DOCUMENTS)
    target = rows[:, 1:].reshape(-1).long().contiguous()
    calibration = load_rows(FIT192_11K, 128)
    mean_value, mean_n = attention_mean(model, calibration)
    factor_cpu = torch.load(FACTOR, map_location="cpu", weights_only=True)
    factor = {key: value.cuda().float() for key, value in factor_cpu.items()}
    native_hidden, native_calls = capture_hidden(model, rows, "native", None, None)
    replay_hidden, replay_calls = capture_hidden(model, rows, "native", None, None)
    knockout_hidden, knockout_calls = capture_hidden(model, rows, "knockout", mean_value, None)
    partner_hidden, partner_calls = capture_hidden(model, rows, "partner", None, factor)
    hidden_replay_max = float((replay_hidden - native_hidden).abs().max())
    if native_calls or replay_calls or knockout_calls != 24 or partner_calls != 24:
        raise RuntimeError("native consequence capture counts changed")

    uniform = fit_uniform(embedding, output)
    uniform["residual_code"] = uniform["residual"] @ uniform["residual_vectors"][:, :512]
    uniform["output_code"] = output @ uniform["output_vectors"][:, :537]
    fit = load_rows(FIT480, 480)
    counts = torch.bincount(fit[:, 1:].reshape(-1), minlength=W).float()
    weights = {
        "sqrt_count_plus_1": (counts.cuda() + 1.0).sqrt(),
        "count_plus_1": counts.cuda() + 1.0,
    }
    weighted = {}
    for name, weight in weights.items():
        weight = weight / weight.mean()
        fitted = distributed._fit_full(embedding, output, weight)
        fitted["residual_code"] = fitted["residual"] @ fitted["residual_vectors"][:, :768]
        fitted["output_code"] = output @ fitted["output_vectors"][:, :793]
        weighted[name] = fitted

    count_fit = weighted["count_plus_1"]
    count_basis = count_fit["residual_vectors"][:, :512]
    count_code = count_fit["residual"] @ count_basis
    approximation = embedding @ count_fit["mapping"] + count_code @ count_basis.T
    correction = output - approximation
    rare_allowed = torch.zeros(W, dtype=torch.bool)
    rare_allowed[:V_REAL] = counts[:V_REAL].cpu() <= 2
    selection_hidden, _ = capture_hidden(model, fit[:32], "native", None, None)
    def shared_function(hidden: torch.Tensor) -> torch.Tensor:
        return ((hidden @ count_fit["mapping"].T) @ embedding.T
                + (hidden @ count_basis) @ count_code.T)
    fisher_score = rare._fisher_scores(selection_hidden, output, shared_function)
    norm_score = correction.square().sum(1).double().cpu()
    fisher_ids_cpu = rare._top_allowed(fisher_score, rare_allowed, SPARSE_K)
    norm_ids_cpu = rare._top_allowed(norm_score, rare_allowed, SPARSE_K)
    rare_ids = torch.nonzero(rare_allowed, as_tuple=False).flatten()
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    random_ids_cpu = rare_ids[torch.randperm(len(rare_ids), generator=generator)[:SPARSE_K]]
    sparse = {}
    selections = {"fisher": fisher_ids_cpu, "norm": norm_ids_cpu, "random": random_ids_cpu}
    for name, ids_cpu in selections.items():
        ids = ids_cpu.cuda()
        sparse[f"{name}_ids"] = ids
        sparse[f"{name}_rows"] = correction[ids].contiguous()

    native = score_native(native_hidden, target, output)
    native_replay = score_native(replay_hidden, target, output)
    native_ko = score_native(knockout_hidden, target, output)
    partner = score_native(partner_hidden, target, output)
    condition_scores = {
        "unablated": score_all(native_hidden, target, embedding, uniform, weighted, sparse),
        "knockout": score_all(knockout_hidden, target, embedding, uniform, weighted, sparse),
        "partner": score_all(partner_hidden, target, embedding, uniform, weighted, sparse),
    }

    all_index = torch.arange(native.numel())
    waves = [torch.arange(0, 48 * TOKENS), torch.arange(48 * TOKENS, 96 * TOKENS)]
    arms = {}; bundle_arms = {}
    for name in IDS:
        p = condition_scores["unablated"][name]
        pko = condition_scores["knockout"][name]
        pq = condition_scores["partner"][name]
        arms[name] = {"price_scalars": PRICES[name],
                      "full": metrics(native, native_ko, p, pko, partner, pq, all_index),
                      "waves": [metrics(native, native_ko, p, pko, partner, pq, index)
                                for index in waves]}
        bundle_arms[name] = {"candidate_ce": p.half(), "candidate_ko_ce": pko.half(),
                             "candidate_partner_ce": pq.half()}

    full_maps = {metric: {name: arms[name]["full"][f"{metric}_normalized_error"] for name in IDS}
                 for metric in ("removal", "composition")}
    wave_maps = [{metric: {name: arms[name]["waves"][wave][f"{metric}_normalized_error"] for name in IDS}
                  for metric in ("removal", "composition")} for wave in range(2)]
    structured = {metric: edge_accuracy(full_maps[metric]) for metric in ("removal", "composition")}
    wave_structured = [{metric: edge_accuracy(wave_maps[wave][metric])
                        for metric in ("removal", "composition")} for wave in range(2)]
    wave_correlation = {
        metric: pearson([wave_maps[0][metric][name] for name in IDS],
                        [wave_maps[1][metric][name] for name in IDS])
        for metric in ("removal", "composition")}
    spans = {metric: max(values.values()) - min(values.values()) for metric, values in full_maps.items()}
    selection_receipts = {
        name: {"count": len(ids), "semantic_sha256": tensor_sha256(ids),
               "fit_count_histogram": {str(value): int((counts[ids].long() == value).sum())
                                       for value in (0, 1, 2)}}
        for name, ids in selections.items()
    }
    replay_ce_max = float((native_replay - native).abs().max())
    pred_a = bool(hidden_replay_max == 0 and replay_ce_max == 0 and mean_n == 128 * TOKENS
                  and knockout_calls == partner_calls == 24
                  and all(item["count"] == SPARSE_K for item in selection_receipts.values()))
    pred_b = bool(all(structured[metric]["accuracy"] >= .85
                      for metric in ("removal", "composition")))
    pred_c = bool(spans["removal"] >= .05 and spans["composition"] >= .05)
    pred_d = bool(all(wave_structured[wave][metric]["accuracy"] >= .70
                      for wave in range(2) for metric in ("removal", "composition"))
                  and all(wave_correlation[metric] >= .70 for metric in ("removal", "composition")))
    native_effect = native_ko - native
    candidate_live = all(abs(arms[name]["full"]["candidate_ce_damage"]) >= 1e-4 for name in IDS)
    strong_null = bool(not pred_a or native_effect.norm() <= 1e-6
                       or abs(float((partner - native).mean())) < 1e-4 or not candidate_live
                       or (spans["removal"] < .01 and spans["composition"] < .01)
                       or any(structured[metric]["accuracy"] < .50 for metric in ("removal", "composition"))
                       or any(wave_correlation[metric] < 0 for metric in ("removal", "composition")))

    bundle = {"schema": "simplicity_vocabulary_complete_candidate_bundle_v1",
              "native_ce": native.half(), "native_ko_ce": native_ko.half(),
              "partner_ce": partner.half(), "arms": bundle_arms}
    torch.save(bundle, BUNDLE)
    result = {
        "status": "complete", "rung": 454,
        "claim_level": "prospective_complete_vocabulary_candidate_teaching_label_generation",
        "sealed_opened": False, "candidate_ids": list(IDS), "candidate_prices": PRICES,
        "program_scope": "exact_input_embedding_candidate_output_vocabulary_map",
        "native_hidden_replay_max_abs": hidden_replay_max, "native_ce_replay_max_abs": replay_ce_max,
        "capture_counts": {"attention_mean_positions": mean_n, "knockout_calls": knockout_calls,
                           "partner_calls": partner_calls},
        "sparse_selection_receipts": selection_receipts, "arms": arms,
        "structured_rank_edges": [list(edge) for edge in rank_edges()],
        "structured_rank_accuracy": structured, "wave_structured_rank_accuracy": wave_structured,
        "wave_continuous_pearson": wave_correlation, "error_spans": spans,
        "bundle": {"path": str(BUNDLE), "sha256": sha256(BUNDLE)},
        'pred_a_complete_instrument': pred_a,
        'pred_b_structured_rank_edges_order': pred_b,
        'pred_c_labels_vary': pred_c,
        'pred_d_wave_reliability': pred_d,
        "strong_null_complete_family_unusable": strong_null,
        "next_step": ("count_vocabulary_family_and_fit_consequence_predictors"
                      if pred_a and pred_b and pred_c and pred_d and not strong_null
                      else "do_not_count_vocabulary_family_add_new_teaching_family"),
        "new_deployed_values": 0, "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
