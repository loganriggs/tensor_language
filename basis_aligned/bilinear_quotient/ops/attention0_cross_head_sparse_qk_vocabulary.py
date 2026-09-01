"""RUNG 426 -- GLOBAL CROSS-HEAD SPARSE Q/K TOKEN VOCABULARY.

Rung 418 tested pairwise complete subspace sharing, not the user's proposed
global sparse dictionary.  The old layer-0 QK MDL program established the
strong comparator: 18 independent dictionaries over exact folded [q|k]
rows beat SVD, while a global hard token partition and within-head
cross-branch regrouping failed.  Its cross-head shared-ATOM item remained
open.  This rung tests it without treating architectural heads as the true
basis.

Exact object: concatenate all 18 (head, score-branch) rows [q|k], each 256
wide, into X[token] in R^4608.  Atoms/encoders fit token IDs mod5 != 4;
mod5 == 4 is SELECT.  All arms use 512 signed-top-k atoms and the same
coordinate MSE recipe.

Arms
----
I72: 18 independent 512x256 dictionaries, k4 each = 72 stored nonzeros/token.
G54: one global 512x4608 dictionary, one k54 code/token (25% fewer codes).
G72: the same global learned family decoded at k72 (equal code count to I72).
P54: independently row-permuted entry slices within FIT/SELECT, global k54.
D54: G54 codes with atom identities independently permuted per entry slice.

All deployed decoder/bias/coefficient tensors are physically FP16 and sparse
indices uint16.  Literal raw tensor bills (no entropy-coded indices):
I72 = G72 = 19,201,824 bytes; G54 = 15,583,320 bytes, 18.8446% lower.
Native layer-0 Q/K/Q2/K2 maps are not used by sparse score generation.

Frozen predictions
------------------
pred_a_exact_physical_instrument:
    exact fold <=1e-10; token split exact/disjoint; each learned loss falls
    >=20%; artifact shapes/dtypes and byte bills exact; candidate logits are
    invariant to zeroing all native layer-0 QK maps at relative sq <=1e-12.
pred_b_same_token_sparse_sharing:
    G54 SELECT balanced factor FVU <=.90*P54; median global-atom head
    participation rank >=3; >=25% atoms have head participation rank >=3.
pred_c_composed_equal_or_lower_price:
    G72 complete-pattern rel-sq <=1.10*I72 and CE damage <=I72+.002 nat;
    G54 pattern rel-sq <=1.35*I72, full attention0-write rel-sq <=1.50*I72,
    and CE damage <=I72+.005 nat (CE ADDED ABOVE NATIVE; LOWER IS BETTER).
pred_d_atom_coupling_specific:
    D54 pattern and full-write errors are each >=1.25*G54, and D54 CE damage
    is >=G54+.01 nat.

Strong null: A failure; G54 has <2% FVU advantage over P54; G72 pattern
error >=1.25*I72 or CE >I72+.01; G54 full-write error >=2*I72; or D54 is
within 2% of G54 on pattern or full-write error.  A full pass is a physical
sparse-generator identification screen, not adoption: fresh/OOD documents,
62 behaviors, frontier composition, and signed interventions remain.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language")
BQ = ROOT / "basis_aligned/bilinear_quotient"
POLY = ROOT / "basis_aligned/polynomial_causal"
QK = ROOT / "basis_aligned/qk_mdl"
OPS = BQ / "ops"
OUT = BQ / "attention0_cross_head_sparse_qk_vocabulary_results.json"
BUNDLE = BQ / "attention0_cross_head_sparse_qk_vocabulary_bundle.pt"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
OV_BASE = OPS / "attention0_ov_downstream_codebook.py"

VOCAB = 50_257
N_HEAD = 9
N_BRANCH = 2
N_ENTRY = N_HEAD * N_BRANCH
HD = 128
ENTRY_DIM = 2 * HD
GLOBAL_DIM = N_ENTRY * ENTRY_DIM
N_ATOM = 512
K_PRIVATE = 4
K_GLOBAL = 54
K_EQUAL = N_ENTRY * K_PRIVATE
STEPS = 1_500
BATCH = 128
LR = 3e-3
DOC_BATCH = 4
POSITIONS = tuple(range(16, 241, 16))
OFFSETS = (1, 2, 4, 8, 16, 32, 64, 128)
PAIR_COUNT = 32_768
ARMS = ("I72", "G54", "G72", "D54")
CONSUMERS = ("mlp0", "q1", "k1", "q2", "k2", "fresh_v")

DECODER_BYTES = N_ATOM * GLOBAL_DIM * 2
BIAS_BYTES = GLOBAL_DIM * 2
I72_BYTES = DECODER_BYTES + BIAS_BYTES + VOCAB * K_EQUAL * 4
G72_BYTES = DECODER_BYTES + BIAS_BYTES + VOCAB * K_EQUAL * 4
G54_BYTES = DECODER_BYTES + BIAS_BYTES + VOCAB * K_GLOBAL * 4


def _unit_rms(value: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(value, (value.shape[-1],))


def _entries_from_factors(factors: dict[int, tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    rows = []
    for head in range(N_HEAD):
        for branch in range(1, N_BRANCH + 1):
            query, key = factors[branch]
            rows.append(torch.cat([query[:VOCAB, head], key[:VOCAB, head]], -1))
    return torch.stack(rows, 1).float().contiguous()


def _quantized_decoder(raw: torch.Tensor) -> torch.Tensor:
    normalized = raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return normalized.half().float()


def _train_global(x: torch.Tensor, fit_ids: torch.Tensor, k: int, seed: int) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial_ids = fit_ids[torch.randperm(len(fit_ids), generator=generator)[:N_ATOM].to(x.device)]
    decoder = x[initial_ids].clone()
    decoder /= decoder.norm(dim=1, keepdim=True).clamp_min(1e-8)
    encoder = decoder.clone()
    bias = x[fit_ids].mean(0)
    params = [torch.nn.Parameter(value) for value in (decoder, encoder, bias)]
    decoder_p, encoder_p, bias_p = params
    optimizer = torch.optim.Adam(params, lr=LR)
    losses = []
    gradient_max = 0.0
    for _step in range(STEPS):
        pick = torch.randint(len(fit_ids), (BATCH,), generator=generator).to(x.device)
        batch = x[fit_ids[pick]]
        dictionary = decoder_p / decoder_p.norm(dim=1, keepdim=True).clamp_min(1e-8)
        code = (batch - bias_p) @ encoder_p.T
        support = code.abs().topk(k, dim=1).indices
        coefficient = code.gather(1, support)
        reconstruction = bias_p + (
            coefficient[..., None] * dictionary[support]).sum(1)
        loss = (reconstruction - batch).square().mean()
        optimizer.zero_grad()
        loss.backward()
        gradient_max = max(gradient_max, max(float(p.grad.abs().max()) for p in params))
        optimizer.step()
        losses.append(float(loss.detach()))
    return {
        "decoder": (decoder_p / decoder_p.norm(dim=1, keepdim=True).clamp_min(1e-8)).detach(),
        "encoder": encoder_p.detach(),
        "bias": bias_p.detach(),
        "initial_loss": sum(losses[:50]) / 50,
        "final_loss": sum(losses[-50:]) / 50,
        "gradient_max": gradient_max,
    }


def _train_independent(x: torch.Tensor, fit_ids: torch.Tensor, seed: int) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    choices = []
    for _entry in range(N_ENTRY):
        choices.append(fit_ids[torch.randperm(
            len(fit_ids), generator=generator)[:N_ATOM].to(x.device)])
    choices = torch.stack(choices)
    decoder = torch.stack([x[choices[e], e] for e in range(N_ENTRY)])
    decoder /= decoder.norm(dim=2, keepdim=True).clamp_min(1e-8)
    encoder = decoder.clone()
    bias = x[fit_ids].mean(0)
    params = [torch.nn.Parameter(value) for value in (decoder, encoder, bias)]
    decoder_p, encoder_p, bias_p = params
    optimizer = torch.optim.Adam(params, lr=LR)
    losses = []
    gradient_max = 0.0
    entry_offset = (torch.arange(N_ENTRY, device=x.device) * N_ATOM)[None, :, None]
    for _step in range(STEPS):
        pick = torch.randint(len(fit_ids), (BATCH,), generator=generator).to(x.device)
        batch = x[fit_ids[pick]]
        dictionary = decoder_p / decoder_p.norm(dim=2, keepdim=True).clamp_min(1e-8)
        code = torch.einsum("bed,end->ben", batch - bias_p, encoder_p)
        support = code.abs().topk(K_PRIVATE, dim=2).indices
        coefficient = code.gather(2, support)
        selected = dictionary.reshape(N_ENTRY * N_ATOM, ENTRY_DIM)[
            support + entry_offset]
        reconstruction = bias_p + (coefficient[..., None] * selected).sum(2)
        loss = (reconstruction - batch).square().mean()
        optimizer.zero_grad()
        loss.backward()
        gradient_max = max(gradient_max, max(float(p.grad.abs().max()) for p in params))
        optimizer.step()
        losses.append(float(loss.detach()))
    return {
        "decoder": (decoder_p / decoder_p.norm(dim=2, keepdim=True).clamp_min(1e-8)).detach(),
        "encoder": encoder_p.detach(),
        "bias": bias_p.detach(),
        "initial_loss": sum(losses[:50]) / 50,
        "final_loss": sum(losses[-50:]) / 50,
        "gradient_max": gradient_max,
    }


def _encode_global(x: torch.Tensor, model: dict, k: int,
                   decoder_override: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # A control override is already a materialized FP16 decoder. Renormalizing
    # after per-entry atom derangement would alter its decoder-slice marginals.
    decoder = (_quantized_decoder(model["decoder"])
               if decoder_override is None else decoder_override.half().float())
    bias = model["bias"].half().float()
    reconstructions, indices, coefficients = [], [], []
    for start in range(0, len(x), 512):
        batch = x[start:start + 512]
        code = (batch - model["bias"]) @ model["encoder"].T
        support = code.abs().topk(k, dim=1).indices
        coefficient = code.gather(1, support).half()
        reconstruction = bias + (
            coefficient.float()[..., None] * decoder[support]).sum(1)
        reconstructions.append(reconstruction.reshape(-1, N_ENTRY, ENTRY_DIM))
        indices.append(support.to(torch.uint16).cpu())
        coefficients.append(coefficient.cpu())
    return torch.cat(reconstructions), torch.cat(indices), torch.cat(coefficients)


def _encode_independent(x: torch.Tensor, model: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    decoder = _quantized_decoder(model["decoder"])
    bias = model["bias"].half().float()
    entry_offset = (torch.arange(N_ENTRY, device=x.device) * N_ATOM)[None, :, None]
    reconstructions, indices, coefficients = [], [], []
    for start in range(0, len(x), 512):
        batch = x[start:start + 512]
        code = torch.einsum("bed,end->ben", batch - model["bias"], model["encoder"])
        support = code.abs().topk(K_PRIVATE, dim=2).indices
        coefficient = code.gather(2, support).half()
        selected = decoder.reshape(N_ENTRY * N_ATOM, ENTRY_DIM)[support + entry_offset]
        reconstruction = bias + (coefficient.float()[..., None] * selected).sum(2)
        reconstructions.append(reconstruction)
        indices.append(support.to(torch.uint16).cpu())
        coefficients.append(coefficient.cpu())
    return torch.cat(reconstructions), torch.cat(indices), torch.cat(coefficients)


def _tables(entries: torch.Tensor) -> dict[str, torch.Tensor]:
    result = {name: torch.empty(
        VOCAB, N_HEAD, HD, device=entries.device) for name in ("q1", "k1", "q2", "k2")}
    for head in range(N_HEAD):
        for branch in range(N_BRANCH):
            entry = entries[:, head * N_BRANCH + branch]
            result[f"q{branch + 1}"][:, head] = _unit_rms(entry[:, :HD])
            result[f"k{branch + 1}"][:, head] = _unit_rms(entry[:, HD:])
    return result


def _balanced_fvu(target: torch.Tensor, predicted: torch.Tensor,
                  select_ids: torch.Tensor) -> tuple[float, list[float]]:
    values = []
    for entry in range(N_ENTRY):
        truth = target[select_ids, entry].double()
        estimate = predicted[select_ids, entry].double()
        numerator = (estimate - truth).square().sum()
        denominator = (truth - truth.mean(0)).square().sum().clamp_min(1e-30)
        values.append(float(numerator / denominator))
    return sum(values) / len(values), values


def _score_metrics(target: dict[str, torch.Tensor], candidates: dict[str, dict[str, torch.Tensor]],
                   select_ids: torch.Tensor, rope_tables, apply_rot) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(426_777)
    query = select_ids[torch.randint(
        len(select_ids), (PAIR_COUNT,), generator=generator).to(select_ids.device)]
    key = select_ids[torch.randint(
        len(select_ids), (PAIR_COUNT,), generator=generator).to(select_ids.device)]
    cos, sin = rope_tables(max(OFFSETS) + 1, HD, select_ids.device, torch.float32, "bf16")

    def scores(tables):
        branch_values = []
        for branch in (1, 2):
            q = tables[f"q{branch}"][query]
            k = tables[f"k{branch}"][key]
            columns = []
            for offset in OFFSETS:
                columns.append((apply_rot(q, cos[offset], sin[offset]) * k).sum(-1) / HD)
            branch_values.append(torch.stack(columns, 1))
        return branch_values

    truth = scores(target)
    denominator_branch = [float(value.double().square().sum()) for value in truth]
    truth_product = truth[0] * truth[1]
    denominator_product = float(truth_product.double().square().sum())
    report = {}
    for name, tables in candidates.items():
        estimate = scores(tables)
        report[name] = {
            "branch_relative_squared_error": [
                float((estimate[b].double() - truth[b].double()).square().sum())
                / denominator_branch[b] for b in range(2)],
            "complete_pattern_relative_squared_error": float(
                (estimate[0].double() * estimate[1].double()
                 - truth_product.double()).square().sum()) / denominator_product,
        }
    return report


def _attention_from_tables(model, tokens: torch.Tensor, state: torch.Tensor,
                           tables: dict[str, torch.Tensor], scores_from_factors) -> torch.Tensor:
    block = model.transformer.h[0]
    score1 = scores_from_factors(tables["q1"], tables["k1"], tokens, HD, "bf16")
    score2 = scores_from_factors(tables["q2"], tables["k2"], tokens, HD, "bf16")
    causal = torch.tril(torch.ones(
        tokens.shape[1], tokens.shape[1], device=tokens.device, dtype=torch.bool))
    pattern = (score1 * score2).masked_fill(~causal, 0)
    value = block.attn.c_v(state).view(*tokens.shape, N_HEAD, HD)
    mixed = torch.einsum("bhqk,bkhd->bqhd", pattern, value).reshape(
        tokens.shape[0], tokens.shape[1], -1)
    return block.attn.c_proj(mixed)


def _document_metrics(model, rows: torch.Tensor, candidates: dict[str, dict[str, torch.Tensor]],
                      base, edge_mod, scoring, scores_from_factors) -> tuple[dict, float]:
    write_num = {arm: 0.0 for arm in ARMS}
    write_den = 0.0
    consumer_num = {arm: {name: 0.0 for name in CONSUMERS} for arm in ARMS}
    consumer_den = {name: 0.0 for name in CONSUMERS}
    ce = {"native": [], **{arm: [] for arm in ARMS}}
    block0, block1 = model.transformer.h[:2]
    replay_relative = 0.0
    for start in range(0, len(rows), DOC_BATCH):
        batch = rows[start:start + DOC_BATCH].to("cuda")
        tokens = batch[:, :-1]
        x0 = F.rms_norm(model.transformer.wte(tokens), (GLOBAL_DIM // 4,))
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * x0
        state = F.rms_norm(token_base, (GLOBAL_DIM // 4,))
        native_attention, first_value = block0.attn(state, None)
        zero_attention = torch.zeros_like(native_attention)
        native_fields = base._consumer_fields(
            block0, block1, x0, token_base, native_attention)
        zero_fields = base._consumer_fields(
            block0, block1, x0, token_base, zero_attention)
        write_den += float(native_attention[:, POSITIONS].double().square().sum())
        for name in CONSUMERS:
            response = (native_fields[name].float().flatten(2)[:, POSITIONS]
                        - zero_fields[name].float().flatten(2)[:, POSITIONS])
            consumer_den[name] += float(response.double().square().sum())
        native_logits = edge_mod._suffix_logits(
            model, tokens, x0, token_base, native_attention, first_value)
        for row in range(len(batch)):
            ce["native"].append(scoring.document_mean_ce(
                native_logits[row], batch[row, 1:]))
        for arm in ARMS:
            changed = _attention_from_tables(
                model, tokens, state, candidates[arm], scores_from_factors)
            write_num[arm] += float(
                (changed[:, POSITIONS].double()
                 - native_attention[:, POSITIONS].double()).square().sum())
            changed_fields = base._consumer_fields(
                block0, block1, x0, token_base, changed)
            for name in CONSUMERS:
                error = (changed_fields[name].float().flatten(2)[:, POSITIONS]
                         - native_fields[name].float().flatten(2)[:, POSITIONS])
                consumer_num[arm][name] += float(error.double().square().sum())
            logits = edge_mod._suffix_logits(
                model, tokens, x0, token_base, changed, first_value)
            for row in range(len(batch)):
                ce[arm].append(scoring.document_mean_ce(logits[row], batch[row, 1:]))

        # One-batch physical independence check: sparse generation is upstream
        # of this suffix and never reads native layer-0 Q/K maps.
        if start == 0:
            before = edge_mod._suffix_logits(
                model, tokens, x0, token_base,
                _attention_from_tables(model, tokens, state, candidates["G54"], scores_from_factors),
                first_value)
            saved = [getattr(block0.attn, name).weight.detach().clone()
                     for name in ("c_q", "c_k", "c_q2", "c_k2")]
            for name in ("c_q", "c_k", "c_q2", "c_k2"):
                getattr(block0.attn, name).weight.zero_()
            after = edge_mod._suffix_logits(
                model, tokens, x0, token_base,
                _attention_from_tables(model, tokens, state, candidates["G54"], scores_from_factors),
                first_value)
            for name, weight in zip(("c_q", "c_k", "c_q2", "c_k2"), saved):
                getattr(block0.attn, name).weight.copy_(weight)
            replay_relative = float(
                (after.double() - before.double()).square().sum()
                / before.double().square().sum().clamp_min(1e-30))

    ce_values = {name: torch.stack(values).double().cpu() for name, values in ce.items()}
    ce_report = {}
    for name, values in ce_values.items():
        ce_report[name] = {
            "mean": float(values.mean()),
            "damage": float(values.mean() - ce_values["native"].mean()),
            "wave_damage": [
                float(values[:48].mean() - ce_values["native"][:48].mean()),
                float(values[48:].mean() - ce_values["native"][48:].mean()),
            ],
        }
    return {
        "full_attention0_write_relative_squared_error": {
            arm: write_num[arm] / write_den for arm in ARMS},
        "consumer_relative_squared_error": {
            arm: {name: consumer_num[arm][name] / consumer_den[name]
                  for name in CONSUMERS} for arm in ARMS},
        "mean_consumer_relative_squared_error": {
            arm: sum(consumer_num[arm][name] / consumer_den[name]
                     for name in CONSUMERS) / len(CONSUMERS) for arm in ARMS},
        "ce": ce_report,
    }, replay_relative


def _permuted_entries(x: torch.Tensor, fit_ids: torch.Tensor,
                      select_ids: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
    result = torch.empty_like(x)
    hashes = []
    for entry in range(N_ENTRY):
        generator = torch.Generator(device="cpu").manual_seed(426_100 + entry)
        fit_perm = fit_ids[torch.randperm(len(fit_ids), generator=generator).to(x.device)]
        select_perm = select_ids[torch.randperm(len(select_ids), generator=generator).to(x.device)]
        result[fit_ids, entry] = x[fit_perm, entry]
        result[select_ids, entry] = x[select_perm, entry]
        hashes.append(hashlib.sha256(torch.cat(
            [fit_perm.cpu(), select_perm.cpu()]).numpy().tobytes()).hexdigest())
    return result, hashes


def _deranged_decoder(model: dict) -> tuple[torch.Tensor, list[str]]:
    # Start from the deployed decoder. Entry-wise permutations then preserve
    # every slice byte-for-byte while destroying cross-entry atom identity.
    decoder = _quantized_decoder(model["decoder"]).reshape(
        N_ATOM, N_ENTRY, ENTRY_DIM).clone()
    hashes = []
    for entry in range(N_ENTRY):
        generator = torch.Generator(device="cpu").manual_seed(426_900 + entry)
        permutation = torch.randperm(N_ATOM, generator=generator).to(decoder.device)
        decoder[:, entry] = decoder[permutation, entry]
        hashes.append(hashlib.sha256(permutation.cpu().numpy().tobytes()).hexdigest())
    return decoder.reshape(N_ATOM, GLOBAL_DIM), hashes


def _head_participation(decoder: torch.Tensor) -> dict:
    atom = decoder.reshape(N_ATOM, N_HEAD, N_BRANCH, ENTRY_DIM).double()
    energy = atom.square().sum((2, 3))
    probability = energy / energy.sum(1, keepdim=True).clamp_min(1e-30)
    rank = 1 / probability.square().sum(1).clamp_min(1e-30)
    return {
        "mean": float(rank.mean()),
        "median": float(rank.median()),
        "fraction_ge_3": float((rank >= 3).double().mean()),
        "fraction_ge_5": float((rank >= 5).double().mean()),
        "min": float(rank.min()),
        "max": float(rank.max()),
    }


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert VOCAB == 50_257 and GLOBAL_DIM == 4_608 and N_ENTRY == 18
        assert K_EQUAL == 72 and K_GLOBAL == 54 and N_ATOM == 512
        assert I72_BYTES == G72_BYTES == 19_201_824
        assert G54_BYTES == 15_583_320
        assert abs(1 - G54_BYTES / I72_BYTES - .1884458476444738) < 1e-12
        assert ROWS_RECEIPT.exists() and OV_BASE.exists()
        print("ATTENTION0 CROSS-HEAD SPARSE QK | dry run: I72/G54/G72/P54/D54, literal bytes")
        return

    started = time.time()
    sys.path.insert(0, str(QK))
    sys.path.insert(0, str(POLY))
    sys.path.insert(0, str(OPS))
    from tier2_model import load_elriggs, reference_forward, rope_tables, apply_rot
    from tier2_folding import branch_factors, scores_from_factors
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent
    import scoring

    edge_path = OPS / "attention0_realized_edge_block_term.py"
    edge_spec = importlib.util.spec_from_file_location("edge_mod", edge_path)
    edge_mod = importlib.util.module_from_spec(edge_spec)
    edge_spec.loader.exec_module(edge_mod)
    base_spec = importlib.util.spec_from_file_location("ov_base", OV_BASE)
    base = importlib.util.module_from_spec(base_spec)
    base_spec.loader.exec_module(base)

    device = torch.device("cuda")
    receipt = json.loads(ROWS_RECEIPT.read_text())
    select_rows = rows_parent.load_role(receipt["entries"]["SELECT"])

    # Exact float64 fold gate, then release before the training model is loaded.
    exact_model, _config = load_elriggs("bilin18", device=device, dtype=torch.float64)
    exact_factors = {branch: branch_factors(exact_model, branch, dtype=torch.float64)
                     for branch in (1, 2)}
    captured = {}

    def capture(layer, score1, score2):
        if layer == 0:
            captured[1] = score1.detach()
            captured[2] = score2.detach()
        return score1, score2

    gate_tokens = select_rows[:1, :-1].to(device)
    reference_forward(exact_model, gate_tokens, "bf16", capture)
    fold_errors = {}
    for branch in (1, 2):
        folded = scores_from_factors(
            *exact_factors[branch], gate_tokens, HD, table_dtype="bf16")
        fold_errors[str(branch)] = float((folded - captured[branch]).abs().max())
    del exact_model, exact_factors, captured, folded
    torch.cuda.empty_cache()

    model, config = load_elriggs("bilin18", device=device, dtype=torch.float32)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    factors = {branch: branch_factors(model, branch, dtype=torch.float32)
               for branch in (1, 2)}
    target_entries = _entries_from_factors(factors)
    x_global = target_entries.reshape(VOCAB, GLOBAL_DIM)
    token_ids = torch.arange(VOCAB, device=device)
    fit_ids = token_ids[token_ids.remainder(5) != 4]
    select_ids = token_ids[token_ids.remainder(5) == 4]
    permuted_entries, permutation_hashes = _permuted_entries(
        target_entries, fit_ids, select_ids)

    print("TRAIN I72 independent dictionaries", flush=True)
    independent = _train_independent(target_entries, fit_ids, seed=426)
    print("TRAIN G54 global dictionary", flush=True)
    global_model = _train_global(x_global, fit_ids, K_GLOBAL, seed=427)
    print("TRAIN P54 permuted-token global control", flush=True)
    permuted_model = _train_global(
        permuted_entries.reshape(VOCAB, GLOBAL_DIM), fit_ids, K_GLOBAL, seed=428)

    print("ENCODE physical FP16/uint16 artifacts", flush=True)
    rec_i72, idx_i72, coef_i72 = _encode_independent(target_entries, independent)
    rec_g54, idx_g54, coef_g54 = _encode_global(x_global, global_model, K_GLOBAL)
    rec_g72, idx_g72, coef_g72 = _encode_global(x_global, global_model, K_EQUAL)
    rec_p54, _idx_p54, _coef_p54 = _encode_global(
        permuted_entries.reshape(VOCAB, GLOBAL_DIM), permuted_model, K_GLOBAL)
    deranged_decoder, derangement_hashes = _deranged_decoder(global_model)
    rec_d54, _idx_d54, _coef_d54 = _encode_global(
        x_global, global_model, K_GLOBAL, decoder_override=deranged_decoder)

    fvu = {}
    fvu["I72"], fvu_i_entries = _balanced_fvu(target_entries, rec_i72, select_ids)
    fvu["G54"], fvu_g_entries = _balanced_fvu(target_entries, rec_g54, select_ids)
    fvu["G72"], fvu_g72_entries = _balanced_fvu(target_entries, rec_g72, select_ids)
    fvu["D54"], fvu_d_entries = _balanced_fvu(target_entries, rec_d54, select_ids)
    fvu["P54"], fvu_p_entries = _balanced_fvu(
        permuted_entries, rec_p54, select_ids)

    target_tables = {
        "q1": factors[1][0][:VOCAB], "k1": factors[1][1][:VOCAB],
        "q2": factors[2][0][:VOCAB], "k2": factors[2][1][:VOCAB],
    }
    candidate_tables = {
        "I72": _tables(rec_i72), "G54": _tables(rec_g54),
        "G72": _tables(rec_g72), "D54": _tables(rec_d54),
    }
    with torch.no_grad():
        score_metrics = _score_metrics(
            target_tables, candidate_tables, select_ids, rope_tables, apply_rot)
        document_metrics, no_native_qk_relative = _document_metrics(
            model, select_rows, candidate_tables, base, edge_mod,
            scoring, scores_from_factors)

    # Materialized shipped objects.  Encoders and native QK maps are absent.
    bundle = {
        "schema": "attention0_cross_head_sparse_qk_vocabulary_bundle_v1",
        "global_decoder_fp16": _quantized_decoder(global_model["decoder"]).half().cpu(),
        "global_bias_fp16": global_model["bias"].half().cpu(),
        "global54_indices_uint16": idx_g54,
        "global54_coefficients_fp16": coef_g54,
        "global72_indices_uint16": idx_g72,
        "global72_coefficients_fp16": coef_g72,
        "independent_decoder_fp16": _quantized_decoder(independent["decoder"]).half().cpu(),
        "independent_bias_fp16": independent["bias"].half().cpu(),
        "independent_indices_uint16": idx_i72,
        "independent_coefficients_fp16": coef_i72,
    }
    torch.save(bundle, BUNDLE)
    bundle_sha = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()

    loss_reports = {
        "I72": {key: independent[key] for key in
                ("initial_loss", "final_loss", "gradient_max")},
        "G54": {key: global_model[key] for key in
                ("initial_loss", "final_loss", "gradient_max")},
        "P54": {key: permuted_model[key] for key in
                ("initial_loss", "final_loss", "gradient_max")},
    }
    loss_decrease = {
        name: 1 - values["final_loss"] / values["initial_loss"]
        for name, values in loss_reports.items()}
    participation = _head_participation(
        _quantized_decoder(global_model["decoder"]))
    g54_advantage = (fvu["P54"] - fvu["G54"]) / fvu["P54"]
    pattern = {arm: score_metrics[arm]["complete_pattern_relative_squared_error"]
               for arm in ARMS}
    write = document_metrics["full_attention0_write_relative_squared_error"]
    ce = document_metrics["ce"]

    artifact_checks = {
        "global_decoder_fp16": bundle["global_decoder_fp16"].dtype == torch.float16
            and tuple(bundle["global_decoder_fp16"].shape) == (N_ATOM, GLOBAL_DIM),
        "global_bias_fp16": bundle["global_bias_fp16"].dtype == torch.float16
            and tuple(bundle["global_bias_fp16"].shape) == (GLOBAL_DIM,),
        "global54_indices_uint16": bundle["global54_indices_uint16"].dtype == torch.uint16
            and tuple(bundle["global54_indices_uint16"].shape) == (VOCAB, K_GLOBAL),
        "global54_coefficients_fp16": bundle["global54_coefficients_fp16"].dtype == torch.float16
            and tuple(bundle["global54_coefficients_fp16"].shape) == (VOCAB, K_GLOBAL),
        "independent_decoder_fp16": bundle["independent_decoder_fp16"].dtype == torch.float16
            and tuple(bundle["independent_decoder_fp16"].shape) == (
                N_ENTRY, N_ATOM, ENTRY_DIM),
        "independent_indices_uint16": bundle["independent_indices_uint16"].dtype == torch.uint16
            and tuple(bundle["independent_indices_uint16"].shape) == (
                VOCAB, N_ENTRY, K_PRIVATE),
        "literal_bills": I72_BYTES == G72_BYTES == 19_201_824
            and G54_BYTES == 15_583_320,
    }
    pred_a = (
        max(fold_errors.values()) <= 1e-10
        and len(fit_ids) == 40_206 and len(select_ids) == 10_051
        and not bool(torch.isin(fit_ids, select_ids).any())
        and all(value >= .20 for value in loss_decrease.values())
        and all(values["gradient_max"] > 0 for values in loss_reports.values())
        and all(artifact_checks.values())
        and no_native_qk_relative <= 1e-12)
    pred_b = (
        g54_advantage >= .10
        and participation["median"] >= 3
        and participation["fraction_ge_3"] >= .25)
    pred_c = (
        pattern["G72"] <= 1.10 * pattern["I72"]
        and ce["G72"]["damage"] <= ce["I72"]["damage"] + .002
        and pattern["G54"] <= 1.35 * pattern["I72"]
        and write["G54"] <= 1.50 * write["I72"]
        and ce["G54"]["damage"] <= ce["I72"]["damage"] + .005)
    pred_d = (
        pattern["D54"] >= 1.25 * pattern["G54"]
        and write["D54"] >= 1.25 * write["G54"]
        and ce["D54"]["damage"] >= ce["G54"]["damage"] + .01)
    strong_null = (
        not pred_a
        or g54_advantage < .02
        or pattern["G72"] >= 1.25 * pattern["I72"]
        or ce["G72"]["damage"] > ce["I72"]["damage"] + .01
        or write["G54"] >= 2 * write["I72"]
        or pattern["D54"] <= 1.02 * pattern["G54"]
        or write["D54"] <= 1.02 * write["G54"])

    result = {
        "status": "attention0_cross_head_sparse_qk_vocabulary_complete",
        "rung": 426,
        "claim_level": "heldout_physical_sparse_qk_generator_screen_not_adoption",
        "convention": "CE added above native; lower is better",
        "definition": {
            "entry": "one head and one QK branch, exact folded [q|k] row in R256",
            "global_row": "concatenation of 18 entries in R4608 for one token",
            "I72": "18 independent 512-atom k4 signed codes",
            "G54": "one global 512-atom k54 signed code",
            "G72": "same global learned family decoded at k72",
            "P54": "entry-wise token-row permutation within FIT/SELECT",
            "D54": "G54 code with per-entry decoder atom identities permuted",
        },
        "tokens": {"real": VOCAB, "FIT": len(fit_ids), "SELECT": len(select_ids),
                   "FINAL_opened": 0},
        "training": {"steps": STEPS, "batch": BATCH, "lr": LR,
                     "losses": loss_reports, "loss_decrease_fraction": loss_decrease},
        "instrument": {
            "fold_max_abs_by_branch": fold_errors,
            "no_native_layer0_qk_logits_relative_squared_error": no_native_qk_relative,
            "artifact_checks": artifact_checks,
            "permutation_hashes": permutation_hashes,
            "derangement_hashes": derangement_hashes,
        },
        "literal_raw_tensor_bytes": {
            "I72": I72_BYTES, "G72": G72_BYTES, "G54": G54_BYTES,
            "G54_saving_vs_I72": I72_BYTES - G54_BYTES,
            "G54_fraction_smaller": 1 - G54_BYTES / I72_BYTES,
            "native_layer0_qk_values_removed": 4 * (GLOBAL_DIM // 4) ** 2,
            "native_layer0_qk_retained": False,
        },
        "bundle": {"path": str(BUNDLE), "file_sha256": bundle_sha,
                   "contains_multiple_arms": True,
                   "arm_bills_are_logical_subartifact_bills_not_whole_bundle_size": True},
        "select_balanced_factor_fvu": fvu,
        "select_factor_fvu_by_entry": {
            "I72": fvu_i_entries, "G54": fvu_g_entries,
            "G72": fvu_g72_entries, "P54": fvu_p_entries,
            "D54": fvu_d_entries},
        "G54_factor_advantage_over_P54": g54_advantage,
        "global_atom_head_participation": participation,
        "select_random_pair_score_metrics": score_metrics,
        "select_document_metrics": document_metrics,
        'pred_a_exact_physical_instrument': bool(pred_a),
        'pred_b_same_token_sparse_sharing': bool(pred_b),
        'pred_c_composed_equal_or_lower_price': bool(pred_c),
        'pred_d_atom_coupling_specific': bool(pred_d),
        "strong_null_no_efficient_cross_head_sparse_qk_vocabulary": bool(strong_null),
        "compression_or_adoption_licensed": False,
        "next_step": (
            "product_metric_physical_sparse_vs_continuous_vs_rank_matched_price"
            if pred_a and pred_b and pred_c and pred_d and not strong_null
            else "descriptive_sparse_atoms_only_continue_continuous_generator"
            if pred_a and pred_b and not pred_c
            else "generic_global_capacity_not_shared_vocabulary"
            if pred_a and pred_c and not pred_b
            else "close_cross_head_sparse_atoms_continue_continuous_generator"
            if pred_a else "instrument_repair_only"),
        "FINAL_opened": 0,
        "config": config,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"], "loss_decrease": loss_decrease,
        "fvu": fvu, "G54_advantage_over_P54": g54_advantage,
        "participation": participation, "score_metrics": score_metrics,
        "document_metrics": document_metrics,
        "pred_a": pred_a, "pred_b": pred_b, "pred_c": pred_c, "pred_d": pred_d,
        "strong_null": strong_null, "next_step": result["next_step"],
        "runtime_s": result["runtime_s"],
    }, indent=2), flush=True)
    print("ATTENTION0 CROSS-HEAD SPARSE QK DONE", flush=True)


if __name__ == "__main__":
    main()
