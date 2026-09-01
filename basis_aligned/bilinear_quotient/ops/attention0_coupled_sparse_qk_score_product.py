"""RUNG 430 -- COUPLED SPARSE QUERY/KEY SCORE-PRODUCT GENERATOR.

User-directed second section-14.3 experiment.  Rung426 established that one
global sparse [q|k] token code beats 18 independent codes at lower bytes, but
trained only factor reconstruction and missed one atom-identity specificity
bar.  This rung separates query and key codes at the identical total price,
then trains exact branch scores and their multiplicative two-branch product.

Arms (all 512 query + 512 key atoms)
------------------------------------
SQ54 factor-only k27+k27 baseline; SC54 branch-score fine-tune; CP54 complete
branch+product fine-tune; CP72 same learned family at k36+k36; PP54 matched
pair-label-permuted fine-tune; WH54 wrong branch2 head h+4 contraction.
CP54-R is an independent restart used only for atom/pair stability.

Literal FP16 decoder/bias/coeff + uint16 index bills:
QK54 = 15,583,320 bytes; QK72 = 19,201,824 bytes.  Encoders and native layer0
Q/K maps are absent from deployed execution.

Frozen predictions
------------------
A: fold<=1e-10; exact split; every learned loss decreases>=20%; artifact
   dtypes/shapes/bills exact; no-native-QK replay relsq<=1e-12.
B: CP54 SELECT product<=.80*SQ54 and <=.90*SC54; branch<=1.10*SC54;
   product<=.85*PP54; write<=.90*SQ54; CE<=SQ54+.002 nat.
C: CP54 product<=.90*r426G54(.6437466217802437); CP72<=.90*r426G72
   (.483746191580884); matched-price CE<=r426+.002; WH54 write>=1.25*CP54
   and CE>=CP54+.01.
D: active top64 atom pairs reconstruct branch scores at relsq<=.30; restart
   median decoder cosine>=.50 q/k; matched top256 pair Jaccard>=.20.

Strong null: A fail; CP54 product gain<5% vs SQ54; CP54 within2% of PP54;
CP72 product>=1.25*r426G72 or CE>r426G72+.01; or WH54 within2% of CP54
on write or CE.  A pass is physical-generator/semantic-candidate evidence,
not adoption; fresh/OOD, 62 behaviors, frontier composition, and signed
interventions remain.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import copy
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
OUT = BQ / "attention0_coupled_sparse_qk_score_product_results.json"
BUNDLE = BQ / "attention0_coupled_sparse_qk_score_product_bundle.pt"
R426 = BQ / "attention0_cross_head_sparse_qk_vocabulary_results.json"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
BASE_PATH = OPS / "attention0_cross_head_sparse_qk_vocabulary.py"
OV_BASE = OPS / "attention0_ov_downstream_codebook.py"

VOCAB = 50_257
N_HEAD = 9
N_BRANCH = 2
N_ENTRY = 18
HD = 128
SIDE_DIM = N_ENTRY * HD
N_ATOM = 512
K_SIDE = 27
K_SIDE_EQUAL = 36
WARM_STEPS = 1_200
FINE_STEPS = 1_000
BATCH = 128
LR = 3e-3
ANCHOR = .25
OFFSETS = (1, 2, 4, 8, 16, 32, 64, 128)
PAIR_SAMPLE = 1_024
ARMS = ("SQ54", "SC54", "CP54", "CP72", "PP54", "WH54")

DECODER_BYTES = 2 * N_ATOM * SIDE_DIM * 2
BIAS_BYTES = 2 * SIDE_DIM * 2
QK54_BYTES = DECODER_BYTES + BIAS_BYTES + VOCAB * (2 * K_SIDE) * 4
QK72_BYTES = DECODER_BYTES + BIAS_BYTES + VOCAB * (2 * K_SIDE_EQUAL) * 4


def _unit(value: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(value, (value.shape[-1],))


def _decoder(value: torch.Tensor) -> torch.Tensor:
    return (value / value.norm(dim=1, keepdim=True).clamp_min(1e-8))


def _split(entries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return entries[..., :HD].contiguous(), entries[..., HD:].contiguous()


def _initial_side(x: torch.Tensor, fit_ids: torch.Tensor,
                  generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = fit_ids[torch.randperm(
        len(fit_ids), generator=generator)[:N_ATOM].to(x.device)]
    flat = x.reshape(VOCAB, SIDE_DIM)
    decoder = _decoder(flat[ids].clone())
    return decoder, decoder.clone(), flat[fit_ids].mean(0)


def _live_decode(batch: torch.Tensor, decoder: torch.Tensor,
                 encoder: torch.Tensor, bias: torch.Tensor,
                 k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dictionary = _decoder(decoder)
    code = (batch - bias) @ encoder.T
    support = code.abs().topk(k, dim=1).indices
    coefficient = code.gather(1, support)
    reconstruction = bias + (
        coefficient[..., None] * dictionary[support]).sum(1)
    return reconstruction, support, coefficient


def _warm(q: torch.Tensor, k: torch.Tensor, fit_ids: torch.Tensor, seed: int) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    qd, qe, qb = _initial_side(q, fit_ids, generator)
    kd, ke, kb = _initial_side(k, fit_ids, generator)
    params = [torch.nn.Parameter(v) for v in (qd, qe, qb, kd, ke, kb)]
    optimizer = torch.optim.Adam(params, lr=LR)
    qflat, kflat = q.reshape(VOCAB, SIDE_DIM), k.reshape(VOCAB, SIDE_DIM)
    losses = []
    gradient_max = 0.0
    for _step in range(WARM_STEPS):
        pick = torch.randint(len(fit_ids), (BATCH,), generator=generator).to(q.device)
        ids = fit_ids[pick]
        qr, _, _ = _live_decode(qflat[ids], *params[:3], K_SIDE)
        kr, _, _ = _live_decode(kflat[ids], *params[3:], K_SIDE)
        loss = .5 * ((qr - qflat[ids]).square().mean()
                     + (kr - kflat[ids]).square().mean())
        optimizer.zero_grad()
        loss.backward()
        gradient_max = max(gradient_max, max(float(p.grad.abs().max()) for p in params))
        optimizer.step()
        losses.append(float(loss.detach()))
    names = ("q_decoder", "q_encoder", "q_bias", "k_decoder", "k_encoder", "k_bias")
    model = {name: value.detach() for name, value in zip(names, params)}
    model["q_decoder"] = _decoder(model["q_decoder"])
    model["k_decoder"] = _decoder(model["k_decoder"])
    model["initial_loss"] = sum(losses[:50]) / 50
    model["final_loss"] = sum(losses[-50:]) / 50
    model["gradient_max"] = gradient_max
    return model


def _fine(q: torch.Tensor, k: torch.Tensor, fit_ids: torch.Tensor, warm: dict,
          mode: str, seed: int, rope_tables, apply_rot,
          permuted_target: bool = False) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    names = ("q_decoder", "q_encoder", "q_bias", "k_decoder", "k_encoder", "k_bias")
    params = [torch.nn.Parameter(warm[name].clone()) for name in names]
    optimizer = torch.optim.Adam(params, lr=LR)
    qflat, kflat = q.reshape(VOCAB, SIDE_DIM), k.reshape(VOCAB, SIDE_DIM)
    cos, sin = rope_tables(max(OFFSETS) + 1, HD, q.device, torch.float32, "bf16")
    permutation = torch.arange(VOCAB, device=q.device)
    if permuted_target:
        shuffled = fit_ids[torch.randperm(len(fit_ids), generator=generator).to(q.device)]
        permutation[fit_ids] = shuffled
    losses = []
    gradient_max = 0.0
    for _step in range(FINE_STEPS):
        qi = fit_ids[torch.randint(
            len(fit_ids), (BATCH,), generator=generator).to(q.device)]
        ki = fit_ids[torch.randint(
            len(fit_ids), (BATCH,), generator=generator).to(q.device)]
        delta_index = torch.randint(
            len(OFFSETS), (BATCH,), generator=generator).to(q.device)
        delta = torch.tensor(OFFSETS, device=q.device)[delta_index]
        qr, _, _ = _live_decode(qflat[qi], *params[:3], K_SIDE)
        kr, _, _ = _live_decode(kflat[ki], *params[3:], K_SIDE)
        qr = _unit(qr.reshape(BATCH, N_ENTRY, HD))
        kr = _unit(kr.reshape(BATCH, N_ENTRY, HD))
        target_q = q[qi]
        target_k = k[permutation[ki]] if permuted_target else k[ki]
        c, s = cos[delta][:, None], sin[delta][:, None]
        estimated = (apply_rot(qr, c, s) * kr).sum(-1) / HD
        target = (apply_rot(target_q, c, s) * target_k).sum(-1) / HD
        branch_loss = (estimated - target).square().mean() / target.square().mean().clamp_min(1e-8)
        ep = estimated.reshape(BATCH, N_HEAD, N_BRANCH).prod(-1)
        tp = target.reshape(BATCH, N_HEAD, N_BRANCH).prod(-1)
        product_loss = (ep - tp).square().mean() / tp.square().mean().clamp_min(1e-8)
        anchor = .5 * (
            (qr - q[qi]).square().mean() + (kr - k[ki]).square().mean())
        loss = branch_loss + ANCHOR * anchor
        if mode == "product":
            loss = loss + product_loss
        optimizer.zero_grad()
        loss.backward()
        gradient_max = max(gradient_max, max(float(p.grad.abs().max()) for p in params))
        optimizer.step()
        losses.append(float(loss.detach()))
    model = {name: value.detach() for name, value in zip(names, params)}
    model["q_decoder"] = _decoder(model["q_decoder"])
    model["k_decoder"] = _decoder(model["k_decoder"])
    model["initial_loss"] = sum(losses[:50]) / 50
    model["final_loss"] = sum(losses[-50:]) / 50
    model["gradient_max"] = gradient_max
    return model


def _encode_side(flat: torch.Tensor, decoder: torch.Tensor, encoder: torch.Tensor,
                 bias: torch.Tensor, k: int):
    physical_decoder = _decoder(decoder).half().float()
    physical_bias = bias.half().float()
    reconstructions, indices, coefficients = [], [], []
    for start in range(0, VOCAB, 512):
        batch = flat[start:start + 512]
        code = (batch - bias) @ encoder.T
        support = code.abs().topk(k, dim=1).indices
        coefficient = code.gather(1, support).half()
        reconstruction = physical_bias + (
            coefficient.float()[..., None] * physical_decoder[support]).sum(1)
        reconstructions.append(_unit(reconstruction.reshape(-1, N_ENTRY, HD)))
        indices.append(support.to(torch.uint16).cpu())
        coefficients.append(coefficient.cpu())
    return (torch.cat(reconstructions), torch.cat(indices),
            torch.cat(coefficients), physical_decoder, physical_bias)


def _encode(q: torch.Tensor, k: torch.Tensor, model: dict, code_k: int) -> dict:
    qr = _encode_side(q.reshape(VOCAB, SIDE_DIM), model["q_decoder"],
                      model["q_encoder"], model["q_bias"], code_k)
    kr = _encode_side(k.reshape(VOCAB, SIDE_DIM), model["k_decoder"],
                      model["k_encoder"], model["k_bias"], code_k)
    return {"q": qr[0], "k": kr[0], "q_indices": qr[1], "q_coefficients": qr[2],
            "k_indices": kr[1], "k_coefficients": kr[2],
            "q_decoder": qr[3], "q_bias": qr[4],
            "k_decoder": kr[3], "k_bias": kr[4]}


def _tables(encoded: dict) -> dict[str, torch.Tensor]:
    result = {}
    for branch in range(N_BRANCH):
        indices = torch.arange(branch, N_ENTRY, N_BRANCH, device=encoded["q"].device)
        result[f"q{branch + 1}"] = encoded["q"][:, indices]
        result[f"k{branch + 1}"] = encoded["k"][:, indices]
    return result


def _factor_fvu(target: torch.Tensor, estimate: torch.Tensor,
                select_ids: torch.Tensor) -> float:
    values = []
    for entry in range(N_ENTRY):
        truth = target[select_ids, entry].double()
        pred = estimate[select_ids, entry].double()
        values.append(float((pred - truth).square().sum()
                            / (truth - truth.mean(0)).square().sum().clamp_min(1e-30)))
    return sum(values) / len(values)


def _checkpoint_hash(model: dict) -> str:
    digest = hashlib.sha256()
    for name in ("q_decoder", "q_encoder", "q_bias", "k_decoder", "k_encoder", "k_bias"):
        value = model[name].detach().float().cpu().contiguous()
        digest.update(name.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _pair_concentration(encoded: dict, select_ids: torch.Tensor,
                        rope_tables, apply_rot, seed: int) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    qi = select_ids[torch.randint(
        len(select_ids), (PAIR_SAMPLE,), generator=generator).to(select_ids.device)]
    ki = select_ids[torch.randint(
        len(select_ids), (PAIR_SAMPLE,), generator=generator).to(select_ids.device)]
    delta_i = torch.randint(len(OFFSETS), (PAIR_SAMPLE,), generator=generator)
    delta = torch.tensor(OFFSETS, device=select_ids.device)[delta_i.to(select_ids.device)]
    cos, sin = rope_tables(max(OFFSETS) + 1, HD, select_ids.device, torch.float32, "bf16")
    qidx = encoded["q_indices"][qi.cpu()].long().to(select_ids.device)
    kidx = encoded["k_indices"][ki.cpu()].long().to(select_ids.device)
    qcoef = encoded["q_coefficients"][qi.cpu()].float().to(select_ids.device)
    kcoef = encoded["k_coefficients"][ki.cpu()].float().to(select_ids.device)
    constant = torch.full((PAIR_SAMPLE, 1), N_ATOM, device=select_ids.device, dtype=torch.long)
    qidx = torch.cat([qidx, constant], 1)
    kidx = torch.cat([kidx, constant], 1)
    qcoef = torch.cat([qcoef, torch.ones(PAIR_SAMPLE, 1, device=select_ids.device)], 1)
    kcoef = torch.cat([kcoef, torch.ones(PAIR_SAMPLE, 1, device=select_ids.device)], 1)
    width = K_SIDE + 1
    numerators = {keep: 0.0 for keep in (32, 64, 128)}
    masses = {keep: 0.0 for keep in (32, 64, 128)}
    denominator = 0.0
    total_mass = 0.0
    pair_mass = torch.zeros((N_ATOM + 1) ** 2, dtype=torch.float64, device=select_ids.device)
    for entry in range(N_ENTRY):
        sl = slice(entry * HD, (entry + 1) * HD)
        qdict = torch.cat([encoded["q_decoder"][:, sl], encoded["q_bias"][None, sl]], 0)
        kdict = torch.cat([encoded["k_decoder"][:, sl], encoded["k_bias"][None, sl]], 0)
        for start in range(0, PAIR_SAMPLE, 128):
            stop = min(start + 128, PAIR_SAMPLE)
            qv = qdict[qidx[start:stop]]
            kv = kdict[kidx[start:stop]]
            qraw = (qv * qcoef[start:stop, :, None]).sum(1)
            kraw = (kv * kcoef[start:stop, :, None]).sum(1)
            qscale = qraw.square().mean(-1).add(
                torch.finfo(qraw.dtype).eps).rsqrt()
            kscale = kraw.square().mean(-1).add(
                torch.finfo(kraw.dtype).eps).rsqrt()
            c = cos[delta[start:stop]][:, None]
            s = sin[delta[start:stop]][:, None]
            qv = apply_rot(qv, c, s)
            contribution = torch.einsum("biu,bju->bij", qv, kv) / HD
            contribution = contribution * qcoef[start:stop, :, None] * kcoef[start:stop, None, :]
            contribution = contribution * qscale[:, None, None] * kscale[:, None, None]
            full = contribution.sum((1, 2))
            denominator += float(full.double().square().sum())
            absolute = contribution.abs()
            total_mass += float(absolute.double().sum())
            flat = contribution.flatten(1)
            for keep in numerators:
                chosen = flat.abs().topk(keep, dim=1).indices
                partial = flat.gather(1, chosen).sum(1)
                numerators[keep] += float((partial.double() - full.double()).square().sum())
                masses[keep] += float(flat.abs().gather(1, chosen).double().sum())
            ids = (qidx[start:stop, :, None] * (N_ATOM + 1)
                   + kidx[start:stop, None, :]).expand(-1, width, width)
            pair_mass.scatter_add_(0, ids.reshape(-1), absolute.double().reshape(-1))
    top_pairs = pair_mass.topk(256).indices.cpu().tolist()
    return {
        "relative_squared_error": {str(k): numerators[k] / max(denominator, 1e-30)
                                   for k in numerators},
        "absolute_mass_fraction": {str(k): masses[k] / max(total_mass, 1e-30)
                                   for k in masses},
        "top256_pair_ids": top_pairs,
    }


def _restart_stability(primary: dict, repeat: dict,
                       primary_pairs: list[int], repeat_pairs: list[int]) -> dict:
    from scipy.optimize import linear_sum_assignment

    def match(left, right):
        left = F.normalize(left.double(), dim=1)
        right = F.normalize(right.double(), dim=1)
        cosine = (left @ right.T).abs().cpu().numpy()
        rows, cols = linear_sum_assignment(-cosine)
        inverse = torch.empty(N_ATOM, dtype=torch.long)
        inverse[torch.tensor(cols)] = torch.tensor(rows)
        return float(torch.tensor(cosine[rows, cols]).median()), inverse

    qcos, qinverse = match(primary["q_decoder"], repeat["q_decoder"])
    kcos, kinverse = match(primary["k_decoder"], repeat["k_decoder"])

    def remap(pair):
        q, k = divmod(pair, N_ATOM + 1)
        q = N_ATOM if q == N_ATOM else int(qinverse[q])
        k = N_ATOM if k == N_ATOM else int(kinverse[k])
        return q * (N_ATOM + 1) + k

    left = set(primary_pairs)
    right = {remap(pair) for pair in repeat_pairs}
    return {"query_median_matched_abs_cosine": qcos,
            "key_median_matched_abs_cosine": kcos,
            "top256_pair_jaccard": len(left & right) / max(len(left | right), 1)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert SIDE_DIM == 2_304 and N_ENTRY == 18
        assert QK54_BYTES == 15_583_320 and QK72_BYTES == 19_201_824
        assert R426.exists() and ROWS_RECEIPT.exists() and BASE_PATH.exists()
        print("ATTENTION0 COUPLED SPARSE QK PRODUCT | dry run: SQ/SC/CP/PP/WH/restart")
        return

    started = time.time()
    sys.path[:0] = [str(QK), str(POLY), str(OPS)]
    from tier2_model import load_elriggs, reference_forward, rope_tables, apply_rot
    from tier2_folding import branch_factors, scores_from_factors
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent
    import scoring

    def load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    parent = load_module("sparse_parent", BASE_PATH)
    ov_base = load_module("ov_base", OV_BASE)
    edge_mod = load_module("edge_mod", OPS / "attention0_realized_edge_block_term.py")
    device = torch.device("cuda")
    receipt = json.loads(ROWS_RECEIPT.read_text())
    select_rows = rows_parent.load_role(receipt["entries"]["SELECT"])
    r426 = json.loads(R426.read_text())

    exact_model, _ = load_elriggs("bilin18", device=device, dtype=torch.float64)
    exact_factors = {b: branch_factors(exact_model, b, dtype=torch.float64) for b in (1, 2)}
    captured = {}

    def capture(layer, score1, score2):
        if layer == 0:
            captured[1], captured[2] = score1.detach(), score2.detach()
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
    factors = {b: branch_factors(model, b, dtype=torch.float32) for b in (1, 2)}
    entries = parent._entries_from_factors(factors)
    q, k = _split(entries)
    ids = torch.arange(VOCAB, device=device)
    fit_ids, select_ids = ids[ids.remainder(5) != 4], ids[ids.remainder(5) == 4]

    print("TRAIN SQ54 factor warm start", flush=True)
    warm = _warm(q, k, fit_ids, 428)
    warm_hash = _checkpoint_hash(warm)
    print("TRAIN SC54 branch-score objective", flush=True)
    score_model = _fine(q, k, fit_ids, warm, "score", 429, rope_tables, apply_rot)
    print("TRAIN CP54 complete-product objective", flush=True)
    product_model = _fine(q, k, fit_ids, warm, "product", 430, rope_tables, apply_rot)
    print("TRAIN PP54 pair-permuted control", flush=True)
    permuted_model = _fine(q, k, fit_ids, warm, "product", 431, rope_tables, apply_rot, True)
    print("TRAIN CP54-R independent restart", flush=True)
    repeat_warm = _warm(q, k, fit_ids, 432)
    repeat_model = _fine(q, k, fit_ids, repeat_warm, "product", 433, rope_tables, apply_rot)

    print("ENCODE physical token tables", flush=True)
    encoded = {
        "SQ54": _encode(q, k, warm, K_SIDE),
        "SC54": _encode(q, k, score_model, K_SIDE),
        "CP54": _encode(q, k, product_model, K_SIDE),
        "CP72": _encode(q, k, product_model, K_SIDE_EQUAL),
        "PP54": _encode(q, k, permuted_model, K_SIDE),
    }
    repeat_encoded = _encode(q, k, repeat_model, K_SIDE)
    tables = {name: _tables(value) for name, value in encoded.items()}
    tables["WH54"] = {name: value.clone() for name, value in tables["CP54"].items()}
    tables["WH54"]["q2"] = torch.roll(tables["WH54"]["q2"], -4, 1)
    tables["WH54"]["k2"] = torch.roll(tables["WH54"]["k2"], -4, 1)
    target_tables = {"q1": factors[1][0][:VOCAB], "k1": factors[1][1][:VOCAB],
                     "q2": factors[2][0][:VOCAB], "k2": factors[2][1][:VOCAB]}

    with torch.no_grad():
        score_metrics = parent._score_metrics(
            target_tables, tables, select_ids, rope_tables, apply_rot)
        parent.ARMS = ARMS
        tables["G54"] = tables["CP54"]
        document_metrics, no_native = parent._document_metrics(
            model, select_rows, tables, ov_base, edge_mod, scoring, scores_from_factors)
        del tables["G54"]
        concentration = _pair_concentration(
            encoded["CP54"], select_ids, rope_tables, apply_rot, 428_800)
        repeat_concentration = _pair_concentration(
            repeat_encoded, select_ids, rope_tables, apply_rot, 428_800)
        stability = _restart_stability(
            encoded["CP54"], repeat_encoded,
            concentration["top256_pair_ids"], repeat_concentration["top256_pair_ids"])

    fvu = {name: {"query": _factor_fvu(q, value["q"], select_ids),
                  "key": _factor_fvu(k, value["k"], select_ids)}
           for name, value in encoded.items()}
    bundle = {
        "schema": "attention0_coupled_sparse_qk_score_product_bundle_v1",
        "query_decoder_fp16": encoded["CP54"]["q_decoder"].half().cpu(),
        "query_bias_fp16": encoded["CP54"]["q_bias"].half().cpu(),
        "key_decoder_fp16": encoded["CP54"]["k_decoder"].half().cpu(),
        "key_bias_fp16": encoded["CP54"]["k_bias"].half().cpu(),
        "query54_indices_uint16": encoded["CP54"]["q_indices"],
        "query54_coefficients_fp16": encoded["CP54"]["q_coefficients"],
        "key54_indices_uint16": encoded["CP54"]["k_indices"],
        "key54_coefficients_fp16": encoded["CP54"]["k_coefficients"],
        "query72_indices_uint16": encoded["CP72"]["q_indices"],
        "query72_coefficients_fp16": encoded["CP72"]["q_coefficients"],
        "key72_indices_uint16": encoded["CP72"]["k_indices"],
        "key72_coefficients_fp16": encoded["CP72"]["k_coefficients"],
    }
    torch.save(bundle, BUNDLE)
    artifact_checks = {
        "query_decoder": bundle["query_decoder_fp16"].dtype == torch.float16
            and tuple(bundle["query_decoder_fp16"].shape) == (N_ATOM, SIDE_DIM),
        "key_decoder": bundle["key_decoder_fp16"].dtype == torch.float16
            and tuple(bundle["key_decoder_fp16"].shape) == (N_ATOM, SIDE_DIM),
        "query_bias": bundle["query_bias_fp16"].dtype == torch.float16
            and tuple(bundle["query_bias_fp16"].shape) == (SIDE_DIM,),
        "key_bias": bundle["key_bias_fp16"].dtype == torch.float16
            and tuple(bundle["key_bias_fp16"].shape) == (SIDE_DIM,),
        "query54_indices": bundle["query54_indices_uint16"].dtype == torch.uint16
            and tuple(bundle["query54_indices_uint16"].shape) == (VOCAB, K_SIDE),
        "query54_coefficients": bundle["query54_coefficients_fp16"].dtype == torch.float16
            and tuple(bundle["query54_coefficients_fp16"].shape) == (VOCAB, K_SIDE),
        "key54_indices": bundle["key54_indices_uint16"].dtype == torch.uint16
            and tuple(bundle["key54_indices_uint16"].shape) == (VOCAB, K_SIDE),
        "key54_coefficients": bundle["key54_coefficients_fp16"].dtype == torch.float16
            and tuple(bundle["key54_coefficients_fp16"].shape) == (VOCAB, K_SIDE),
        "query72_indices": bundle["query72_indices_uint16"].dtype == torch.uint16
            and tuple(bundle["query72_indices_uint16"].shape) == (VOCAB, K_SIDE_EQUAL),
        "query72_coefficients": bundle["query72_coefficients_fp16"].dtype == torch.float16
            and tuple(bundle["query72_coefficients_fp16"].shape) == (VOCAB, K_SIDE_EQUAL),
        "key72_indices": bundle["key72_indices_uint16"].dtype == torch.uint16
            and tuple(bundle["key72_indices_uint16"].shape) == (VOCAB, K_SIDE_EQUAL),
        "key72_coefficients": bundle["key72_coefficients_fp16"].dtype == torch.float16
            and tuple(bundle["key72_coefficients_fp16"].shape) == (VOCAB, K_SIDE_EQUAL),
        "bills": QK54_BYTES == 15_583_320 and QK72_BYTES == 19_201_824,
    }
    training = {"SQ54": warm, "SC54": score_model, "CP54": product_model,
                "PP54": permuted_model, "SQ54_R": repeat_warm, "CP54_R": repeat_model}
    loss_reports = {name: {key: value[key] for key in
                           ("initial_loss", "final_loss", "gradient_max")}
                    for name, value in training.items()}
    decreases = {name: 1 - value["final_loss"] / value["initial_loss"]
                 for name, value in loss_reports.items()}
    pattern = {name: score_metrics[name]["complete_pattern_relative_squared_error"]
               for name in ARMS}
    branch = {name: sum(score_metrics[name]["branch_relative_squared_error"]) / 2
              for name in ARMS}
    write = document_metrics["full_attention0_write_relative_squared_error"]
    ce = document_metrics["ce"]
    r426_g54_pattern = r426["select_random_pair_score_metrics"]["G54"][
        "complete_pattern_relative_squared_error"]
    r426_g72_pattern = r426["select_random_pair_score_metrics"]["G72"][
        "complete_pattern_relative_squared_error"]
    r426_g54_ce = r426["select_document_metrics"]["ce"]["G54"]["damage"]
    r426_g72_ce = r426["select_document_metrics"]["ce"]["G72"]["damage"]

    warm_hash_recheck = _checkpoint_hash(warm)
    pred_a = (max(fold_errors.values()) <= 1e-10
              and len(fit_ids) == 40_206 and len(select_ids) == 10_051
              and not bool(torch.isin(fit_ids, select_ids).any())
              and all(v >= .20 for v in decreases.values())
              and all(v["gradient_max"] > 0 for v in loss_reports.values())
              and all(artifact_checks.values()) and warm_hash == warm_hash_recheck
              and no_native <= 1e-12)
    pred_b = (pattern["CP54"] <= .80 * pattern["SQ54"]
              and pattern["CP54"] <= .90 * pattern["SC54"]
              and branch["CP54"] <= 1.10 * branch["SC54"]
              and pattern["CP54"] <= .85 * pattern["PP54"]
              and write["CP54"] <= .90 * write["SQ54"]
              and ce["CP54"]["damage"] <= ce["SQ54"]["damage"] + .002)
    pred_c = (pattern["CP54"] <= .90 * r426_g54_pattern
              and pattern["CP72"] <= .90 * r426_g72_pattern
              and ce["CP54"]["damage"] <= r426_g54_ce + .002
              and ce["CP72"]["damage"] <= r426_g72_ce + .002
              and write["WH54"] >= 1.25 * write["CP54"]
              and ce["WH54"]["damage"] >= ce["CP54"]["damage"] + .01)
    pred_d = (concentration["relative_squared_error"]["64"] <= .30
              and stability["query_median_matched_abs_cosine"] >= .50
              and stability["key_median_matched_abs_cosine"] >= .50
              and stability["top256_pair_jaccard"] >= .20)
    strong_null = (not pred_a
                   or pattern["CP54"] >= .95 * pattern["SQ54"]
                   or pattern["CP54"] >= .98 * pattern["PP54"]
                   or pattern["CP72"] >= 1.25 * r426_g72_pattern
                   or ce["CP72"]["damage"] > r426_g72_ce + .01
                   or write["WH54"] <= 1.02 * write["CP54"]
                   or abs(ce["WH54"]["damage"] - ce["CP54"]["damage"])
                   <= .02 * max(abs(ce["CP54"]["damage"]), 1e-8))

    result = {
        "status": "attention0_coupled_sparse_qk_score_product_complete",
        "rung": 430,
        "claim_level": "heldout_physical_generator_and_semantic_candidate_screen_not_adoption",
        "convention": "CE added above native; lower is better",
        "literal_raw_tensor_bytes": {"CP54": QK54_BYTES, "CP72": QK72_BYTES,
                                     "native_layer0_qk_retained": False},
        "instrument": {"fold_max_abs_by_branch": fold_errors,
                       "no_native_qk_logits_relative_squared_error": no_native,
                       "warm_checkpoint_sha256": warm_hash,
                       "warm_checkpoint_recheck_sha256": warm_hash_recheck,
                       "artifact_checks": artifact_checks},
        "training": {"warm_steps": WARM_STEPS, "fine_steps": FINE_STEPS,
                     "batch": BATCH, "anchor": ANCHOR,
                     "losses": loss_reports, "loss_decrease_fraction": decreases},
        "select_factor_fvu": fvu,
        "select_random_pair_score_metrics": score_metrics,
        "select_document_metrics": document_metrics,
        "active_atom_pair_concentration": concentration,
        "restart_atom_pair_concentration": repeat_concentration,
        "restart_stability": stability,
        "r426_anchors": {"G54_pattern": r426_g54_pattern,
                         "G72_pattern": r426_g72_pattern,
                         "G54_ce_damage": r426_g54_ce,
                         "G72_ce_damage": r426_g72_ce},
        "bundle": {"path": str(BUNDLE),
                   "file_sha256": hashlib.sha256(BUNDLE.read_bytes()).hexdigest()},
        'pred_a_valid_physical_instrument': bool(pred_a),
        'pred_b_product_coupling_learns_real_relation': bool(pred_b),
        'pred_c_improves_r426_at_matched_price': bool(pred_c),
        'pred_d_stable_sparse_compositions': bool(pred_d),
        "strong_null_no_efficient_coupled_sparse_composition": bool(strong_null),
        "compression_or_adoption_licensed": False,
        "next_step": ("fresh_ood_62_behaviors_then_continuous_comparison"
                      if pred_a and pred_b and pred_c and pred_d and not strong_null
                      else "useful_sparse_generator_nonunique_atoms"
                      if pred_a and pred_b and pred_c and not pred_d and not strong_null
                      else "close_sparse_composition_prioritize_continuous_generator"
                      if pred_a else "instrument_repair_only"),
        "FINAL_opened": 0,
        "config": config,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "loss_decrease": decreases,
                      "factor_fvu": fvu, "pattern": pattern, "branch": branch,
                      "write": write, "ce": ce, "concentration": concentration,
                      "stability": stability, "pred_a": pred_a, "pred_b": pred_b,
                      "pred_c": pred_c, "pred_d": pred_d, "strong_null": strong_null,
                      "next_step": result["next_step"], "runtime_s": result["runtime_s"]},
                     indent=2), flush=True)
    print("ATTENTION0 COUPLED SPARSE QK PRODUCT DONE", flush=True)


if __name__ == "__main__":
    main()
