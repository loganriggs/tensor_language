"""RUNG 301 -- ERROR-RELATIVE SIGNED CAUSAL-RESPONSE RANK FOR MLP0.

Motivation
----------
The earlier observability Gramian found a stable but large (r90 677--825)
first-order suffix-sensitive subspace at every stream site.  That closes a small
global observability quotient, but leaves open the result's stated next object:
the response directions relative to errors a concrete compressed program makes.

For every native MLP0 output y_i on 32 frozen FineWeb fit documents, backpropagate
the complete next-token loss through the native suffix and capture g_i=dL/dy_i.
With centered output z_i=y_i-mean(y), define the signed response operator

    K = mean_i z_i^T g_i,       S = (K + K^T)/2.

For a unit direction q, q^T S q is exactly the mean first-order loss response to
deleting that centered output component.  Rank response coordinates by absolute
eigenvalue of S.  This is not the gradient Gramian and not output PCA: it measures
where MLP0's actual error-bearing writes align with what the suffix reads.

Executable candidates
---------------------
At ranks r in {64,128,256}, replace the Down output by

    y_hat = mean_y + (y-mean_y) Q_r Q_r^T.

This is executable without native Down: factor Q_r^T Down and Q_r, with adjusted
bias.  Left/Right remain native.  Literal MLP0 price is

    2*4608*1152 + r*(4608+1152) + 1152 scalars.

Compare at identical rank and price with (i) activation-PCA output directions and
(ii) the top left singular directions of native Down.  Fit and evaluation are
disjoint; evaluation uses 16 FineWeb skip1200 rows and the same 16 frozen
WikiText-2 rows/fingerprint as rungs 300/300B.  No evaluation label selects rank.

Stability control
-----------------
Build signed-response and activation-PCA bases independently on the two 16-row
fit halves.  Report projector overlap ||Qa^T Qb||_F^2/r; the random expectation
at r=128 is 128/1152=0.1111.

Frozen predictions
------------------
pred_a_response_beats_noncausal_bases:
    At some r<=128, response-basis CE damage is at least 25% smaller than BOTH
    PCA and weight-SVD damage on BOTH FineWeb and WikiText (controls must have
    nonnegative damage).
pred_b_response_rank_is_predictive_and_priced:
    Response r128 damage <=0.10 on FineWeb and <=0.12 on WikiText while its
    literal MLP0 price is <=72% of native MLP0.
pred_c_response_basis_is_split_stable:
    Rank-128 split projector overlap >=0.60, more than five times random.

Null: response fails to beat both controls at every r<=128, or rank-128 split
overlap <=0.25.  This is a two-corpus screen; even a pass needs full census,
certificates, composition, and finite signed interventions before adoption.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mlp0_signed_response_rank_screen_results.json"
DEV = "cuda"
D = 1152
H = 4608
RANKS = (64, 128, 256)
FIT_ROWS = 32
EVAL_ROWS = 16


def _load_rows(path: Path, n: int) -> torch.Tensor:
    value = torch.load(path, map_location="cpu")
    rows = value["rows"] if isinstance(value, dict) else value
    assert rows.ndim == 2 and rows.shape[1] >= 257
    return rows[:n, :257].long().contiguous()


def _wikitext_rows(n: int = EVAL_ROWS, width: int = 257, skip: int = 1024) -> tuple[torch.Tensor, str]:
    from datasets import load_dataset
    import tiktoken

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())
    tokens = tiktoken.get_encoding("gpt2").encode_ordinary(text)
    stop = skip + n * width
    assert len(tokens) >= stop
    return torch.tensor(tokens[skip:stop], dtype=torch.long).reshape(n, width), str(dataset._fingerprint)


def _manual_logits(model: torch.nn.Module, index: torch.Tensor) -> torch.Tensor:
    x = F.rms_norm(model.transformer.wte(index), (D,))
    x0 = x
    value0 = None
    for block in model.transformer.h:
        x, value0 = block(x, value0, x0)
    return 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30.0)


def _fit_moments(model: torch.nn.Module, rows: torch.Tensor) -> dict[str, torch.Tensor]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    output_rows = []
    gradient_rows = []
    capture: dict[str, torch.Tensor] = {}

    def leaf_hook(_module, _args, output):
        leaf = output.detach().requires_grad_(True)
        capture["leaf"] = leaf
        return leaf

    handle = model.transformer.h[0].mlp.register_forward_hook(leaf_hook)
    try:
        for start in range(0, len(rows), 2):
            batch = rows[start:start + 2]
            index = batch[:, :-1].to(DEV)
            target = batch[:, 1:].to(DEV)
            logits = _manual_logits(model, index)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1), reduction="sum")
            loss.backward()
            leaf = capture["leaf"]
            assert leaf.grad is not None and leaf.shape == leaf.grad.shape
            output_rows.append(leaf.detach().float().cpu().reshape(-1, D))
            gradient_rows.append(leaf.grad.detach().float().cpu().reshape(-1, D))
            capture.clear()
    finally:
        handle.remove()
    output = torch.cat(output_rows).double()
    gradient = torch.cat(gradient_rows).double()
    assert output.shape == gradient.shape == (len(rows) * 256, D)
    return {"output": output, "gradient": gradient}


def _bases(moment: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    output = moment["output"]
    gradient = moment["gradient"]
    mean = output.mean(0)
    centered = output - mean
    response = (centered.T @ gradient) / len(output)
    response = 0.5 * (response + response.T)
    response_values, response_vectors = torch.linalg.eigh(response)
    response_order = torch.argsort(response_values.abs(), descending=True)
    covariance = centered.T @ centered / len(output)
    pca_values, pca_vectors = torch.linalg.eigh(covariance)
    pca_order = torch.argsort(pca_values, descending=True)
    return {
        "mean": mean.float(),
        "response_values": response_values[response_order].float(),
        "response_vectors": response_vectors[:, response_order].float(),
        "pca_values": pca_values[pca_order].float(),
        "pca_vectors": pca_vectors[:, pca_order].float(),
    }


def _overlap(left: torch.Tensor, right: torch.Tensor, rank: int) -> float:
    return float((left[:, :rank].T @ right[:, :rank]).square().sum() / rank)


@torch.no_grad()
def _score(
    model: torch.nn.Module,
    rows: torch.Tensor,
    basis: torch.Tensor | None,
    mean: torch.Tensor | None,
) -> float:
    state: dict[str, torch.Tensor] = {}
    handle = None
    if basis is not None:
        q = basis.to(DEV)
        mu = mean.to(DEV)

        def project_hook(_module, _args, output):
            centered = output.float() - mu
            return (mu + (centered @ q) @ q.T).to(output.dtype)

        handle = model.transformer.h[0].mlp.register_forward_hook(project_hook)
    loss_sum = 0.0
    count = 0
    try:
        for start in range(0, len(rows), 2):
            batch = rows[start:start + 2]
            index = batch[:, :-1].to(DEV)
            target = batch[:, 1:].to(DEV)
            logits = _manual_logits(model, index)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                target.reshape(-1),
                reduction="sum",
            )
            loss_sum += float(loss)
            count += target.numel()
    finally:
        if handle is not None:
            handle.remove()
    return loss_sum / count


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert (ROOT / ".rowcache/fineweb_n480_skip80.pt").exists()
        assert (ROOT / ".rowcache/fineweb_n96_skip1200.pt").exists()
        print("MLP0 SIGNED RESPONSE RANK SCREEN | dry run: rows, bases, prices, and bars valid")
        return
    started = time.time()
    sys.path.insert(0, "/workspace/tensor_language/basis_aligned/qk_mdl")
    from tier2_model import load_elriggs

    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D
    fit = _load_rows(ROOT / ".rowcache/fineweb_n480_skip80.pt", FIT_ROWS)
    fineweb = _load_rows(ROOT / ".rowcache/fineweb_n96_skip1200.pt", EVAL_ROWS)
    wikitext, fingerprint = _wikitext_rows()
    moment = _fit_moments(model, fit)
    full = _bases(moment)
    half_a = _bases({"output": moment["output"][:16 * 256], "gradient": moment["gradient"][:16 * 256]})
    half_b = _bases({"output": moment["output"][16 * 256:], "gradient": moment["gradient"][16 * 256:]})
    response_overlap_128 = _overlap(half_a["response_vectors"], half_b["response_vectors"], 128)
    pca_overlap_128 = _overlap(half_a["pca_vectors"], half_b["pca_vectors"], 128)

    down = model.transformer.h[0].mlp.Down.weight.detach().float()
    weight_vectors = torch.linalg.svd(down, full_matrices=False).U.cpu()
    native_fine = _score(model, fineweb, None, None)
    native_wiki = _score(model, wikitext, None, None)
    native_price = H * 3 * D + D
    arms = {}
    for rank in RANKS:
        factor_price = 2 * H * D + rank * (H + D) + D
        row = {"rank": rank, "literal_mlp0_scalars": factor_price,
               "storage_fraction_native_mlp0": factor_price / native_price}
        for name, vectors in (
            ("response", full["response_vectors"]),
            ("activation_pca", full["pca_vectors"]),
            ("weight_svd", weight_vectors),
        ):
            basis = vectors[:, :rank]
            fine_ce = _score(model, fineweb, basis, full["mean"])
            wiki_ce = _score(model, wikitext, basis, full["mean"])
            row[name] = {
                "fineweb_ce": fine_ce,
                "fineweb_damage": fine_ce - native_fine,
                "wikitext_ce": wiki_ce,
                "wikitext_damage": wiki_ce - native_wiki,
            }
        arms[str(rank)] = row
        print(
            f"r={rank} price={factor_price/native_price:.3f} | FW resp/pca/wsvd "
            f"{row['response']['fineweb_damage']:+.4f}/{row['activation_pca']['fineweb_damage']:+.4f}/"
            f"{row['weight_svd']['fineweb_damage']:+.4f} | WT "
            f"{row['response']['wikitext_damage']:+.4f}/{row['activation_pca']['wikitext_damage']:+.4f}/"
            f"{row['weight_svd']['wikitext_damage']:+.4f}", flush=True,
        )

    eligible = [arms[str(rank)] for rank in RANKS if rank <= 128]
    pred_a = any(
        all(arm[control][corpus + "_damage"] >= 0 for control in ("activation_pca", "weight_svd")
            for corpus in ("fineweb", "wikitext"))
        and all(
            arm["response"][corpus + "_damage"]
            <= 0.75 * min(arm["activation_pca"][corpus + "_damage"], arm["weight_svd"][corpus + "_damage"])
            for corpus in ("fineweb", "wikitext")
        )
        for arm in eligible
    )
    arm128 = arms["128"]
    pred_b = bool(
        arm128["response"]["fineweb_damage"] <= 0.10
        and arm128["response"]["wikitext_damage"] <= 0.12
        and arm128["storage_fraction_native_mlp0"] <= 0.72
    )
    pred_c = bool(response_overlap_128 >= 0.60)
    no_control_win = all(
        not all(
            arm["response"][corpus + "_damage"]
            < min(arm["activation_pca"][corpus + "_damage"], arm["weight_svd"][corpus + "_damage"])
            for corpus in ("fineweb", "wikitext")
        )
        for arm in eligible
    )
    null = bool(no_control_win or response_overlap_128 <= 0.25)
    response_abs = full["response_values"].abs()
    result = {
        "status": "mlp0_signed_response_rank_screen_complete",
        "rung": 301,
        "claim_level": "empirical_signed_response_two_corpus_screen_only",
        "populations": {
            "fit_rows": len(fit),
            "fit_positions": int(moment["output"].shape[0]),
            "fineweb_eval_rows": len(fineweb),
            "wikitext_eval_rows": len(wikitext),
            "wikitext_fingerprint": fingerprint,
        },
        "native": {
            "mlp0_scalars": native_price,
            "fineweb_ce": native_fine,
            "wikitext_ce": native_wiki,
        },
        "response_geometry": {
            "rank128_abs_eigenvalue_fraction": float(response_abs[:128].sum() / response_abs.sum()),
            "rank128_split_projector_overlap": response_overlap_128,
            "rank128_random_overlap_expectation": 128 / D,
            "activation_pca_rank128_split_overlap": pca_overlap_128,
        },
        "arms": arms,
        'pred_a_response_beats_noncausal_bases': bool(pred_a),
        'pred_b_response_rank_is_predictive_and_priced': bool(pred_b),
        'pred_c_response_basis_is_split_stable': bool(pred_c),
        "null_no_stable_response_advantage": null,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "response_geometry": result["response_geometry"],
        "predicates": [pred_a, pred_b, pred_c],
        "null": null,
        "runtime_s": result["runtime_s"],
    }, indent=2), flush=True)
    print("MLP0 SIGNED RESPONSE RANK SCREEN DONE", flush=True)


if __name__ == "__main__":
    main()
