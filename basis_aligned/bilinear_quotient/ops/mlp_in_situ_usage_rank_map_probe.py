#!/usr/bin/env python3
"""How many dimensions does each MLP block actually USE on real text? In-situ rank map, all 18 blocks (CPU forward).

# BQGATE: EXPERIMENT
# pred_a_instrument
# pred_b_no_low_rank_mlp_output_in_situ
# pred_c_weight_map_predicts_in_situ_usage

Exact full-model CPU forward (tt_model semantics) on the frozen copy-induction v2 row caches; centred covariances of
each block's MLP output write, product state and attention write at sampled positions; effective ranks.
Preregistration: polynomial_causal/MLP_IN_SITU_USAGE_RANK_MAP_PROBE_PREREGISTRATION.md
"""
from __future__ import annotations
import hashlib, json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from receipt import dump

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
PREREG = POLY / "MLP_IN_SITU_USAGE_RANK_MAP_PROBE_PREREGISTRATION.md"
SNAP = Path("/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240")
BLOB = SNAP / "pytorch_model.bin"
NAT = ROOT / ".rowcache_terminal_copy_induction_v2/fit_natural.pt"
CODE = ROOT / ".rowcache_terminal_copy_induction_v2/ood_code.pt"
WEIGHTMAP = ROOT / "all_mlp_operator_family_rank_results.json"
OUT = ROOT / "mlp_in_situ_usage_rank_map_probe_results.json"
HASHES = {
    PREREG: "fb63d0d7253dbff1b1ae6e450e460a95d5c80f1b4c0b91c9fe78f42747e344dd",
    BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
    NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1",
    CODE: "6cf514e75dfd03399f223a9ba5f6ebe5f4b1315bcb839a515e1c19e7b5474bd9",
    WEIGHTMAP: "e237ca67f30d9a6aca5f5ce52c6aae6258dce2c0530471252f3dbbd7b180b9c4",
}
RUNG = "mlp_in_situ_usage_rank_map_probe"
D, NH, HD, T, NL, V = 1152, 9, 128, 257, 18, 50304
POS = torch.tensor(list(range(1, 256, 4)))       # 64 sampled positions per doc
DOCS_PER_CHUNK = 16
# 2026-09-03 19:57Z ops fix: honour the lane's cap (bqrunner2 exports OMP_NUM_THREADS=4). Before this line grabbed every core, so
# every probe importing this module ran 16 threads on lane 2 (measured 592-1102% CPU, box load 30 on 16 cores).
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS") or 0) or max(1, os.cpu_count() or 1))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_hashes():
    for p, e in HASHES.items():
        if not p.is_file() or sha256(p) != e:
            raise RuntimeError(f"frozen hash mismatch: {p}")


def load_model():
    sys.path.insert(0, "/workspace/tensor_language")
    import jacclust.tt_model as TT
    cfg = json.load(open(SNAP / "config.json")); cfg.pop("step", None)
    m = TT.GPT(TT.GPTConfig(**cfg)).float().eval()
    sd = torch.load(BLOB, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    m.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def rope(Tn):
    inv = 1.0 / (10000 ** (torch.arange(0, HD, 2).float() / HD))
    fr = torch.outer(torch.arange(Tn).float(), inv)
    return fr.cos().bfloat16().float()[None, :, None, :], fr.sin().bfloat16().float()[None, :, None, :]


def rot(x, c, s):
    d = x.shape[-1] // 2
    return torch.cat([x[..., :d] * c + x[..., d:] * s, -x[..., :d] * s + x[..., d:] * c], -1)


@torch.no_grad()
def manual_forward(m, idx, collect=None):
    """tt_model semantics. collect: callable(l, a_write, xhat, g, m_write) on sampled positions."""
    B, Tn = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope(Tn)
    mask = torch.tril(torch.ones(Tn, Tn, dtype=torch.bool))
    for l, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn; h = F.rms_norm(x, (D,))
        def pr(lin):
            return rot(F.rms_norm(lin(h).view(B, Tn, NH, HD), (HD,)), cos, sin)
        v = a.c_v(h).view(B, Tn, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        pat = (torch.einsum("bqhd,bkhd->bhqk", pr(a.c_q), pr(a.c_k)) / HD) * (torch.einsum("bqhd,bkhd->bhqk", pr(a.c_q2), pr(a.c_k2)) / HD)
        pat = pat.masked_fill(~mask, 0.0)
        aw = a.c_proj(torch.einsum("bhqk,bkhd->bqhd", pat, v).reshape(B, Tn, D))
        x = x + aw
        xhat = F.rms_norm(x, (D,)); mlp = blk.mlp
        g = mlp.Left(xhat) * mlp.Right(xhat)
        mw = mlp.Down(g)
        if collect is not None:
            collect(l, aw[:, POS].reshape(-1, D), xhat[:, POS].reshape(-1, D), g[:, POS].reshape(-1, 4 * D), mw[:, POS].reshape(-1, D))
        x = x + mw + mlp.Down_bias
    logits = 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)
    return logits


class Acc:
    def __init__(self, n):
        self.S = torch.zeros(n, n); self.mu = torch.zeros(n); self.cnt = 0; self.e = 0.0
    def add(self, X):
        self.S += X.T @ X; self.mu += X.sum(0); self.cnt += X.shape[0]; self.e += float(X.pow(2).sum())
    def cov(self):
        mu = self.mu / self.cnt
        return (self.S / self.cnt - torch.outer(mu, mu)).double(), self.e / self.cnt


def spectrum(C):
    ev = torch.linalg.eigvalsh(C)
    mx = float(ev.max()); psd = float(ev.min() / mx)
    ev = ev.clamp_min(0); pr = ev / ev.sum()
    eff = float(torch.exp(-(pr[pr > 0] * pr[pr > 0].log()).sum()))
    cs = torch.cumsum(pr.flip(0), 0)
    r90 = int((cs < 0.90).sum().item()) + 1
    return {"eff_rank": eff, "rank_90": r90, "top1": float(pr.max()), "psd_min_over_max": psd}


def spearman(a, b):
    ra = torch.argsort(torch.argsort(torch.tensor(a))).double(); rb = torch.argsort(torch.argsort(torch.tensor(b))).double()
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()))


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed", "rung": RUNG, "out": str(OUT)})); return
    check_hashes()
    m = load_model()
    nat = torch.load(NAT, map_location="cpu"); nat = (nat["rows"] if isinstance(nat, dict) else nat).long()
    code = torch.load(CODE, map_location="cpu"); code = (code["rows"] if isinstance(code, dict) else code).long()
    # instrument: manual forward vs the model's own module, CE on 4 natural docs
    idx = nat[:4, :T - 1]; tgt = nat[:4, 1:T]
    lg = manual_forward(m, idx)
    ce_manual = float(F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1)))
    ce_module = float(m(idx.contiguous(), tgt.contiguous()))
    per = {}
    for name, rows in (("natural", nat), ("code", code)):
        acc = {l: {"a": Acc(D), "m": Acc(D), "g": Acc(4 * D)} for l in range(NL)}
        ce_sum, ce_n = 0.0, 0
        def collect(l, aw, xhat, g, mw):
            acc[l]["a"].add(aw); acc[l]["m"].add(mw); acc[l]["g"].add(g)
        for i in range(0, rows.shape[0], DOCS_PER_CHUNK):
            idx = rows[i:i + DOCS_PER_CHUNK, :T - 1]; tgt = rows[i:i + DOCS_PER_CHUNK, 1:T]
            lg = manual_forward(m, idx, collect)
            ce_sum += float(F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction="sum")); ce_n += tgt.numel()
        blocks = []
        for l in range(NL):
            Ca, ea = acc[l]["a"].cov(); Cm, em = acc[l]["m"].cov(); Cg, eg = acc[l]["g"].cov()
            blocks.append({"block": l, "mlp_write": spectrum(Cm), "product_state": spectrum(Cg), "attn_write": spectrum(Ca),
                           "mean_energy": {"attn_write": ea, "mlp_write": em}, "attn_over_mlp_write_energy": ea / em})
        per[name] = {"n_samples": acc[0]["m"].cnt, "model_ce_nat": ce_sum / ce_n, "blocks": blocks}
        del acc
    wm = json.load(open(WEIGHTMAP))["per_block"]
    weight_ranks = [next(b["eff_rank_entropy"] for b in wm if b["block"] == l) for l in range(NL)]
    nat_m = [b["mlp_write"]["eff_rank"] for b in per["natural"]["blocks"]]
    rho_s = spearman(weight_ranks, nat_m)
    psd_ok = all(b[k]["psd_min_over_max"] >= -1e-6 for c in per for b in per[c]["blocks"] for k in ("mlp_write", "product_state", "attn_write"))
    pred_a = bool(abs(ce_manual - ce_module) <= 1e-4 and all(per[c]["n_samples"] >= 12000 for c in per) and psd_ok)
    pred_b = bool(pred_a and min(nat_m) >= 100)
    pred_c = bool(pred_a and rho_s >= 0.5)
    strong_null = bool(not (pred_a and pred_b and pred_c))
    if not pred_a:
        verdict = "instrument_invalid"
    elif pred_b and pred_c:
        verdict = "no_low_rank_mlp_write_in_situ_and_weight_map_orders_usage"
    elif pred_b:
        verdict = "no_low_rank_mlp_write_in_situ_but_weight_map_does_not_order_usage"
    elif pred_c:
        verdict = "in_situ_low_rank_block_exists_weight_map_orders_usage"
    else:
        verdict = "in_situ_low_rank_block_exists_weight_map_does_not_order_usage"
    result = {
        "status": "complete", "rung": RUNG, "owner_lane": "claude_parallel_probe",
        "claim_level": "in_situ_activation_rank_map_descriptive_no_circuit_claim",
        "source_hashes": {str(k): v for k, v in HASHES.items()},
        "instrument": {"ce_manual_4docs": ce_manual, "ce_module_4docs": ce_module, "abs_diff": abs(ce_manual - ce_module)},
        "per_corpus": per, "weight_map_eff_rank_by_block": weight_ranks, "natural_mlp_write_eff_rank_by_block": nat_m,
        "spearman_weight_map_vs_in_situ_mlp_write": rho_s, "min_natural_mlp_write_eff_rank": min(nat_m),
        "bars": {"ce_tol": 1e-4, "min_samples": 12000, "min_eff_rank": 100, "spearman_min": 0.5, "nulls": {"eff_rank": 50, "spearman": 0.0}},
        'pred_a_instrument': pred_a,
        'pred_b_no_low_rank_mlp_output_in_situ': pred_b,
        'pred_c_weight_map_predicts_in_situ_usage': pred_c,
        "strong_null": strong_null, "verdict": verdict,
        "execution_price": {"full_model_forwards_cpu_docs": int(nat.shape[0] + code.shape[0] + 8), "gpu_forwards": 0, "backwards": 0,
                            "deployed_parameters": 0, "cpu_only": True},
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({"verdict": verdict, "strong_null": strong_null, "instrument": result["instrument"],
                      "min_eff_rank": min(nat_m), "spearman": rho_s, "nat_mlp_write_eff": [round(x) for x in nat_m],
                      **{k: v for k, v in result.items() if k.startswith("pred_")}, "runtime_s": result["runtime_s"]}, indent=1))


if __name__ == "__main__":
    main()
