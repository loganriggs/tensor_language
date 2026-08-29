# COVERAGE-SPLIT MLP16 IDENTITY AUDIT.
#
# Frozen before execution, 2026-08-29 08:20 UTC. DISCOVERY ONLY.
# S1898/S1900 count changed predictions at all positions, whereas S1899/S1901 compare
# tensors only where the current token is in the 16,110-token table. This successor
# directly partitions both changed predictions and live-output/compiled-row error by
# current-token coverage in the same restored MLP16 forward.
#
#   pred_a COVERAGE LOCALISES CHANGES: >=99% of changes are on uncovered tokens.
#   pred_b COVERED ROW IDENTITY: covered relative L2 error is <1e-5 on every role.
#   pred_c FALLBACK IS NONIDENTICAL: uncovered relative L2 error is >1e-3 on every role.
#   pred_d CONTROLS: coverage is 16,110 and total change counts reproduce S1898.
import json
import os
import time

import torch

from live_vs_table_output import (
    D, DEV, EVAL_SETS, FIT_ROWS, H, NCOV, RIDGE, T, V, W,
    forward_logits, load, m, row_hook,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "coverage_split_mlp16_identity_results.json")
SITE = ("mlp", 16)
EXPECTED = {"skip7000": 1321, "skip11000": 1350, "skip1200": 650}


@torch.no_grad()
def main():
    if os.path.exists(OUT):
        raise RuntimeError(f"refusing to overwrite {OUT}")
    started = time.time()
    fit = load(FIT_ROWS)
    seen_cpu = torch.zeros(V, dtype=torch.bool)
    seen_cpu[fit[:, :T].reshape(-1).long()] = True
    assert int(seen_cpu.sum()) == NCOV
    seen = seen_cpu.to(DEV)
    tk = seen_cpu.nonzero(as_tuple=True)[0].to(DEV)
    unc = (~seen).nonzero(as_tuple=True)[0]
    sites = [(kind, layer) for kind in ("mlp", "attn") for layer in range(18)]

    # Exact settled full-rank tables and rank-64 learned fallback rows.
    lpc = torch.zeros(NCOV, W, device=DEV)
    for i in range(0, NCOV, 256):
        token = tk[i:i + 256].unsqueeze(1)
        lpc[i:i + token.shape[0]] = torch.log_softmax(forward_logits(token)[:, 0].float(), -1)
    pcn = torch.softmax(lpc, -1)
    pcn = (pcn / pcn.norm(dim=-1, keepdim=True).clamp_min(1e-9)).half()
    del lpc
    nnrow = torch.zeros(V, dtype=torch.long, device=DEV)
    nnrow[tk] = torch.arange(NCOV, device=DEV)
    for i in range(0, unc.numel(), 512):
        u = unc[i:i + 512]
        p = torch.softmax(forward_logits(u.unsqueeze(1))[:, 0].float(), -1)
        p = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        nnrow[u] = (p.half() @ pcn.T).float().argmax(-1)
    del pcn

    tables = {site: torch.zeros(NCOV, D, device=DEV) for site in sites}
    captured = {}

    def table_capture(site):
        def hook(_module, _args, output):
            captured[site] = (output[0] if isinstance(output, tuple) else output)[:, 0].float()
            return None
        return hook

    for i in range(0, NCOV, 256):
        token = tk[i:i + 256].unsqueeze(1)
        forward_logits(token, [(site, table_capture(site)) for site in sites])
        for site in sites:
            tables[site][i:i + token.shape[0]] = captured[site]

    emb_cov = m.transformer.wte.weight.detach()[tk].float().double()
    gram = emb_cov.T @ emb_cov
    gram += RIDGE * torch.eye(D, device=DEV, dtype=torch.float64) * (NCOV / D)
    emb_unc = m.transformer.wte.weight.detach()[unc].float().double()
    full_rows = {}
    for site in sites:
        weights = torch.linalg.solve(gram, emb_cov.T @ tables[site].double())
        u, s, vh = torch.linalg.svd(weights, full_matrices=False)
        rank64 = (u[:, :64] * s[:64]) @ vh[:64]
        site_rows = torch.zeros(V, D, device=DEV)
        site_rows[tk] = tables[site]
        site_rows[unc] = (emb_unc @ rank64).float()
        full_rows[site] = site_rows
    print(f"COVERAGE SPLIT MLP16 | settled rows built in {time.time() - started:.1f}s", flush=True)

    all_hooks = [(site, row_hook(full_rows[site])) for site in sites]
    restored_hooks = [(site, row_hook(full_rows[site])) for site in sites if site != SITE]
    live = {}

    def capture_live(_module, _args, output):
        live["output"] = (output[0] if isinstance(output, tuple) else output).detach().float()
        return None

    results = {}
    for role, path, _ in EVAL_SETS:
        eval_rows = load(path)
        counts = {"covered": 0, "uncovered": 0,
                  "changed_covered": 0, "changed_uncovered": 0}
        sums = {"covered_num2": 0.0, "covered_den2": 0.0,
                "uncovered_num2": 0.0, "uncovered_den2": 0.0}
        maxima = {"covered": 0.0, "uncovered": 0.0}
        for i in range(0, eval_rows.shape[0], 8):
            idx = eval_rows[i:i + 8, :-1].to(DEV).contiguous()
            baseline = forward_logits(idx, all_hooks)[:, 64:].argmax(-1)
            restored = forward_logits(idx, restored_hooks + [(SITE, capture_live)])[:, 64:].argmax(-1)
            current = idx[:, 64:]
            expected = full_rows[SITE][current]
            actual = live["output"][:, 64:]
            difference = actual - expected
            changed = restored != baseline
            covered = seen[current]
            for name, mask in (("covered", covered), ("uncovered", ~covered)):
                count = int(mask.sum())
                counts[name] += count
                counts[f"changed_{name}"] += int((changed & mask).sum())
                if count:
                    diff_part = difference[mask]
                    actual_part = actual[mask]
                    sums[f"{name}_num2"] += float((diff_part.double() ** 2).sum())
                    sums[f"{name}_den2"] += float((actual_part.double() ** 2).sum())
                    maxima[name] = max(maxima[name], float(diff_part.abs().max()))
        changed_total = counts["changed_covered"] + counts["changed_uncovered"]
        results[role] = {
            **counts,
            "changed_total": changed_total,
            "uncovered_share_of_changes": counts["changed_uncovered"] / max(changed_total, 1),
            "relative_l2_covered": (sums["covered_num2"] / max(sums["covered_den2"], 1e-30)) ** 0.5,
            "relative_l2_uncovered": (sums["uncovered_num2"] / max(sums["uncovered_den2"], 1e-30)) ** 0.5,
            "max_abs_covered": maxima["covered"],
            "max_abs_uncovered": maxima["uncovered"],
        }
        print(role, json.dumps(results[role], sort_keys=True), flush=True)

    pred_a = all(results[r]["uncovered_share_of_changes"] >= 0.99 for r in results)
    pred_b = all(results[r]["relative_l2_covered"] < 1e-5 for r in results)
    pred_c = all(results[r]["relative_l2_uncovered"] > 1e-3 for r in results)
    pred_d = int(seen_cpu.sum()) == NCOV and all(
        results[r]["changed_total"] == EXPECTED[r] for r in results)
    payload = {
        "status": "discovery_complete",
        "scope": "restored MLP16 output versus compiled row, split by current-token coverage",
        "coverage": NCOV,
        "results": results,
        "predictions": {
            'pred_a_99pct_changes_uncovered': bool(pred_a),
            'pred_b_covered_row_identity': bool(pred_b),
            'pred_c_fallback_nonidentity': bool(pred_c),
            'pred_d_controls': bool(pred_d),
        },
        "runtime_s": time.time() - started,
    }
    temporary = OUT + ".tmp"
    with open(temporary, "x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, OUT)
    print(json.dumps(payload["predictions"], sort_keys=True), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
