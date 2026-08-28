# HYBRID TENSOR-CLASS ORACLE -- which deleted contraction causes the class gap?
#
# Frozen protocol and exact arm semantics:
#   ../polynomial_causal/HYBRID_TENSOR_CLASS_ORACLE_PREREGISTRATION.md
#
# This is discovery-only. It uses native halves as causal class oracles, not as a
# claimed simple endpoint. It is intentionally separate from the concurrent
# nonlocal_program_class lag/prefix-mean family.
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
POLY = os.path.join(ROOT, "polynomial_causal")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, POLY)

import downstream_rank_sweep as base
from hybrid_tensor_class_oracle_stats import analyze_hybrid_losses


D = 1152
RANK = 8
STEPS = 180
EVERY = 30
BATCH = 4
LR = 1e-3
PT = os.path.dirname(HERE) + "/"
OUT = HERE + "/hybrid_tensor_class_oracle_results.json"
FIT_ROWS = PT + ".rowcache/fineweb_n96_skip80.pt"
EVAL_SETS = (
    ("skip7000", PT + ".rowcache/fineweb_n192_skip7000.pt", 3.29205),
    ("skip11000", PT + ".rowcache/fineweb_n192_skip11000.pt", 3.09711),
)
S1738_TABLE_CE = 7.35114
S1748_RECOVERY = {"skip7000": 0.40631, "skip11000": 0.38578}
S1750_PEAK = {"skip7000": 0.59388, "skip11000": 0.57418}

ALL_SITES = tuple((kind, layer) for kind in ("mlp", "attn") for layer in range(18))
DEPTH_ORDER = tuple(
    (kind, layer) for layer in range(18) for kind in ("attn", "mlp")
)
ARM_SITES = {
    "both_compiled": ALL_SITES,
    "attention_native": tuple(("mlp", layer) for layer in range(18)),
    "mlp_native": tuple(("attn", layer) for layer in range(18)),
    "both_native": (),
}


def hooks_for(sites, tables, seen, factors):
    return [
        (site, base.factor_hook(tables[site], seen, *factors[site]))
        for site in sites
    ]


@torch.no_grad()
def initialize_arm(fit, sites, tables, seen):
    installed = {}
    fits_ok = True
    site_set = set(sites)
    for site in DEPTH_ORDER:
        if site not in site_set:
            continue
        candidates, count = base.fit_one(
            fit, site, tables, seen, installed,
        )
        fits_ok = fits_ok and count == 24576
        installed[site] = candidates[RANK]
    return installed, fits_ok


def train_arm(name, fit, eval_rows, live_ce, sites, tables, seen, seed):
    if not sites:
        values = {role: live_ce[role] for role in eval_rows}
        return {
            "compiled_sites": 0,
            "start_ce": dict(values),
            "best_step": 0,
            "best_ce": dict(values),
            "trajectory": [{"step": 0, **values}],
            "factor_reals_M": 0.0,
            "active_table_reals_M": 0.0,
            "fits_ok": True,
        }

    initial, fits_ok = initialize_arm(fit, sites, tables, seen)
    factors = {}
    parameters = []
    for site in sites:
        left, right = initial[site]
        left = left.clone().requires_grad_(True)
        right = right.clone().requires_grad_(True)
        factors[site] = (left, right)
        parameters.extend((left, right))

    def current_hooks():
        return hooks_for(sites, tables, seen, factors)

    with torch.no_grad():
        start = {role: base.ce(rows, current_hooks()) for role, rows in eval_rows.items()}
    best = {"selection_ce": start["skip7000"], "step": 0, "values": dict(start)}
    trajectory = [{"step": 0, **{key: round(value, 5) for key, value in start.items()}}]
    optimizer = torch.optim.Adam(parameters, lr=LR)
    generator = torch.Generator().manual_seed(seed)
    for step in range(STEPS):
        selection = torch.randint(0, fit.shape[0], (BATCH,), generator=generator)
        batch = fit[selection]
        tokens = batch[:, :-1].to(base.DEV).contiguous()
        logits = base.forward(tokens, current_hooks())
        targets = batch[:, 1:].to(base.DEV)
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape)[:, 64:]
        coverage = base.COV["seen"][tokens[:, 64:]]
        loss = losses[coverage].mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % EVERY == 0:
            with torch.no_grad():
                values = {
                    role: base.ce(rows, current_hooks())
                    for role, rows in eval_rows.items()
                }
            trajectory.append({
                "step": step + 1,
                **{key: round(value, 5) for key, value in values.items()},
            })
            if values["skip7000"] < best["selection_ce"]:
                best = {
                    "selection_ce": values["skip7000"],
                    "step": step + 1,
                    "values": dict(values),
                }
    count = len(sites)
    result = {
        "compiled_sites": count,
        "start_ce": {key: round(value, 5) for key, value in start.items()},
        "best_step": best["step"],
        "best_ce": {key: round(value, 5) for key, value in best["values"].items()},
        "trajectory": trajectory,
        "factor_reals_M": round(count * 2 * RANK * D / 1e6, 4),
        "active_table_reals_M": round(count * int(seen.sum()) * D / 1e6, 4),
        "fits_ok": bool(fits_ok),
    }
    del optimizer, parameters, factors, initial
    torch.cuda.empty_cache()
    return result


def main():
    started = time.time()
    base.RANKS = (RANK,)
    base.RIDGE = 1e-2
    for parameter in base.m.parameters():
        parameter.requires_grad_(False)
    fit = base.load(FIT_ROWS)
    print(
        "HYBRID TENSOR-CLASS ORACLE | rank 8 | final CE | "
        "CC / native-attn / native-MLP / native-both | DISCOVERY ONLY",
        flush=True,
    )

    base.COV["seen"] = torch.zeros(50257, dtype=torch.bool, device=base.DEV)
    with torch.no_grad():
        tables, seen = base.fit_tables(fit, ALL_SITES)
    base.COV["seen"] = seen
    covered_types = int(seen.sum())

    eval_rows = {}
    live_ce = {}
    for role, path, reference in EVAL_SETS:
        rows = base.load(path)
        eval_rows[role] = rows
        with torch.no_grad():
            live_ce[role] = base.ce(rows, [])
        if abs(live_ce[role] - reference) > 1e-2:
            raise RuntimeError(f"{role} live CE changed")
    with torch.no_grad():
        table_ce = {
            role: base.ce(rows, [
                (site, base.table_hook(tables[site], seen)) for site in ALL_SITES
            ])
            for role, rows in eval_rows.items()
        }

    arms = {}
    shared_seed = 2026082876
    for name, sites in ARM_SITES.items():
        arm_started = time.time()
        arms[name] = train_arm(
            name, fit, eval_rows, live_ce, sites, tables, seen, shared_seed,
        )
        print(
            f"  {name:18s}: heldout CE {arms[name]['best_ce']['skip11000']:.5f} "
            f"at step {arms[name]['best_step']:3d} | {len(sites):2d} compiled sites "
            f"[{time.time() - arm_started:.0f}s]",
            flush=True,
        )

    heldout_ce = {
        name: arms[name]["best_ce"]["skip11000"] for name in ARM_SITES
    }
    analysis = analyze_hybrid_losses(
        heldout_ce, live_ce=live_ce["skip11000"], native_atol=1e-4,
        dominance_atol=0.01,
    )
    attention_gain = analysis.attention_restoration_gain
    mlp_gain = analysis.mlp_restoration_gain
    interaction = analysis.interaction_harm
    pred_a = attention_gain > mlp_gain
    pred_b = (
        heldout_ce["attention_native"] < heldout_ce["both_compiled"]
        and heldout_ce["mlp_native"] < heldout_ce["both_compiled"]
    )
    pred_c = interaction >= 0.10
    cc_start_recovery = {
        role: table_ce[role] - arms["both_compiled"]["start_ce"][role]
        for role in eval_rows
    }
    cc_best_recovery = {
        role: table_ce[role] - arms["both_compiled"]["best_ce"][role]
        for role in eval_rows
    }
    pred_d = (
        abs(arms["both_native"]["best_ce"]["skip7000"] - live_ce["skip7000"]) <= 1e-4
        and abs(cc_start_recovery["skip11000"] - S1748_RECOVERY["skip11000"]) <= 0.002
        and abs(cc_best_recovery["skip11000"] - S1750_PEAK["skip11000"]) <= 0.03
        and abs(table_ce["skip7000"] - S1738_TABLE_CE) <= 0.005
        and covered_types == 5419
        and all(arm["fits_ok"] for arm in arms.values())
    )

    payload = {
        "config": {
            "rank": RANK, "steps": STEPS, "every": EVERY, "batch": BATCH,
            "lr": LR, "shared_training_seed": shared_seed,
            "arms": {name: [f"{kind}{layer}" for kind, layer in sites]
                     for name, sites in ARM_SITES.items()},
            "objective": "final CE on covered fit positions 64:256",
            "checkpoint": "minimum skip7000 CE; skip11000 reported at the same step",
            "role_note": "DISCOVERY ONLY; both compiler eval roles were already spent",
        },
        "live_ce": {key: round(value, 5) for key, value in live_ce.items()},
        "table_ce": {key: round(value, 5) for key, value in table_ce.items()},
        "covered_token_types": covered_types,
        "arms": arms,
        "heldout_analysis": {
            "harm": {key: round(value, 5) for key, value in analysis.harm.items()},
            "attention_restoration_gain": round(attention_gain, 5),
            "mlp_restoration_gain": round(mlp_gain, 5),
            "interaction_harm": round(interaction, 5),
            "dominant_missing_contraction": analysis.dominant_missing_contraction,
        },
        "controls": {
            "both_compiled_start_recovery": {
                key: round(value, 5) for key, value in cc_start_recovery.items()
            },
            "both_compiled_best_recovery": {
                key: round(value, 5) for key, value in cc_best_recovery.items()
            },
        },
        "predictions": {
            "pred_a_attention_dominates": bool(pred_a),
            "pred_b_each_native_half_helps": bool(pred_b),
            "pred_c_superadditive_harm": bool(pred_c),
            "pred_d_controls": bool(pred_d),
        },
        "runtime_s": round(time.time() - started, 1),
    }
    with open(OUT, "w") as handle:
        json.dump(payload, handle, indent=1)
    print(
        f"\n  attention restoration {attention_gain:+.5f} | "
        f"MLP restoration {mlp_gain:+.5f} | interaction {interaction:+.5f} | "
        f"dominant {analysis.dominant_missing_contraction}",
        flush=True,
    )
    print(
        f"pred_a {pred_a} | pred_b {pred_b} | pred_c {pred_c} | pred_d {pred_d}",
        flush=True,
    )
    print(f"wrote {OUT} ({payload['runtime_s']}s)", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
