"""Render the per-(run, penalty weight) table from run_checks.py JSONs as Markdown.

  python scripts/summarize.py results/*.json
"""
import json, sys
import numpy as np


def med(xs):
    xs = [x for x in xs if x == x]
    return float(np.median(xs)) if xs else float("nan")


def table(path):
    d = json.load(open(path)); a = d["args"]; rows = []
    name = f"{a['arch']} / {a['texts_source']} / {a['train_contexts']}+{a['heldout_contexts']} ctx, {a['factors']} factors, {a['iterations']} it, seeds {a['seeds']}"
    base = {f["seed"]: f["heldout"]["mean_total_energy"] for f in d["fits"] if f["penalty_weight"] == 0}
    by_w = {}
    for f in d["fits"]:
        by_w.setdefault(f["penalty_weight"], []).append(f)
    out = [f"### {name}", "", f"model step {d['model'].get('step')}; AJ PR range {sum(1 for _ in [0])}× layers {a['source_layer']+1}..{a['target_layer']-1} ({d['feature_dim']} units); E2 null (random pair / span) "
           + ", ".join(f"L{L}: {v['mean']:.3f}/{v['span_mean']:.3f}" for L, v in d["E2_null"].items()), "",
           "| w | median held-out PR | held-out energy / baseline | E1 top-1 | E1 top-5 | E1 all-AJ-MLPs | E1 source MLP | E1 attention | E2 span align (best) | E2 read-off energy ratio | E3 top-1 @0.05 | E3 top-5 @0.05 | E3 random-5 @0.05 | E3 top-1 @0.2 | cos(l,r) | E4 seeds (pair / span) | E4 split (pair / span) | top-5 Jaccard |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    st = d.get("stability", {})
    for w in sorted(by_w):
        F = [x for f in by_w[w] for x in f["factors"]]
        ratio = np.mean([f["heldout"]["mean_total_energy"] / base[f["seed"]] for f in by_w[w]])
        c = lambda k: med([x["E1_completeness"][k] for x in F])
        e3 = lambda k: med([x["E3_ablation"].get(k, float("nan")) for x in F])
        span = med([max(v["span_score"] for v in x["E2_alignment"].values()) for x in F if "E2_alignment" in x])
        ro = med([x["E2_readoff_top_unit"]["energy_ratio"] for x in F if "E2_readoff_top_unit" in x])
        seeds = st.get(f"seeds_w{w}", []); seeds_s = st.get(f"seeds_w{w}_span", [])
        s4 = f"{np.mean([np.mean(v) for v in seeds]):.2f} / {np.mean([np.mean(v) for v in seeds_s]):.2f}" if seeds else "—"
        sp = st.get(f"split_w{w}"); sps = st.get(f"split_w{w}_span"); jac = st.get(f"split_w{w}_top5_jaccard")
        s5 = f"{np.mean(sp):.2f} / {np.mean(sps):.2f}" if sp else "—"; j5 = f"{np.mean(jac):.2f}" if jac else "—"
        out.append(f"| {w:g} | {med([f['heldout']['median_participation_ratio'] for f in by_w[w]]):.1f} | {ratio:.2f} | {c('top1_aj_range'):.2f} | {c('top5_aj_range'):.2f} | {c('all_aj_range_mlps'):.2f} | {c('source_block_mlp'):.2f} | {c('all_attention'):.2f} | {span:.2f} | {ro:.2f} | {e3('top1@0.05'):.2f} | {e3('top5@0.05'):.2f} | {e3('random5@0.05'):.2f} | {e3('top1@0.2'):.2f} | {med([x['cos_l_r'] for x in F]):.2f} | {s4} | {s5} | {j5} |")
    if st:
        out += ["", f"E4 random-dictionary null: pair {st.get('random_null_mean', float('nan')):.4f} (max {st.get('random_null_max', float('nan')):.4f}), span {st.get('span_random_null_mean', float('nan')):.4f} (max {st.get('span_random_null_max', float('nan')):.4f}). Held-out energy is (u·H[l,r])² averaged over factors and contexts; medians are over factors (and seeds)."]
    return "\n".join(out)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(table(p)); print()
