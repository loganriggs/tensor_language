"""Circuits lane rung C3 (plans/CIRCUITS_PLAN_V5.md): the four properties (removal, selectivity, composition, generalisation) on
creator-unit circuits at finite scale.

  python scripts/run_circuits_c3.py --smoke
  python scripts/run_circuits_c3.py --device cuda --c1a results/circuits_c1a.json --blocks results/circuits_c1c_blocks.json --out results/circuits_c3.json
"""
import argparse, collections, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from circuit_checks.core import FreezeSpec
from circuit_checks.checks import output_score, hidden_response, top_units, clean_cache
from circuit_checks.tensorgpt import TensorGPTSpans
from circuit_checks.inrepo_model import load, state_before_block, ROOT
from scripts.run_symmetric_dct_v1 import readers_for
from scripts.run_circuits_c1a import power_top
from scripts.run_checks import advbench_texts

BARS = dict(zero=1e-6, all=0.99, removal=0.3, removal_heads=0.5, selective=2.0, compose=0.1, general=0.7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--device", default="cuda"); ap.add_argument("--source-layer", type=int, default=8)
    ap.add_argument("--sequence-length", type=int, default=32); ap.add_argument("--contexts", type=int, default=8); ap.add_argument("--n-tokens", type=int, default=16)
    ap.add_argument("--c1a", default="results/circuits_c1a.json"); ap.add_argument("--blocks", default="results/circuits_c1c_blocks.json"); ap.add_argument("--n-creators", type=int, default=30)
    ap.add_argument("--scale", type=float, default=0.01); ap.add_argument("--out", default="results/circuits_c3.json")
    a = ap.parse_args(); t_start = time.time(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.smoke:
        sys.path.insert(0, ROOT); import jacclust.tt_model as TT
        a.device = "cpu"; a.source_layer = 1; a.contexts = 2; a.sequence_length = 8; a.n_tokens = 1; a.n_creators = 4
        cfg = TT.GPTConfig(vocab_size=64, n_layer=4, n_head=2, n_embd=16, bilinear=True, bilinear_attn=True, squared_attn=True)
        torch.manual_seed(0); model = TT.GPT(cfg)
        for p in model.parameters():
            if p.dim() == 2: torch.nn.init.normal_(p, std=0.3)
        model = model.eval(); model.requires_grad_(False); readers = F.normalize(torch.randn(cfg.n_embd, 2), dim=0); toks = [3]
        rows_fw = torch.randint(0, 64, (2, a.sequence_length), generator=torch.Generator().manual_seed(20)); rows_adv = torch.randint(0, 64, (2, a.sequence_length), generator=torch.Generator().manual_seed(30))
        g0 = torch.Generator().manual_seed(4); creators = {k: [(1 + (i % 3), int(torch.randint(0, 64, (1,), generator=g0))) for i in range(a.n_creators)] for k in range(3)}
        heads = {k: [(1, 0), (2, 1), (3, 0)] for k in range(3)}
    else:
        model, cfg, meta = load("bilinear-attn", a.device); readers = readers_for(model, a.device)
        rows = torch.load(os.path.join(ROOT, "basis_aligned/bilinear_quotient/.rowcache/fineweb_n192_skip7000.pt"), map_location="cpu"); rows_fw = rows[96:96 + a.contexts, :a.sequence_length].long()
        from transformers import GPT2Tokenizer; tok = GPT2Tokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
        _, adv = advbench_texts(32, a.contexts); rows_adv = torch.cat([tok(t, return_tensors="pt", truncation=True, padding="max_length", max_length=a.sequence_length).input_ids for t in adv])
        c1a = json.load(open(a.c1a)); toks = c1a["tokens"][:a.n_tokens]; per_dir = {p["name"]: p for p in c1a["per_direction"]}
        blk = json.load(open(a.blocks)); cnt = collections.defaultdict(collections.Counter)
        for r in blk["records"]:
            for L, lst in r["top_units"].items():
                if int(L) <= a.source_layer + 4:
                    for h, e in lst: cnt[r["direction"]][(int(L), int(h))] += 1
        creators = {k: [u for u, _ in cnt[k].most_common(a.n_creators)] for k in cnt}
        heads = {}
    dev = a.device; d = cfg.n_embd; n_layer = cfg.n_layer; H = cfg.n_head
    W = model.lm_head.weight.float(); tokdirs = F.normalize(W[toks].T, dim=0)
    U = torch.cat([readers.to(dev), tokdirs.to(dev)], 1); names = [f"reader{k}" for k in range(readers.shape[1])] + [f"tok{t}" for t in toks]; K = U.shape[1]
    if not a.smoke:
        heads = {k: [tuple(int(z) for z in h.split(".")) for h in per_dir[names[k]]["top3_heads"]] for k in range(K)}
    spans = TensorGPTSpans(model, a.source_layer, n_layer, slice(-3, None), final_norm=True, per_head=True); layers = spans.all_layers
    widths = {L: spans.mlp_width(L) for L in layers}
    core_count = collections.Counter(u for k in range(K) for u in creators[k]); core = {u for u, n in core_count.items() if n >= 0.5 * K}
    specific = {k: [u for u in creators[k] if u not in core] for k in range(K)}
    def ctx_of(ids):
        with torch.no_grad():
            st = state_before_block(model, ids.to(dev), a.source_layer)
        return (st["values"].float(), st["initial_values"].float(), st["first_values"].float())
    sets = {"fineweb": [ctx_of(rows_fw[i:i + 1]) for i in range(len(rows_fw))], "advbench": [ctx_of(rows_adv[i:i + 1]) for i in range(len(rows_adv))]}
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        spans.make_span(sets["fineweb"][0])(torch.zeros(d, device=dev), None)
    rho = float(torch.stack([c[0].norm(dim=-1).mean() for c in sets["fineweb"]]).mean()); al = a.scale * rho
    def unit_spec(units, cache):
        masks = {L: torch.zeros(widths[L]) for L in layers}
        for (L, h) in units: masks[L][h] = 1
        return FreezeSpec(mlp_masks={L: m.to(dev) for L, m in masks.items() if m.sum() > 0}, clean_hidden=cache["hidden"])
    def head_spec(hl, cache):
        masks = {L: torch.zeros(H) for L in layers}
        for (L, h) in hl: masks[L][h] = 1
        return FreezeSpec(head_value_masks={L: m.to(dev) for L, m in masks.items() if m.sum() > 0}, clean_values=cache["values"])
    def both(us, hs):
        return FreezeSpec(mlp_masks=us.mlp_masks, clean_hidden=us.clean_hidden, head_value_masks=hs.head_value_masks, clean_values=hs.clean_values)
    g = torch.Generator().manual_seed(0); all_units_early = [(L, h) for L in layers if L <= a.source_layer + 4 for h in range(widths[L])]
    rec = []; zero_ctrl = []; all_ctrl = []; t0 = time.time()
    with sdpa_kernel(SDPBackend.MATH):
        for setname, ctxs in sets.items():
            for c, ctx in enumerate(ctxs):
                span = spans.make_span(ctx); cache = clean_cache(span, d); zero = torch.zeros(d, device=dev)
                probes = {}; f0 = {}; D0 = {}
                with torch.no_grad(): pass
                for k in range(K):
                    u = U[:, k]; f = lambda th, fz=None, u=u: u @ span(th, fz)[0]; v, _ = power_top(lambda th: f(th), d, dev); probes[k] = (v, f)
                    with torch.no_grad():
                        f0[k] = f(zero); D0[k] = float(f(2 * al * v) - 2 * f(al * v) + f0[k])
                D = lambda k, spec: float(probes[k][1](2 * al * probes[k][0], spec) - 2 * probes[k][1](al * probes[k][0], spec) + f0[k])
                fr = lambda k, spec: (1 - D(k, spec) / D0[k]) if abs(D0[k]) > 1e-30 else float("nan")
                with torch.no_grad():
                    if c == 0 and setname == "fineweb":
                        zero_ctrl.append(float(abs(D(0, unit_spec([], cache)) - D0[0]) / max(abs(D0[0]), 1e-30)))
                        all_ctrl.append(fr(0, both(unit_spec([(L, h) for L in layers for h in range(widths[L])], cache), head_spec([(L, h) for L in layers for h in range(H)], cache))))
                    cspec = {k: unit_spec(creators[k], cache) for k in range(K)}; sspec = {k: unit_spec(specific[k], cache) for k in range(K)}
                    core_spec = unit_spec(sorted(core), cache)
                    for k in range(K):
                        r = dict(set=setname, context=c, direction=k, name=names[k], D0=D0[k])
                        r["creators"] = fr(k, cspec[k]); r["creators_heads"] = fr(k, both(cspec[k], head_spec(heads[k], cache))); r["heads"] = fr(k, head_spec(heads[k], cache))
                        r["specific"] = fr(k, sspec[k]) if specific[k] else float("nan"); r["core"] = fr(k, core_spec) if core else float("nan")
                        resp = hidden_response(span, probes[k][0], probes[k][0], layers); t100 = top_units(resp, 100 if not a.smoke else 5); t100.clean_hidden = cache["hidden"]; r["readers100"] = fr(k, t100)
                        ridx = torch.randperm(len(all_units_early), generator=g)[:len(creators[k])].tolist(); r["random"] = fr(k, unit_spec([all_units_early[i] for i in ridx], cache))
                        if setname == "fineweb":
                            r["cross_specific"] = {}; r["cross_creators"] = {}; r["union"] = {}
                            for j in range(K):
                                if j == k: continue
                                r["cross_specific"][j] = fr(k, sspec[j]) if specific[j] else float("nan"); r["cross_creators"][j] = fr(k, cspec[j])
                                r["union"][j] = fr(k, unit_spec(sorted(set(creators[k]) | set(creators[j])), cache))
                        rec.append(r)
                m = lambda key: float(np.nanmedian([x[key] for x in rec if x["set"] == setname]))
                print(f"[props] {setname} context {c + 1}/{len(ctxs)} ({time.time() - t0:.0f}s): creators {m('creators'):.2f} +heads {m('creators_heads'):.2f} heads {m('heads'):.2f} readers100 {m('readers100'):.2f} core {m('core'):.2f} specific {m('specific'):.2f} random {m('random'):.2f}", flush=True)
                json.dump(dict(partial=True, records=rec), open(a.out + ".partial", "w"), default=float)
    fw = [r for r in rec if r["set"] == "fineweb"]; adv = [r for r in rec if r["set"] == "advbench"]
    med = lambda rows, key: float(np.nanmedian([r[key] for r in rows]))
    sel = []; comp = []
    for r in fw:
        own = r["specific"]; others = [v for v in r["cross_specific"].values() if v == v]
        if own == own and others and abs(np.mean(others)) > 1e-9: sel.append(abs(own) / abs(np.mean(others)))
        for j, uv in r["union"].items():
            comp.append(abs(uv - (r["creators"] + r["cross_creators"][j])))
    preds = dict(pred_a_instrument=max(zero_ctrl) <= BARS["zero"] and min(all_ctrl) >= BARS["all"],
                 pred_b_removal=med(fw, "creators") >= BARS["removal"] and med(fw, "creators_heads") >= BARS["removal_heads"],
                 pred_c_selective=(float(np.median(sel)) if sel else float("nan")) >= BARS["selective"], pred_d_composable=(float(np.median(comp)) if comp else float("nan")) <= BARS["compose"],
                 pred_e_generalises=med(adv, "creators") >= BARS["general"] * med(fw, "creators"))
    costs = dict(creators_per_direction=a.n_creators, shared_core=len(core), heads=3, creator_read_multiply_adds=a.n_creators * 2 * d, head_value_map_multiply_adds=3 * 2 * (d // H) * d)
    res = dict(args=vars(a), rho=rho, predictions=preds, controls=dict(zero_mask_max=max(zero_ctrl), all_patched_min=min(all_ctrl)), core=sorted(core), creators={names[k]: creators[k] for k in range(K)},
               summary=dict(fineweb={key: med(fw, key) for key in ("creators", "creators_heads", "heads", "readers100", "core", "specific", "random")},
                            advbench={key: med(adv, key) for key in ("creators", "creators_heads", "heads", "readers100", "core", "specific", "random")},
                            selectivity_ratio_median=(float(np.median(sel)) if sel else None), cross_specific_median=float(np.nanmedian([v for r in fw for v in r["cross_specific"].values()])),
                            cross_creators_median=float(np.nanmedian([v for r in fw for v in r["cross_creators"].values()])), composition_deviation_median=(float(np.median(comp)) if comp else None), costs=costs),
               records=rec, seconds=time.time() - t_start)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(dict(predictions=preds, controls=res["controls"], summary=res["summary"]), indent=1, default=float), flush=True); print("wrote", a.out)


if __name__ == "__main__":
    main()
