"""Embedding-forward folding, rung 49 (v661): three more table readings tested by restricted edits — 0.4, 1.1, 1.5.

v660: a table ordering nominates, a restricted edit decides (0.3 supported, 0.7 falsified). Same instrument for: head 0.4 "recalls the recent
topic word" — cut on CAPITALISED word-initial keys vs other keys; head 1.1 "of / possessive attaches to its noun" — cut at queries in
{of, his, her, their, your, its, our, my} vs other queries; head 1.5 "a noun-phrase opener looks back at its verb" — cut at DETERMINER queries
{a, an, the, some, these, those, this, that, several, many, few, each, every} vs other queries. Per-position CE on 192 x 512 skip7000. CE ADDED.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays              native CE within 0.002 of 3.13241 (instrument)
    pred_b_04_cost_on_capitalised_keys total cost of cutting 0.4 on capitalised keys >= 1.5 x the total on all other keys (capitalised keys are ~8% of keys). Prior: unsure
    pred_c_11_cost_on_of_queries       per-query cost of cutting 1.1 at of/possessive queries >= 3 x that at other queries. Prior: unsure
    pred_d_15_cost_on_determiner_queries per-query cost of cutting 1.5 at determiner queries >= 2 x that at other queries. Prior: unsure
    pred_e_class_fractions_sane        each class is between 0.02 and 0.5 of positions / keys (instrument)
PRICE (registered maximum): 7 configs x 6 eval batches = 42 forwards (per-position losses); 0 backwards; 0 fits. Bar <= 46.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_gate_reading_edit2_v661_result.json"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "embedding_forward.gate_reading_edit2_v661"
FORWARDS_MAX = 46
EBATCH, Q_MIN = 32, 8
REPLAY_TOL, R04, R11, R15, FRAC = 0.002, 1.5, 3.0, 2.0, (0.02, 0.5)
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_04_cost_on_capitalised_keys": ">= 1.5x total", "pred_c_11_cost_on_of_queries": ">= 3x per query",
               "pred_d_15_cost_on_determiner_queries": ">= 2x per query", "pred_e_class_fractions_sane": "[0.02, 0.5] x 3"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"replay_tol": REPLAY_TOL, "r04": R04, "r11": R11, "r15": R15, "frac": FRAC}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    V = model.config.vocab_size; dev = "cuda"; b0 = model.transformer.h[0]; b1 = model.transformer.h[1]; forwards = 0
    ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    cap = torch.zeros(V, dtype=torch.bool); ofq = torch.zeros(V, dtype=torch.bool); det = torch.zeros(V, dtype=torch.bool)
    OF = {" of", " his", " her", " their", " your", " its", " our", " my"}; DET = {" a", " an", " the", " some", " these", " those", " this", " that", " several", " many", " few", " each", " every"}
    for i in range(V):
        try:
            s = enc.decode([i])
        except Exception:
            continue
        if s.startswith(" ") and len(s) > 2 and s[1].isupper() and s[1:].isalpha():
            cap[i] = True
        if s in OF:
            ofq[i] = True
        if s in DET:
            det[i] = True
    cap, ofq, det = cap.to(dev), ofq.to(dev), det.to(dev)
    with torch.no_grad():
        state = {"idx": None, "mode": None}
        pre = model.register_forward_pre_hook(lambda m, args: state.__setitem__("idx", args[0]))
        native_sq = {0: b0.attn.squared_attention, 1: b1.attn.squared_attention}

        def make_patched(layer):
            def patched(q, k, v, q2, k2):
                Bn, Tn, Hn, Dn = q.shape
                pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
                causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat.masked_fill_(~causal, 0.0)
                if state["mode"] and ((layer == 0) == state["mode"].startswith("04")):
                    idx = state["idx"]; off = (causal & ~torch.eye(Tn, device=pat.device, dtype=torch.bool))[None].expand(Bn, -1, -1)
                    m = state["mode"]
                    if m.startswith("04"):
                        kill = off & (cap[idx][:, None, :] if m == "04_cap" else ~cap[idx][:, None, :]); h = 4
                    elif m.startswith("11"):
                        kill = off & (ofq[idx][:, :, None] if m == "11_of" else ~ofq[idx][:, :, None]); h = 1
                    else:
                        kill = off & (det[idx][:, :, None] if m == "15_det" else ~det[idx][:, :, None]); h = 5
                    pat[:, h] = torch.where(kill, torch.zeros_like(pat[:, h]), pat[:, h])
                return torch.einsum("bhqk,bkhd->bhqd", pat, v)
            return patched

        b0.attn.squared_attention = make_patched(0); b1.attn.squared_attention = make_patched(1)

        def per_position(mode):
            state["mode"] = mode; losses = []; fw = 0
            for s in range(0, ev.shape[0], EBATCH):
                idx = ev[s:s + EBATCH, :-1].to(dev); tgt = ev[s:s + EBATCH, 1:].to(dev)
                cache = {}; hh = model.lm_head.register_forward_hook(lambda m, a, o: cache.__setitem__("z", o)); model(idx, tgt); hh.remove(); fw += 1
                logits = 30.0 * torch.tanh(cache["z"].float() / 30.0)
                losses.append(F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), reduction="none").view(idx.shape))
            state["mode"] = None
            return torch.cat(losses), fw

        L = {}
        for mode in (None, "04_cap", "04_other", "11_of", "11_other", "15_det", "15_other"):
            L[mode], fw = per_position(mode); forwards += fw
        native = float(L[None].mean())
        idx_all = ev[:, :-1].to(dev); Te = idx_all.shape[1]; qm = (torch.arange(Te, device=dev) >= Q_MIN)[None].expand_as(idx_all)
        rep = {}
        tot_cap, tot_ncap = float((L["04_cap"] - L[None])[qm].mean()), float((L["04_other"] - L[None])[qm].mean()); cap_frac = float(cap[idx_all].float().mean())
        rep["h04"] = {"total_cost_capitalised_keys": tot_cap, "total_cost_other_keys": tot_ncap, "capitalised_key_fraction": cap_frac}
        for name, mask, key in (("h11", ofq, "11"), ("h15", det, "15")):
            is_q = mask[idx_all] & qm; is_o = ~mask[idx_all] & qm
            d_q = (L[f"{key}_{'of' if key == '11' else 'det'}"] - L[None]); d_o = (L[f"{key}_other"] - L[None])
            rep[name] = {"per_query_cost_class": float(d_q[is_q].mean()), "per_query_cost_other": float(d_o[is_o].mean()), "class_fraction": float(is_q[qm].float().mean()),
                         "total_class": float(d_q[qm].mean()), "total_other": float(d_o[qm].mean())}
        for l_, sq in native_sq.items():
            (b0 if l_ == 0 else b1).attn.squared_attention = sq
        pre.remove()
        print(f"native {native:.5f} | 0.4: cut on capitalised keys {tot_cap:+.4f} vs other keys {tot_ncap:+.4f} (fraction {cap_frac:.3f})")
        for name in ("h11", "h15"):
            r = rep[name]; print(f"{name}: per-query cost at class queries {r['per_query_cost_class']:+.4f} vs other {r['per_query_cost_other']:+.4f} (class fraction {r['class_fraction']:.3f}; totals {r['total_class']:+.4f} / {r['total_other']:+.4f})")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_04_cost_on_capitalised_keys": rep["h04"]["total_cost_capitalised_keys"] >= R04 * rep["h04"]["total_cost_other_keys"],
                   "pred_c_11_cost_on_of_queries": rep["h11"]["per_query_cost_class"] >= R11 * rep["h11"]["per_query_cost_other"],
                   "pred_d_15_cost_on_determiner_queries": rep["h15"]["per_query_cost_class"] >= R15 * rep["h15"]["per_query_cost_other"],
                   "pred_e_class_fractions_sane": all(FRAC[0] <= f <= FRAC[1] for f in (rep["h04"]["capitalised_key_fraction"], rep["h11"]["class_fraction"], rep["h15"]["class_fraction"]))}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_gate_reading_edit2_result_v661", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, **rep},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
