#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_03_cost_on_continuations pred_c_03_continuation_fraction_sane pred_d_07_cost_on_punctuation_keys pred_e_07_punctuation_key_fraction_sane
"""Embedding-forward folding, rung 48 (v660): two of the table readings, tested by key- and query-restricted edits.

v659 read the gate tables: head 0.3 "attaches a subword to the start of its word" (opens on continuation pieces), head 0.7 "measures distance to
the last sentence boundary" (weights punctuation keys). Causal test on the 192 x 512 skip7000 rows, per position: zero head 0.3's off-diagonal
pattern only at queries whose current token is a WORD-CONTINUATION piece (GPT-2 token with no leading space, alphabetic) vs only at all other
queries; zero head 0.7's off-diagonal pattern only on PUNCTUATION keys (token string made of punctuation) vs only on all other keys. If the
readings are right, the cost per affected position concentrates where the table says. CE ADDED, lower is better.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays                   native CE within 0.002 of 3.13241 (instrument)
    pred_b_03_cost_on_continuations         per-query cost of cutting 0.3 at continuation queries >= 3 x the per-query cost of cutting it at other queries. Prior: unsure
    pred_c_03_continuation_fraction_sane    continuation queries are 0.1-0.6 of positions (instrument)
    pred_d_07_cost_on_punctuation_keys      total cost of cutting 0.7 on punctuation keys >= 2 x the total cost of cutting it on all other keys. Prior: unsure
    pred_e_07_punctuation_key_fraction_sane punctuation keys are 0.03-0.3 of keys (instrument)
PRICE (registered maximum): 5 configs x 6 eval batches = 30 forwards (per-position losses); 0 backwards; 0 fits. Bar <= 34.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_gate_reading_edit_v660_result.json"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "embedding_forward.gate_reading_edit_v660"
FORWARDS_MAX = 34
EBATCH, Q_MIN = 32, 8
REPLAY_TOL, RATIO_Q, RATIO_K, CONT_FRAC, PUNC_FRAC = 0.002, 3.0, 2.0, (0.1, 0.6), (0.03, 0.3)
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_03_cost_on_continuations": ">= 3x per query", "pred_c_03_continuation_fraction_sane": "[0.1, 0.6]",
               "pred_d_07_cost_on_punctuation_keys": ">= 2x total", "pred_e_07_punctuation_key_fraction_sane": "[0.03, 0.3]"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"replay_tol": REPLAY_TOL, "ratio_q": RATIO_Q, "ratio_k": RATIO_K, "cont_frac": CONT_FRAC, "punc_frac": PUNC_FRAC}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    V = model.config.vocab_size; dev = "cuda"; b0 = model.transformer.h[0]; forwards = 0
    ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    cont = torch.zeros(V, dtype=torch.bool); punc = torch.zeros(V, dtype=torch.bool)
    for i in range(V):
        try:
            s = enc.decode([i])
        except Exception:
            continue
        if s and not s.startswith(" ") and s.isalpha():
            cont[i] = True
        st = s.strip()
        if st and all(not c.isalnum() and not c.isspace() for c in st):
            punc[i] = True
    cont, punc = cont.to(dev), punc.to(dev)
    with torch.no_grad():
        state = {"idx": None, "mode": None}
        pre = model.register_forward_pre_hook(lambda m, args: state.__setitem__("idx", args[0]))
        native_sq = b0.attn.squared_attention

        def patched(q, k, v, q2, k2):
            Bn, Tn, Hn, Dn = q.shape
            pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
            causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat.masked_fill_(~causal, 0.0)
            if state["mode"]:
                idx = state["idx"]; off = (causal & ~torch.eye(Tn, device=pat.device, dtype=torch.bool))[None].expand(Bn, -1, -1)
                if state["mode"] == "03_cont":
                    kill = off & cont[idx][:, :, None]; h = 3
                elif state["mode"] == "03_other":
                    kill = off & ~cont[idx][:, :, None]; h = 3
                elif state["mode"] == "07_punc":
                    kill = off & punc[idx][:, None, :]; h = 7
                else:
                    kill = off & ~punc[idx][:, None, :]; h = 7
                pat[:, h] = torch.where(kill, torch.zeros_like(pat[:, h]), pat[:, h])
            return torch.einsum("bhqk,bkhd->bhqd", pat, v)

        b0.attn.squared_attention = patched

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
        for mode in (None, "03_cont", "03_other", "07_punc", "07_other"):
            L[mode], fw = per_position(mode); forwards += fw
        native = float(L[None].mean())
        idx_all = ev[:, :-1].to(dev); Te = idx_all.shape[1]; qm = (torch.arange(Te, device=dev) >= Q_MIN)[None].expand_as(idx_all)
        is_cont = cont[idx_all] & qm; is_other = ~cont[idx_all] & qm
        d_cont = L["03_cont"] - L[None]; d_other = L["03_other"] - L[None]
        cost_cont_per_q = float(d_cont[is_cont].mean()); cost_other_per_q = float(d_other[is_other].mean()); cont_frac = float(is_cont[qm].float().mean())
        d_punc = L["07_punc"] - L[None]; d_nonpunc = L["07_other"] - L[None]
        tot_punc, tot_nonpunc = float(d_punc[qm].mean()), float(d_nonpunc[qm].mean()); punc_frac = float(punc[idx_all].float().mean())
        b0.attn.squared_attention = native_sq; pre.remove()
        print(f"native {native:.5f} | 0.3: per-query cost at continuation queries {cost_cont_per_q:+.4f} vs other queries {cost_other_per_q:+.4f} (continuation fraction {cont_frac:.3f}; totals {float(d_cont[qm].mean()):+.4f} / {float(d_other[qm].mean()):+.4f})")
        print(f"0.7: total cost cutting punctuation keys {tot_punc:+.4f} vs all other keys {tot_nonpunc:+.4f} (punctuation key fraction {punc_frac:.3f})")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_03_cost_on_continuations": cost_cont_per_q >= RATIO_Q * cost_other_per_q,
                   "pred_c_03_continuation_fraction_sane": CONT_FRAC[0] <= cont_frac <= CONT_FRAC[1], "pred_d_07_cost_on_punctuation_keys": tot_punc >= RATIO_K * tot_nonpunc,
                   "pred_e_07_punctuation_key_fraction_sane": PUNC_FRAC[0] <= punc_frac <= PUNC_FRAC[1]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_gate_reading_edit_result_v660", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "h03": {"per_query_cost_continuation": cost_cont_per_q, "per_query_cost_other": cost_other_per_q, "continuation_fraction": cont_frac},
                                          "h07": {"total_cost_punctuation_keys": tot_punc, "total_cost_other_keys": tot_nonpunc, "punctuation_key_fraction": punc_frac}},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
