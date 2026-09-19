#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_head_sum_closure pred_b_readers_carry_the_final_contrast pred_c_chain_cut_lowers_readers pred_d_9_6_loses_most pred_e_cut_matches_margin
"""The attention share: what the reader heads lose when the MLP chain is cut (v380). v370: zeroing the six named MLP units costs 14% of the they - he margin;
v373: the MLP populations saturate near a third. The readers named in sections 4.1-4.3 -- heads 9.6, 11.3, 12.4, 15.1 (and every other head, reported) --
write the number to the final token. Per head of blocks 9-15 at the FINAL token: the write projected on the they - he unembedding direction (u_they - u_he,
the logit reader), plural - singular over aligned pairs, native vs the six-unit chain zeroed; which readers lose how much.
PREDICTIONS (scored as written; failures preserved; priors from sections 4.1-4.3 / v370)
    pred_a_head_sum_closure             per-head writes sum to each block's attention output within relative 1e-4
    pred_b_readers_carry_the_final_contrast  heads 9.6, 11.3, 12.4, 15.1 together carry >= 0.50 of the summed |head contrast| on the logit reader at the final token (native)
    pred_c_chain_cut_lowers_readers     under the chain cut, the four readers' summed contrast falls (negative change)
    pred_d_9_6_loses_most               head 9.6 loses the largest absolute contrast among the four (it reads MLP 8's state at the noun). Prior: unsure.
    pred_e_cut_matches_margin           the four readers' summed contrast change is within a factor 2 of the margin change (the margin loss flows through them)
PRICE (registered maximum): 3 row batches x (1 native + 1 edited) = 6 forwards; 0 backwards; 0 fits. Bar <= 8.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import run_mlp1_token_table_scaling_v287 as v287
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/reader_heads_under_chain_cut_v380_result.json"
CANDIDATE_ID = "pronoun_number.reader_heads_under_chain_cut_v380"
N_HEAD = 9; BLOCKS = tuple(range(9, 16)); READERS = ("9.6", "11.3", "12.4", "15.1")
CHAIN = {3: (3465, 493), 5: (1036,), 6: (2483,), 7: (1779,), 8: (829,)}
CLOSURE_TOL, READER_MIN, BATCH = 1e-4, 0.50, 32
FORWARDS_MAX = 8
PREDICTIONS = {"pred_a_head_sum_closure": "<= 1e-4", "pred_b_readers_carry_the_final_contrast": ">= 0.50", "pred_c_chain_cut_lowers_readers": "negative", "pred_d_9_6_loses_most": "largest loss", "pred_e_cut_matches_margin": "within 2x"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "reader_min": READER_MIN}, "blocks": list(BLOCKS), "chain": {str(k): list(v) for k, v in CHAIN.items()}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0; blocks = model.transformer.h
    fw = L.ManualForward(backend)
    D = model.config.n_embd; hd = D // N_HEAD; u = (model.lm_head.weight.detach().float()[L._single(" they")] - model.lm_head.weight.detach().float()[L._single(" he")]).cpu()   # logit reader direction
    def run(edits):
        proj = {l: [] for l in BLOCKS}; margins = []; closure = 0.0
        with torch.no_grad():
            for start in range(0, len(rows), BATCH):
                chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); idx = torch.arange(len(chunk)); pf = torch.tensor([r_.final for r_ in chunk])
                x = F.rms_norm(model.transformer.wte(tokens), (D,)); x0, v1_ = x, None
                for l, block in enumerate(blocks):
                    live = block.lambdas[0] * x + block.lambdas[1] * x0; xin_a = F.rms_norm(live, (D,))
                    if l in BLOCKS:
                        captured = {}; hook = block.attn.c_proj.register_forward_pre_hook(lambda m, args: captured.setdefault("y", args[0]))
                        try: attention, v1_ = block.attn(xin_a, v1_)
                        finally: hook.remove()
                        Wp = block.attn.c_proj.weight.detach().float(); y = captured["y"][idx, pf].float(); per_head = torch.stack([(y[:, h * hd:(h + 1) * hd] @ Wp[:, h * hd:(h + 1) * hd].T) for h in range(N_HEAD)], 1)
                        closure = max(closure, float(((per_head.sum(1) - attention[idx, pf].float()).norm(dim=1) / attention[idx, pf].float().norm(dim=1)).max())); proj[l].append((per_head @ u.to(per_head.device)).cpu())
                    else: attention, v1_ = block.attn(xin_a, v1_)
                    x = live + attention; xin = F.rms_norm(x, (D,))
                    if edits and l in edits:
                        h = dod_units.hidden(model, block.mlp, xin); h[:, :, torch.tensor(list(edits[l]), device=h.device)] = 0; x = x + block.mlp.Down(h) + block.mlp.Down_bias
                    else: x = x + block.mlp(xin)
                logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30); margins += [float(logits[i, r_.final, L._single(" they")] - logits[i, r_.final, L._single(" he")]) for i, r_ in enumerate(chunk)]
        return {l: torch.cat(v) for l, v in proj.items()}, margins, closure
    P0, m0, c0 = run(None); forwards += 3; P1, m1, c1 = run(CHAIN); forwards += 3; closure = max(c0, c1)
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; plural = [i for i, r_ in enumerate(rows) if r_.present]; sing = [partner[(rows[i].construction, rows[i].group, False)] for i in plural]
    def contrast(P): return {f"{l}.{h}": float((P[l][plural, h] - P[l][sing, h]).sum()) / len(plural) for l in BLOCKS for h in range(N_HEAD)}
    c_nat, c_edit = contrast(P0), contrast(P1); total = sum(abs(v) for v in c_nat.values()); reader_share = sum(abs(c_nat[k]) for k in READERS) / total
    change = {k: c_edit[k] - c_nat[k] for k in c_nat}; readers_change = sum(change[k] for k in READERS)
    def oriented(m): return sum((m[i] if row.present else -m[i]) for i, row in enumerate(rows)) / len(rows)
    dm = oriented(m1) - oriented(m0)
    top = sorted(c_nat, key=lambda k: -abs(c_nat[k]))[:8]
    report = {"closure_max": closure, "native_margin": oriented(m0), "margin_change": dm, "reader_share_native": reader_share, "top8_heads_native": {k: c_nat[k] for k in top}, "readers_native": {k: c_nat[k] for k in READERS}, "readers_change": {k: change[k] for k in READERS}, "readers_change_sum": readers_change,
              "largest_reader_loss": min(READERS, key=lambda k: change[k])}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_head_sum_closure": closure <= CLOSURE_TOL, "pred_b_readers_carry_the_final_contrast": reader_share >= READER_MIN, "pred_c_chain_cut_lowers_readers": readers_change < 0, "pred_d_9_6_loses_most": report["largest_reader_loss"] == "9.6",
                   "pred_e_cut_matches_margin": 0.5 <= abs(readers_change) / max(abs(dm), 1e-6) <= 2.0}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "reader_heads_under_chain_cut_result_v380", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
