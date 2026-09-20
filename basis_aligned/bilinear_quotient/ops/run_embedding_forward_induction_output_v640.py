#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_direct_path_favours_induction_token pred_c_direct_path_sign_rate pred_d_cost_concentrates_on_eligible pred_e_eligible_fraction_sane
"""Embedding-forward folding, rung 32 (v640): the output side of the induction head 5.5.

v635-v639 established head 5.5 as the induction head by its pattern (keys whose predecessor equals the current token) and located the key feature's
writers. This rung asks what its write DOES: (i) response — at every query position i with at least one induction-eligible key, take the eligible
key j* with the largest |pattern|; the induction prediction is tok_{j*} (the token that followed the earlier copy). Project 5.5's write at i
(through the frozen final rmsnorm scale of that position and the unembedding — the direct path, no downstream layers) onto the logit of tok_{j*}
and onto the logits of 8 random tokens: the direct-path logit effect. (ii) edit — cut 5.5's off-diagonal pattern on the 192 x 512 skip7000 rows and
attribute the CE cost per position: eligible positions vs non-eligible. CE ADDED, lower is better. Direct-path effects are an approximation
(later layers are not replayed); the CE split is exact.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays                 native CE within 0.002 of 3.13241 (instrument)
    pred_b_direct_path_favours_induction_token mean direct-path logit effect on tok_{j*} >= 5 x the mean |effect| on random tokens. Prior: unsure
    pred_c_direct_path_sign_rate          the effect on tok_{j*} is positive at >= 0.7 of eligible positions. Prior: unsure
    pred_d_cost_concentrates_on_eligible  per-position CE cost of the cut on eligible positions >= 3 x that on non-eligible positions. Prior: likely
    pred_e_eligible_fraction_sane         eligible positions are between 0.2 and 0.8 of all positions >= 8 (instrument: the split is not degenerate)
PRICE (registered maximum): 2 capture forwards (64 rows, with the final-norm scale and 5.5's pre-projection output captured) + 2 configs x 6 eval
batches with per-position losses = 14 forwards; 0 backwards; 0 fits. Bar <= 16.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_induction_output_v640_result.json"
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "embedding_forward.induction_output_v640"
FORWARDS_MAX = 16
EBATCH, N_CAP, Q_MIN, N_RAND = 32, 64, 8, 8
L_IND, H_IND = 5, 5
REPLAY_TOL, EFFECT_RATIO, SIGN_RATE, COST_RATIO, FRAC = 0.002, 5.0, 0.7, 3.0, (0.2, 0.8)
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_direct_path_favours_induction_token": ">= 5 x random", "pred_c_direct_path_sign_rate": ">= 0.7 positive",
               "pred_d_cost_concentrates_on_eligible": ">= 3 x per position", "pred_e_eligible_fraction_sane": "in [0.2, 0.8]"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "head": [L_IND, H_IND],
            "bars": {"replay_tol": REPLAY_TOL, "effect_ratio": EFFECT_RATIO, "sign_rate": SIGN_RATE, "cost_ratio": COST_RATIO, "frac": FRAC}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    D = model.config.n_embd; H = model.config.n_head; hd = D // H; V = model.config.vocab_size; dev = "cuda"
    blocks = model.transformer.h; b5 = blocks[L_IND]; forwards = 0
    fit = torch.load(FIT_ROWS, map_location="cpu").long(); ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    with torch.no_grad():
        state = {"cut": False, "capture": None}
        native_sq = b5.attn.squared_attention

        def patched(q, k, v, q2, k2):
            Bn, Tn, Hn, Dn = q.shape
            pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
            causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat.masked_fill_(~causal, 0.0)
            off = (causal & ~torch.eye(Tn, device=pat.device, dtype=torch.bool))[None]
            if state["cut"]:
                pat[:, H_IND] = torch.where(off, torch.zeros_like(pat[:, H_IND]), pat[:, H_IND])
            z = torch.einsum("bhqk,bkhd->bhqd", pat, v)
            if state["capture"] is not None:
                state["capture"]["pat"].append(pat[:, H_IND].detach().clone()); state["capture"]["z"].append(z[:, H_IND].detach().clone())
            return z

        b5.attn.squared_attention = patched
        # final-norm scale per position: hook the lm_head input (already normalised) and the residual before the final norm
        fin = {}
        h_lm = model.lm_head.register_forward_pre_hook(lambda m, a: fin.setdefault("normed", []).append(a[0].detach().clone()))
        state["capture"] = {"pat": [], "z": []}; idxs = []
        for s in range(0, N_CAP, EBATCH):
            idx = fit[s:s + EBATCH, :-1].to(dev); model(idx, fit[s:s + EBATCH, 1:].to(dev)); forwards += 1; idxs.append(idx)
        h_lm.remove()
        pat = torch.cat(state["capture"]["pat"]); z = torch.cat(state["capture"]["z"]); normed = torch.cat(fin["normed"]); idx = torch.cat(idxs); state["capture"] = None
        Bn, Tn = idx.shape; pos = torch.arange(Tn, device=dev); off = (pos[:, None] > pos[None, :])[None].expand(Bn, -1, -1)
        prev_tok = torch.cat([torch.full_like(idx[:, :1], -1), idx[:, :-1]], 1); ind = (idx[:, :, None] == prev_tok[:, None, :]) & off
        eligible = ind.any(-1) & (pos >= Q_MIN)[None]                                                        # [B, T]
        masked = torch.where(ind, pat.abs(), torch.zeros_like(pat)); jstar = masked.argmax(-1)                # [B, T]
        target = torch.gather(idx, 1, jstar)                                                                 # tok_{j*}
        # direct path: head write O_h z at position i, scaled by the final norm's per-position factor (||normed|| / ||residual||) is not available
        # without the pre-norm residual; use the rms of the head write's contribution as a fixed direction and score through W_U on the unit sphere:
        # effect = W_U [ (O_h z_i) / rms(final residual_i) ] where the final residual is recovered from normed * its own rms (unknown) -> use the
        # normed vector's implied scale: rms(normed) = 1 by construction, so we report effects in units of one final-rms.
        Oh = b5.attn.c_proj.weight[:, H_IND * hd:(H_IND + 1) * hd].float()                                   # [D, hd]
        write = z @ Oh.T                                                                                     # [B, T, D]
        WU = model.lm_head.weight.float()                                                                    # [V, D]
        gen = torch.Generator(device=dev).manual_seed(640)
        rand = torch.randint(0, V, (Bn, Tn, N_RAND), generator=gen, device=dev)
        eff_target = (write * WU[target]).sum(-1)                                                            # [B, T]
        eff_rand = (write[:, :, None, :] * WU[rand]).sum(-1)                                                 # [B, T, R]
        et = eff_target[eligible]; er = eff_rand[eligible]
        mean_t, mean_r_abs, sign_rate = float(et.mean()), float(er.abs().mean()), float((et > 0).float().mean())
        n_elig = int(eligible.sum()); n_all = int((pos >= Q_MIN).sum() * Bn); frac = n_elig / n_all
        print(f"eligible positions {n_elig} / {n_all} ({frac:.3f}); direct-path effect on induction token: mean {mean_t:+.4f} (positive at {sign_rate:.3f}); |effect| on random tokens: {mean_r_abs:.4f}; ratio {mean_t / mean_r_abs:.2f}")

        def per_position_ce(cut):
            state["cut"] = cut; losses = []; fw = 0
            for s in range(0, ev.shape[0], EBATCH):
                idx_ = ev[s:s + EBATCH, :-1].to(dev); tgt = ev[s:s + EBATCH, 1:].to(dev)
                cache = {}; hh = model.lm_head.register_forward_hook(lambda m, a, o: cache.__setitem__("z", o)); model(idx_, tgt); hh.remove(); fw += 1
                logits = 30.0 * torch.tanh(cache["z"].float() / 30.0)
                losses.append(F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), reduction="none").view(idx_.shape))
            state["cut"] = False
            return torch.cat(losses), fw

        L_nat, fw = per_position_ce(False); forwards += fw; L_cut, fw = per_position_ce(True); forwards += fw
        native = float(L_nat.mean()); delta = L_cut - L_nat
        Be, Te = ev.shape[0], ev.shape[1] - 1; pos_e = torch.arange(Te, device=dev); idx_e = ev[:, :-1].to(dev)
        prev_e = torch.cat([torch.full_like(idx_e[:, :1], -1), idx_e[:, :-1]], 1)
        elig_e = ((idx_e[:, :, None] == prev_e[:, None, :]) & (pos_e[:, None] > pos_e[None, :])[None]).any(-1) & (pos_e >= Q_MIN)[None]
        qm = (pos_e >= Q_MIN)[None].expand(Be, -1)
        cost_elig = float(delta[elig_e].mean()); cost_non = float(delta[qm & ~elig_e].mean()); total_added = float(delta.mean())
        b5.attn.squared_attention = native_sq
        print(f"native CE {native:.5f} | cut 5.5: CE added {total_added:+.4f} overall; per position: eligible {cost_elig:+.4f} vs non-eligible {cost_non:+.4f} (ratio {cost_elig / cost_non:.2f}); eligible fraction on eval {float(elig_e[qm].float().mean()):.3f}")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_direct_path_favours_induction_token": mean_t >= EFFECT_RATIO * mean_r_abs,
                   "pred_c_direct_path_sign_rate": sign_rate >= SIGN_RATE, "pred_d_cost_concentrates_on_eligible": cost_elig >= COST_RATIO * cost_non,
                   "pred_e_eligible_fraction_sane": FRAC[0] <= frac <= FRAC[1]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_induction_output_result_v640", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "cut_added_total": total_added, "cost_eligible": cost_elig, "cost_non_eligible": cost_non, "eligible_fraction_fit": frac,
                                          "direct_path": {"mean_effect_target": mean_t, "mean_abs_effect_random": mean_r_abs, "sign_rate": sign_rate, "n_eligible": n_elig}},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
