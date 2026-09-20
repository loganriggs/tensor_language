#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_coefficient_core_reproduces pred_b_token_metric_core_gain pred_c_quarter_at_256 pred_d_frequency_beats_uniform pred_e_input_side_compresses_more
"""Embedding-forward folding, rung 3 (v611): HOSVD-projected TUCKER RETAINED ENERGY of the centred token-folded early-MLP tensors.

Front-side twin of Codex's readout-side symmetric Tucker object. v610 established that the centred fold T_C = T x_2 C^1/2 x_3 C^1/2 (C = the
token covariance of the normalised MLP input; uniform or unigram-frequency weighted) IS the quadratic part of the token-folded tensor, and that
this part carries 89-91% (MLP-0) / 39-62% (MLP-1) of the single-token write energy. The Aug-28 coefficient-metric HOSVD found a 64^3 core of MLP-1
retaining 0.33% of the tensor. This rung asks how much a symmetric Tucker projection retains in the TOKEN metric instead: with W_r = top-r output
eigenframe and P_r = top-r whitened input eigenframe of T_C (exact HOSVD frames, no fitting), the core
    G[a,p,q] = sym_pq sum_k (W^T Down)[a,k] (L C^1/2 P)[k,p] (R C^1/2 P)[k,q]
and retained(r) = ||G||^2 / ||sym T_C||^2 at r in {32, 64, 128, 256, 512}. Also reported: single-mode curves (top-r eigenvalue fractions, input and
output), the two-input-mode projection with the output mode kept whole, and the identical curves in the coefficient metric (C = I) for comparison.
Exact projections only; cores built through 4608 x r^2 chunks in fp64; no 1152^3 tensor.
PREDICTIONS (scored as written; failures preserved)
    pred_a_coefficient_core_reproduces   MLP-1's coefficient-metric 64^3 retained energy is in [0.001, 0.01] (Aug-28: 0.0033; band allows the
                                         symmetrisation / Down-gauge convention) (instrument)
    pred_b_token_metric_core_gain        MLP-1 uniform-centred 64^3 retained energy >= 10 x its coefficient-metric 64^3 value. Prior: likely
    pred_c_quarter_at_256                MLP-0 frequency-centred 256^3 core retains >= 0.25 of the energy. Prior: unsure
    pred_d_frequency_beats_uniform       at every r, frequency-centred retained >= uniform-centred retained, for the cubic core, both MLPs. Prior: likely
    pred_e_input_side_compresses_more    at r = 128 (uniform-centred, both MLPs) the two-input-mode projection with the output whole retains MORE
                                         than the output-only projection with the inputs whole (the token metric compresses the input side more
                                         than the output side; v610 ranks 507 vs 746 at MLP-1). Prior: likely for MLP-1, unsure for MLP-0
PRICE (registered maximum): 13 batched T=1 manual table forwards (v609 tables); 0 real model forwards; 0 backwards; 0 fits. Bar <= 14.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard
import embedding_forward_lib as EF

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_tucker_energy_v611_result.json"
OUT_PT = ROOT / "circuits/followups/embedding_forward_tucker_energy_v611_frames.pt"
FREQ_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
CANDIDATE_ID = "embedding_forward.tucker_energy_v611"
FORWARDS_MAX = 14
BATCH = 4096
RANKS = (32, 64, 128, 256, 512)
COEF_BAND = (0.001, 0.01)
GAIN_MIN = 10.0
QUARTER_MIN = 0.25
CHUNK = 64
PREDICTIONS = {"pred_a_coefficient_core_reproduces": "[0.001, 0.01]", "pred_b_token_metric_core_gain": ">= 10x", "pred_c_quarter_at_256": ">= 0.25",
               "pred_d_frequency_beats_uniform": "freq >= uniform at every r", "pred_e_input_side_compresses_more": "input-2mode(128) > output(128)"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "layers": [0, 1], "ranks": list(RANKS),
            "bars": {"coef_band": COEF_BAND, "gain_min": GAIN_MIN, "quarter_min": QUARTER_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model; blocks = model.transformer.h
    D = model.config.n_embd; V = model.config.vocab_size; dev = "cuda"
    with torch.no_grad():
        tabs = EF.single_token_tables(model, BATCH, dev); forwards = tabs["forwards"]
        p_freq, n_tok, n_seen = EF.unigram_weights(FREQ_ROWS, V, dev=dev)
        p_unif = torch.full((V,), 1.0 / V, device=dev, dtype=torch.float64)
        eye = torch.eye(D, device=dev, dtype=torch.float64)

        def core_energy(WD, LP, RP):
            """||sym core||^2 for core[a,p,q] = sum_k WD[a,k] LP[k,p] RP[k,q]; chunked over p."""
            r = LP.shape[1]; e = 0.0
            core = torch.empty(WD.shape[0], r, r, dtype=torch.float64, device=dev)
            for s in range(0, r, CHUNK):
                blk = (LP[:, s:s + CHUNK, None] * RP[:, None, :]).reshape(LP.shape[0], -1)            # [K, chunk*r]
                core[:, s:s + CHUNK, :] = (WD @ blk).reshape(WD.shape[0], -1, r)
            core = 0.5 * (core + core.transpose(1, 2))
            return float(core.square().sum())

        def curves(block, C):
            L, R, Dw = EF.mlp_weights(block); Gd = Dw.T @ Dw
            w, U = torch.linalg.eigh(C); S_half = (U * w.clamp_min(0).sqrt()) @ U.T
            GLL, GRR, GLR = L @ C @ L.T, R @ C @ R.T, L @ C @ R.T
            total = 0.5 * (float((Gd * GLL * GRR).sum()) + float((Gd * GLR * GLR.T).sum()))
            M2 = 0.25 * (L.T @ (Gd * GRR) @ L + L.T @ (Gd * GLR.T) @ R + R.T @ (Gd * GLR) @ L + R.T @ (Gd * GLL) @ R)
            in_ev, in_U = torch.linalg.eigh(S_half @ M2 @ S_half); in_ev, in_U = in_ev.flip(0), in_U.flip(1)
            Gout = 0.5 * Dw @ (GLL * GRR + GLR * GLR.T) @ Dw.T
            out_ev, out_U = torch.linalg.eigh(Gout); out_ev, out_U = out_ev.flip(0), out_U.flip(1)
            LS, RS = L @ S_half, R @ S_half
            res = {"total_energy": total, "single_mode_input": {}, "single_mode_output": {}, "two_input_modes": {}, "cubic": {}}
            for r in RANKS:
                P, W = in_U[:, :r], out_U[:, :r]
                res["single_mode_input"][str(r)] = float(in_ev[:r].sum() / in_ev.sum())
                res["single_mode_output"][str(r)] = float(out_ev[:r].sum() / out_ev.sum())
                LP, RP = LS @ P, RS @ P
                res["two_input_modes"][str(r)] = core_energy(Dw, LP, RP) / total
                res["cubic"][str(r)] = core_energy(W.T @ Dw, LP, RP) / total
            return res, {"in_frame": in_U[:, :max(RANKS)].float().cpu(), "out_frame": out_U[:, :max(RANKS)].float().cpu(), "in_evals": in_ev.float().cpu(), "out_evals": out_ev.float().cpu()}

        report = {"unigram": {"tokens": n_tok, "vocab_seen": n_seen}, "layers": {}}; frames = {}
        for l, Xin in ((0, tabs["X0"]), (1, tabs["X1"])):
            xh = EF.rms(Xin).double(); rep = {}
            for name, C in (("coefficient", eye),
                            ("uniform_centered", EF.second_moment(xh - (xh * p_unif[:, None]).sum(0), xh - (xh * p_unif[:, None]).sum(0), p_unif)),
                            ("frequency_centered", EF.second_moment(xh - (xh * p_freq[:, None]).sum(0), xh - (xh * p_freq[:, None]).sum(0), p_freq))):
                rep[name], fr = curves(blocks[l], C); frames[f"mlp{l}_{name}"] = fr
                print(f"MLP-{l} {name:18s}: cubic " + " ".join(f"r{r}={rep[name]['cubic'][str(r)]:.4f}" for r in RANKS)
                      + " | 2-input " + " ".join(f"{rep[name]['two_input_modes'][str(r)]:.3f}" for r in RANKS)
                      + " | in1 " + " ".join(f"{rep[name]['single_mode_input'][str(r)]:.3f}" for r in RANKS)
                      + " | out1 " + " ".join(f"{rep[name]['single_mode_output'][str(r)]:.3f}" for r in RANKS))
            report["layers"][str(l)] = rep
        disk_guard.guard_torch_save(frames, str(OUT_PT), "v611 HOSVD frames")

    L0, L1 = report["layers"]["0"], report["layers"]["1"]
    predictions = {"pred_a_coefficient_core_reproduces": COEF_BAND[0] <= L1["coefficient"]["cubic"]["64"] <= COEF_BAND[1],
                   "pred_b_token_metric_core_gain": L1["uniform_centered"]["cubic"]["64"] >= GAIN_MIN * L1["coefficient"]["cubic"]["64"],
                   "pred_c_quarter_at_256": L0["frequency_centered"]["cubic"]["256"] >= QUARTER_MIN,
                   "pred_d_frequency_beats_uniform": all(Lx["frequency_centered"]["cubic"][str(r)] >= Lx["uniform_centered"]["cubic"][str(r)] for Lx in (L0, L1) for r in RANKS),
                   "pred_e_input_side_compresses_more": all(Lx["uniform_centered"]["two_input_modes"]["128"] > Lx["uniform_centered"]["single_mode_output"]["128"] for Lx in (L0, L1))}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_tucker_energy_result_v611", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
