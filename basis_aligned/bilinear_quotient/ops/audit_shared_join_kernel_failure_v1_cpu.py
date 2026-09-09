#!/usr/bin/env python3
"""Post-result CPU diagnosis on opened cases; no new fitted candidate or promotion.
Compare original and exact-native-score field-support gating, splitting support
failure from fitted amplitude failure. All results are diagnostic and reused.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import types

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path('/workspace/tensor_language'); POLY = ROOT/'basis_aligned/polynomial_causal'
sys.path[:0] = [str(ROOT), str(POLY), str(ROOT/'basis_aligned/bilinear_quotient/ops')]
import torch
import shared_join_kernel_reference as R
import run_equality_router_v1 as score
from hop_ablate import load


def main():
    os.chdir(ROOT); torch.set_num_threads(2); signal.alarm(300); started = time.perf_counter()
    out = POLY/'SHARED_JOIN_KERNEL_FAILURE_AUDIT_V1.json'
    assert not out.exists()
    path = POLY/'SHARED_JOIN_KERNEL_V2_COVERAGE_ROWS.pt'
    coeff_path = POLY/'SHARED_JOIN_KERNEL_V2_COVERAGE_COEFFICIENTS.pt'
    rows = torch.load(path, map_location='cpu', weights_only=True)
    coeff = torch.load(coeff_path, map_location='cpu', weights_only=True)
    model, _ = load('attn4-rms-seed0'); model = model.double().eval()
    assert next(model.parameters()).device.type == 'cpu'
    layer = model.layers[2]; original = layer.pattern; data = {}
    def gated(_self, x):
        p = original(x).clone()
        for h in range(2):
            p[:, h+1] *= R.indices(data['tokens'], h)[2]
        return p
    result = {}
    with torch.inference_mode():
        for pop, record in rows.items():
            tokall = record['tokens']; support_logits = []; extremes = []; native_error = 0.
            radial_native, radial_rule = [], []
            for i in range(0, len(tokall), 4):
                tok = tokall[i:i+4]; data['tokens'] = tok
                x = model.embed(tok)
                for prefix in model.layers[:2]: x = prefix(x)
                gain = R.rms_gain(x, layer.norm); native_pattern = original(x)[:, 1:3]
                approx = R.fitted_pattern(tok, gain, coeff['theta'], coeff['gamma'], coeff['seen'])
                flat = approx.abs().flatten(1).argmax(-1)
                for j, idx in enumerate(flat.tolist()):
                    head, rem = divmod(idx, 51*51); q, s = divmod(rem, 51)
                    extremes.append({'row': i+j, 'head': head+1, 'query_position': q, 'source_position': s,
                                     'query_gain': float(gain[j, q]), 'source_gain': float(gain[j, s]),
                                     'candidate_score': float(approx[j, head, q, s]),
                                     'native_same_score': float(native_pattern[j, head, q, s]),
                                     'native_max_abs_score': float(native_pattern[j].abs().max())})
                # Sensitivity of formulas to residual rescaling, not a legal token counterfactual.
                if i == 0:
                    p2 = original(2*x)[:, 1:3]
                    a2 = R.fitted_pattern(tok, R.rms_gain(2*x, layer.norm), coeff['theta'], coeff['gamma'], coeff['seen'])
                    radial_native.append(float((p2-native_pattern).square().sum().sqrt()/native_pattern.square().sum().sqrt()))
                    radial_rule.append(float(a2.square().sum().sqrt()/approx.square().sum().sqrt()))
                native_error = max(native_error, float((model(tok)-record['native']['full'][i:i+4]).abs().max()))
                layer.pattern = types.MethodType(gated, layer)
                try:
                    support_logits.append(model(tok))
                finally:
                    del layer.pattern
            support = torch.cat(support_logits); native = record['native']['full']
            delta = support-native
            rms_by_position = delta.sub(delta.mean(-1, keepdim=True)).square().mean((0, 2)).sqrt()
            result[pop] = {'support_only': {'all': score.distribution(support, native, torch),
                                          'query': score.distribution(support[:, -1], native[:, -1], torch)},
                           'support_query_accuracy': float((support[:, -1].argmax(-1) == record['answers']).double().mean()),
                           'native_cpu_gpu_max_error': native_error,
                           'support_centered_error_rms_by_position': rms_by_position.tolist(),
                           'largest_candidate_score_cases': sorted(extremes, key=lambda v: abs(v['candidate_score']), reverse=True)[:5],
                           'native_score_relative_change_under_2x_residual': radial_native,
                           'rule_score_norm_ratio_under_2x_residual': radial_rule}
            print(json.dumps({'population': pop, 'support_only': result[pop]['support_only'],
                              'largest_candidate_score_case': result[pop]['largest_candidate_score_cases'][0]}), flush=True)
    receipt = {'analysis': 'shared_join_kernel_failure_audit_v1', 'status': 'opened_case_diagnostic_no_promotion',
               'source_rows_sha256': score.digest(path), 'coefficients_sha256': score.digest(coeff_path),
               'code_sha256': score.digest(Path(__file__)), 'populations': result,
               'formula_note': 'Ignoring epsilon, native RMS-normalized scores are invariant to x->2x; the fixed kernel scales by1/16. This synthetic sensitivity is not a token intervention.',
               'wall_seconds': time.perf_counter()-started}
    out.write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__': main()
