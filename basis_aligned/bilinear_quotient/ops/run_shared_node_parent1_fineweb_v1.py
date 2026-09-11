#!/usr/bin/env python3
# BQGATE: 11forwards88seq128tokens; 72 frozen FineWeb endpoints, no fitting.
"""Frozen native MLP17 branch removal screen.
pred_a: finite/count/row checks and native/branch0 physical logit replay <=1e-5.
pred_b: native actual-target top20 fraction >=.5 in both target families.
pred_c, conditional on A/B: each own CE damage >=.02, own-minus-other >=.01,
and each single branch mean absolute control CE change <=.05.
Null: algebraic branches need not selectively support their weight-token families.
Price: 11 body forwards, 88 sequences of 128 inputs; native background retained.
"""
import os
os.environ['OMP_NUM_THREADS'] = '2'
import hashlib
import json
import signal
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F

RUNNER = Path(__file__).resolve()
ROOT = RUNNER.parents[3]
P = ROOT / 'basis_aligned/polynomial_causal'
STEM = 'SHARED_NODE_PARENT1_FINEWEB_V1'
sys.path[:0] = [str(RUNNER.parent), str(ROOT)]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    binding = json.loads((P / (STEM + '_BINDING.json')).read_text())
    assert all(digest(path) == sha for path, sha in binding['files'].items())
    panel = torch.load(P / (STEM + '_ROWS.pt'), weights_only=True, map_location='cpu')
    rows = panel['rows']
    assert rows.shape == (72, 129) and rows.dtype == torch.long
    assert len(set(panel['documents'])) == 24
    source = torch.load(ROOT / binding['row_source'], weights_only=True, map_location='cpu')
    for i, record in enumerate(panel['metadata']):
        family, doc, pos = record['family'], record['document'], record['target_position']
        assert family == i // 24 and doc == panel['documents'][i % 24]
        assert torch.equal(rows[i], source[doc, pos-128:pos+1])
        target = int(rows[i, -1])
        assert target == record['target_id']
        assert (target in panel['families'][0]) == (family == 0)
        assert (target in panel['families'][1]) == (family == 1)
    saved = torch.load(P / 'SHARED_NODE_CANONICAL_BRANCHES_V1_SPECTRAL.pt', weights_only=True, map_location='cpu')
    node = saved['nodes'][1]
    assert node['reader'].shape == (1152,) and node['partners'].shape == (1152, 2)
    assert node['writers'].shape == (1152, 2)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(model_loaded=False, gpu_accessed=False, fitting=False,
            body_forwards=11, sequences=88, input_tokens=128, endpoints=72,
            document_units=24, interventions=['branch0', 'branch1', 'both']))); return
    output = P / (STEM + '_RESULT.json')
    assert not output.exists()
    signal.alarm(900)
    started = time.perf_counter()
    torch.set_grad_enabled(False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    from fastload import load_model_fast
    model = load_model_fast().cuda().eval()
    last = model.transformer.h[17]
    mlp = last.mlp
    assert len(model.transformer.h) == 18 and not mlp.config.gated
    assert mlp.Left.weight.shape == (4608, 1152)
    u, partners = node['reader'].double().cuda(), node['partners'].double().cuda()
    writers = torch.linalg.solve_triangular(saved['output_whitener'].double(),
        node['writers'].double(), upper=True).cuda()
    counts = [0, 0]
    def count(module, args):
        counts[0] += 1; counts[1] += len(args[0])
        assert args[0].shape[1:] == (128, 1152) and counts[0] <= 11
    handle = model.transformer.h[0].attn.register_forward_pre_hook(count)
    def delta(x, branch):
        xx = x.double()
        return (((xx @ u) * (xx @ partners[:, branch]))[..., None] * writers[:, branch]).float()
    def logits(h):
        return 30 * torch.tanh(model.lm_head(F.rms_norm(h, (1152,))) / 30)
    def prefix(tokens):
        x = F.rms_norm(model.transformer.wte(tokens), (1152,)); x0 = x; v1 = None
        for block in model.transformer.h[:17]:
            x, v1 = block(x, v1, x0)
        x = last.lambdas[0] * x + last.lambdas[1] * x0
        attention, v1 = last.attn(F.rms_norm(x, (1152,)), v1)
        pre = x + attention
        xin = F.rms_norm(pre, (1152,))
        return xin, pre, mlp(xin)
    controls, scores, ranks, ports = [], [], [], []
    try:
        for offset in range(0, 72, 8):
            tokens = rows[offset:offset+8, :128].contiguous().cuda()
            targets = rows[offset:offset+8, 1:].contiguous().cuda()
            xin, pre, native = prefix(tokens)
            xx, pp, nn = xin[:, -1], pre[:, -1], native[:, -1]
            d0, d1 = delta(xx, 0), delta(xx, 1)
            base = logits(pp + nn)
            alternatives = [logits(pp + (nn-d0)), logits(pp + (nn-d1)), logits(pp + (nn-(d0+d1)))]
            if offset == 0:
                for branch in (None, 0):
                    capture = {}
                    def head_hook(module, args, value):
                        capture['raw'] = value[:, -1].detach().clone()
                    def mlp_hook(module, args, value):
                        capture['input_error'] = float((args[0]-xin).norm()/xin.norm())
                        return value if branch is None else value-delta(args[0], branch)
                    hooks = [model.lm_head.register_forward_hook(head_hook), mlp.register_forward_hook(mlp_hook)]
                    try:
                        model(tokens, targets)
                    finally:
                        for hook in hooks: hook.remove()
                    physical = 30 * torch.tanh(capture['raw']/30)
                    manual = base if branch is None else alternatives[branch]
                    controls.append(dict(branch=branch, input_error=capture['input_error'],
                        logit_relative_error=float((physical-manual).norm()/physical.norm())))
            target = targets[:, -1]
            native_ce = F.cross_entropy(base.double(), target, reduction='none')
            effects = [F.cross_entropy(a.double(), target, reduction='none')-native_ce for a in alternatives]
            scores.append(torch.stack([native_ce]+effects, -1).cpu())
            ranks.append((1+(base > base.gather(1, target[:, None])).sum(1)).cpu())
            ports.append(dict(input=xx.cpu(), pre=pp.cpu(), native_output=nn.cpu()))
            print(json.dumps(dict(endpoints_completed=offset+8, body_counts=counts)), flush=True)
    finally:
        handle.remove()
    scores = torch.cat(scores).reshape(3, 24, 4)
    ranks = torch.cat(ranks).reshape(3, 24)
    effects = scores[:, :, 1:]
    means = effects.mean(1)
    capability = (ranks <= 20).double().mean(1)
    valid = bool(torch.isfinite(scores).all()) and counts == [11, 88] and all(
        c['input_error'] <= 1e-6 and c['logit_relative_error'] <= 1e-5 for c in controls)
    pred_b = valid and bool((capability[:2] >= .5).all())
    own = torch.stack([means[0, 0], means[1, 1]])
    contrast = torch.stack([means[0, 0]-means[0, 1], means[1, 1]-means[1, 0]])
    collateral = effects[2, :, :2].abs().mean(0)
    pred_c = pred_b and bool((own >= .02).all() and (contrast >= .01).all() and (collateral <= .05).all())
    indices = torch.randint(24, (2000, 24), generator=torch.Generator().manual_seed(4302))
    bootstrap = effects[:, indices, :].mean(2)
    intervals = torch.quantile(bootstrap, torch.tensor([.025, .975], dtype=bootstrap.dtype), dim=1)
    cache = P / (STEM + '_ENDPOINTS.pt')
    assert not cache.exists()
    torch.save(dict(ports={k:torch.cat([v[k] for v in ports]) for k in ports[0]}, scores=scores,
        ranks=ranks, scope='Frozen validation only; no fitting authorization.'), cache)
    result = dict(pred_a=valid, pred_b=pred_b, pred_c=pred_c,
        family_names=['branch0_weight_tokens', 'branch1_weight_tokens', 'nearby_control'],
        intervention_names=['remove_branch0', 'remove_branch1', 'remove_both'],
        mean_ce_added=means.tolist(), native_ce=scores[:, :, 0].mean(1).tolist(),
        native_top20_fraction=capability.tolist(), own_ce_added=own.tolist(), own_minus_other=contrast.tolist(),
        mean_absolute_control_ce_change=collateral.tolist(),
        joint_minus_sum_ce_added=(effects[:, :, 2]-effects[:, :, 0]-effects[:, :, 1]).mean(1).tolist(),
        descriptive_document_bootstrap_95=intervals.tolist(), row_scores=scores.tolist(),
        physical_controls=controls, price=dict(body_forwards=counts[0], sequences=counts[1], input_tokens=128,
        distinct_endpoints=72, document_units=24, standalone_branch_bank_coefficients=5760,
        native_background_retained=True), cache=dict(path=str(cache), sha256=digest(cache)),
        binding_sha256=digest(P / (STEM + '_BINDING.json')), seconds=time.perf_counter()-started,
        scope='Historical FineWeb frozen weight-branch validation; no fitting, fresh/OOD, sufficiency, or circuit-identification claim.')
    with output.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({k:result[k] for k in ('pred_a', 'pred_b', 'pred_c', 'mean_ce_added', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
