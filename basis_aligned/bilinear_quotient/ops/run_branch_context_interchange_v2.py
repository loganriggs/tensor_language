#!/usr/bin/env python3
# BQGATE: zero body forwards; 12288 cached endpoint tail rows plus <=32 controls.
"""Frozen amplitude interchange; no factor fitting.
pred_a: cached native/removal CE and identity <=1e-5; gradient FP64 <=1e-9;
finite, hash, pairing and derangement checks.
pred_b: each single exact swap CE >=.005 and document CI lower>0, both panels.
pred_c: each single linear swap CE >=.001 and CI lower>0, both panels.
Null: no beneficial within-family/domain amplitude-context alignment.
"""
import os
os.environ['OMP_NUM_THREADS'] = '2'
import hashlib
import json
from pathlib import Path
import signal
import time
import torch
import torch.nn.functional as F

RUNNER = Path(__file__).resolve()
ROOT = RUNNER.parents[3]
P = ROOT / 'basis_aligned/polynomial_causal'
STEM = 'BRANCH_CONTEXT_INTERCHANGE_V2'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()


def permutations(panel):
    n = len(panel['documents'])
    labels = panel.get('domain_labels', ['all'] * n)
    generator = torch.Generator().manual_seed(5701)
    result = torch.empty(4, 3*n, dtype=torch.long)
    for family in range(3):
        for domain in sorted(set(labels)):
            ids = torch.tensor([family*n+i for i, label in enumerate(labels) if label == domain])
            assert len(ids) >= 32
            order = ids[torch.randperm(len(ids), generator=generator)]
            for k, shift in enumerate((1, 7, 13, 19)):
                result[k, order] = order.roll(shift)
    for donor in result:
        assert torch.equal(donor.sort().values, torch.arange(3*n))
        assert bool((donor != torch.arange(3*n)).all())
        for i, j in enumerate(donor.tolist()):
            assert i//n == j//n and labels[i % n] == labels[j % n]
            assert panel['documents'][i % n] != panel['documents'][j % n]
    return result


def tail(h, unembedding):
    return 30 * torch.tanh(F.linear(F.rms_norm(h, (1152,), eps=torch.finfo(torch.float32).eps), unembedding)/30)


def loss_gradient(h, target, unembedding):
    # Analytic VJP of final RMS, linear map and capped logits; accumulation dtype follows h.
    r = (h.square().mean(1, keepdim=True)+torch.finfo(torch.float32).eps).rsqrt()
    t = torch.tanh(F.linear(h*r, unembedding)/30)
    logits = 30*t
    prob = logits.softmax(1)
    prob[torch.arange(len(h), device=h.device), target] -= 1
    g = (prob*(1-t.square())) @ unembedding
    gh = r*g-h*r.pow(3)*(g*h).mean(1, keepdim=True)
    return gh


def summarize(effects, panel, threshold):
    # effects: donor, family, document, arm. Donors are fixed for this diagnostic.
    averaged = effects.mean(0)
    doc = averaged.mean(0)
    indices = torch.randint(len(doc), (2000, len(doc)), generator=torch.Generator().manual_seed(5702))
    ci = torch.quantile(doc[indices].mean(1), torch.tensor([.025, .975], dtype=doc.dtype), dim=0)
    means = doc.mean(0)
    domains = panel.get('domain_labels', ['all']*len(doc))
    return dict(mean_ce_change=means.tolist(), descriptive_document_95=ci.tolist(),
        held=bool((means[:2] >= threshold).all() and (ci[0, :2] > 0).all()),
        per_family=averaged.mean(1).tolist(),
        per_domain={name: averaged[:, torch.tensor([i for i, label in enumerate(domains) if label==name])].mean((0, 1)).tolist()
                    for name in sorted(set(domains))},
        joint_minus_singles=float(means[2]-means[0]-means[1]))


def main():
    binding = json.loads((P / (STEM+'_BINDING.json')).read_text())
    assert all(digest(path)==sha for path, sha in binding['files'].items())
    data = []
    for label, suffix in [('fineweb', 'SUPPRESSION_V1'), ('corpus_shift', 'CORPUS_SHIFT_V1')]:
        panel = torch.load(P / f'SHARED_NODE_PARENT1_{suffix}_ROWS.pt', weights_only=True, map_location='cpu')
        cache = torch.load(P / f'SHARED_NODE_PARENT1_{suffix}_ENDPOINTS.pt', weights_only=True, map_location='cpu')
        assert len(panel['documents'])==128 and len(set(panel['documents']))==128
        assert cache['ports']['input'].shape==(384, 1152)
        assert panel['rows'].shape==(384, 129)
        for i, meta in enumerate(panel['metadata']):
            assert meta['family']==i//128 and meta['document']==panel['documents'][i % 128]
            assert meta['target_id']==int(panel['rows'][i, -1])
        data.append((label, panel, cache, permutations(panel)))
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False, body_forwards=0, cached_endpoints=768,
            tail_rows=12288, control_rows_max=32, fitting=False))); return
    output = P / (STEM+'_RESULT.json')
    assert not output.exists()
    signal.alarm(900)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    weights = torch.load(binding['checkpoint'], weights_only=True, mmap=True, map_location='cpu')
    unembedding = weights['lm_head.weight'].float().cuda()
    assert unembedding.shape==(50304, 1152)
    saved = torch.load(P / 'SHARED_NODE_CANONICAL_BRANCHES_V1_SPECTRAL.pt', weights_only=True, map_location='cpu')
    node = saved['nodes'][1]
    writers = torch.linalg.solve_triangular(saved['output_whitener'].double(), node['writers'].double(), upper=True).cuda()
    results, replay, gradient_errors, identity_errors = {}, [], [], []
    for label, panel, cache, donors in data:
        x = cache['ports']['input'].double()
        amplitudes = (x @ node['reader'].double())[:, None]*(x @ node['partners'].double())
        h = (cache['ports']['pre']+cache['ports']['native_output']).cuda()
        amp = amplitudes.cuda()
        targets = panel['rows'][:, -1].cuda()
        exact = torch.empty(4, 384, 3, dtype=torch.float64)
        linear = torch.empty_like(exact)
        donors_gpu = donors.cuda()
        # Independent autograd check uses FP64, explicitly retaining native FP32 RMS epsilon.
        hcheck = h[:4].double().detach().requires_grad_(True)
        ucheck = unembedding.double()
        loss = F.cross_entropy(tail(hcheck, ucheck), targets[:4], reduction='sum')
        auto = torch.autograd.grad(loss, hcheck)[0]
        analytic = loss_gradient(hcheck.detach(), targets[:4], ucheck)
        gradient_errors.append(float((auto-analytic).norm()/auto.norm()))
        del ucheck, hcheck, loss, auto, analytic
        with torch.no_grad():
            for offset in range(0, 384, 8):
                end = offset+8
                hh, tt = h[offset:end], targets[offset:end]
                baseline = F.cross_entropy(tail(hh, unembedding).double(), tt, reduction='none')
                deltas = [(amp[offset:end, j:j+1]*writers[:, j]).float() for j in range(2)]
                removal = []
                # Match original pre+(native-delta) arithmetic, not just equivalent h-delta.
                pre = cache['ports']['pre'][offset:end].cuda()
                native = cache['ports']['native_output'][offset:end].cuda()
                for delta in (deltas[0], deltas[1], deltas[0]+deltas[1]):
                    removal.append(F.cross_entropy(tail(pre+(native-delta), unembedding).double(), tt, reduction='none')-baseline)
                computed = torch.stack([baseline]+removal, 1).cpu()
                reference = cache['scores'].reshape(384, 4)[offset:end]
                replay.append(float((computed-reference).abs().max()))
                g = loss_gradient(hh, tt, unembedding).double()
                for k in range(4):
                    difference = amp[donors_gpu[k, offset:end]]-amp[offset:end]
                    shifts = [difference[:, j:j+1]*writers[:, j] for j in range(2)]
                    for j, shift in enumerate((shifts[0], shifts[1], shifts[0]+shifts[1])):
                        exact[k, offset:end, j] = (F.cross_entropy(tail(hh+shift.float(), unembedding).double(), tt, reduction='none')-baseline).cpu()
                        linear[k, offset:end, j] = (g*shift).sum(1).cpu()
                if offset==0:
                    identity = hh + ((amp[:8]-amp[:8])@writers.T).float()
                    identity_errors.append(float((identity-hh).abs().max()))
        assert bool(torch.isfinite(exact).all() and torch.isfinite(linear).all())
        exact = exact.reshape(4, 3, 128, 3)
        linear = linear.reshape(4, 3, 128, 3)
        results[label] = dict(exact=summarize(exact, panel, .005), linear=summarize(linear, panel, .001),
            donor_indices=donors.tolist(), exact_document_effects=exact.mean((0, 1)).tolist(),
            linear_document_effects=linear.mean((0, 1)).tolist())
        print(json.dumps(dict(panel=label, exact=results[label]['exact'], linear=results[label]['linear'])), flush=True)
    a = max(replay)<=1e-5 and max(gradient_errors)<=1e-9 and max(identity_errors)<=1e-5
    result = dict(pred_a=a, pred_b=a and all(r['exact']['held'] for r in results.values()),
        pred_c=a and all(r['linear']['held'] for r in results.values()),
        max_cached_ce_replay_error=max(replay), gradient_errors=gradient_errors, identity_errors=identity_errors,
        panels=results, seconds=time.perf_counter()-started,
        price=dict(body_forwards=0, cached_endpoints=768, tail_rows=12288, native_background_retained=True),
        binding_sha256=digest(P / (STEM+'_BINDING.json')),
        scope='Frozen amplitude interchange within family/domain on previously inspected panels. '
              'No parameter fitting, fresh confirmation, standalone sufficiency or semantic circuit identification. '
              'Bootstrap intervals condition on fixed donors; donor dependence is not fully captured.')
    with output.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({k:result[k] for k in ('pred_a','pred_b','pred_c','seconds','max_cached_ce_replay_error')}), flush=True)


if __name__=='__main__': main()
