"""Frozen native checkpoint: 32 independent full-tensor support probes.

pred_a CP replay<=1e-9 normalized and monotone finite128supports;
pred_b >=24 swap gains>=1e-7; pred_c median swap>=10x median refit;
pred_d compute<=120s. No installation of concurrent swaps or additive claim.
"""
import hashlib
import io
import json
import time
from pathlib import Path
import torch
from folded_sparse_dictionary_v1 import decode
from folded_support_exchange_v1 import quadratic, best_exchange
from structured_branch_amplitudes_v1 import inner

P = Path(__file__).resolve().parent
CK = Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def main():
    torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    output = P / 'NATIVE_SUPPORT_EXCHANGE_V1_AUDIT.json'
    assert not output.exists()
    source = Path('/dev/shm/bilin18_projected_sparse_dictionary_fit_v1_s0.pt')
    content = source.read_bytes()  # Writer replaces atomically; this inode is one accepted checkpoint.
    sha = hashlib.sha256(content).hexdigest()
    frozen = Path('/dev/shm/bilin18_native_support_exchange_v1_source.pt')
    with frozen.open('xb') as f: f.write(content)
    saved = torch.load(io.BytesIO(content), map_location='cpu', weights_only=True)
    del content
    sd = torch.load(CK, mmap=True, weights_only=True, map_location='cpu')
    l, r, down = [sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ('Left','Right','Down')]
    metric = torch.load('/dev/shm/bilin18_native_product_energy_v1.pt', weights_only=True,
                        map_location='cpu')['unembedding_gram'].double()
    wh = torch.linalg.cholesky(metric).T
    native = (l, r, wh @ down)
    basis = saved['analysis_basis'].double(); ids = saved['code_indices'].long()
    values = saved['code_values'].double()
    readers, basis, _, _ = decode(basis, ids, values, torch.ones(len(ids)))
    a, b = readers.chunk(2); w = wh @ saved['down'].double()
    candidate = (a, b, w)
    total = json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())['native_total']
    started = time.perf_counter(); basegram = basis @ basis.T; rows = []
    for side in ('Left','Right'):
        aa, bb = (a,b) if side == 'Left' else (b,a)
        nn = native if side == 'Left' else (r,l,native[2])
        for j in [i*4608//16 for i in range(16)]:
            index = j if side == 'Left' else j+4608
            support = ids[index].tolist()
            gram, rhs = quadratic(nn, aa, bb, w, j, basis, basegram)
            old = values[index]
            ss = gram[support][:,support]
            beta = torch.linalg.solve(ss, rhs[support])
            initial_energy = old @ ss @ old - 2 * old @ rhs[support]
            refit_energy = -rhs[support] @ beta
            refit_gain = float((initial_energy-refit_energy)/total)
            new_ids, new_values, report = best_exchange(gram, rhs, support)
            replacement = new_values @ basis[new_ids]
            delta = ((replacement-aa[j])[None],bb[j:j+1],w[:,j:j+1])
            cp_gain = -float((inner(delta,delta)+2*inner(candidate,delta)-2*inner(native,delta))/total)
            swap_gain = report['gain']/total
            replay = abs(cp_gain-refit_gain-swap_gain)
            row = dict(side=side, product=j, refit_gain=refit_gain, swap_gain=swap_gain,
                       independent_cp_total_gain=cp_gain, replay_error=replay,
                       support_cardinality=len(set(new_ids)), **report)
            rows.append(row)
            print(json.dumps(row), flush=True)
    seconds = time.perf_counter()-started
    median_swap = float(torch.tensor([r['swap_gain'] for r in rows]).median())
    median_refit = float(torch.tensor([r['refit_gain'] for r in rows]).median())
    passed = dict(pred_a_instrument=all(r['replay_error']<=1e-9 and r['support_cardinality']==128
                      and r['independent_cp_total_gain']>=-1e-9
                      and bool(torch.isfinite(torch.tensor(list((r['refit_gain'],r['swap_gain'],r['replay_error'])))).all())
                      for r in rows),
                  pred_b_coverage=sum(r['swap_gain']>=1e-7 for r in rows)>=24,
                  pred_c_graph_dominates=median_swap>=10*max(median_refit,0.),
                  pred_d_cost=seconds<=120)
    result = dict(predictions=passed, rows=rows, median_swap_gain=median_swap,
                  median_refit_gain=median_refit, compute_seconds=seconds,
                  source=dict(path=str(frozen),sha256=sha,iteration=saved['history'][-1]['iteration'],
                              loss=saved['loss']),body_forwards=0,
                  scope='Independent single-reader fixed-writer conditional swaps; gains not additive. '
                        'Snapshot of unconverged native fit; not global support recovery or circuit identification.')
    with output.open('x') as f: json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__': main()
