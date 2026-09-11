"""CPU native-checkpoint gauge audit, registered on AGENT_BOARD 06:51.
pred_a map replay1e-10; pred_b 2xenergy reduction both; pred_c balance1e-6.
Does not measure optimization speed or change the live fit.
"""
import hashlib
import json
from pathlib import Path
import time
import torch
from structured_bilinear_bank_v1 import StructuredBank
from structured_stage_balance_v1 import balance

P = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    started = time.perf_counter()
    reports = []
    native = json.loads((P / 'STRUCTURED_BILINEAR_NATIVE_V2_RESULT.json').read_text())
    for prior in native['starts']:
        assert digest(prior['cache']['path']) == prior['cache']['sha256']
        saved = torch.load(prior['cache']['path'], map_location='cpu', weights_only=True)
        model = StructuredBank([2]*7+[3,3], branches=4)
        model.load_state_dict(saved['model'])
        rows = []
        for branch_index, branch in enumerate(model.maps):
            for map_index, transform in enumerate(branch):
                before = transform.matrix()
                row = balance(transform)
                after = transform.matrix()
                row.update(branch=branch_index, map=map_index,
                    map_relative_error=float((before-after).norm()/before.norm()),
                    max_absolute_error=float((before-after).abs().max()))
                rows.append(row)
        cache = Path(f'/dev/shm/bilin18_structured_balanced_v1_s{prior["seed"]}.pt')
        assert not cache.exists()
        torch.save(dict(model=model.state_dict(), seed=prior['seed'],
            prior_cache=prior['cache'], optimizer_history_compatible=False), cache)
        result = dict(seed=prior['seed'], maps=rows,
            total_energy_reduction_factor=sum(r['initial_energy'] for r in rows)/sum(r['final_energy'] for r in rows),
            max_map_error=max(r['map_relative_error'] for r in rows),
            max_final_imbalance=max(r['final_imbalance'] for r in rows),
            cache=dict(path=str(cache), sha256=digest(cache), bytes=cache.stat().st_size),
            prior_cache=prior['cache'])
        reports.append(result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('maps','cache','prior_cache')}),flush=True)
    result = dict(predictions={
        'pred_a_replay': all(r['max_map_error'] <= 1e-10 for r in reports),
        'pred_b_energy_halved': all(r['total_energy_reduction_factor'] >= 2 for r in reports),
        'pred_c_balanced': all(r['max_final_imbalance'] <= 1e-6 for r in reports)},
        starts=reports, seconds=time.perf_counter()-started,
        source_hashes={str(p):digest(p) for p in (Path(__file__),P/'structured_stage_balance_v1.py')},
        gpu_access=False, corpus_access=False,
        scope='Same-function stage reparameterization; not coefficient fit improvement, optimization speed evidence or circuit identification.')
    with (P/'STRUCTURED_STAGE_BALANCE_V1_AUDIT.json').open('x') as f:
        json.dump(result,f,indent=2)
        f.write('\n')
    print(json.dumps(dict(predictions=result['predictions'],seconds=result['seconds'])),flush=True)


if __name__ == '__main__':
    main()
