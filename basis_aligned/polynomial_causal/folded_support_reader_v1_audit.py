"""One exact conditional Left/Right update of a frozen sparse native program.
Predictions registered on AGENT_BOARD; CPU only, zero text/model forwards.
"""
import hashlib
import json
from pathlib import Path
import time
import torch
from native_reader_msp_generalization_v1 import P, CK
from symmetric_product_als_v1 import normal_operator, native_rhs
from folded_support_reader_v1 import decode, project, solve
from structured_branch_amplitudes_v1 import inner


def dense(a, b, w):
    return (torch.einsum('oj,ja,jb->oab', w, a, b) +
            torch.einsum('oj,jb,ja->oab', w, a, b)) / 2


def control():
    torch.manual_seed(711)
    n, d, k, o = 3, 7, 4, 5
    atoms = torch.randn(n, k, d)
    b = torch.randn(n, d); w = torch.randn(o, n); z = torch.randn(n, k)
    l = torch.randn(6, d); r = torch.randn(6, d); down = torch.randn(o, 6)
    design = []
    for i in range(n * k):
        zi = torch.zeros_like(z); zi.flatten()[i] = 1
        design.append(dense(decode(zi, atoms), b, w).flatten())
    design = torch.stack(design, 1); target = dense(l, r, down).flatten()
    rhs = native_rhs(l, r, down, b, w, torch.eye(o))
    hz = project(normal_operator(decode(z, atoms), b, w.T @ w), atoms).flatten()
    errors = [float((hz - design.T @ design @ z.flatten()).norm() / hz.norm()),
              float((project(rhs, atoms).flatten() - design.T @ target).norm() /
                    (design.T @ target).norm())]
    fitted, report = solve(atoms, b, w.T @ w, rhs, z)
    reference = torch.linalg.lstsq(design, target).solution
    errors.append(float((fitted.flatten() - reference).norm() / reference.norm()))
    return dict(errors=errors, solver=report, passed=max(errors) <= 1e-9 and report['converged'])


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    started = time.perf_counter(); check = control(); assert check['passed'], check
    parent = json.loads((P / 'OBLIQUE_READER_DICTIONARY_V1_ordinary_covariance_SEED_0.json').read_text())
    cache = Path(parent['cache']['path'])
    assert hashlib.sha256(cache.read_bytes()).hexdigest() == parent['cache']['sha256']
    saved = torch.load(cache, weights_only=True, map_location='cpu')
    selected = torch.tensor(json.loads((P / 'NATIVE_OBLIQUE_ENCODER_V1_AUDIT.json').read_text())['selected_products'])
    sd = torch.load(CK, weights_only=True, map_location='cpu', mmap=True)
    l, r, down = [sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ('Left', 'Right', 'Down')]
    metric = torch.load('/dev/shm/bilin18_native_product_energy_v1.pt', weights_only=True, map_location='cpu')['unembedding_gram']
    vals, vecs = torch.linalg.eigh(metric); root = vals.clamp_min(0).sqrt()[:, None] * vecs.T
    writers = root @ down; chosen_w = writers[:, selected]
    output_gram = chosen_w.T @ chosen_w
    basis = saved['analysis_basis']; indices = saved['code_indices'].long()
    values = saved['code_values'].clone()
    readers = torch.cat([decode(values[i:i+64], basis[indices[i:i+64]]) for i in range(0, len(values), 64)])
    a, b = readers[:4608].clone(), readers[4608:].clone()
    original_a, original_b = a.clone(), b.clone()
    target_energy = float(json.loads((P / 'FULLU_TRACE_METRIC_V1_AUDIT.json').read_text())['full_coefficient_metric']['total_energy'])
    rows = []; deltas = []
    for name in ('left', 'right'):
        tic = time.perf_counter()
        variable, partner = (a, b) if name == 'left' else (b, a)
        offset = 0 if name == 'left' else 4608
        ids = selected + offset; atoms = basis[indices[ids]]
        other = partner[selected]
        rhs = native_rhs(l, r, down, other, down[:, selected], metric)
        rhs -= native_rhs(a, b, down, other, down[:, selected], metric)
        rhs += normal_operator(variable[selected], other, output_gram)
        fitted, report = solve(atoms, other, output_gram, rhs, values[ids])
        before = variable[selected].clone(); after = decode(fitted, atoms); change = after - before
        operator_change = normal_operator(change, other, output_gram)
        delta_error = float((change * operator_change).sum() +
                            2 * (change * (normal_operator(before, other, output_gram) - rhs)).sum())
        deltas.append(delta_error)
        variable[selected] = after; values[ids] = fitted
        row = dict(operand=name, solver=report, relative_full_error_change=delta_error / target_energy,
                   seconds=time.perf_counter()-tic)
        rows.append(row); print(json.dumps(row), flush=True)
    # Independent CP contraction: delta = new selected products - old selected products.
    change_cp = (torch.cat((a[selected], original_a[selected])),
                 torch.cat((b[selected], original_b[selected])),
                 torch.cat((chosen_w, -chosen_w), 1))
    original_cp = (original_a, original_b, writers); native_cp = (l, r, writers)
    independent_delta = float(inner(change_cp, change_cp) + 2*inner(original_cp, change_cp) - 2*inner(native_cp, change_cp))
    replay = abs(independent_delta - sum(deltas)) / target_energy
    before_capture = parent['scores']['coefficient_capture']; gain = -independent_delta / target_energy
    output = dict(control=check, source=parent['cache'], selected_products=selected.tolist(), arms=rows,
        independent_error_change_replay=replay, capture_before=before_capture, capture_after=before_capture+gain,
        capture_gain=gain, predictions=dict(
            pred_a_instrument=check['passed'] and replay<=1e-9 and all(x['solver']['converged'] for x in rows),
            pred_b_nonworsening=all(x['relative_full_error_change']<=1e-10 for x in rows),
            pred_c_gain=gain>=.0001), seconds=time.perf_counter()-started,
        scope='One conditional Left/Right sweep, full-U weight loss, fixed basis/supports/Down. No joint convergence or heldout behavioral claim; same program price.')
    artifact = Path('/dev/shm/bilin18_folded_support_reader_v1.pt'); assert not artifact.exists()
    torch.save(dict(selected_products=selected, left_values=values[selected], right_values=values[selected+4608],
                    source=parent['cache']), artifact)
    output['cache'] = dict(path=str(artifact), sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
    (P/'FOLDED_SUPPORT_READER_V1_AUDIT.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k: v for k, v in output.items() if k not in ('selected_products','arms')}), flush=True)


if __name__ == '__main__':
    main()
