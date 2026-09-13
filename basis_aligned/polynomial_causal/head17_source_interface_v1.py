"""Weight-only interface membership of the existing four regional source readers."""
from pathlib import Path
import json
import torch
from compiled_shared_head_v1 import execute_head
from folded_normalized_router_v1 import rotary
from shared_head_native_ports_v1 import execute as execute_ports

P = Path(__file__).resolve().parent
CHECKPOINT = Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def main():
    torch.set_num_threads(2)
    weights = torch.load(CHECKPOINT, map_location='cpu', weights_only=True, mmap=True)
    def head(layer, name):
        return weights[f'transformer.h.{layer}.attn.{name}.weight'].reshape(9, 128, 1152)[2].double()
    mu = float(weights['transformer.h.17.attn.lamb'])
    k1, k2 = head(17, 'c_k'), head(17, 'c_k2')
    value = torch.cat(((1-mu)*head(17, 'c_v'), mu*head(0, 'c_v')), dim=1)
    stack = torch.cat((torch.cat((k1, torch.zeros_like(k1)), 1),
                       torch.cat((k2, torch.zeros_like(k2)), 1), value))
    program = torch.load(P/'COMPILED_SHARED_HEAD2_V1_ARTIFACT.pt', weights_only=True)
    readers = torch.cat((program['parent'], program['children'])).double()
    u, singular, vh = torch.linalg.svd(stack, full_matrices=False)
    assert singular[-1] > singular[0]*1e-12
    coefficients = ((readers @ vh.T)/singular) @ u.T
    projected = coefficients @ stack
    residual = readers-projected
    # An independent QR projection discriminates solver error from absent membership.
    q, _ = torch.linalg.qr(stack.T, mode='reduced')
    qr_projected = (readers @ q) @ q.T
    torch.manual_seed(170203)
    x = torch.randn(2048, 2304, dtype=torch.float64)
    original = x @ readers.T
    candidate = (x @ stack.T) @ coefficients.T
    def features(read):
        return (read[:, 0]*read[:, 1])[:, None]*read[:, 2:]
    f, fp = features(original), features(candidate)
    result = dict(
        source_stack_shape=list(stack.shape), reader_shape=list(readers.shape),
        original_reader_dtype=str(program['parent'].dtype), mu=mu,
        stack_condition=float(singular[0]/singular[-1]),
        relative_reader_errors=(residual.norm(dim=1)/readers.norm(dim=1)).tolist(),
        qr_svd_projection_disagreement=float((qr_projected-projected).norm()/readers.norm()),
        residual_normal_equation_error=float((residual @ stack.T).norm()/(readers.norm()*stack.norm())),
        random_probe_cubic_feature_errors=((fp-f).norm(dim=0)/f.norm(dim=0)).tolist(),
        scope='Untruncated weight-space membership and Gaussian feature check, not native causal equivalence. The four readers belong to a globally fitted source block; dominance of head17.2 writes does not imply head17.2 support of each reader.',
    )
    cache = torch.load(P/'COMMON_QUADRATIC_NATIVE_V1_PORTS.pt', weights_only=True, mmap=True)
    current, first = cache['current'].double(), cache['first'].double()
    query = current[:, 8]
    changed = dict(program, parent=projected[:2], children=projected[2:])
    sums = [torch.zeros_like(query), torch.zeros_like(query)]
    source_errors = []
    reading_errors = []
    interface_errors = []
    corrected_errors = []
    for pos in range(9):
        source = torch.cat((current[:, pos], first[:, pos]), dim=1)
        rotation = rotary(8, 128).T @ rotary(pos, 128)
        ref = execute_head(query, source, rotation, program)
        candidate_write = execute_head(query, source, rotation, changed)
        qa, qb = query @ program['q1'].T, query @ program['q2'].T
        ka, kb, vv = (source @ stack.T).split(128, dim=-1)
        port_write = execute_ports(qa, qb, ka, kb, vv, rotation, program, coefficients)
        corrected = execute_ports(qa, qb, ka, kb, vv, rotation, program, coefficients,
                                  source @ residual.T)
        interface_errors.append(float((port_write-candidate_write).norm()/candidate_write.norm()))
        corrected_errors.append(float((corrected-ref).norm()/ref.norm()))
        sums[0] += ref
        sums[1] += candidate_write
        source_errors.append(float((candidate_write-ref).norm()/ref.norm()))
        true_read = source @ readers.T
        compact_read = (source @ stack.T) @ coefficients.T
        reading_errors.append(float((compact_read-true_read).norm()/true_read.norm()))
    result['cached_native_write_check'] = dict(
        prefixes=len(query), target_position=8, source_positions=list(range(9)),
        relative_all_source_write_error=float((sums[1]-sums[0]).norm()/sums[0].norm()),
        per_source_write_errors=source_errors, per_source_reader_errors=reading_errors,
        max_port_interface_replay_error=max(interface_errors),
        max_exact_four_read_correction_error=max(corrected_errors),
        scope='Existing FineWeb native normalized states; old selected branch only. Final behavioral effect and equality to new mixed-write circuit remain untested.',
    )
    result['interface_price'] = dict(
        query_inputs=256, source_inputs=384, optional_exact_correction_inputs=4,
        stored_scalars=coefficients.numel()+sum(program[k].numel() for k in ('dual','atom_k1','atom_k2','atom_write')),
        scope='Local port executor only. Native query/key/value producers remain charged; exact correction also requires four residual readers of the full source. This is not a whole-model saving.',
    )
    assert max(interface_errors+corrected_errors) < 1e-10
    (P/'HEAD17_SOURCE_INTERFACE_V1_RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
