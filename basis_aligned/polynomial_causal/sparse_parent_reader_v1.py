"""Weight-first full-rank sparse parent reader, fixed original basis gauge.

Pred_a projector/factored replay <=1e-10 in FP64.
Pred_b every family/context and actual endpoint direct-field error <=10%.
Pred_c packed FP32 basis storage saves >=10%, charging dense64x64 correction.
Packed execution tested after value rounding; CPU reference expands orthobasis.
No native suffix validation or faster sparse executor claim.
"""
from pathlib import Path
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from regional_even_routing_v1 import routing


def unpack(item):
    shape = tuple(item['shape'])
    mask = torch.from_numpy(np.unpackbits(item['mask'].numpy(), bitorder='little')[:np.prod(shape)].copy()).bool()
    s = torch.zeros(mask.numel(), dtype=torch.float64)
    s[mask] = item['values'].double()
    s = s.reshape(shape)
    # Runtime correction is charged, rather than pretending sparse S is orthonormal.
    correction = torch.linalg.inv(torch.linalg.cholesky(s.T @ s)).T
    return s, correction, s @ correction


def main():
    torch.set_num_threads(2)
    start = time.perf_counter()
    p = Path(__file__).resolve().parent
    native = torch.load(p/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt', weights_only=True)
    basis = native['key_basis'][1].double()
    candidates = {}
    for fraction in (.25, .4):
        keep = basis.numel() - round(fraction*basis.numel())
        indices = basis.abs().flatten().argsort(descending=True, stable=True)[:keep]
        mask = torch.zeros(basis.numel(), dtype=torch.bool)
        mask[indices] = True
        candidates[str(fraction)] = dict(shape=list(basis.shape),
            mask=torch.from_numpy(np.packbits(mask.numpy(), bitorder='little')),
            values=basis.flatten()[mask].float())
    # Freeze candidate payloads before reading behavioral ports.
    torch.save(candidates, p/'SPARSE_PARENT_READER_V1_PROGRAM.pt')
    candidates = torch.load(p/'SPARSE_PARENT_READER_V1_PROGRAM.pt', weights_only=True)
    cache = torch.load(p/'SELECTIVE_INTERACTION_PORTS_V1_ARTIFACT.pt', weights_only=True, mmap=True)
    panel = json.loads((p/'SCALAR_CUE_CHANNEL_SHIFT_V1_ROWS.json').read_text())['rows']
    ends = torch.tensor([len(row['ids'])-1 for row in panel])
    controls, rows, prices = [], [], []
    expanded = {}
    for name, item in candidates.items():
        s, correction, q = unpack(item)
        expanded[name] = dict(native, key_basis=[native['key_basis'][0], q])
        k = native['k1'][1].double()
        x = basis.T  # deterministic nonzero weight-derived probes
        explicit = (x @ q) @ (k @ q).T
        factored = (x @ s) @ (correction @ correction.T) @ (k @ s).T
        controls.append(dict(method=name,
            orthogonality_error=float((q.T @ q-torch.eye(64)).norm()/8),
            factored_replay_error=float((explicit-factored).norm()/explicit.norm()),
            projector_relative_error=float((q@q.T-basis@basis.T).norm()/(basis@basis.T).norm())))
        old = basis.numel()*4
        packed = item['values'].numel()*4+item['mask'].numel()+64*64*4+16
        prices.append(dict(method=name, basis_bytes_before=old, packed_basis_bytes_after=packed,
            basis_storage_saving=1-packed/old, retained_reader_nodes=64,
            sparse_reader_edges=item['values'].numel(), correction_entries=64*64,
            full_interface_storage_saving=(old-packed)/(665856*4),
            resident_expanded_basis_bytes=basis.numel()*8,
            scope='FP32 payload plus bitmask, 16 shape bytes and charged FP32 runtime correction; archive container overhead excluded. Reference execution expands dense FP64 basis; no resident-memory/runtime saving.'))
    for context in range(2):
        current = F.rms_norm(cache['raw9'][context].float(), (1152,), eps=torch.finfo(torch.float32).eps)
        values = (current.double() @ native['current_value_reader'])[..., None]
        ref = (routing(current, native, 1) @ values)[..., 0]
        for name, candidate in expanded.items():
            pred = (routing(current, candidate, 1) @ values)[..., 0]
            for family in range(3):
                sl = slice(24*family,24*(family+1))
                x,y = ref[sl],pred[sl]
                ix=torch.arange(24);end=ends[sl]
                rows.append(dict(method=name,context=context,family=family,
                    relative_error=float((x-y).norm()/x.norm()),
                    endpoint_relative_error=float((x[ix,end]-y[ix,end]).norm()/x[ix,end].norm())))
    result=dict(pred_a=all(max(c['orthogonality_error'],c['factored_replay_error'])<=1e-10 for c in controls),
        pred_b={name:all(max(r['relative_error'],r['endpoint_relative_error'])<=.1 for r in rows if r['method']==name) for name in candidates},
        pred_c={x['method']:x['basis_storage_saving']>=.1 for x in prices},
        controls=controls,prices=prices,rows=rows,seconds=time.perf_counter()-start,scope=__doc__)
    (p/'SPARSE_PARENT_READER_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
