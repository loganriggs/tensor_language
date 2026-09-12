"""Weight-only physical lifts of frozen producer components; no behavioral inference."""
import json
from pathlib import Path
import torch

P = Path(__file__).resolve().parent

@torch.no_grad()
def main():
    torch.set_num_threads(2)
    output = P / 'SCALAR_PRODUCER_NATIVE_LIFT_V1_RESULT.json'
    assert not output.exists()
    binding = json.loads((P / 'STRUCTURED_PRODUCER_CACHE_V1_BINDING.json').read_text())['files']
    state = torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')), weights_only=True, mmap=True, map_location='cpu')
    bank = torch.load(P / 'STRUCTURED_PRODUCER_BANK_WEIGHTS_V1_ARTIFACT.pt', weights_only=True, map_location='cpu')
    C = torch.load(P / 'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt', weights_only=True, map_location='cpu')['current_readers'].double()
    records, writers = [], []
    for layer, head, index in ((8, 2, 2), (9, 8, 17)):
        prefix = f'transformer.h.{layer}.attn.'
        sl = slice(128 * head, 128 * (head + 1))
        mix = float(state[prefix + 'lamb'])
        V = torch.cat(((1-mix)*state[prefix+'c_v.weight'][sl].double(), mix*state['transformer.h.0.attn.c_v.weight'][sl].double()), dim=1)
        O = state[prefix+'c_proj.weight'][:, sl].double()
        v = bank['source_readers'][index].double()
        d = O @ (V @ v)
        scale = 1.0
        for j in range(layer+1, 18):
            scale *= float(state[f'transformer.h.{j}.lambdas'][0])
        expected = bank['output_writers'][index].double()
        error = float((scale*C@d-expected).norm()/expected.norm())
        records.append(dict(head=f'{layer}.{head}', commutation_relative_error=error, residual_scale=scale, writer_norm=float(d.norm()), source_reader_norm=float(v.norm())))
        writers.append(d)
    result = dict(pred_a=all(r['commutation_relative_error'] <= 1e-10 for r in records), records=records,
                  physical_writer_cosine=float(torch.nn.functional.cosine_similarity(writers[0], writers[1], dim=0)),
                  scope='Exact natural residual lift of the unmerged value components. Common downstream writer does not imply common full-residual writer. Recursive removal not executed; no preservation, sufficiency or OOD conclusion.')
    output.write_text(json.dumps(result, indent=2)+'\n')
    torch.save(dict(writers=torch.stack(writers)), P/'SCALAR_PRODUCER_NATIVE_LIFT_V1_ARTIFACT.pt')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
