"""Conservative native graph closure and literal parameter footprint.

The result is a syntactic dependency audit, not a lower bound over equivalent
programs or a proof that every native edge is behaviorally necessary.
"""
import json,hashlib
from pathlib import Path
import torch
P=Path(__file__).resolve().parent


def main():
    binding=json.loads((P/'REGIONAL_RECURSIVE_KEY_V1_BINDING.json').read_text())['files']
    path=next(k for k in binding if k.endswith('pytorch_model.bin'))
    sd=torch.load(path,weights_only=True,mmap=True,map_location='cpu')
    supports=json.loads((P/'REGIONAL_KEY_SUPPORT_V1_FROZEN.json').read_text())['supports']
    def footprint(support):
        keys={'transformer.wte.weight','transformer.h.0.attn.c_v.weight'}
        for j in range(14):keys.add(f'transformer.h.{j}.lambdas')
        for j in (8,9,13):
            for name in ('c_k','c_k2'):keys.add(f'transformer.h.{j}.attn.{name}.weight')
        for port in support:
            prefix=f'transformer.h.{port//2}.'+('attn.' if port%2==0 else 'mlp.')
            keys.update(k for k in sd if k.startswith(prefix))
        assert all(k in sd for k in keys)
        return dict(scalars=sum(sd[k].numel() for k in keys),native_storage_bytes=sum(sd[k].numel()*sd[k].element_size() for k in keys),fp32_bytes=4*sum(sd[k].numel() for k in keys),tensor_count=len(keys))
    cells=[]
    for name,support in supports.items():
        # Native serial residual updates: MLPj reads attentionj's output; either
        # update at j reads all previous blocks. No graph-level cancellation assumed.
        closure=list(range(max(support)+1))
        cells.append(dict(name=name,selected_support=support,syntactic_closure=closure,
                          omitted_dependencies=sorted(set(closure)-set(support)),
                          selected_module_footprint=footprint(support),native_producer_closure_footprint=footprint(closure)))
    price=json.loads((P/'REGIONAL_RECURSIVE_KEY_V1_RESULT.json').read_text())['generator_price']
    result=dict(cells=cells,all26_key_program_footprint=footprint(range(26)),
                cached_zero_scalars=price['cached_zero_scalars'],cached_zero_fp32_bytes=4*price['cached_zero_scalars'],
                exact_full_denominator_native_update_closure=list(range(26)),
                excludes='Native full-prompt queries, downstream17 shared feature/writers, final MLP/unembedding, temporary state and runtime compute; counts are not a whole-circuit price.',
                scope='Opaque native module graph closure with unique checkpoint tensors, including full-vocabulary embedding, shared first value map, residual reentry and needed key maps. Alternative algebraic decompositions can have different dependencies. Zero-cache regeneration additionally requires omitted native modules unless supplied as charged constants.',
                source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (P/'REGIONAL_KEY_DEPENDENCY_PRICE_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
