"""Post-reconstruction CPU weight identities; fixed seed9111270,12x7streams."""
import json,time
from pathlib import Path
import torch
import correlative_route_read_write_v1 as C
from audit_normalized_router_fold_v1 import CHECKPOINT,EXPECTED,digest

ROOT=Path(__file__).resolve().parent


def main():
    out=ROOT/'CORRELATIVE_WEIGHT_PORTS_V1_RESULT.json';artifact=ROOT/'CORRELATIVE_WEIGHT_PORTS_V1_MAPS.pt'
    assert not out.exists() and not artifact.exists()
    torch.set_num_threads(2);start=time.perf_counter();assert digest(CHECKPOINT)==EXPECTED
    source=ROOT/'CORRELATIVE_REPLAYABLE_INTERFACE_V1_ARTIFACT.pt'
    saved=torch.load(source,map_location='cpu',weights_only=True)
    state=torch.load(CHECKPOINT,map_location='cpu',weights_only=True,mmap=True)
    blocks=C.fold(saved,state)
    generator=torch.Generator().manual_seed(9111270)
    x,x0=[torch.randn(12,7,1152,dtype=torch.float64,generator=generator) for _ in range(2)]
    results={}
    for layer,block in blocks.items():
        prefix=f'transformer.h.{layer}.attn.';lam=block['value_mix_lambda']
        local=x@state[prefix+'c_v.weight'].double().T
        first=x0@state['transformer.h.0.attn.c_v.weight'].double().T
        values=((1-lam)*local+lam*first).reshape(12,7,9,128)
        patterns={port['head']:torch.randn(12,7,dtype=torch.float64,generator=generator) for port in block['ports']}
        z=torch.zeros(12,1152,dtype=torch.float64)
        for port in block['ports']:
            h=port['head'];z[:,h*128:(h+1)*128]=(patterns[h][...,None]*values[:,:,h,:]).sum(1)
        actual=z@block['full_head_reader'];compiled=C.scalar(block,patterns,x,x0)
        delta=torch.randn(12,dtype=torch.float64,generator=generator)
        expanded=(delta[:,None]*block['full_head_reader'])@state[prefix+'c_proj.weight'].double().T
        folded=C.write_delta(block,actual+delta,actual)
        results[str(layer)]={'heads':[p['head'] for p in block['ports']],
                            'lambda':lam,'scalar_maxabs':float((compiled-actual).abs().max()),
                            'scalar_relative':float((compiled-actual).norm()/actual.norm()),
                            'writer_maxabs':float((folded-expanded).abs().max()),
                            'writer_relative':float((folded-expanded).norm()/expanded.norm())}
    with artifact.open('xb') as stream:torch.save(blocks,stream)
    r={'schema':'correlative.weight_ports.v1','blocks':results,
       'passed':all(v['scalar_relative']<=1e-10 and v['scalar_maxabs']<=1e-10 and v['writer_relative']<=1e-10 and v['writer_maxabs']<=1e-10 for v in results.values()),
       'block_count':len(blocks),'head_count':sum(len(b['ports']) for b in blocks.values()),
       'core_scalars':sum(b['writer'].numel()+sum(p[k].numel() for p in b['ports'] for k in ('local_value_reader','first_value_reader')) for b in blocks.values()),
       'artifact_bytes':artifact.stat().st_size,'source_sha256':digest(source),'maps_sha256':digest(artifact),
       'helper_sha256':digest(Path(C.__file__)),'checkpoint_sha256':EXPECTED,'seconds':time.perf_counter()-start,
       'scope':'Synthetic continuous stream identity using trained weights and frozen fitted interface; actual native router and upstream states remain required; no native text test',
       'native_parameters_retained':545902902,'model_forwards':0,'gpu_accessed':False}
    with out.open('x') as stream:json.dump(r,stream,indent=2);stream.write('\n')
    print(json.dumps(r,indent=2))


if __name__=='__main__':main()
