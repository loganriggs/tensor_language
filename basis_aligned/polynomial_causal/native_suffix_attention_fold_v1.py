"""Exact reader pullback for the frozen suffix program, not new tensor algebra.
A scalar/readout/O/current-value/first-value fold replay<=1e-9.
Native QK1*QK2 routing, residual and normalization remain explicit dependencies.
"""
import hashlib
import json
from pathlib import Path
import torch
from native_relation_split_v1 import evaluate


def scalar_from_reads(artifact,reads,radius2):
    values=[]
    for j,c in enumerate(artifact['components']):
        z=reads[...,16*j:16*(j+1)]
        values.append(z[...,0]*z[...,1]+z[...,2:].square()@c['rest_coefficients']+
            (c['leading_radial']+c['rest_radial'])*radius2+c['bias'])
    return torch.stack(values,-1)


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);p=Path(__file__).parent
    out=p/'NATIVE_SUFFIX_ATTENTION_FOLD_V1.json';artifact=p/'NATIVE_SUFFIX_ATTENTION_FOLD_V1.pt'
    assert not out.exists() and not artifact.exists()
    source=p/'NATIVE_RELATION_SPLIT_V1.pt';program=torch.load(source,weights_only=True,map_location='cpu')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==json.loads((p/'NATIVE_RELATION_SPLIT_V1.json').read_text())['artifact_sha256']
    reader=torch.cat([torch.cat([c['a'][None],c['b'][None],c['rest_readers']]) for c in program['components']])
    binding=json.loads((p/'NATIVE_RELATION_OUTPUT_FRESH_V1_BINDING.json').read_text())['files']
    checkpoint=next(k for k in binding if k.endswith('/pytorch_model.bin'))
    state=torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu')
    o=state['transformer.h.17.attn.c_proj.weight'].double()
    current=state['transformer.h.17.attn.c_v.weight'].double();first=state['transformer.h.0.attn.c_v.weight'].double()
    mix=state['transformer.h.17.attn.lamb'].double();folded=reader@o
    torch.manual_seed(6541);x=torch.randn(11,1152);y=torch.randn(11,1152);errors=[];headchecks=[]
    lead,rest=evaluate(program,x);a=scalar_from_reads(program,x@reader.T,x.square().sum(-1))
    errors.append(float((a-lead-rest).norm()/(lead+rest).norm()))
    errors.append(float(((y@o.T)@reader.T-y@folded.T).norm()/((y@o.T)@reader.T).norm()))
    for h in range(9):
        sl=slice(128*h,128*(h+1));fo=folded[:,sl]
        cv=(1-mix)*fo@current[sl];fv=mix*fo@first[sl]
        direct=((1-mix)*(x@current[sl].T)+mix*(y@first[sl].T))@o[:,sl].T@reader.T
        compiled=x@cv.T+y@fv.T
        error=float((compiled-direct).norm()/direct.norm());errors.append(error)
        headchecks.append(dict(head=h,fold_error=error,folded_output_rank=int(torch.linalg.matrix_rank(fo)),
            current_map_norm=float(cv.norm()),first_value_map_norm=float(fv.norm())))
    torch.save(dict(readers=reader,folded_output=folded,components=program['components'],value_mix=mix,
        scope='48 reads; native O/c_v/QK and normalization dependencies retained. Folded source maps recomputable from this artifact and checkpoint.'),artifact)
    result=dict(pred_a=max(errors)<=1e-9,errors=errors,head_checks=headchecks,value_mix=float(mix),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        checkpoint_sha256=binding[checkpoint],price=dict(reader_floats=reader.numel(),folded_output_floats=folded.numel(),
            scalar_products=45,normalization_radius_required=True,native_routing_required=True),
        scope='Existing exact linear-reader pullback applied to a new validated suffix component. No head behavioral importance, source closure, compression adoption or full extraction claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
