"""Exact compilation of the existing 64 producer readers; no factor fitting."""
import json,hashlib,time
from pathlib import Path
import torch
P=Path(__file__).resolve().parent


def run():
    start=time.perf_counter();torch.set_default_dtype(torch.float64)
    source=P/'PRODUCER_METRIC_OUTER32_V1_PROGRAM.pt'
    program=torch.load(source,weights_only=True)
    binding=json.loads((P/'FULLU_PAIRED_PRODUCER_V1_BINDING.json').read_text())['files']
    checkpoint=next(k for k in binding if k.endswith('/pytorch_model.bin'))
    state=torch.load(checkpoint,weights_only=True,mmap=True)
    l,r,d=[state[f'transformer.h.16.mlp.{n}.weight'].double() for n in ('Left','Right','Down')]
    x=torch.load(P/'QUARTIC_GROUP_PORTS_V1_PORTS.pt',weights_only=True)['input16'].double()
    ports=torch.load(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True)['ports']
    den=ports['pre'].double().square().mean(-1)+torch.finfo(torch.float32).eps
    hidden=(x@l.T)*(x@r.T)
    a=program['output_readers'].double();scale=float(program['producer_scale'])
    folded=torch.einsum('dm,odr->omr',d,a)*scale
    original=torch.einsum('nd,odr->nor',hidden@d.T*scale,a)
    new=torch.einsum('nm,omr->nor',hidden,folded)
    weights=program['outer_weights'].double();writers=program['output_writers'].double()
    execute=lambda values:(values.square()*weights).sum(-1)@writers.T/den[:,None]
    old_write=execute(original);new_write=execute(new)
    errors=dict(reader_relative=float((original-new).norm()/original.norm()),write_relative=float((old_write-new_write).norm()/old_write.norm()))
    if 'native_write' in program:
        errors['saved_write_relative']=float((new_write-program['native_write'].double()).norm()/new_write.norm())
    old_values=l.numel()+r.numel()+d.numel()+a.numel()+weights.numel()+writers.numel()
    new_values=l.numel()+r.numel()+folded.numel()+weights.numel()+writers.numel()
    artifact=P/'PRODUCER_READER_COMPILE_V1_PROGRAM.pt'
    assert not artifact.exists()
    torch.save(dict(folded_readers=folded,weights=weights,writers=writers,native_input_layers=['transformer.h.16.mlp.Left.weight','transformer.h.16.mlp.Right.weight'],source=str(source)),artifact)
    result=dict(pred_a=max(errors.values())<=1e-10,errors=errors,old_conditional_values=old_values,new_conditional_values=new_values,saved_values=old_values-new_values,source_sha=hashlib.sha256(source.read_bytes()).hexdigest(),checkpoint_sha=binding[checkpoint],artifact_sha=hashlib.sha256(artifact.read_bytes()).hexdigest(),execution_seconds=time.perf_counter()-start,scope='Exact partial numerator execution; native L16/R16, normalization and background remain. Existing behavioral failures unchanged; no whole-model savings claim.')
    (P/'PRODUCER_READER_COMPILE_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));assert result['pred_a']


if __name__=='__main__':run()
