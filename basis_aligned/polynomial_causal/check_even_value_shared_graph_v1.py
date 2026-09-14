"""Weights-only span freeze and four-context signed S/R/O graph replay."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,itertools,json,time,tempfile,sys
import torch
import torch.nn.functional as F
from check_even_key_value_bank_v1 import P,CHECKPOINT
from even_value_shared_graph_v1 import compile_program,SharedGraph
from even_key_value_native_backend_v1 import FullValueComponents
from key_span_reuse_v1 import KeySpanParent


@torch.no_grad()
def main():
    out=P/'EVEN_VALUE_SHARED_GRAPH_V1_RESULT.json';artifact=P/'EVEN_VALUE_SHARED_GRAPH_V1_PROGRAM.pt'
    assert not out.exists() and not artifact.exists()
    torch.set_num_threads(2);tic=time.perf_counter()
    value=torch.load(P/'EVEN_KEY_VALUE_BANK_V1_PROGRAM.pt',weights_only=True)
    scalar=torch.load(P/'KEY_SPAN_REUSE_V1_PROGRAM.pt',weights_only=True)
    writer=torch.load(P/'SCALAR_PRODUCER_NATIVE_LIFT_V1_ARTIFACT.pt',weights_only=True)['writers'][1]
    reader=scalar['current_value_reader']
    p=compile_program(value,reader,writer)
    rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
    spans=dict(value=rel(value['current_value'].double().T@p['scalar_value_coordinates'],reader),
               output=rel(value['output'].double()@p['scalar_output_coordinates'],writer))
    # Freeze coefficients from weights before loading validation contexts.
    torch.save(p,artifact);sha=hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert max(spans.values())<=1e-10
    graph=SharedGraph(p);old=FullValueComponents(value,'cpu');old_scalar=KeySpanParent(scalar)
    sd=torch.load(CHECKPOINT,weights_only=True,mmap=True)
    cache=torch.load(P/'SELECTIVE_INTERACTION_PORTS_V1_ARTIFACT.pt',weights_only=True,mmap=True)
    rows=json.loads((P/'SCALAR_CUE_CHANNEL_SHIFT_V1_ROWS.json').read_text())['rows']
    gains=list(itertools.product([0,1],repeat=3))+[(-1,.5,2)]
    records=[]
    for i in [0,18,36,54]:
        n=len(rows[i]['ids']);ids=torch.tensor([rows[i]['ids']])
        current=F.rms_norm(cache['raw9'][0,i,:n][None].float(),(1152,))
        initial=F.rms_norm(F.embedding(ids,sd['transformer.wte.weight']),(1152,)).double()
        first=initial@value['first_value'].double().T
        even,odd=old(current,first)
        selected=old_scalar.scalar(current,ids,1)[...,None]*writer
        expected=(selected,even-selected,odd)
        branches=graph.state(current,initial)
        for g in gains:
            reference=sum(float(a)*b for a,b in zip(g,expected))
            prediction=graph.write(branches,g)
            error=rel(prediction,reference) if bool(reference.norm()) else float(prediction.abs().max())
            records.append(dict(row=i,gains=g,relative_error=error))
    base_payload=sum(v.numel()*v.element_size() for v in value.values())
    old_branch=reader.numel()*reader.element_size()+writer.numel()*writer.element_size()
    new_branch=sum(p[k].numel()*p[k].element_size() for k in ['scalar_value_coordinates','scalar_output_coordinates'])
    total_saving=1-(base_payload+new_branch)/(base_payload+old_branch)
    result=dict(utc=datetime.now(timezone.utc).isoformat(),pred_a=max(spans.values())<=1e-10,
                pred_b=all(r['relative_error']<=1e-10 for r in records),
                pred_c=new_branch<=.2*old_branch and total_saving>=.001,
                span_errors=spans,records=records,old_branch_bytes=old_branch,new_branch_bytes=new_branch,
                base_value_payload_bytes=base_payload,total_tensor_saving_fraction=total_saving,
                serialized_program_bytes=artifact.stat().st_size,program_sha256=sha,
                program_scalars=sum(t.numel() for t in p.values()),seconds=time.perf_counter()-tic,
                scope='Exact coordinates of prior scalar in existing native V/output maps. Joint S/R/O writes before unchanged nonlinear suffix; four existing contexts,36gain cases. No fitting/quantization/newnative/OOD/stable-semantic-unit claim. Scalar branch saving is not fullmodel saving; all QK/value/output maps and external normalized inputs retained.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


def pack_audit():
    out=P/'EVEN_VALUE_SHARED_GRAPH_PACKED_V1_RESULT.json'
    artifact=P/'EVEN_VALUE_SHARED_GRAPH_PACKED_V1_PROGRAM.pt'
    assert not out.exists() and not artifact.exists()
    oldfile=P/'EVEN_VALUE_SHARED_GRAPH_V1_PROGRAM.pt'
    old=torch.load(oldfile,weights_only=True)
    packed={k:v.contiguous().clone() for k,v in old.items()}
    # Match archive naming/overhead by using file objects for both representations.
    with artifact.open('wb') as f:torch.save(packed,f)
    reference=torch.load(P/'EVEN_KEY_VALUE_BANK_V1_PROGRAM.pt',weights_only=True)
    scalar=torch.load(P/'KEY_SPAN_REUSE_V1_PROGRAM.pt',weights_only=True)
    reference['private_scalar_reader']=scalar['current_value_reader'].clone()
    reference['private_scalar_writer']=torch.load(P/'SCALAR_PRODUCER_NATIVE_LIFT_V1_ARTIFACT.pt',weights_only=True)['writers'][1].clone()
    with tempfile.TemporaryFile() as f:
        torch.save(reference,f);reference_bytes=f.tell()
    restored=torch.load(artifact,weights_only=True)
    equal=all(torch.equal(v,restored[k]) for k,v in old.items())
    coord=['scalar_value_coordinates','scalar_output_coordinates']
    result=dict(utc=datetime.now(timezone.utc).isoformat(),pred_a=equal,
                pred_b=artifact.stat().st_size<=.999*reference_bytes,
                original_serialized_bytes=oldfile.stat().st_size,
                packed_serialized_bytes=artifact.stat().st_size,
                matched_private_reference_serialized_bytes=reference_bytes,
                serialized_saving_fraction=1-artifact.stat().st_size/reference_bytes,
                coordinate_original_buffer_bytes=sum(old[k].untyped_storage().nbytes() for k in coord),
                coordinate_packed_buffer_bytes=sum(restored[k].untyped_storage().nbytes() for k in coord),
                packed_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                original_sha256=hashlib.sha256(oldfile.read_bytes()).hexdigest(),
                compiler_sha256=hashlib.sha256((P/'even_value_shared_graph_v1.py').read_bytes()).hexdigest(),
                scope='Same tensor values/dtypes, owned contiguous solution storage; no quantization. Matched program-file comparison includes all V/QK/output maps. Runtime source and native prefix/suffix remain separate; not total-model savings.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    pack_audit() if '--pack' in sys.argv else main()
