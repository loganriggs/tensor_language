"""Frozen four-context replay and one-context timing for all value channels."""
import json
import hashlib
import statistics
import time
from datetime import datetime,timezone
from pathlib import Path
import torch
import torch.nn.functional as F
from even_key_value_bank_v1 import compile_program,execute,routing
from key_span_reuse_v1 import KeySpanParent

P=Path(__file__).resolve().parent
CHECKPOINT=Path('/home/loganriggs/.local/share/bilin18/hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


@torch.no_grad()
def main():
    out=P/'EVEN_KEY_VALUE_BANK_V1_RESULT.json';artifact=P/'EVEN_KEY_VALUE_BANK_V1_PROGRAM.pt'
    if out.exists() or artifact.exists():raise FileExistsError('Preserve frozen evidence')
    torch.set_num_threads(2)
    original=torch.load(P/'KEY_SPAN_REUSE_V1_PROGRAM.pt',weights_only=True)
    state=torch.load(CHECKPOINT,weights_only=True,mmap=True)
    p=compile_program(original,state);torch.save(p,artifact)
    cache=torch.load(P/'SELECTIVE_INTERACTION_PORTS_V1_ARTIFACT.pt',weights_only=True,mmap=True)
    rows=json.loads((P/'SCALAR_CUE_CHANNEL_SHIFT_V1_ROWS.json').read_text())['rows']
    records=[];timings=None
    for row_index in [0,18,36,54]:
        row=rows[row_index];n=len(row['ids']);tokens=torch.tensor([row['ids']])
        current=F.rms_norm(cache['raw9'][0,row_index,:n][None].double(),(1152,),eps=torch.finfo(torch.float32).eps)
        initial=F.rms_norm(F.embedding(tokens,state['transformer.wte.weight']),(1152,),eps=torch.finfo(torch.float32).eps).double()
        # The old scalar interface serves as an independent evaluator per value channel.
        # Its first-value lookup below is an explicit test fixture, never exported.
        reference_program={k:torch.stack([p[k].double(),p[k].double()]) for k in ['q1','q2','k1','k2','key_coordinates']}
        reference=KeySpanParent(reference_program)
        position_ids=torch.arange(n)[None]
        v_current=p['current_value'].double();v_first=p['first_value'].double()
        first_values=initial@v_first.T
        def independent():
            ca,fa=[],[]
            for j in range(128):
                reference_program['current_value_reader']=v_current[j]
                ca.append(reference.scalar(current,position_ids,1))
                reference_program['first_token_value']=first_values[0,:,j]
                fa.append(reference.scalar(current,position_ids,0))
            return torch.stack(ca,-1),torch.stack(fa,-1)
        def shared():return execute(current,initial,p)
        ca,fa=independent()
        mix=p['mixture'].double();expected=(1-mix)*ca+mix*fa
        write,channels=shared()
        g=routing(current,p)
        rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
        records.append(dict(row=row_index,length=n,
                            current_error=rel(g@(current@v_current.T),ca),
                            inherited_error=rel(g@first_values,fa),
                            mixture_error=rel(channels,expected),
                            full_write_error=rel(write,expected@p['output'].double().T)))
        if row_index==0:
            # Both arms compute current/inherited values and final output; scalar arm
            # additionally routes each source separately (256 scalar evaluations).
            def independent_complete():
                c,f=independent()
                return ((1-mix)*c+mix*f)@p['output'].double().T
            funcs=[independent_complete,shared]
            for f in funcs:f()
            trials=[[],[]]
            for trial in range(7):
                for j in [(trial+i)%2 for i in range(2)]:
                    start=time.perf_counter();funcs[j]();trials[j].append(time.perf_counter()-start)
            med=list(map(statistics.median,trials))
            timings=dict(trials=trials,median_seconds=med,speedup=med[0]/med[1],
                         comparison='256 independent current/inherited scalar evaluations vs one shared gate and 128 mixed channels; one fixed cached prefix, CPU only')
    result=dict(utc=datetime.now(timezone.utc).isoformat(),
                pred_a=all(max(r[k] for k in ['current_error','inherited_error','mixture_error','full_write_error'])<=1e-10 for r in records),
                pred_b=timings['speedup']>=2,records=records,timing=timings,
                program_scalars=sum(v.numel() for v in p.values()),
                payload_bytes=sum(v.numel()*v.element_size() for v in p.values()),
                serialized_bytes=artifact.stat().st_size,
                source_bytes=(P/'even_key_value_bank_v1.py').stat().st_size,
                program_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256((P/'even_key_value_bank_v1.py').read_bytes()).hexdigest(),
                scope='Full-value conditional even-key component of head9.8, not full native attention or a newly identified circuit. Existing key projection fixed; actual current/first value maps, signed mixture and full output map retained. No native suffix/capability/OOD test, full-model price or speed claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if not result['pred_a']:raise SystemExit(1)


if __name__=='__main__':main()
