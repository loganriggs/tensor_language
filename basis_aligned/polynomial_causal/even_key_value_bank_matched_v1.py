"""Correct the 256-call diagnostic to the registered 128 mixed-channel baseline."""
import hashlib,json,statistics,time
from datetime import datetime,timezone
import torch
import torch.nn.functional as F
from check_even_key_value_bank_v1 import P,CHECKPOINT
from even_key_value_bank_v1 import execute
from key_span_reuse_v1 import KeySpanParent


@torch.no_grad()
def main():
    out=P/'EVEN_KEY_VALUE_BANK_MATCHED_V1_RESULT.json'
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(2)
    p=torch.load(P/'EVEN_KEY_VALUE_BANK_V1_PROGRAM.pt',weights_only=True)
    state=torch.load(CHECKPOINT,weights_only=True,mmap=True)
    cache=torch.load(P/'SELECTIVE_INTERACTION_PORTS_V1_ARTIFACT.pt',weights_only=True,mmap=True)
    row=json.loads((P/'SCALAR_CUE_CHANNEL_SHIFT_V1_ROWS.json').read_text())['rows'][0]
    n=len(row['ids']);ids=torch.tensor([row['ids']]);positions=torch.arange(n)[None]
    current=F.rms_norm(cache['raw9'][0,0,:n][None].double(),(1152,),eps=torch.finfo(torch.float32).eps)
    initial=F.rms_norm(F.embedding(ids,state['transformer.wte.weight']),(1152,),eps=torch.finfo(torch.float32).eps).double()
    rp={k:torch.stack([p[k].double(),p[k].double()]) for k in ['q1','q2','k1','k2','key_coordinates']}
    old=KeySpanParent(rp)
    def independent():
        values=(1-p['mixture'].double())*(current@p['current_value'].double().T)+p['mixture'].double()*(initial@p['first_value'].double().T)
        outputs=[]
        for j in range(128):
            rp['first_token_value']=values[0,:,j]
            outputs.append(old.scalar(current,positions,0))
        return torch.stack(outputs,-1)@p['output'].double().T
    def shared():return execute(current,initial,p)[0]
    ref=independent();value=shared()
    relative=float((ref-value).norm()/ref.norm())
    funcs=[independent,shared];trials=[[],[]]
    for f in funcs:f()
    for trial in range(7):
        for j in [(trial+i)%2 for i in range(2)]:
            start=time.perf_counter();funcs[j]();trials[j].append(time.perf_counter()-start)
    med=list(map(statistics.median,trials))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),pred_a=relative<=1e-10,
                pred_b=med[0]/med[1]>=2,relative_error=relative,
                independent_scalar_calls=128,trials=trials,median_seconds=med,speedup=med[0]/med[1],
                superseded_timing='EVEN_KEY_VALUE_BANK_V1_RESULT.json: pred_b invalid for registered128comparison; original256timing retained descriptively',
                artifact_sha256=hashlib.sha256((P/'EVEN_KEY_VALUE_BANK_V1_PROGRAM.pt').read_bytes()).hexdigest(),
                scope='Matched128independentmixed-channel evaluator versus one shared QK routing. Both include current/initial value projections, signed mixture and full output map. Firstfixed18tokenprefix only. Known computational sharing, not newsemanticcircuit or native/fullmodel speed.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
