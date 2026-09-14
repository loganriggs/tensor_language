import json,time,statistics
from pathlib import Path
from datetime import datetime,timezone
import torch
P=Path(__file__).resolve().parent

@torch.no_grad()
def main():
    out=P/'KEY_SPAN_KERNEL_V1_RESULT.json';assert not out.exists()
    torch.set_num_threads(2);torch.manual_seed(170234100)
    old=torch.load(P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt',weights_only=True)
    new=torch.load(P/'KEY_SPAN_REUSE_V1_PROGRAM.pt',weights_only=True)
    prior=json.loads((P/'KEY_SPAN_RUNTIME_V1_RESULT.json').read_text())['rows'];rows=[]
    x=torch.randn(1,16,1152,dtype=torch.float64)
    for i in [0,1]:
        nums=[x@old[k][i].double().T for k in ['k1','k2']]
        fs=[lambda:x@old['key_basis'][i],lambda:torch.cat(nums,-1)@new['key_coordinates'][i]]
        a,b=[f() for f in fs];err=float((a-b).norm()/a.norm());samples=[[],[]]
        for f in fs:
            for _ in range(5):f()
        for trial in range(7):
            for j in [trial%2,1-trial%2]:
                start=time.perf_counter()
                for _ in range(100):fs[j]()
                samples[j].append((time.perf_counter()-start)/100)
        med=[statistics.median(s) for s in samples]
        whole=next(r['medians'][1] for r in prior if r['batch']==1 and r['length']==16 and r['index']==i)
        rows.append(dict(index=i,relative=err,seconds=med,samples=samples,saved_fraction_of_prior_whole=(med[0]-med[1])/whole))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),rows=rows,pred_a=max(r['relative'] for r in rows)<=1e-10,
                pred_b=all(r['seconds'][1]<=r['seconds'][0] for r in rows),scope='Kernel only, existing key outputs free on both sides because both full programs already compute them. Prior whole-call fraction descriptive; original runtimeB failure remains.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':main()
