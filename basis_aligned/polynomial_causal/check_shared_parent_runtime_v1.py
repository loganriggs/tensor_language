"""Board11:03 registered exact replay and speed; fixed7x10 timing trials."""
import importlib.util
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.nn.functional as F
from shared_parent_runtime_v1 import SharedParent

P=Path(__file__).resolve().parent


@torch.no_grad()
def main():
    out=P/'SHARED_PARENT_RUNTIME_V1_RESULT.json'; assert not out.exists()
    torch.set_num_threads(2); torch.manual_seed(170233000)
    package=P/'extracted_circuits/sparse_even_key_producers_8_2_9_8_v1'
    spec=importlib.util.spec_from_file_location('parent_package',package/'execute.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    p=old.load_program(package/'program.pt'); start=time.perf_counter(); new=SharedParent(p)
    preparation=time.perf_counter()-start; checks=[]; timings=[]
    cache=torch.load(P/'SELECTIVE_INTERACTION_PORTS_V1_ARTIFACT.pt',weights_only=True,mmap=True)
    panel=json.loads((P/'SCALAR_CUE_CHANNEL_SHIFT_V1_ROWS.json').read_text())['rows']
    for context in range(2):
        x=F.rms_norm(cache['raw9'][context].float(),(1152,),eps=torch.finfo(torch.float32).eps)
        tokens=torch.zeros(x.shape[:2],dtype=torch.long)
        a=old.scalar(x,tokens,p,1); b=new.scalar(x,tokens,1)
        checks.append(dict(inputs='cached_native',context=context,relative=float((a-b).norm()/a.norm())))
    for batch in [1,8]:
        x=torch.randn(batch,32,1152); tokens=torch.randint(0,50000,(batch,32))
        for index in [0,1]:
            a=old.scalar(x,tokens,p,index); b=new.scalar(x,tokens,index)
            checks.append(dict(inputs='independent',batch=batch,index=index,relative=float((a-b).norm()/a.norm())))
        arms=[lambda:old.scalar(x,tokens,p,1),lambda:new.scalar(x,tokens,1)]
        for fn in arms:
            for _ in range(3):fn()
        samples=[[],[]]
        for trial in range(7):
            for j in ([0,1] if trial%2==0 else [1,0]):
                start=time.perf_counter()
                for _ in range(10):arms[j]()
                samples[j].append((time.perf_counter()-start)/10)
        med=[statistics.median(s) for s in samples]
        timings.append(dict(batch=batch,length=32,seconds=med,samples=samples,speedup=med[0]/med[1]))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),checks=checks,timings=timings,
                pred_a=all(r['relative']<=1e-10 for r in checks),pred_b=all(r['speedup']>=1.1 for r in timings),
                preparation_seconds=preparation,extra_resident_adapter_bytes=sum(t.numel()*t.element_size() for a in new.adapters for t in a),
                saved_mac_per_call='N*1152*64 + 2*128*1152*64; latter moves to one-time preparation',
                serialized_weight_change=0,
                scope='Exact reuse in existing conditional sparse parent. Native context/suffix external; resident adapters increase, no full-model speed or new circuit claim.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
