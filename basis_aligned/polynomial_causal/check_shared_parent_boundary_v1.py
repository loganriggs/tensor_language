"""Frozen runtime countercheck: caching versus actual shared input read."""
import importlib.util
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
from shared_parent_runtime_v1 import SharedParent

P=Path(__file__).resolve().parent


@torch.no_grad()
def main():
    out=P/'SHARED_PARENT_RUNTIME_BOUNDARY_V1_RESULT.json';assert not out.exists()
    torch.set_num_threads(2);torch.manual_seed(170233100)
    root=P/'extracted_circuits/sparse_even_key_producers_8_2_9_8_v1'
    spec=importlib.util.spec_from_file_location('original',root/'execute.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    p=old.load_program(root/'program.pt')
    cached=SharedParent(p,False);shared=SharedParent(p);rows=[];start_all=time.perf_counter()
    for batch in [1,8]:
        for length in [16,128]:
            x=torch.randn(batch,length,1152);tokens=torch.randint(0,50000,(batch,length))
            for index in [0,1]:
                functions=[lambda:old.scalar(x,tokens,p,index),lambda:cached.scalar(x,tokens,index),lambda:shared.scalar(x,tokens,index)]
                values=[fn() for fn in functions];errors=[float((v-values[0]).norm()/values[0].norm()) for v in values[1:]]
                for fn in functions:
                    for _ in range(3):fn()
                samples=[[],[],[]]
                for trial in range(7):
                    for j in [(trial+i)%3 for i in range(3)]:
                        start=time.perf_counter()
                        for _ in range(5):functions[j]()
                        samples[j].append((time.perf_counter()-start)/5)
                med=[statistics.median(s) for s in samples]
                rows.append(dict(batch=batch,length=length,index=index,replay_relative=errors,
                                 medians=med,samples=samples,total_speedup=med[0]/med[2],read_sharing_speedup=med[1]/med[2]))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),rows=rows,
                pred_a=all(max(r['replay_relative'])<=1e-10 for r in rows),
                pred_b=all(r['total_speedup']>=1.1 for r in rows),
                pred_c=all(r['read_sharing_speedup']>=1.03 for r in rows),
                seconds=time.perf_counter()-start_all,
                arm_order=['original','cached_adapters_only','cached_and_shared_read'],
                scope='Fixed weight runtime discriminator; independent probes, no new behavior test. Both new arms add256KiB resident adapters, zero serialized weights. Shared read saves batch*length*1152*64 MACs per head call beyond cached-only.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    print(json.dumps([{k:v for k,v in r.items() if k not in ['samples','medians']} for r in rows]))


if __name__=='__main__':main()
