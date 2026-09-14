"""Frozen runtime countercheck: caching versus actual shared input read."""
import importlib.util
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
from shared_parent_runtime_v1 import SharedParent
from key_span_reuse_v1 import KeySpanParent

P=Path(__file__).resolve().parent


@torch.no_grad()
def main():
    out=P/'KEY_SPAN_RUNTIME_V1_RESULT.json';assert not out.exists()
    torch.set_num_threads(2);torch.manual_seed(170233100)
    root=P/'extracted_circuits/sparse_even_key_producers_8_2_9_8_v1'
    spec=importlib.util.spec_from_file_location('original',root/'execute.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    p=torch.load(P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt',weights_only=True)
    cached=SharedParent(p);shared=KeySpanParent(torch.load(P/'KEY_SPAN_REUSE_V1_PROGRAM.pt',weights_only=True));rows=[];start_all=time.perf_counter()
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
                                 medians=med,samples=samples,total_speedup=med[0]/med[2],key_span_over_shared_speedup=med[1]/med[2]))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),rows=rows,
                pred_a=all(max(r['replay_relative'])<=1e-10 for r in rows),
                pred_b=all(r['key_span_over_shared_speedup']>=1.03 for r in rows),
                seconds=time.perf_counter()-start_all,
                arm_order=['original_exact','shared_cached_exact','key_span_exact'],
                scope='Exact original-weight runtime discriminator; independent probes, no new behavior test. Both prepared arms retain256KiB resident adapters. Key span additionally replaces2*1152*64 FP64basis scalars with2*256*64 coordinates, saving917504 residentbytes. Perheadcall MAC reduction vsalreadyshared=batch*length*(1152-256)*64.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    print(json.dumps([{k:v for k,v in r.items() if k not in ['samples','medians']} for r in rows]))


if __name__=='__main__':main()
