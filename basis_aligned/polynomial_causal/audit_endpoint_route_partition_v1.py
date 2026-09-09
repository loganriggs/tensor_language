"""Exact two-route knockout census from saved native projection-port outputs."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import time
import torch
import field_intervention_metrics as M
BASE=Path(__file__).resolve().parent


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    signal.alarm(180);torch.set_num_threads(2);started=time.perf_counter()
    source=BASE/'ORIGIN_ENDPOINT_INTERACTION_V1_ROWS.pt';out=BASE/'ENDPOINT_ROUTE_PARTITION_V1_RESULT.json'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest=='4e25c055bc7822349261b88db9ade9bb0cad7c9caaf6d525bc56344285f82f8b' and not out.exists()
    data=torch.load(source,map_location='cpu',weights_only=True);audits=[];results={};cases=[]
    for pop,block in data.items():
        z=block['query_logits'];a,b,c,d=(z[k] for k in ('native','key_cut','value_cut','both_cut'))
        first=b+c-d;second=a-b+d
        audits.extend((M.correspondence(first,z['without_interaction']),M.correspondence(first+second,a+c)))
        logits={'native':a,'without_I':first,'without_J':second,'without_both':c};meta=block['metadata']
        gold=torch.tensor([m['answer'] for m in meta]);bits=torch.stack([v.argmax(-1)==gold for v in logits.values()],-1)
        codes=[''.join(str(int(x)) for x in row) for row in bits];groups={}
        for query in (0,1):
            groups[str(query)]={}
            for hop in range(4):
                selected=[i for i,m in enumerate(meta) if m['query']==query and m['hop']==hop]
                sel=torch.tensor([i in selected for i in range(len(meta))])
                census={''.join(map(str,p)):0 for p in itertools.product((0,1),repeat=4)}
                for i in selected:census[codes[i]]+=1
                patterns={str(order):{code:sum(meta[i]['order']==order and codes[i]==code for i in selected) for code in census} for order in sorted({meta[i]['order'] for i in selected})}
                groups[str(query)][str(hop)]={'n':len(selected),'correctness_census':census,'order_census':patterns,
                    'both_singles_survive_joint_fails':census['1110'],
                    'only_I_single_fails':census['1010']+census['1011'],
                    'only_J_single_fails':census['1100']+census['1101'],
                    'both_singles_fail':census['1000']+census['1001'],
                    'all_survive':census['1111'],'nonmonotone_joint_rescues':census['1001']+census['1011']+census['1101'],
                    'native_errors':sum(v for k,v in census.items() if k[0]=='0'),
                    'panels':{arm:M.panel(logits,arm,meta,sel) for arm in logits}}
        for i,m in enumerate(meta):cases.append({'population':pop,**m,'correctness_bits':codes[i]})
        results[pop]=groups
    result={'scope':'Opened exact route partition and exhaustive correctness census; prior nomination remains failed.',
        'bit_order':['native','without_I','without_J','without_both'],'instrument_passed':all(a['passed'] for a in audits),
        'oracle_max_abs':max(a['max_abs'] for a in audits),'populations':results,'cases':cases,
        'source_sha256':digest,'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'prereg_sha256':hashlib.sha256((BASE/'ENDPOINT_ROUTE_PARTITION_V1_PREREGISTRATION.md').read_bytes()).hexdigest(),
        'wall_seconds':time.perf_counter()-started}
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'instrument_passed':result['instrument_passed'],'oracle_max_abs':result['oracle_max_abs'],
        'wall_seconds':result['wall_seconds'],'forward_hop3':{p:r['0']['3'] for p,r in results.items()}},indent=2))


if __name__=='__main__':main()
