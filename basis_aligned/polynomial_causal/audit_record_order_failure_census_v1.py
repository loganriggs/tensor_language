"""All opened record-order pairs, partitioned by native correctness; no filtering."""
import hashlib
import json
from pathlib import Path
import torch
import entity_equivariance_reference as J

BASE=Path(__file__).resolve().parent
INPUT=BASE/'RECORD_ORDER_OUTPUT_INVARIANCE_V1_ROWS.pt'
OUT=BASE/'RECORD_ORDER_FAILURE_CENSUS_V1.json'


def main():
    torch.set_num_threads(2);assert not OUT.exists()
    assert hashlib.sha256(INPUT.read_bytes()).hexdigest()=='beade7e639dcdb2752a7dfb7419866a2d3339d457417841e67902478ee4506c6'
    data=torch.load(INPUT,map_location='cpu',weights_only=True);results={};cases=[];maximum=0.
    for pop,block in data.items():
        old=block['query_logits']['native'];metadata=block['metadata']
        gold=torch.tensor([m['answer'] for m in metadata]);old_correct=old.argmax(-1)==gold;groups={}
        for name,new in block['query_logits'].items():
            if name=='native':continue
            radius,_=J.information_radius(old,new);radius=radius.clamp_min(0)
            maximum=max(maximum,float((radius-block['radii'][name]).abs().max()))
            new_correct=new.argmax(-1)==gold
            categories={'both_correct':old_correct&new_correct,'one_correct':old_correct^new_correct,'neither_correct':~old_correct&~new_correct}
            groups[name]={}
            for hop in range(4):
                sel=torch.tensor([m['hop']==hop for m in metadata]);n=int(sel.sum());parts={}
                for category,mask in categories.items():
                    selected=sel&mask;count=int(selected.sum())
                    parts[category]={'pairs':count,'contribution_to_all_pair_mean':float(radius[selected].sum())/n,
                        'within_category_mean':float(radius[selected].mean()) if count else None}
                assert sum(x['pairs'] for x in parts.values())==n
                assert abs(sum(x['contribution_to_all_pair_mean'] for x in parts.values())-float(radius[sel].mean()))<1e-12
                groups[name][str(hop)]=parts
            for i,m in enumerate(metadata):
                cases.append({'population':pop,'permutation':name,**m,'old_prediction':int(old[i].argmax()),
                    'new_prediction':int(new[i].argmax()),'pair_bound':float(radius[i]),
                    'category':next(c for c,mask in categories.items() if mask[i])})
        results[pop]=groups
    receipt={'scope':'opened all-pair correctness census, not a filtered bound or new discovery cohort',
        'pairs':len(cases),'saved_radius_max_abs':maximum,'partition_passed':True,'groups':results,'cases':cases,
        'input_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest()}
    OUT.write_text(json.dumps(receipt,indent=2)+'\n');assert maximum<1e-12
    print(json.dumps({'pairs':len(cases),'replay':maximum,'hop3':{pop:{name:g['3'] for name,g in groups.items()} for pop,groups in results.items()}},indent=2))


if __name__=='__main__':main()
