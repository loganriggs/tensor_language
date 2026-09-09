"""Model-free population and prospective counterfactual semantics."""
import torch
from forward_endpoint_program_reference import joins


def populations():
    out={}
    for pop,seed in (('iid',24909),('ood_three_8cycles',24910)):
        g=torch.Generator().manual_seed(seed);worlds=[]
        for world in range(16):
            perm=torch.randperm(24,generator=g);f=torch.empty(24,dtype=torch.long)
            size=24 if pop=='iid' else 8
            for offset in range(0,24,size):
                cycle=perm[offset:offset+size];f[cycle]=cycle.roll(-1)
            keys=torch.randperm(24,generator=g);prefix=torch.stack((keys,f[keys]),-1).flatten()
            positions=torch.empty(24,dtype=torch.long).scatter_(0,keys,torch.arange(24))
            labels=torch.randperm(24,generator=g);sigma=torch.empty(24,dtype=torch.long).scatter_(0,labels,labels.roll(-1))
            tokens=[];metadata=[]
            for query in range(24):
                chain=[query]
                for _ in range(3):chain.append(int(f[chain[-1]]))
                for hop in range(4):
                    eligible=hop>=2 and int(positions[chain[hop-1]])<int(positions[chain[hop-2]])
                    parity=int(positions[chain[hop-2]])%2 if eligible else -1
                    answer=chain[hop]
                    metadata.append({'world':world,'query':query,'hop':hop,'answer':answer,'eligible':eligible,
                                     'parity':parity,'desired_answer':int(sigma[answer]) if eligible else answer})
                    tokens.append(torch.cat((prefix,torch.tensor([24,query,25+hop]))))
            tokens=torch.stack(tokens);mask,records=joins(tokens[:1]);mask=mask[0]
            even=mask.clone();odd=mask.clone()
            target_even=(torch.arange(51)//2)%2==0;even[~target_even]=False;odd[target_even]=False
            worlds.append({'world':world,'tokens':tokens,'metadata':metadata,'sigma':sigma,'function':f,
                           'mask':mask,'masks':{0:even,1:odd},'records':records[0]})
        out[pop]=worlds
    return out


def controls():
    checks={};counts={}
    for pop,worlds in populations().items():
        valid=partition=semantics=deranged=True;counts[pop]={}
        for w in worlds:
            tok=w['tokens'];f=w['function'];sigma=w['sigma'];m=w['masks']
            valid &= bool((tok[0,:48:2].sort().values==torch.arange(24)).all() and (tok[0,1:48:2].sort().values==torch.arange(24)).all())
            deranged &= bool((sigma.sort().values==torch.arange(24)).all() and (sigma!=torch.arange(24)).all())
            partition &= bool(torch.equal(m[0]|m[1],w['mask']) and not (m[0].any(-1)&m[1].any(-1)).any())
            for row,meta in zip(tok,w['metadata']):
                e=int(row[49])
                for _ in range(meta['hop']):e=int(f[e])
                valid &= e==meta['answer']
                if meta['hop']>=2:
                    origin=int(row[49])
                    for _ in range(meta['hop']-2):origin=int(f[origin])
                    found=[r for r in w['records'] if r['origin']==origin]
                    semantics &= bool(found)==meta['eligible']
                    if found:
                        semantics &= found[0]['endpoint']==meta['answer'] and (found[0]['destinations'][0]//2)%2==meta['parity']
                key=f"hop{meta['hop']}_eligible{meta['eligible']}_parity{meta['parity']}"
                counts[pop][key]=counts[pop].get(key,0)+1
        checks.update({pop+'_maps_answers':valid,pop+'_deranged_permutation':deranged,
                       pop+'_disjoint_partition':partition,pop+'_semantic_parser_agreement':semantics})
        checks[pop+'_eligible_groups_nonempty']=all(counts[pop].get(f'hop{h}_eligibleTrue_parity{p}',0)>0 for h in (2,3) for p in (0,1))
    return {'passed':all(checks.values()),'checks':checks,'counts':counts}
