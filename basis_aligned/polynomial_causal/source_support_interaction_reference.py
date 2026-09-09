"""Anchored Boolean interactions for independently edited final-layer sources.

Disjoint source supports imply degree<=2 in Boolean edit variables, regardless
of the pointwise RMS nonlinearity. Overlapping edits do not have that guarantee.
"""
import torch


def populations():
    out={}
    for pop,seed in (('iid',20909),('ood_three_8cycles',20910)):
        g=torch.Generator().manual_seed(seed);worlds=[]
        for world in range(16):
            perm=torch.randperm(24,generator=g);fmap=torch.empty(24,dtype=torch.long)
            size=24 if pop=='iid' else 8
            for start in range(0,24,size):
                cycle=perm[start:start+size];fmap[cycle]=cycle.roll(-1)
            chains=[]
            for start in (0,8,16):
                chain=[int(perm[start])]
                for _ in range(3):chain.append(int(fmap[chain[-1]]))
                chains.append(chain)
            fixed={};masks={};heads={}
            for j,slots in enumerate(((1,8,17),(3,10,19),(5,12,21))):
                forward=j!=1;order=(0,2,1) if forward else (0,1,2)
                fixed.update({s:chains[j][k] for s,k in zip(slots,order)})
                mask=torch.zeros(51,51,dtype=torch.bool)
                mask[2*slots[2]:2*slots[2]+2,2*slots[1]:2*slots[1]+2]=True
                masks[j]=mask;heads[j]=1 if forward else 2
            rest=[int(v) for v in torch.randperm(24,generator=g) if int(v) not in fixed.values()];it=iter(rest)
            keys=torch.tensor([fixed[s] if s in fixed else next(it) for s in range(24)])
            binding=torch.stack((keys,fmap[keys]),-1).flatten();rows=[];answers=[]
            for chain in chains:
                for hop in range(4):
                    rows.append(torch.cat((binding,torch.tensor([24,chain[0],25+hop]))));answers.append(chain[hop])
            worlds.append({'world':world,'tokens':torch.stack(rows),'answers':torch.tensor(answers),'masks':masks,'heads':heads})
        out[pop]=worlds
    return out


def mobius(values):
    n = (len(values)-1).bit_length()
    assert len(values) == 1 << n
    out = [v.clone() for v in values]
    for bit in range(n):
        for mask in range(1 << n):
            if mask&(1 << bit): out[mask] -= out[mask^(1 << bit)]
    return out


def evaluate(function, base, writes):
    return [function(base+sum((v for i,v in enumerate(writes) if mask&(1 << i)), torch.zeros_like(base)))
            for mask in range(1 << len(writes))]


def controls():
    from deep_model import DeepModel
    checks={}; details={}
    data=populations(); disjoint_supports=True; correct_answers=True; valid_maps=True
    for worlds in data.values():
        for w in worlds:
            support=[m.any(-1) for m in w['masks'].values()]
            disjoint_supports &= all(not bool((support[i]&support[j]).any()) for i,j in ((0,1),(0,2),(1,2)))
            tok=w['tokens']; valid_maps &= bool((tok[0,:48:2].sort().values==torch.arange(24)).all() and (tok[0,1:48:2].sort().values==torch.arange(24)).all())
            fmap=torch.empty(24,dtype=torch.long).scatter_(0,tok[0,:48:2],tok[0,1:48:2])
            for row,answer in zip(tok,w['answers']):
                x=row[49]
                for _ in range(int(row[50])-25):x=fmap[x]
                correct_answers &= bool(x==answer)
    checks.update({'generator_disjoint_supports':disjoint_supports,'generator_valid_maps':valid_maps,'generator_answers':correct_answers})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20908)
        model=DeepModel(8,16,4,['attn'],8,norm='rms').double().eval()
        x=torch.randn(2,8,16,dtype=torch.float64)
        disjoint=[]; overlap=[]
        for pos in (1,3,5):
            d=torch.zeros_like(x);d[:,pos]=.4*torch.randn(2,16,dtype=torch.float64);disjoint.append(d)
            o=torch.zeros_like(x);o[:,3]=d[:,pos];overlap.append(o)
    with torch.inference_mode():
        f=lambda z:model.head(model.layers[-1](z))
        vals=evaluate(f,x,disjoint);terms=mobius(vals)
        third=float(terms[7].abs().max()); pair_outside=0.; live=[]
        for mask,later in ((3,3),(5,5),(6,5)):
            outside=torch.ones(8,dtype=torch.bool);outside[later]=False
            pair_outside=max(pair_outside,float(terms[mask][:,outside].abs().max()))
            live.append(float(terms[mask][:,later].abs().max()))
        wrong=mobius(evaluate(f,x,overlap)); overlap_third=float(wrong[7].abs().max())
        predicted=terms[0]+terms[1]+terms[2]+terms[4]+terms[3]+terms[5]+terms[6]
        checks['disjoint_third_zero']=third<=1e-9
        checks['pair_support_only_later_changed_queries']=pair_outside<=1e-9
        checks['every_pair_live']=min(live)>1e-9
        checks['seven_eval_joint_prediction']=bool(torch.allclose(predicted,vals[7],atol=1e-9,rtol=1e-9))
        checks['overlap_third_live']=overlap_third>1e-9
        details.update({'fp64_third_max':third,'pair_outside_max':pair_outside,'pair_live_maxima':live,'overlap_third_max':overlap_third})
    g=torch.Generator().manual_seed(20908)
    base=torch.randint(-1,2,(6,3),generator=g)
    matrices=[torch.randint(-1,2,(3,2),generator=g) for _ in range(6)]
    def integer(z):
        q1,k1,q2,k2,v,u=[z@m for m in matrices]
        p=((q1@k1.T)*(q2@k2.T)).tril()
        return u+p@v
    ds=[];os=[]
    for pos in (0,2,4):
        d=torch.zeros_like(base);d[pos]=torch.randint(-2,3,(3,),generator=g);ds.append(d)
        o=torch.zeros_like(base);o[2]=d[pos];os.append(o)
    exact=mobius(evaluate(integer,base,ds));mixed=mobius(evaluate(integer,base,os))
    checks['integer_disjoint_third_zero']=bool((exact[7]==0).all())
    checks['integer_overlap_third_nonzero']=bool((mixed[7]!=0).any())
    details['integer_overlap_third_max']=int(mixed[7].abs().max())
    return {'passed':all(checks.values()),'checks':checks,'details':details}
