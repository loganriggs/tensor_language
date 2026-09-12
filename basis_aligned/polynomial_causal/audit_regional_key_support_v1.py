"""Preregistered CPU shared-update support localization; no weight fit."""
import json, hashlib, time
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
EPS=torch.finfo(torch.float32).eps


def prefixes(stem):
    rows=json.loads((P/(stem+'_ROWS.json')).read_text())['rows']; result=set()
    for i in range(0,len(rows),2):
        a,b=rows[i]['ids'],rows[i+1]['ids']; c=[k for k in range(len(a)) if a[k]!=b[k]]
        assert len(c)==1
        result.update([tuple(a[:c[0]+1]),tuple(b[:c[0]+1])])
    return result


def main():
    tic=time.perf_counter();torch.set_num_threads(2)
    out=P/'REGIONAL_KEY_SUPPORT_V1_RESULT.json'; frozen=P/'REGIONAL_KEY_SUPPORT_V1_FROZEN.json'
    assert not out.exists() and not frozen.exists()
    train=prefixes('REGIONAL_SOURCE_BLOCK_V2'); test=prefixes('REGIONAL_SOURCE_BLOCK_OOD_V1')
    assert len(train)==4 and len(test)==8 and not train&test
    source=P/'REGIONAL_KEY_PREFIX_V3_ARTIFACT.pt'
    records=torch.load(source,weights_only=True,map_location='cpu')
    binding=json.loads((P/'REGIONAL_KEY_PREFIX_V3_BINDING.json').read_text())['files']
    checkpoint=next(k for k in binding if k.endswith('pytorch_model.bin'))
    sd=torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu')
    readers={j:[sd[f'transformer.h.{j}.attn.{name}.weight'].double().reshape(9,128,1152) for name in ('c_k','c_k2')] for j in (8,9,13)}
    def compile_record(rec):
        z=torch.cat([rec['anchor'][None],rec['parts']],0).double(); n=len(z); j=rec['layer']
        ps=[torch.einsum('fd,hkd->fhk',z,k) for k in readers[j]]
        native=rec['native'].double(); native_rho=native.square().mean()+EPS
        ref=[]
        for k in readers[j]:
            v=torch.einsum('d,hkd->hk',native,k)
            ref.append(v/(v.square().mean(-1,keepdim=True)+EPS*native_rho).sqrt())
        return (n,z@z.T/1152,ps,ref)
    # Heldout projected features are constructed only after support freezing.
    training=[compile_record(r) for r in records if tuple(r['prefix']) in train]
    def errors(supports,data):
        amplitudes=torch.zeros(len(supports),27,dtype=torch.float64);amplitudes[:,0]=1
        for i,s in enumerate(supports): amplitudes[i,[k+1 for k in s]]=1
        all_errors=[]
        for n,g,ps,ref in data:
            a=amplitudes[:,:n]; rho=torch.einsum('nf,fg,ng->n',a,g,a)+EPS
            keys=[]
            for p in ps:
                v=torch.einsum('nf,fhk->nhk',a,p)
                keys.append(v/(v.square().mean(-1,keepdim=True)+EPS*rho[:,None,None]).sqrt())
            u,v=keys; ru,rv=ref
            den=(ru.square().sum(-1)*rv.square().sum(-1)).sum()
            norm=(u.square().sum(-1)*v.square().sum(-1)).sum(-1)
            cross=((u*ru).sum(-1)*(v*rv).sum(-1)).sum(-1)
            all_errors.append(((norm+den-2*cross).clamp_min(0)/den).sqrt())
        return torch.stack(all_errors,1)
    universe=set(range(26)); trajectories={}; supports={}
    for mode in ('forward','backward'):
        selected=set() if mode=='forward' else universe.copy(); trajectory=[]
        for step in range(13):
            choices=sorted(universe-selected if mode=='forward' else selected)
            candidates=[selected|{k} if mode=='forward' else selected-{k} for k in choices]
            scores=errors(candidates,training).max(1).values
            best=int(scores.argmin());selected=candidates[best]
            trajectory.append(dict(changed=choices[best],support=sorted(selected),worst_error=float(scores[best])))
        supports[mode]=sorted(selected);trajectories[mode]=trajectory
    frozen.write_text(json.dumps(dict(supports=supports,trajectories=trajectories,
                                     train_prefixes=[list(p) for p in sorted(train)],test_prefixes=[list(p) for p in sorted(test)]),indent=2)+'\n')
    heldout=[compile_record(r) for r in records if tuple(r['prefix']) in test]
    scores={mode:dict(original_max=float(errors([support],training).max()),heldout_max=float(errors([support],heldout).max())) for mode,support in supports.items()}
    full=float(errors([list(universe)],training+heldout).max())
    # Fixed original case, forward support: compare the contracted metric with dense outer products.
    n,g,ps,ref=training[0];a=torch.zeros(n,dtype=torch.float64);a[0]=1
    for k in supports['forward']:
        if k+1<n:a[k+1]=1
    rho=a@g@a+EPS;keys=[]
    for p in ps:
        v=torch.einsum('f,fhk->hk',a,p);keys.append(v/(v.square().mean(-1,keepdim=True)+EPS*rho).sqrt())
    dense=keys[0][:,:,None]*keys[1][:,None,:];reference=ref[0][:,:,None]*ref[1][:,None,:]
    dense_error=float((dense-reference).norm()/reference.norm())
    agreement=abs(dense_error-float(errors([supports['forward']],training)[0,0]))
    a_ok=full<=2e-5 and agreement<=1e-10;b_ok=a_ok and scores['forward']['original_max']<=.1
    result={'pred_a':a_ok,'pred_b':b_ok,'pred_c':b_ok and scores['forward']['heldout_max']<=.1}
    result.update(scores=scores,supports=supports,all_port_replay_error=full,dense_metric_agreement=agreement,
                  seconds=time.perf_counter()-tic,source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  artifact_sha=hashlib.sha256(source.read_bytes()).hexdigest(),frozen_sha=hashlib.sha256(frozen.read_bytes()).hexdigest(),
                  scope='Original-prefix conditional dependency localization of frozen weight-discovered branch. Geographic support validation, no weights or amplitudes fitted; no recursively pruned execution or behavioral sufficiency claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
