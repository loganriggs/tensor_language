"""Check pair truncation after exact producer substitution, without data fitting."""
from pathlib import Path
import json,hashlib
import torch
from coupled_quartic_writer_v1 import solve
P=Path(__file__).resolve().parent
torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
source=P/'NATIVE_PAIR_PENCIL_V1_PROGRAM.pt';prior=json.loads((P/'NATIVE_PAIR_PENCIL_V1_RESULT.json').read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest()==prior['artifact_sha256']
p=torch.load(source,weights_only=True);dual=p['dual_readers'];cores=torch.zeros(2,1152,1152);small=cores.clone();rows=[];pairs=[]
for block,(ids,values) in enumerate(zip(p['groups'],p['cores'])):
    for m in range(2):
        for a,i in enumerate(ids):
            for b,j in enumerate(ids):
                cores[m,i,j]=values[m,a,b]
                if block in p['selected_blocks']:small[m,i,j]=values[m,a,b]
    if block in p['selected_blocks']:
        offset=len(rows);rows.extend(ids)
        pairs.extend((offset+i,offset+j) for i in range(len(ids)) for j in range(i,len(ids)))
forms=dual.T@cores@dual;truncated=dual.T@small@dual
b=dual[rows];ij=torch.tensor(pairs);u,v=b[ij[:,0]],b[ij[:,1]]
k=((u@u.T)*(v@v.T)+(u@v.T)*(v@u.T))/2;c=torch.einsum('id,mde,ie->im',u,forms,v)
mix,diag=solve(k,c);refit=torch.stack([((u.T*mix[:,m])@v+(v.T*mix[:,m])@u)/2 for m in range(2)])
# Weights fixed above; the cache is only an evaluation panel.
binding=json.loads((P/'COUPLED_QUARTIC_LBFGS_V1_BINDING.json').read_text())['files'];ck=next(k for k in binding if k.endswith('/pytorch_model.bin'))
state=torch.load(ck,weights_only=True,mmap=True)
l,r,d=[state[f'transformer.h.16.mlp.{name}.weight'].double() for name in ('Left','Right','Down')]
scale=float(state['transformer.h.17.lambdas'][0])
x=torch.load(P/'QUARTIC_GROUP_PORTS_V1_PORTS.pt',weights_only=True)['input16'].double()
producer=scale*((x@l.T)*(x@r.T))@d.T
pre=torch.load(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True)['ports']['pre'].double();den=pre.square().mean(-1)+torch.finfo(torch.float32).eps
reference=torch.load(P/'QUARTIC_GROUP_BOUNDARY_V1_WRITES.pt',weights_only=True)['lifted'][0]
panel=json.loads((P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ROWS.json').read_text())['rows'];reports=[];writes=[]
for name,form in [('full',forms),('truncated',truncated),('refit',refit)]:
    write=torch.einsum('nd,mde,ne->nm',producer,form,producer)@p['output_writers'].T/den[:,None]
    families=[]
    for family in sorted({r['family'] for r in panel}):
        ids=torch.tensor([i for i,r in enumerate(panel) if r['family']==family]);change=write[2*ids+1]-write[2*ids];ref=reference[2*ids+1]-reference[2*ids]
        families.append(dict(family=family,paired_write_error=float((change-ref).norm()/ref.norm())))
    reports.append(dict(name=name,write_error=float((write-reference).norm()/reference.norm()),families=families));writes.append(write)
result=dict(pred_a=reports[0]['write_error']<=1e-8 and diag['normal_residual']<=1e-8,pred_b=reports[-1]['write_error']<=.1,pred_c=all(r['paired_write_error']<=.1 for r in reports[-1]['families']),reports=reports,scope='Exact native producer followed by fixed pair approximation, external normalization/background. Reused development inputs, no full model or semantic extraction claim.')
out=P/'NATIVE_PAIR_PENCIL_FOLD_V1_RESULT.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['pred_a']
