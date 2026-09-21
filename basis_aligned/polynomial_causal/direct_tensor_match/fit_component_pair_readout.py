"""Fit the actual conditional component product with fixed399 input directions."""
from pathlib import Path
import json,time,hashlib
import torch
from component_pair_readout import ComponentPairReadout
from shared_private_metric import gram
from global_mixed_source_graph import export,score
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
def main():
 plan=json.loads((P/'COMPONENT_PAIR_READOUT_PLAN_V1.json').read_text())
 for name,digest in plan['hashes'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest
 assert all(json.loads((P/'EXPANDED_COMPONENT_INPUTS_V1.json').read_text())['predictions'].values())
 start=time.perf_counter();output=P/'COMPONENT_PAIR_READOUT_V1.json';assert not output.exists()
 d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);extra=torch.load(P/'EXPANDED_COMPONENT_INPUTS_V1.pt',weights_only=True);oldextra=torch.load(P/'EXPANDED_COVARIANCE_STATES_V1.pt',weights_only=True)
 meta=json.loads((P/'EMPIRICAL_SOURCE_DIRECTIONS_V1.json').read_text());parent=torch.load(P/'EMPIRICAL_SOURCE_DIRECTIONS_PROGRAMS_V1.pt',weights_only=True)[meta['winners']['1']]
 z=torch.cat([d['z'][:1536],oldextra['z'].flatten(0,1).double()]);h=torch.cat([d['h'][:1536],extra['h'].flatten(0,1).double()]);assert len(z)==16384
 target=torch.cat([torch.stack([p['truth'][:1536] for p in d['pairs']],1),extra['native_phi'].flatten(0,1)])
 root=torch.linalg.inv(d['inverse_root']);m=parent['left_reader'].shape[1]
 l=torch.cat([parent['left_reader'],parent['square_reader']],1);r=torch.cat([parent['right_reader'],parent['square_reader']],1);L=root@l;R=root@r
 T=d['teacher'];G=gram(L,R,L,R);C=torch.einsum('ir,oij,jr->ro',L,T,R)
 delta=z-d['mu'];X=(delta@l)*(delta@r)-(l*(d['old_covariance']@r)).sum(0)
 Q=torch.stack([q for pair in d['pairs'] for q in pair['Qs']]);qm=Q@d['mu'];offset=2*z@qm.T+torch.einsum('ij,oji->o',d['old_covariance'],Q)-qm@d['mu']
 s=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt();t=h@torch.stack([pair['a'] for pair in d['pairs']],1)
 weights=torch.zeros((len(G),6),dtype=T.dtype);weights[:m]=parent['product_weights']/d['scales'];weights[m:,5]=parent['square_weights']/d['scales'][5]
 parentq=X@weights*d['scales']+offset
 from compact_source_graph import source_reads
 parent_replay=float((parentq-source_reads(z,parent)).norm()/parentq.norm());assert parent_replay<1e-10
 records=[];programs={}
 for lam in plan['lambdas']:
  for seed in plan['seeds']:
   generator=torch.Generator().manual_seed(seed);W=weights.clone()
   if seed:
    noise=torch.randn(W.shape,dtype=W.dtype,generator=generator)*W.norm(dim=0)/len(W)**.5*.01
    W[:m]+=noise[:m];W[m:,5]+=noise[m:,5]
   histories=[];initials=[];losses=[];train=[]
   for j,pair in enumerate(d['pairs']):
    na=m;nb=len(G) if j==2 else m;energy=T[2*j:2*j+2].square().sum()
    A0=(t[:,j]-.5*offset[:,2*j])/s-pair['alpha'];B0=offset[:,2*j+1]/s-pair['beta']
    Da=-.5*d['scales'][2*j]*X[:,:na]/s[:,None];Db=d['scales'][2*j+1]*X[:,:nb]/s[:,None]
    metric=ComponentPairReadout(G[:na,:na],G[:nb,:nb],C[:na,2*j],C[:nb,2*j+1],energy,Da,Db,A0,B0,target[:,j],lam)
    wa,wb,history=metric.fit(W[:na,2*j],W[:nb,2*j+1],plan['cycles'])
    W[:na,2*j]=wa;W[:nb,2*j+1]=wb;histories.append(history);losses.append(history[-1]);initials.append(history[0])
    y=(A0+Da@wa)*(B0+Db@wb);train.append(float((y-target[:,j]).norm()/(target[:,j]-target[:,j].mean()).norm()))
   big=export(L,R,W.T,d);p={k:v.clone() for k,v in big.items()}
   for n in ('left_reader','right_reader'):p[n]=big[n][:,:m].clone()
   p['product_weights']=big['product_weights'][:m].clone();p['square_reader']=big['left_reader'][:,m:].clone();p['square_weights']=big['product_weights'][m:,5].clone();p['square_output']=torch.tensor(5)
   assert sum(v.numel() for v in p.values() if v.is_floating_point())==896198
   raw=torch.einsum('ir,ro,jr->oij',L,W,R);hat=(raw+raw.transpose(-1,-2))/2;coef=float((hat-T).norm()/T.norm())
   key=f'{lam}_seed{seed}';programs[key]=p;rec=dict(key=key,lam=lam,seed=seed,objective=sum(losses)/3,initial_objective=sum(initials)/3,histories=histories,calibration_component_errors=train,original_coefficient_error=coef,**score(big,d))
   records.append(rec);print('RECORD',json.dumps({k:v for k,v in rec.items() if k!='histories'}),flush=True)
 winners={str(lam):min((r for r in records if r['lam']==lam),key=lambda r:r['objective'])['key'] for lam in plan['lambdas']}
 selected={k:next(r for r in records if r['key']==v) for k,v in winners.items()};primary=selected[str(plan['primary_lambda'])];control=selected['0']
 pred=dict(pred_a_instrument=parent_replay<1e-10,pred_b_values=all(x<=.15 and x<=1.1*y for x,y in zip(primary['per_mode_errors'],plan['scalar_baseline'])),pred_c_coefficient=primary['original_coefficient_error']<=1.1*control['original_coefficient_error'])
 torch.save(programs,P/'COMPONENT_PAIR_READOUT_PROGRAMS_V1.pt');output.write_text(json.dumps(dict(plan=plan,parent_replay=parent_replay,records=records,winners=winners,predictions=pred,seconds=time.perf_counter()-start),indent=2)+'\n');print(pred,flush=True)
if __name__=='__main__':main()
