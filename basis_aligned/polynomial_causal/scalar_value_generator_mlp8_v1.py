"""Absolute head9 value numerator through MLP8, not the prior directional response.
A random coefficient fold<=1e-10; B native value replay<=1e-4;
C rank<=64 approximation<=1% value and intervention-change error.
"""
from pathlib import Path
import json,time,torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);torch.manual_seed(581);tic=time.perf_counter();out=P/'SCALAR_VALUE_GENERATOR_MLP8_V1_RESULT.json';assert not out.exists()
 binding=json.loads((P/'FOLDED_PRODUCER_CUBIC_NATIVE_V1_BINDING.json').read_text())['files'];sd=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu');reader=torch.load(P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt',weights_only=True)['current_value_readers'][1].double();L,R,D=[sd['transformer.h.8.mlp.'+n+'.weight'].double() for n in ['Left','Right','Down']];bias=sd['transformer.h.8.mlp.Down_bias'].double();lam=sd['transformer.h.9.lambdas'].double();weight=D.T@reader;raw=L.T@(weight[:,None]*R);matrix=(raw+raw.T)/2;eps=torch.finfo(torch.float32).eps
 z=torch.randn(32,1152,dtype=torch.float64);direct=((z@L.T)*(z@R.T))@weight;folded=torch.einsum('ni,ij,nj->n',z,matrix,z);control=float((direct-folded).norm()/direct.norm());eig,vec=torch.linalg.eigh(matrix);order=eig.abs().argsort(descending=True);eig=eig[order];vec=vec[:,order]
 cache=torch.load(P/'SCALAR_PRODUCER_MLP_BRIDGE_CACHE_V1_ARTIFACT.pt',weights_only=True);rows=json.loads((P/'STRUCTURED_PRODUCER_CONFIRMATION_V1_ROWS.json').read_text())['rows'];writer=torch.load(P/'SCALAR_PRODUCER_NATIVE_LIFT_V1_ARTIFACT.pt',weights_only=True)['writers'][0].double();zs=[];xs=[];den=[];targets=[]
 for state in [0,1]:
  zstate=[];xstate=[];dstate=[];tstate=[]
  for i,row in enumerate(rows):
   n=len(row['ids']);zz=cache['z8'][i,:n].double()
   if state:zz=zz-cache['a8'][i,:n,None]*writer
   x0=F.rms_norm(sd['transformer.wte.weight'][torch.tensor(row['ids'])],(1152,),eps=eps).double();raw9=cache['r9'][state,i,:n];d9=(raw9.square().mean(-1)+eps).sqrt().double();target=F.rms_norm(raw9,(1152,),eps=eps).double()@reader
   zstate.append(zz);xstate.append(x0);dstate.append(d9);tstate.append(target)
  zs.append(torch.cat(zstate));xs.append(torch.cat(xstate));den.append(torch.cat(dstate));targets.append(torch.cat(tstate))
 zs=torch.stack(zs);xs=torch.stack(xs);den=torch.stack(den);target=torch.stack(targets);rho=zs.square().mean(-1)+eps;linear=lam[0]*(zs@reader)+lam[1]*(xs@reader)+lam[0]*(bias@reader);project=zs@vec;terms=project.square()*eig;pred=(linear+lam[0]*terms.sum(-1)/rho)/den
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));native_errors=[rel(pred[i],target[i]) for i in [0,1]];change_reference=target[1]-target[0];records=[]
 for rank in [0,1,4,16,64,256,1152]:
  approx=(linear+lam[0]*terms[...,:rank].sum(-1)/rho)/den;records.append(dict(rank=rank,coefficient_energy_fraction=float(eig[:rank].square().sum()/eig.square().sum()),native_value_errors=[rel(approx[i],target[i]) for i in [0,1]],intervention_change_error=rel(approx[1]-approx[0],change_reference),positive_modes=int((eig[:rank]>0).sum()),negative_modes=int((eig[:rank]<0).sum())))
 result=dict(pred_a=control<=1e-10,pred_b=max(native_errors)<=1e-4,pred_c=max(native_errors)<=1e-4 and any(r['rank']<=64 and max(r['native_value_errors'])<=.01 and r['intervention_change_error']<=.01 for r in records),random_coefficient_relative_error=control,native_value_relative_errors=native_errors,change_reference_norm=float(change_reference.norm()),records=records,spectrum_top10=eig[:10].tolist(),positions_per_state=target.shape[1],seconds=time.perf_counter()-tic,scope='Exact absolute numerator of head9.8 currentvalue throughMLP8 plusresidual/reentry/bias. Native9RMSdenominator supplied, z8/x0 inputs supplied. Cachedbroaderhead8removal states, not a newselectivecomponent intervention. Eigenrank selected onlybyweights; nativefield checks notend-task adoption.')
 torch.save(dict(reader=reader,matrix=matrix,eigenvalues=eig,eigenvectors=vec,lambdas=lam,bias_read=reader@bias),P/'SCALAR_VALUE_GENERATOR_MLP8_V1_PROGRAM.pt');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
