"""Fixed-writer MLP response predicts whole head9 change from pristine ports.
A baseline<=1e-5; B fullmatrix response/statechange<=1e-4;
C rank<=64 whole scalarresponse androutingchange<=5% eachcuefamily.
"""
from pathlib import Path
import json,time,torch
import torch.nn.functional as F
from regional_even_routing_v1 import routing
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter();out=P/'PRISTINE_MLP_PORT_V1_RESULT.json';assert not out.exists();bridge=torch.load(P/'SCALAR_PRODUCER_DIRECTIONAL_MLP_V1_PROGRAM.pt',weights_only=True);j=bridge['mixed_map'];d=bridge['direction'];low=torch.load(P/'DIRECTIONAL_ROUTING_PREDICTOR_V1_TOP64.pt',weights_only=True);cache=torch.load(P/'SELECTIVE_INTERACTION_PORTS_V1_ARTIFACT.pt',weights_only=True);rows=json.loads((P/'SCALAR_CUE_CHANNEL_SHIFT_V1_ROWS.json').read_text())['rows'];p=torch.load(P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt',weights_only=True);binding=json.loads((P/'FOLDED_PRODUCER_CUBIC_NATIVE_V1_BINDING.json').read_text())['files'];sd=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu');lam=sd['transformer.h.9.lambdas'].double();bias=sd['transformer.h.8.mlp.Down_bias'].double();raw=cache['raw9'].double();z=cache['z8'][0].double();a=cache['amplitude8'][0];eps=torch.finfo(torch.float32).eps;ids=torch.zeros(z.shape[:2],dtype=torch.long);valid=torch.zeros_like(ids,dtype=torch.bool)
 for i,r in enumerate(rows):n=len(r['ids']);ids[i,:n]=torch.tensor(r['ids']);valid[i,:n]=True
 x0=F.rms_norm(sd['transformer.wte.weight'][ids],(1152,),eps=eps).double();u0=(raw[0]-lam[1]*x0)/lam[0]-z-bias
 rho=z.square().mean(-1,keepdim=True)+eps;zm=z-a[...,None]*d;rhom=zm.square().mean(-1,keepdim=True)+eps;mid=z-a[...,None]*d/2;base=-a[...,None]*d+(rho/rhom-1)*u0
 def evaluate(r):
  current=F.rms_norm(r.float(),(1152,),eps=eps);gamma=routing(current,p,1);value=current.double()@p['current_value_reader'];scalar=(gamma@value[...,None])[...,0];return gamma,scalar
 g0,s0=evaluate(raw[0]);rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-30));baseline=max(rel(g0,cache['gamma'][0]),rel(s0[valid],cache['scalar'][0][valid]));records=[];rank_data={}
 radial=raw[0]*(u0*raw[0]).sum(-1,keepdim=True)/raw[0].square().sum(-1,keepdim=True).clamp_min(1e-30)
 for mode in ['full','rank64','omit_u0','radial_u0','transverse_u0']:
  kept={'full':u0,'rank64':u0,'omit_u0':torch.zeros_like(u0),'radial_u0':radial,'transverse_u0':u0-radial}[mode];base=-a[...,None]*d+(rho/rhom-1)*kept;mixed=mid@j.T if mode=='full' else (mid@low['right'].T)@low['left'].T;change=lam[0]*(base-a[...,None]/rhom*mixed);predraw=raw[0]+change;gp,sp=evaluate(predraw);cells=[]
  for family in range(3):
   ix=[i for i,r in enumerate(rows) if r['family']==family];ts=[len(rows[i]['ids'])-1 for i in ix];mask=valid[ix];target=cache['scalar'][1,ix,ts]-cache['scalar'][0,ix,ts];prediction=sp[ix,ts]-s0[ix,ts];dg=cache['gamma'][1,ix]-cache['gamma'][0,ix];gmask=valid[ix,:,None]&valid[ix,None,:];response_error=rel(prediction,target);route_error=rel((gp[ix]-g0[ix])[gmask],dg[gmask]);state_error=rel(change[ix][mask],(raw[1,ix]-raw[0,ix])[mask]);cells.append(dict(family=family,name=rows[ix[0]]['family_name'],response_reference_norm=float(target.norm()),scalar_response_error=response_error,routing_change_error=route_error,raw_state_change_error=state_error,passed=response_error<=.05 and route_error<=.05))
  records.append(dict(mode=mode,cells=cells));print(json.dumps(records[-1]),flush=True)
 A=baseline<=1e-5 and all(c['scalar_response_error']<=1e-4 for c in records[0]['cells']) and all(c['passed'] for c in records[1]['cells']);B=A and all(c['passed'] for c in records[2]['cells']);C=A and any(all(c['passed'] for c in r['cells']) for r in records[3:]);result={'pred_a':A,'pred_b':B,'pred_c':C,'records':records,'seconds':time.perf_counter()-tic,'scope':'CPU dependency audit on existing shortcue cache. Frozen weights. u0 radial/transverse split is a diagnostic using native raw9; eliminating u0 port does not generate raw9 independently.'};out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
