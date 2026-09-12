"""Frozen compact response on fresh selective-removal cache, pristine inputs only.
A edited-z reconstruction<=1e-5; B value-port prediction<=5% eachfamily.
"""
from pathlib import Path
import torch,json
from compiled_value_interaction_v1 import execute,EPS
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);out=P/'COMPILED_VALUE_INTERACTION_SELECTIVE_V1_RESULT.json';assert not out.exists();p=torch.load(P/'COMPILED_VALUE_INTERACTION_V1_PROGRAM.pt',weights_only=True);cache=torch.load(P/'SELECTIVE_INTERACTION_PORTS_V1_ARTIFACT.pt',weights_only=True);rows=json.loads((P/'SCALAR_CUE_CHANNEL_SHIFT_V1_ROWS.json').read_text())['rows'];z0=cache['z8'][0];amplitude=cache['amplitude8'][0];den=(cache['raw9'][0].square().mean(-1)+EPS).sqrt().double();pred=execute(z0,amplitude,den,p);dv=cache['value'][1]-cache['value'][0];avg=(cache['gamma'][1]+cache['gamma'][0])/2;vport=(avg@dv[...,None])[...,0];estimate=(avg@pred[...,None])[...,0];records=[];valid=torch.zeros_like(pred,dtype=torch.bool)
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
 for i,row in enumerate(rows):valid[i,:len(row['ids'])]=True
 z1=z0.double()-amplitude[...,None]*p['writer'];zerror=rel(z1[valid],cache['z8'][1].double()[valid])
 for family in range(3):
  ix=[i for i,r in enumerate(rows) if r['family']==family];ts=[len(rows[i]['ids'])-1 for i in ix];mask=valid[ix];r=dict(family=family,name=rows[ix[0]]['family_name'],all_position_value_change_error=rel(pred[ix][mask],dv[ix][mask]),last_query_value_port_error=rel(estimate[ix,ts],vport[ix,ts]));records.append(r)
 result=dict(pred_a=zerror<=1e-5,pred_b=all(r['last_query_value_port_error']<=.05 for r in records),changed_z_reconstruction_error=zerror,records=records,tensor_scalars=sum(x.numel() for x in p.values()),scope='Compiled5771scalar predictor uses pristinez8, baseline9norm andinterventionamplitude only; no changedstate/changednorm passed. Actualaveragerouting used only to evaluate value-mediated scalarport, not predicted. Fresh selective/newcue cache; fullhead response still needsrouting prediction.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
