from pathlib import Path
import json,time,torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter();gen=torch.load(P/'SCALAR_VALUE_GENERATOR_MLP8_V1_PROGRAM.pt',weights_only=True);cache=torch.load(P/'SELECTIVE_INTERACTION_PORTS_V1_ARTIFACT.pt',weights_only=True);rows=json.loads((P/'SCALAR_CUE_CHANNEL_SHIFT_V1_ROWS.json').read_text())['rows'];binding=json.loads((P/'DIRECTIONAL_INTERACTION_LOGIT_V1_BINDING.json').read_text())['files'];sd=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True);z=cache['z8'][0].double();raw=cache['raw9'][0];ids=torch.zeros(z.shape[:2],dtype=torch.long);valid=torch.zeros_like(ids,dtype=torch.bool)
 for i,row in enumerate(rows):n=len(row['ids']);ids[i,:n]=torch.tensor(row['ids']);valid[i,:n]=True
 eps=torch.finfo(torch.float32).eps;x0=F.rms_norm(sd['transformer.wte.weight'][ids],(1152,)).double();rho=z.square().mean(-1)+eps;den=(raw.square().mean(-1)+eps).sqrt().double();lam=gen['lambdas'];linear=lam[0]*(z@gen['reader'])+lam[1]*(x0@gen['reader'])+lam[0]*gen['bias_read'];terms=(z@gen['eigenvectors']).square()*gen['eigenvalues'];target=cache['value'][0];gamma=cache['gamma'][0];rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));records=[];source_groups=[]
 for rank in [0,4,16,64,1152]:
  pred=(linear+lam[0]*terms[...,:rank].sum(-1)/rho)/den;cells=[]
  for family in range(3):
   ix=[i for i,r in enumerate(rows) if r['family']==family];ui=ix[::2];ai=ix[1::2];mask=valid[ui];dv=target[ui]-target[ai];dp=pred[ui]-pred[ai];mean_g=(gamma[ui]+gamma[ai])/2;truth=(mean_g@dv[...,None])[...,0];approx=(mean_g@dp[...,None])[...,0];ts=[len(rows[i]['ids'])-1 for i in ui];vi=torch.arange(len(ui));cells.append(dict(family=family,name=rows[ix[0]]['family_name'],absolute_value_error=rel(pred[ix][valid[ix]],target[ix][valid[ix]]),cue_value_error=rel(dp[mask],dv[mask]),cue_value_port_error=rel(approx[vi,ts],truth[vi,ts]),reference_port_norm=float(truth[vi,ts].norm())))
  records.append(dict(rank=rank,cells=cells))
 A=all(c['absolute_value_error']<=1e-4 for c in records[-1]['cells']);B=A and all(max(c['cue_value_error'],c['cue_value_port_error'])<=1e-4 for c in records[-1]['cells']);C=B and all(max(c['cue_value_error'],c['cue_value_port_error'])<=.05 for c in records[1]['cells']);result={'pred_a':A,'pred_b':B,'pred_c':C,'records':records,'seconds':time.perf_counter()-tic,'scope':'Existingweight eigenfactors on nativecue variation, not fixedhead8removal. Allvalid sourcevalues and symmetric native-routing value-mediated scalarcontrast. z8/x0/native9norm stillsupplied; routeusedforreadoutscoring notgenerated.16/64are descriptive diagnostics, rank4was primary.'};(P/'VALUE_GENERATOR_CUE_CONTRAST_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
