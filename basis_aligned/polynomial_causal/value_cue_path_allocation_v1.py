from pathlib import Path
import json,time,torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter();gen=torch.load(P/'SCALAR_VALUE_GENERATOR_MLP8_V1_PROGRAM.pt',weights_only=True);cache=torch.load(P/'SELECTIVE_INTERACTION_PORTS_V1_ARTIFACT.pt',weights_only=True);rows=json.loads((P/'SCALAR_CUE_CHANNEL_SHIFT_V1_ROWS.json').read_text())['rows'];binding=json.loads((P/'DIRECTIONAL_INTERACTION_LOGIT_V1_BINDING.json').read_text())['files'];sd=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True);z=cache['z8'][0].double();raw=cache['raw9'][0];ids=torch.zeros(z.shape[:2],dtype=torch.long);valid=torch.zeros_like(ids,dtype=torch.bool)
 for i,row in enumerate(rows):n=len(row['ids']);ids[i,:n]=torch.tensor(row['ids']);valid[i,:n]=True
 eps=torch.finfo(torch.float32).eps;x0=F.rms_norm(sd['transformer.wte.weight'][ids],(1152,)).double();rho=z.square().mean(-1)+eps;den=(raw.square().mean(-1)+eps).sqrt().double();lam=gen['lambdas'];linear=lam[0]*(z@gen['reader'])+lam[1]*(x0@gen['reader'])+lam[0]*gen['bias_read'];terms=(z@gen['eigenvectors']).square()*gen['eigenvalues'];target=cache['value'][0];gamma=cache['gamma'][0];rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));records=[];source_groups=[]
 components=[linear/den,lam[0]*terms[...,:4].sum(-1)/rho/den,lam[0]*terms[...,4:].sum(-1)/rho/den];records=[]
 for family in range(3):
  ix=[i for i,r in enumerate(rows) if r['family']==family];ui=ix[::2];ai=ix[1::2];mean_g=(gamma[ui]+gamma[ai])/2;ts=[len(rows[i]['ids'])-1 for i in ui];vi=torch.arange(len(ui));d=(mean_g@(target[ui]-target[ai])[...,None])[...,0][vi,ts];parts=[]
  for label,v in zip(['direct_reentry_bias','four_quadratic_modes','quadratic_remainder'],components):
   e=(mean_g@(v[ui]-v[ai])[...,None])[...,0][vi,ts];parts.append(e);records.append(dict(family=family,part=label,aligned_fraction=float((e@d)/d.square().sum()),norm_over_target=float(e.norm()/d.norm()),cosine=float((e@d)/(e.norm()*d.norm()))))
  records.append(dict(family=family,full_accounting_relative_error=rel(sum(parts),d)))
 result={'records':records,'scope':'Symmetricnative-routing allocation of the sourcevalue-mediated cuecontrast, not fullhead/logit effect. Direct/reentry/bias versus fourmodequadratic versus remainder, all fromexistingweightfold. Not a physicalbranch removal; routing and pristineinputs supplied.'};(P/'VALUE_CUE_PATH_ALLOCATION_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
