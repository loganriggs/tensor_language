"""Weight-only embedding generator and oracle token-only limitation certificate."""
from pathlib import Path
import json,torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent
old=json.loads((P/'SCALAR_NEW_ENDPOINTS_V1_ROWS.json').read_text())['rows'][:24];fresh=json.loads((P/'MLP8_VALUE_FRESH_V1_ROWS.json').read_text())['rows'];rows=old+[dict(r,donor_id=r['donor_id']+24) for r in fresh]
a=torch.load(P/'PHI4_PROVENANCE_V1_ARTIFACT.pt',weights_only=True);gen=torch.load(P/'SCALAR_VALUE_GENERATOR_MLP8_V1_PROGRAM.pt',weights_only=True)
binding=json.loads((P/'PHI4_PROVENANCE_V1_BINDING.json').read_text())['files'];path=next(k for k in binding if k.endswith('pytorch_model.bin'));sd=torch.load(path,weights_only=True,mmap=True)
ids=torch.tensor([r['ids'][int(a['cue_positions'][i])] for i,r in enumerate(rows)])
x=F.rms_norm(sd['transformer.wte.weight'][ids],(1152,)).double();phi=((x@gen['eigenvectors'][:,:4]).square()*gen['eigenvalues'][:4]).sum(-1)
native=torch.stack([a['modes'][i,int(a['cue_positions'][i])].sum() for i in range(96)])
y=native[::2]-native[1::2];pred=phi[::2]-phi[1::2];pairs=[(int(ids[i]),int(ids[i+1])) for i in range(0,96,2)];oracle=torch.zeros_like(y)
for key in set(pairs):
 ix=[i for i,k in enumerate(pairs) if k==key];oracle[ix]=y[ix].mean()
rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-30))
records=[]
for group in range(4):
 ix=slice(group*12,(group+1)*12);records.append(dict(group=group,embedding_generator_error=rel(pred[ix],y[ix]),correct_difference_signs=int((pred[ix]*y[ix]>0).sum()),pairs=12,native_mean_difference=float(y[ix].mean()),embedding_mean_difference=float(pred[ix].mean()),oracle_group_error=rel(oracle[ix],y[ix])))
res=dict(records=records,embedding_total_error=rel(pred,y),oracle_best_token_pair_constant_error=rel(oracle,y),oracle_fresh_error=rel(oracle[12:],y[12:]),scope='Fixed phi4 applied directly to RMS token embedding, no fitted weights. Oracle pairwise mean is an in-sample lower bound for any context-independent paired-token contrast; not a deployed learned circuit or held-out prediction. RMS-normalized city-state generation remains the target; downstream rho9/routing also remain context-dependent.')
(P/'PHI4_CITY_TOKEN_GENERATOR_V1_RESULT.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
