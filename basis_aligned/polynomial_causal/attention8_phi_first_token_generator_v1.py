"""Weight-derived token generator for the first-layer value input port, not full phi8."""
from pathlib import Path
import json,torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);files=json.loads((P/'PHI4_PROVENANCE_V1_BINDING.json').read_text())['files'];sd=torch.load(next(k for k in files if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True);fold=torch.load(P/'ATTENTION8_PHI_READER_FOLD_V1_PROGRAM.pt',weights_only=True);a=torch.load(P/'ATTENTION8_PHI_VALUE_ROUTING_V1_ARTIFACT.pt',weights_only=True);prov=torch.load(P/'PHI4_PROVENANCE_V1_ARTIFACT.pt',weights_only=True);old=json.loads((P/'SCALAR_NEW_ENDPOINTS_V1_ROWS.json').read_text())['rows'][:24];fresh=json.loads((P/'MLP8_VALUE_FRESH_V1_ROWS.json').read_text())['rows'];rows=old+[dict(r,donor_id=r['donor_id']+24) for r in fresh];unique=sorted({t for row in rows for t in row['ids']});ids=torch.tensor(unique);emb=sd['transformer.wte.weight'][ids].float();x0=F.rms_norm(emb,(1152,));lam=sd['transformer.h.0.lambdas'].float();raw=lam[0]*x0+lam[1]*x0;xin=F.rms_norm(raw,(1152,));pred=xin.double()@fold['first_value_readers'][2].T;index={t:i for i,t in enumerate(unique)};replay=torch.zeros_like(a['source_values'][:,:,1]);offcity=0.;total=0.
 for i,row in enumerate(rows):
  n=len(row['ids']);j=row['donor_id'];cue=int(prov['cue_positions'][i]);replay[i,:n]=pred[[index[t] for t in row['ids']]];delta=a['source_values'][j,:n,1]-a['source_values'][i,:n,1];total+=float(delta.square().sum());delta=delta.clone();delta[cue]=0;offcity+=float(delta.square().sum())
 ref=a['source_values'][:,:,1];rel=float((replay-ref).norm()/ref.norm());result=dict(pred_a=rel<=1e-5,pred_b=offcity==0,token_generator_relative_error=rel,offcity_delta_norm_fraction=(offcity/max(total,1e-30))**.5,unique_tokens=len(unique),folded_map_shape=[4,1152],block0_lambdas=lam.tolist(),scope='Fixedweights embedding->initialRMS->block0 learnedreentry->attention0RMS->foldedsharedfirstvalue map. No fitted token table. Only first-value port is closed; embedding matrix/nativeQK/Q7/norm/head9/suffix remain external.')
 (P/'ATTENTION8_PHI_FIRST_TOKEN_GENERATOR_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
