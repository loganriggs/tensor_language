"""Exact context-conditioned four-reader contraction for the city value path."""
from pathlib import Path
import json,torch
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);a=torch.load(P/'ATTENTION8_PHI_VALUE_ROUTING_V1_ARTIFACT.pt',weights_only=True);par=torch.load(P/'MLP7_PHI_PARENTS_V1_ARTIFACT.pt',weights_only=True);prov=torch.load(P/'PHI4_PROVENANCE_V1_ARTIFACT.pt',weights_only=True);gen=torch.load(P/'SCALAR_VALUE_GENERATOR_MLP8_V1_PROGRAM.pt',weights_only=True);target=torch.load(P/'ATTENTION8_PHI_ROUTING_VALUE_DELTA_V1_ARTIFACT.pt',weights_only=True)['fields'][:,:,2];physical=torch.load(P/'ATTENTION8_PHI_VALUE_PORTS_V1_ARTIFACT.pt',weights_only=True)['regional'];old=json.loads((P/'SCALAR_NEW_ENDPOINTS_V1_ROWS.json').read_text())['rows'][:24];fresh=json.loads((P/'MLP8_VALUE_FRESH_V1_ROWS.json').read_text())['rows'];rows=old+[dict(r,donor_id=r['donor_id']+24) for r in fresh];readers=torch.zeros(96,target.shape[1],4,dtype=torch.float64);prediction=torch.zeros_like(target);final=torch.zeros(96,dtype=torch.float64)
 for i,row in enumerate(rows):
  n=len(row['ids']);j=row['donor_id'];c=int(prov['cue_positions'][i]);mask=torch.arange(n)>c;weightedq=2*gen['lambdas'][0]*a['routing'][i,:n,c,None]*par['parents'][i,:n,1]*gen['eigenvalues'][:4]/(par['rho8_squared'][i,:n,None]*prov['rho9'][i,:n,None]);weightedq*=mask[:,None];readers[i,:n]=prov['gamma'][i,:n,:n]@weightedq;df=a['source_values'][j,c,1]-a['source_values'][i,c,1];prediction[i,:n]=readers[i,:n]@df;final[i]=prediction[i,n-1]
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));effect=physical[:,6,0]-physical[:,0,0];direction=torch.ones(96,dtype=torch.float64);direction[::2]=-1;groups=[]
 for k in range(4):
  w=final[24*k:24*(k+1)]*direction[24*k:24*(k+1)];e=effect[24*k:24*(k+1)]*direction[24*k:24*(k+1)];corr=float(torch.corrcoef(torch.stack([w,e]))[0,1]);groups.append(dict(group=k,directed_final_write_mean=float(w.mean()),positive_final_writes=int((w>0).sum()),negative_final_writes=int((w<0).sum()),positive_effects=int((e>0).sum()),negative_effects=int((e<0).sum()),write_effect_correlation=corr))
 result=dict(replay_error=rel(prediction,target),effective_reader_shape=list(readers.shape),groups=groups,scope='Exact conditional context contraction with fixedcity source c. EffectiveT-by4readers still require native gamma8/gamma9/Q7/norms. Finalwrite comparison descriptive, not final-position-only causal sufficiency or autonomous prediction.')
 assert result['replay_error']<1e-10;torch.save(dict(effective_readers=readers),P/'FIRST_TOKEN_EFFECTIVE_CONTEXT_V1_ARTIFACT.pt');(P/'FIRST_TOKEN_EFFECTIVE_CONTEXT_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
