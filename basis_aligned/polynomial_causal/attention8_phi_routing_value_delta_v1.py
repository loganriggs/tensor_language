"""Exact five-term head8.2 routing/value change, before physical port swaps."""
from pathlib import Path
import json,torch
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2)
 a=torch.load(P/'ATTENTION8_PHI_VALUE_ROUTING_V1_ARTIFACT.pt',weights_only=True);par=torch.load(P/'MLP7_PHI_PARENTS_V1_ARTIFACT.pt',weights_only=True);prov=torch.load(P/'PHI4_PROVENANCE_V1_ARTIFACT.pt',weights_only=True);gen=torch.load(P/'SCALAR_VALUE_GENERATOR_MLP8_V1_PROGRAM.pt',weights_only=True);ref=torch.load(P/'ATTENTION8_PHI_HEADS_V1_ARTIFACT.pt',weights_only=True)['head_donor_fields'][:,:,2];eig=gen['eigenvalues'][:4];gain=float(gen['lambdas'][0]);old=json.loads((P/'SCALAR_NEW_ENDPOINTS_V1_ROWS.json').read_text())['rows'][:24];fresh=json.loads((P/'MLP8_VALUE_FRESH_V1_ROWS.json').read_text())['rows'];rows=old+[dict(r,donor_id=r['donor_id']+24) for r in fresh];fields=torch.zeros(96,ref.shape[1],5,dtype=torch.float64)
 for i,row in enumerate(rows):
  n=len(row['ids']);j=row['donor_id'];g=a['routing'][i,:n,:n];dg=a['routing'][j,:n,:n]-g;c=a['source_values'][i,:n,0];f=a['source_values'][i,:n,1];dc=a['source_values'][j,:n,0]-c;df=a['source_values'][j,:n,1]-f
  terms=torch.stack([dg@(c+f),g@dc,g@df,dg@dc,dg@df],1);q=par['parents'][i,:n,1];v=2*(terms*q[:,None,:]*eig).sum(-1)/par['rho8_squared'][i,:n,None];v[:int(prov['cue_positions'][i])+1]=0;fields[i,:n]=prov['gamma'][i,:n,:n]@(v*gain/prov['rho9'][i,:n,None])
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));groups=[]
 for k in range(4):
  z=fields[k*24:(k+1)*24];r=ref[k*24:(k+1)*24];norm=r.square().sum();groups.append(dict(group=k,aligned_fractions=[float((z[:,:,i]*r).sum()/norm) for i in range(5)],routing_only_error=rel(z[:,:,0],r),current_only_error=rel(z[:,:,1],r),first_only_error=rel(z[:,:,2],r),both_values_error=rel(z[:,:,1]+z[:,:,2],r),without_mixed_error=rel(z[:,:,:3].sum(-1),r)))
 result=dict(replay_error=rel(fields.sum(-1),ref),terms=['routing_only','current_value_only','first_value_only','routing_current_mixed','routing_first_mixed'],groups=groups,scope='Exact all-position conditional donorwrite allocation with recipient Q7/norm/head9 routing. Aligned fractions can be negative; not native causal effects. No data fitting.')
 assert result['replay_error']<1e-4
 torch.save(dict(fields=fields),P/'ATTENTION8_PHI_ROUTING_VALUE_DELTA_V1_ARTIFACT.pt');(P/'ATTENTION8_PHI_ROUTING_VALUE_DELTA_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
