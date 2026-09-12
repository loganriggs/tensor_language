"""Exact computational source partitions; no fitted attribution or semantic promotion."""
from pathlib import Path
import json,torch,tiktoken
P=Path(__file__).resolve().parent

def main():
 old=json.loads((P/'SCALAR_NEW_ENDPOINTS_V1_ROWS.json').read_text())['rows'][:24]
 fresh=json.loads((P/'MLP8_VALUE_FRESH_V1_ROWS.json').read_text())['rows']
 rows=old+[dict(r,donor_id=r['donor_id']+24) for r in fresh]
 a=torch.load(P/'PHI4_PROVENANCE_V1_ARTIFACT.pt',weights_only=True)
 assert len(rows)==96
 enc=tiktoken.get_encoding('gpt2');mode=a['modes'];contrib=a['final_source_mode_contributions'];cue=a['cue_positions'];reg=a['regional'];checks=[];prechecks=[];records=[];row_receipts=[]
 for group in range(4):
  ix=list(range(24*group,24*(group+1)));parts=[];totals=[];modeparts=[];rawcuemodes=[];logits=[]
  for i in ix:
   r=rows[i];n=len(r['ids']);pos=int(cue[i]);j=r['donor_id'];sgn=-1 if i%2==0 else 1
   part=contrib[i,:n].sum(-1)*sgn;total=a['donor_scalar'][i,n-1]*sgn
   checks.append(float(abs(part.sum()-total)/abs(total).clamp_min(1e-30)))
   prechecks.append(float((mode[j,:pos]-mode[i,:pos]).abs().max()) if pos else 0.)
   parts.append(torch.stack([part[:pos].sum(),part[pos],part[pos+1:].sum()]))
   totals.append(total);modeparts.append(contrib[i,:n].sum(0)*sgn)
   rawcuemodes.append((mode[j,pos]-mode[i,pos])*sgn)
   logits.append((reg[i,1,0]-reg[i,0,0])*sgn)
   top=part.abs().topk(min(4,n)).indices.tolist()
   row_receipts.append(dict(row=i,group=group,cue_position=pos,top_source_positions=[dict(position=k,token=enc.decode([r['ids'][k]]),directed_contribution=float(part[k])) for k in top],directed_final_write=float(total),directed_logit=float(logits[-1])))
  parts=torch.stack(parts);totals=torch.stack(totals);modeparts=torch.stack(modeparts);rawcuemodes=torch.stack(rawcuemodes);logits=torch.stack(logits)
  den=totals.square().sum().clamp_min(1e-30)
  records.append(dict(group=group,source_names=['before_city','city','after_city'],source_aligned_fractions=(parts.T@totals/den).tolist(),mode_aligned_fractions=(modeparts.T@totals/den).tolist(),mean_directed_city_mode_differences=rawcuemodes.mean(0).tolist(),mean_directed_city_value_difference=float(rawcuemodes.sum(-1).mean()),mean_directed_final_write=float(totals.mean()),positive_final_writes=int((totals>0).sum()),mean_directed_logit=float(logits.mean()),positive_logits=int((logits>0).sum()),write_logit_cosine=float(torch.nn.functional.cosine_similarity(totals,logits,dim=0))))
 assert max(checks)<1e-9
 assert max(prechecks)<1e-10
 result=dict(max_source_sum_error=max(checks),max_pre_city_mode_difference=max(prechecks),groups=records,rows=row_receipts,scope='Exact donor-write computational allocation on reused panels. Aligned fractions can be negative or exceed one. Final-position write alone doesnot specify downstream effect; all-position field remains external interface. No new fit or semantic mode identification.')
 (P/'PHI4_PROVENANCE_ANALYSIS_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
