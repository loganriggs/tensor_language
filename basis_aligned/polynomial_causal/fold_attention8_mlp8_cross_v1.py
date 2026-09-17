"""Native ordered attention-head/write crosses; fold attribution, not edits.
Pred_a headsum/crosssum relative closure<=1e-5. Pred_b head8.2-only cross error
<=.35 in every construction family. No omission is installed in the model.
"""
from pathlib import Path
import torch,json,sys
from attention8_context_head_terms_v1 import heads
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);sys.path.insert(0,str(P/'extracted_circuits/typed_face_mlp8_coupled_v1'));import head8
 local=torch.load(P/'extracted_circuits/typed_face_mlp8_coupled_v1/program.pt',weights_only=True);context=torch.load(P/'ATTENTION8_CONTEXT_GENERATOR_V1_PROGRAM.pt',weights_only=True)
 fixtures=torch.load(P/'TYPED_FACE_MLP8_COUPLED_V1_ARTIFACT.pt',weights_only=True)['fixtures'];sources=torch.load(P/'MLP8_CONTEXT_SOURCES_V1_ARTIFACT.pt',weights_only=True)['sources'];sys.path.insert(0,str(P.parent/'bilinear_quotient/ops'));from regional_endpoint_batching_v1 import group_rows
 groups,_=group_rows(json.loads((P/'TYPED_FACE_NATIVE8_FRESH_V1_ROWS.json').read_text())['rows']);L,R,D=[local['mlp8'][x].double() for x in ['left','right','down']];stats={};head_errors=[];cross_errors=[];family={}
 for row,f,source in zip(groups,fixtures,sources):
  x=f['inputs'];a=heads(context,x['current8'],torch.tensor([row['ids']])).double();head_errors.append(float((a.sum(0)-source[2].double()).norm()/source[2].norm()))
  delta=.5*head8.execute(local['head8'],x['current8'],x['donor_city8'],x['recipient_token'],x['donor_token'],x['city'],x['destination']).double();g=x['post_attention8'].double();s1=(g+delta).square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps;ld=delta@L.T;rd=delta@R.T
  lr=((ld[None]*(a@R.T))@D.T)/s1;rl=(((a@L.T)*rd[None])@D.T)/s1;terms=lr+rl
  native_a=source[2].double();target=(ld*(native_a@R.T)+(native_a@L.T)*rd)@D.T/s1;cross_errors.append(float((terms.sum(0)-target).norm()/target.norm()));den=float(target.square().sum())
  fam=family.setdefault(row['variant'],{'den':0.,'err':0.});fam['den']+=den;fam['err']+=float((terms[2]-target).square().sum())
  for h in range(9):
   for label,t in [('delta*head',lr[h]),('head*delta',rl[h]),('sum',terms[h])]:
    acc=stats.setdefault(f'{h}:{label}',[0.,0.,0.]);acc[0]+=float(t.square().sum());acc[1]+=float((t*target).sum());acc[2]+=den
 errors={k:(v['err']/v['den'])**.5 for k,v in family.items()};result={'pred_a':max(head_errors+cross_errors)<=1e-5,'pred_b':max(errors.values())<=.35,'max_head_sum_error':max(head_errors),'max_cross_sum_error':max(cross_errors),'head2_only_cross_errors':errors,'terms':{k:{'norm_ratio':(v[0]/v[2])**.5,'aligned_fraction':v[1]/v[2]} for k,v in stats.items()},'scope':'Exact ordered cross attribution on40opened states. Reference is attention8/background cross contribution to local MLP8 response, not full response or spelling effect. All normalization terms and other sources remain; no causal source omission or circuit promotion.'}
 (P/'ATTENTION8_MLP8_CROSS_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
