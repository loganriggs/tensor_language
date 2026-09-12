"""Frozen spectral blocks on pristine/changed native head9.8 cached contexts.
Checks implicit querywriter against separately reduced operator at eachposition.
A executor<=1e-8; B each target/state field relativeerror<=.1 (screen only).
"""
from pathlib import Path
import json,time,torch
import torch.nn.functional as F
from fixed_value_quadratic_operator_v1 import metric_power
from folded_normalized_router_v1 import rotary,EPS
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64);tic=time.perf_counter();out=P/'FIXED_VALUE_QUADRATIC_NATIVE_V1_FIELDS.json';assert not out.exists()
 programs=[p for p in torch.load(P/'FIXED_VALUE_QUADRATIC_SVD_V1_ARTIFACT.pt',weights_only=True) if p['seed']==17];p=torch.load(P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt',weights_only=True);q1,k1,q2,k2=[p[n][1].double() for n in ['q1','k1','q2','k2']];band=torch.load(P/'SCALAR_JOINT_KEY_BANK_V1_ARTIFACT.pt',weights_only=True)['bands'][1,0].double();cache=torch.load(P/'SCALAR_PRODUCER_MLP_BRIDGE_CACHE_V1_ARTIFACT.pt',weights_only=True)['r9'];current=F.rms_norm(cache,(1152,),eps=EPS).double();rows=json.loads((P/'STRUCTURED_PRODUCER_CONFIRMATION_V1_ROWS.json').read_text())['rows'];records=[];maxcheck=0.
 for program in programs:
  bs=program['source_basis'];bq=program['query_basis'];v=program['value'];xs=program['quadratics'];mx=metric_power(xs,v,1);mode=program['target'];keys=[(k1,k2)] if mode=='full' else [((k1@band)@band.T,(k2@band)@band.T),(k1-(k1@band)@band.T,k2-(k2@band)@band.T)];jbank=sum((a@bs)[None]@mx@(b@bs).T[None] for a,b in keys)
  for state in [0,1]:
   references=[];predictions=[]
   for i,row in enumerate(rows):
    n=len(row['ids']);s=current[state,i,:n];q=s[-1];qa=q@q1.T;qb=q@q2.T;ka=s@k1.T;kb=s@k2.T;rr=torch.stack([rotary(n-1,128).T@rotary(j,128) for j in range(n)]);ra=torch.einsum('i,nij->nj',qa,rr);rb=torch.einsum('i,nij->nj',qb,rr);gate=1/(128**2*((qa.square().mean()+EPS)*(qb.square().mean()+EPS)*(ka.square().mean(-1)+EPS)*(kb.square().mean(-1)+EPS)).sqrt());sl=s@bs;features=torch.einsum('ni,rij,nj->nr',sl,xs,sl)*(sl@v)[:,None];queryweights=torch.einsum('ni,rij,nj->nr',ra,jbank,rb);pred=(features*queryweights).sum(-1)*gate
    gamma=sum((ra*(s@a.T)).sum(-1)*(rb*(s@b.T)).sum(-1) for a,b in keys);ref=gamma*(s@p['current_value_readers'][1].double())*gate;references.append(float(ref.sum()));predictions.append(float(pred.sum()))
    # First real row: independent reduced-coordinate writer matrices for everyposition.
    if i==0:
     ql=bq.T@q;check=[]
     for rotation in rr:
      writer=sum(((q1@bq).T@rotation@(a@bs))[None]@mx@((q2@bq).T@rotation@(b@bs)).T[None] for a,b in keys);check.append(torch.einsum('i,rij,j->r',ql,writer,ql))
     check=torch.stack(check);maxcheck=max(maxcheck,float((check-queryweights).norm()/queryweights.norm()))
   ref=torch.tensor(references);pred=torch.tensor(predictions);records.append(dict(target=mode,state=state,relative_field_error=float((pred-ref).norm()/ref.norm()),cosine=float((pred@ref)/(pred.norm()*ref.norm())),reference_rms=float(ref.square().mean().sqrt()),prediction_rms=float(pred.square().mean().sqrt())))
 result=dict(pred_a=maxcheck<=1e-8,pred_b=all(r['relative_field_error']<=.1 for r in records),querywriter_relative_error=maxcheck,rows=records,seconds=time.perf_counter()-tic,scope='Frozen seed17 rank8 blocks on48pristine and48after8-removal cachedhead9states. FP64 nativeweights/norms/rotarytables, endpointcausalsum only. This is field approximation, not new LM intervention or nativeFP32operation-order equivalence. Same contexts as previous discovery panels.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
