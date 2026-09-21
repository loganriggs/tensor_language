"""Frozen programs, changed historical source pairings, exact centered response metric.
Predictions: replay fit<1e-5; learned error>1.5fit on>=3/4unfitted offsets;
learned beats finite_fixed on>=3/4. No fitting; no independent-text claim.
"""
import json,time
from pathlib import Path
import torch

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();p=Path(__file__).resolve().parent
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,mmap=True,map_location='cpu')
 def w(k):return state[k].double()
 data=torch.load(p/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);h,z=data['h'].double(),data['z'].double();source=w('transformer.h.17.lambdas')[0]*((z@w('transformer.h.16.mlp.Left.weight').T)*(z@w('transformer.h.16.mlp.Right.weight').T))@w('transformer.h.16.mlp.Down.weight').T
 norm=lambda x:x/(x.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
 x0=norm(h);mu=x0.mean(0);L,R,D=[w('transformer.h.17.mlp.'+k+'.weight') for k in ['Left','Right','Down']]
 RU=torch.load(p/'MIDPOINT_CONTINUATION_NATIVE_OPERATOR_V1.pt',weights_only=True)['R_U'].double();teacher=lambda x:((x@L.T)*(x@R.T))@D.T
 names=['FINITE_FIXED','FINITE_LEARNED','MIXED'];programs={name:{k:v.double() for k,v in torch.load(p/f'FULL_QUADRATIC_{name}_PROGRAM_V1.pt',weights_only=True).items()} for name in names}
 def student(x,p):return ((x@p['a'].T)*(x@p['b'].T))@p['writer'].T+x@p['linear'].T+p['bias']
 t0=teacher(x0);ct0=teacher(x0-mu);s0={name:student(x0,pp) for name,pp in programs.items()};rows=[]
 for shift in [257,64,509,1021,1537]:
  x1=norm(h+source.roll(shift,0)-source);t1=teacher(x1);den=((teacher(x1-mu)-ct0)@RU.T).norm();errors={name:float((((student(x1,pp)-s0[name])-(t1-t0))@RU.T).norm()/den) for name,pp in programs.items()};rows.append(dict(donor_shift=shift,errors=errors));print(json.dumps(rows[-1]),flush=True)
 fit=rows[0]['errors']['FINITE_LEARNED'];held=rows[1:];pred=dict(pred_a_replay=abs(fit-.032640071575735626)<1e-5,pred_b_pair_overfit=sum(r['errors']['FINITE_LEARNED']>1.5*fit for r in held)>=3,pred_c_learning=sum(r['errors']['FINITE_LEARNED']<r['errors']['FINITE_FIXED'] for r in held)>=3)
 result=dict(predictions=pred,records=rows,seconds=time.monotonic()-start,scope='Same historical2048states, unfitted donor recombinations only; not independent documents, OOD or semantically matched donors. Direct exported affine-corrected differences; centered teacher quadratic denominator and QR output metric. No new fitting.')
 (p/'HISTORICAL_PAIR_RECOMBINATION_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(predictions=pred,seconds=result['seconds'])))
if __name__=='__main__':main()
