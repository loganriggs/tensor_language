"""Frozen conditional-program residual accounting; opened panel, no fitting."""
import hashlib,json,time
from pathlib import Path
import torch
from conditional_quartic_cp import evaluate as full_eval
from lean_conditional_cp import evaluate as lean_eval
P=Path(__file__).resolve().parent
SCALE=19054614563.464127

def load(name):
 path=P/name
 obj=torch.load(path,weights_only=True)
 obj={k:[t.double() for t in v] if isinstance(v,list) else v.double() if isinstance(v,torch.Tensor) else v for k,v in obj.items()}
 return obj,hashlib.sha256(path.read_bytes()).hexdigest()

def account(y,parent,pred):
 e=parent-y;d=pred-parent;r=pred-y
 norm=y.square().sum();a=e.square().sum();b=d.square().sum();cross=2*(e*d).sum();total=r.square().sum()
 replay=float((total-a-b-cross).abs()/norm);assert replay<1e-12
 n=y.shape[0];state=r.square().sum(1);top=state.topk((n+9)//10).indices
 den=y.square().sum(0)
 docerr=(r.reshape(-1,64,16).square().sum((1,2))/y.reshape(-1,64,16).square().sum((1,2))).sqrt()
 return dict(pooled_error=float((total/norm).sqrt()),parent_error=float((a/norm).sqrt()),edit_error_over_native=float((b/norm).sqrt()),residual_edit_cosine=float((e*d).sum()/(a*b).sqrt()),squared_error_identity=dict(parent=float(a/norm),edit=float(b/norm),cross=float(cross/norm),total=float(total/norm),replay=replay),feature_errors=(r.square().sum(0)/den).sqrt().tolist(),parent_feature_errors=(e.square().sum(0)/den).sqrt().tolist(),feature_edit_errors=(d.square().sum(0)/den).sqrt().tolist(),feature_cross_terms=(2*(e*d).sum(0)/den).tolist(),features_improved=(r.square().sum(0)<e.square().sum(0)).tolist(),documents_improved=int((r.reshape(-1,64,16).square().sum((1,2))<e.reshape(-1,64,16).square().sum((1,2))).sum()),document_error_quantiles=torch.quantile(docerr,torch.tensor([.5,.9,.99,1.],dtype=docerr.dtype)).tolist(),worst_10pct_error_share=float(state[top].sum()/state.sum()),worst_10pct_target_share=float(y[top].square().sum()/norm))

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic()
 data=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True);x=data['rows'].double();y=data['target'].double()/SCALE
 assert x.shape==(16384,1152) and y.shape==(16384,16)
 rows=[]
 for seed in [1001,1002]:
  parent,ph=load(f'MIXED_CP_FEATURES_SEED{seed}_V1.pt');parent['coefficients']/=SCALE
  pv=torch.cat([torch.stack([xx@f.T for f in parent['factors']]).prod(0)@parent['coefficients'].T for xx in x.split(2048)])
  for kind,fn in [('CONDITIONAL_CP',full_eval),('LEAN_CONDITIONAL_CP',lean_eval)]:
   program,h=load(f'{kind}_SEED{seed}_RANK256_V1.pt');program['coefficients']/=SCALE;program['constant']/=SCALE
   pred=torch.cat([fn(program,xx) for xx in x.split(2048)])
   assert torch.isfinite(pred).all()
   row=dict(kind=kind,seed=seed,parent_sha256=ph,program_sha256=h,**account(y,pv,pred));rows.append(row)
   print(kind,seed,'error',row['pooled_error'],'edit',row['edit_error_over_native'],'cos',row['residual_edit_cosine'],'improved docs',row['documents_improved'],flush=True)
 result=dict(rows=rows,seconds=time.monotonic()-start,scope='Opened 256-document panel, 64 states each, selected purequartic16outputpath. No fitting or candidate selection. Negative cross term means compression compensates parent errors; it is not native algebra equality. Squared residual identity checked. Float64 evaluation of frozen float32 exports.',token_content_sha256=data.get('token_content_sha256'))
 (P/'CONDITIONAL_RESIDUAL_ACCOUNTING_V1.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
