"""Delete correction nodes; select global pairing by synthetic Gaussian fit only."""
import hashlib,json,time
from pathlib import Path
import torch
from lean_conditional_cp import compile_program,evaluate,price,MATCHINGS,PAIR_INDEX
from conditional_quartic_cp import evaluate as full_evaluate
P=Path(__file__).resolve().parent;SCALE=19054614563.464127

def cast(p,dtype):return {k:[a.to(dtype) for a in v] if isinstance(v,list) else v.to(dtype) if isinstance(v,torch.Tensor) else v for k,v in p.items()}

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();torch.manual_seed(17000)
 toy=dict(input_projection=torch.randn(5,7,dtype=torch.float64),factors=[torch.randn(11,5,dtype=torch.float64) for _ in range(4)],biases=[torch.randn(11,dtype=torch.float64) for _ in range(4)],pair_weights=torch.randn(6,11,dtype=torch.float64),coefficients=torch.randn(3,11,dtype=torch.float64),constant=torch.randn(3,dtype=torch.float64));tx=torch.randn(9,7,dtype=torch.float64);checks=[]
 for m,pairs in enumerate(MATCHINGS):
  lean=compile_program(toy,m);masked=dict(toy);mask=torch.zeros_like(toy['pair_weights']);idx=[PAIR_INDEX[p] for p in pairs];mask[idx]=toy['pair_weights'][idx];masked['pair_weights']=mask
  a=evaluate(lean,tx);b=full_evaluate(masked,tx);err=float((a-b).norm()/b.norm());assert err<1e-12;checks.append(err)
 cache=torch.load(P/'GAUSSIAN_CP_DATA_PROJECTIONS_V1.pt',weights_only=True);S=cache['projections']['covariance']['whitener'].double();mu=cache['mean'].double()
 synthetic=[torch.randn(4096,1152,generator=torch.Generator().manual_seed(seed),dtype=torch.float64)@S.T+mu for seed in [17020,17021]]
 data=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True);x=data['rows'].double();y=data['target'].double()/SCALE
 old=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'].double();labels=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'][1];target=labels['target'].double()/SCALE;weights=labels['weight'].double()
 matched=json.loads((P/'ROOT_MATCHED_READER_V1.json').read_text())['rows'][1];rec,don=torch.tensor(matched['pairs_flat']).T;reference=torch.tensor([r['reference'] for r in matched['pair_rows']],dtype=torch.float64)/SCALE;rows=[]
 for seed in [1001,1002]:
  parent=cast(torch.load(P/f'MIXED_CP_FEATURES_SEED{seed}_V1.pt',weights_only=True),torch.float64)
  def parent_value(xx):return torch.stack([xx@f.T for f in parent['factors']]).prod(0)@(parent['coefficients']/SCALE).T
  refs=[parent_value(xx) for xx in synthetic]
  for rank in [64,128,256]:
   full=cast(torch.load(P/f'CONDITIONAL_CP_SEED{seed}_RANK{rank}_V1.pt',weights_only=True),torch.float64);full['coefficients']/=SCALE;full['constant']/=SCALE
   choices=[]
   for m in range(3):
    p=compile_program(full,m);pg=evaluate(p,synthetic[0]);err=float((pg-refs[0]).norm()/refs[0].norm());choices.append((err,m,p))
   train,m,p=min(choices,key=lambda a:a[0]);held=evaluate(p,synthetic[1]);fp=full_evaluate(full,synthetic[1]);pred=torch.cat([evaluate(p,xx) for xx in x.split(2048)]);op=evaluate(p,old)
   archive=cast(p,torch.float32);archive['coefficients']*=SCALE;archive['constant']*=SCALE;export=torch.cat([evaluate(archive,xx.float()).double()/SCALE for xx in x.split(2048)]);drift=float((export-pred).norm()/pred.norm());assert drift<1e-4
   path=P/f'LEAN_CONDITIONAL_CP_SEED{seed}_RANK{rank}_V1.pt';torch.save(archive,path)
   r=dict(seed=seed,input_rank=rank,matching=m,pairing_fit_errors=[a[0] for a in choices],fit_parent_error=train,held_parent_error=float((held-refs[1]).norm()/refs[1].norm()),full_conditional_held_parent_error=float((fp-refs[1]).norm()/refs[1].norm()),opened_value_error=float((pred-y).norm()/y.norm()),feature_errors=((pred-y).square().sum(0)/y.square().sum(0)).sqrt().tolist(),root1_response_error=float(((op[don,1]-op[rec,1])-reference).norm()/reference.norm()),root1_sensitivity_error=float(((weights[:,1]*(op[:,1]-target[:,1]).square()).sum()/(weights[:,1]*target[:,1].square()).sum()).sqrt()),cost=price(p),export_error=drift,artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
   rows.append(r);print(json.dumps(r),flush=True)
 (P/'LEAN_CONDITIONAL_CP_V1.json').write_text(json.dumps(dict(rows=rows,masked_full_replays=checks,seconds=time.monotonic()-start,scope='Two parents×threeinputranks. Threeglobalpairings selected ONLYby4096artificialGaussianprobes; independent4096syntheticheldprobes. Openednativepanelsdiagnostic, no freshvalidation/circuitadoption. Quadraticcorrectionnodesdeletedexcepttwoalreadyusedbyquartic; constantsretained;1536variableproducts.'),indent=2)+'\n')
if __name__=='__main__':main()
