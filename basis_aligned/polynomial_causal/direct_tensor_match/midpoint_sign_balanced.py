from pathlib import Path
from datetime import datetime,timezone
import torch,json,itertools,time,os
from scipy.optimize import linear_sum_assignment
PER_SIGN=int(os.environ.get("MIDPOINT_PER_SIGN","2"));WIDTH=2*PER_SIGN
STEM=os.environ.get("MIDPOINT_OUTPUT_STEM","MIDPOINT_SIGN_BALANCED")
p=Path('/workspace/tensor_language/basis_aligned/polynomial_causal/direct_tensor_match');torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.perf_counter();rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);n=rows['n'].double();m=rows['m'].double();old=torch.load(p/'MIDPOINT_EXTRACTED_PROGRAM_V1.pt',weights_only=True);K=torch.load(p/'MIDPOINT_FACTOR_PROGRAMS_V1.pt',weights_only=True)['K'].double();K=(K+K.transpose(1,2))/2;M=torch.load(p/'MIDPOINT_TIED_READERS_V1.pt',weights_only=True)['common_moment'];e,V=torch.linalg.eigh(M+1e-6*M.trace()/1152*torch.eye(1152));S=(V*e.sqrt()[None,:])@V.T;parts=json.loads((p/'MIDPOINT_STABILITY_V1.json').read_text())['partitions']
def factors(ids):
 nn=n[ids].flatten(0,1);mm=m[ids].flatten(0,1);a=nn.T@nn/len(nn);b=mm.T@mm/len(mm);moment=(a/a.trace()+b/b.trace())/2;e,V=torch.linalg.eigh(moment+1e-6*moment.trace()/1152*torch.eye(1152));s=(V*e.sqrt()[None,:])@V.T;inv=(V*e.rsqrt()[None,:])@V.T;result=[]
 for k in K:
  z=s@k@s;vals,U=torch.linalg.eigh((z+z.T)/2);idx=torch.cat((torch.where(vals>0)[0][-PER_SIGN:].flip(0),torch.where(vals<0)[0][:PER_SIGN]));assert len(idx)==WIDTH;v=(inv@U[:,idx])*vals[idx].abs().sqrt()[None,:];result.append((v,vals[idx].sign()))
 return result
full=factors(list(range(32)));directions=torch.cat([v for v,sign in full],dim=1);readout=torch.zeros(4*WIDTH,4,dtype=torch.float64)
for g,(_,sign) in enumerate(full):readout[WIDTH*g:WIDTH*g+WIDTH,g]=sign
nn=n.flatten(0,1);mm=m.flatten(0,1);truth=rows['y'].flatten(0,1).double()@old['scalar_readers']-old['offset'].double();prediction=((nn@directions)*(mm@directions))@readout-old['offset'].double();err=(prediction-truth).square().sum(0);den=truth.square().sum(0);calibration=dict(relative_errors=(err/den).sqrt().tolist(),aggregate_error=float((err.sum()/den.sum()).sqrt()));records=[]
for part in parts:
 a=factors(part['first_documents']);b=factors(part['second_documents']);features=[]
 for g,((va,sa),(vb,sb)) in enumerate(zip(a,b)):
  va=S@va;vb=S@vb;ua=va/va.norm(dim=0);ub=vb/vb.norm(dim=0);corr=(ua.T@ub).square()*sa[:,None]*sb[None,:];ii,jj=linear_sum_assignment(-corr.numpy());matched=corr[ii,jj].tolist();blocks=[]
  for sl in [slice(0,PER_SIGN),slice(PER_SIGN,WIDTH)]:
   x=va[:,sl];y=vb[:,sl];dot=(x.T@y).square().sum();norm=((x.T@x).square().sum()*(y.T@y).square().sum()).sqrt();blocks.append(float(dot/norm))
  features.append(dict(feature=g,matched_product_cosines=matched,minimum_product_cosine=min(matched),positive_block_cosine=blocks[0],negative_block_cosine=blocks[1]))
 records.append(dict(partition=part['partition'],features=features));print(part['partition'],[(round(f['minimum_product_cosine'],3),round(min(f['positive_block_cosine'],f['negative_block_cosine']),3)) for f in features],flush=True)
result=dict(per_sign=PER_SIGN,products=4*WIDTH,input_coefficients=1152*4*WIDTH,calibration=calibration,predictions=dict(individual_stability=all(f['minimum_product_cosine']>.9 for r in records for f in r['features']),signed_block_stability=all(min(f['positive_block_cosine'],f['negative_block_cosine'])>.95 for r in records for f in r['features']),calibration_preservation=calibration['aggregate_error']<=1.25*.07180493443762649),records=records,seconds=time.perf_counter()-start,scope=f'Fixedprior {PER_SIGN} positive/{PER_SIGN} negative terms per fixed outputfeature. Conditionalproduct andsignedblock stability. Calibrationmetric only; nativebehavior notvalidated. Comparison doesnotchange earliercutofffailure.')
out=p/f'{STEM}_V1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');torch.save(dict(program=dict(directions=directions.float(),readout=readout.float(),offset=old['offset'],scalar_readers=old['scalar_readers'],reduced_writers=old['reduced_writers'])),p/f'{STEM}_V1.pt');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
