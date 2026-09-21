"""Fixed learned-product least-squares control, after managed fit is terminal.
Primary is the already coefficient-selected fit, not chosen on native outcomes.
"""
from pathlib import Path
import json,torch,time
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.perf_counter()
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);fit=json.loads((P/'SHARED_PRODUCT_NATIVE_FIT_V1.json').read_text());oldprograms=torch.load(P/'SHARED_PRODUCT_NATIVE_PROGRAMS_V1.pt',weights_only=True);root=torch.linalg.inv(d['inverse_root']);T=d['teacher'];records=[];programs={}
base=json.loads((P/'MULTIMODE_PROJECTION_SHARING_V1.json').read_text())
for record in fit['records']:
 old=oldprograms[record['key']];V=root@old['shared_reader'];norm=V.norm(dim=0);V=V/norm;reader=old['shared_reader']/norm
 gram=(V.T@V).square();rhs=torch.einsum('ir,oij,jr->ro',V,T,V)
 ev,E=torch.linalg.eigh(gram);keep=ev>ev[-1]*1e-12;coeff=E[:,keep]@((E[:,keep].T@rhs)/ev[keep,None])
 normal=float((gram@coeff-rhs).norm()/rhs.norm());assert normal<1e-8
 reconstructed=torch.einsum('ro,ir,jr->oij',coeff,V,V);coef=float((reconstructed-T).norm()/T.norm())
 assert coef<=record['coefficient_error']+1e-8
 readout=coeff*d['scales'][None,:];Qhat=torch.einsum('ro,ir,jr->oij',readout,reader,reader)
 p={k:v for k,v in old.items() if k not in ['shared_reader','product_weights','source_linear','source_bias']};p['shared_reader']=reader;p['product_weights']=readout
 linears=[];biases=[]
 for j,pair in enumerate(d['pairs']):
  for k,Q in enumerate(pair['Qs']):
   pos=2*j+k;linear=2*Q@d['mu'];constant=d['mu']@Q@d['mu']+torch.trace(d['old_covariance']@Q)
   linears.append(linear-2*Qhat[pos]@d['mu']);biases.append(constant-d['mu']@linear+d['mu']@Qhat[pos]@d['mu']-torch.trace(d['old_covariance']@Qhat[pos]))
 p['source_linear']=torch.stack(linears,1);p['source_bias']=torch.stack(biases)
 reads=(d['z']@reader).square()@readout+d['z']@p['source_linear']+p['source_bias']
 dense=torch.einsum('ni,oij,nj->no',d['z'],Qhat,d['z'])+d['z']@p['source_linear']+p['source_bias'];replay=float((reads-dense).norm()/dense.norm());assert replay<1e-8
 phi=((d['h']@p['h_readers']-.5*reads[:,::2])/d['scale'][:,None]-p['alpha'])*(reads[:,1::2]/d['scale'][:,None]-p['beta'])
 errors=[]
 for j,pair in enumerate(d['pairs']):
  truth=pair['truth'][d['indices']];errors.append(float((phi[d['indices'],j]-truth).norm()/(truth-truth.mean()).norm()))
 ratios=[a/b for a,b in zip(errors,base['baseline_errors'])]
 records.append(dict(key=record['key'],coefficient_error_before=record['coefficient_error'],coefficient_error_after=coef,per_mode_errors_before=record['per_mode_errors'],per_mode_errors_after=errors,ratios_to_baseline=ratios,fidelity_pass=max(errors)<=.15 and max(ratios)<=1.1,discarded_gram_directions=int((~keep).sum()),gram_condition=float(ev[-1]/ev[keep][0]),normal_equation_error=normal,dense_program_replay=replay,stored_floats=sum(v.numel() for v in p.values()),source_products=reader.shape[1]))
 programs[record['key']]=p;print(records[-1],flush=True)
primary=next(r for r in records if r['key']==fit['winner'])
out=dict(records=records,primary=fit['winner'],predictions=dict(pred_a_least_squares=all(r['normal_equation_error']<1e-8 and r['dense_program_replay']<1e-8 for r in records),pred_b_fidelity=primary['fidelity_pass'],pred_c_same_cost=primary['source_products']==512),seconds=time.perf_counter()-start,scope='Exact fixed-product readout refit at numerical rank cutoff1e-12, preserving centered affine target. All same opened rows; primary unchanged from prior coefficient selection.')
(P/'SHARED_PRODUCT_READOUT_REFIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');torch.save(programs,P/'SHARED_PRODUCT_READOUT_PROGRAMS_V1.pt');print(out['predictions'])
