"""Independent pair-bank baselines near both shared graph storage budgets.
Reuse the regular symmetric-pencil compiler; retain instrument failures explicitly.
"""
from pathlib import Path
import json,time,torch
from quadratic_pair_blocks import compile_pair,products
from source_interface import residual_write
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
def main():
 start=time.perf_counter();d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);S=torch.linalg.inv(d['inverse_root']);I=torch.eye(1152,dtype=S.dtype);ids=d['indices'];z=d['z'][ids];h=d['h'][ids];rms=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt();programs={};rows=[]
 for metric in ['native_isotropic','calibration_shaped']:
  transform=I if metric=='native_isotropic' else S;inverse=I if metric=='native_isotropic' else d['inverse_root']
  for width in [256,384]:
   bundle={};records=[];writer=d['residual_writer'].clone();instrument=True
   for j,pair in enumerate(d['pairs']):
    Q=pair['Qs'];T=[transform@q@transform for q in Q];_,basis=torch.linalg.eigh(sum(t@t for t in T));basis=basis[:,-width:];cores=[basis.T@t@basis for t in T];native_basis=inverse@basis;hats=[native_basis@c@native_basis.T for c in cores]
    try:compiled=compile_pair(*cores)
    except (ValueError,RuntimeError) as error:
     records.append(dict(mode=j+1,compiler_failure=str(error)));instrument=False;continue
    p=dict(shared_reader=native_basis@compiled['input_transform'],product_indices=compiled['product_indices'],product_weights=compiled['product_weights'],h_reader=pair['a'].clone(),residual_writer=writer,alpha=pair['alpha'].clone(),beta=pair['beta'].clone())
    for key,true,hat in zip(['a','b'],Q,hats):
     delta=true-hat;p[key+'_linear']=2*delta@d['mu'];p[key+'_bias']=torch.trace(d['old_covariance']@delta)-d['mu']@delta@d['mu']
    reads=products(z@p['shared_reader'],p['product_indices'])@p['product_weights']+torch.stack([z@p[k+'_linear']+p[k+'_bias'] for k in ['a','b']],1)
    direct=torch.stack([torch.einsum('ni,ij,nj->n',z,hat,z)+z@p[k+'_linear']+p[k+'_bias'] for k,hat in zip(['a','b'],hats)],1);replay=float((reads-direct).norm()/direct.norm());assert replay<1e-8
    phi=((h@pair['a']-.5*reads[:,0])/rms-pair['alpha'])*(reads[:,1]/rms-pair['beta']);truth=pair['truth'][ids];value=float((phi-truth).norm()/(truth-truth.mean()).norm())
    E=torch.stack([hat-true for hat,true in zip(hats,Q)]);native=float(E.norm()/torch.stack(Q).norm());cov=float(torch.stack([S@e@S for e in E]).norm()/torch.stack([S@q@S for q in Q]).norm())
    write=residual_write(z,h,p);assert float((write-phi[:,None]*writer).norm()/write.norm())<1e-10
    bundle[str(j)]=p;records.append(dict(mode=j+1,source_products=len(p['product_weights']),execution_replay=replay,component_error=value,native_isotropic_error=native,calibration_shaped_error=cov,compiler=compiled['diagnostics']))
   key=f'{metric}_{width}';row=dict(key=key,width=width,metric=metric,instrument_pass=instrument,components=records)
   if instrument:
    # Exactly one shared output writer; all other coefficients occupy separate arrays.
    unique={v.data_ptr():v for p in bundle.values() for v in p.values() if v.is_floating_point()};price=sum(v.numel() for v in unique.values());assert price==3462*width+11532
    row.update(source_products=sum(r['source_products'] for r in records),stored_floats=price,index_integers=sum(p['product_indices'].numel() for p in bundle.values()),per_mode_errors=[r['component_error'] for r in records],native_isotropic_equal_pair_error=(sum(r['native_isotropic_error']**2 for r in records)/3)**.5,calibration_shaped_error=(sum(r['calibration_shaped_error']**2 for r in records)/3)**.5)
    if metric=='calibration_shaped' and width==256:
     expected=[.03058409729022641,.027558449717507608,.11941807478056159];assert max(abs(a-b) for a,b in zip(row['per_mode_errors'],expected))<1e-7
    programs[key]=bundle
   rows.append(row);print(json.dumps(row),flush=True)
 out=dict(records=rows,seconds=time.perf_counter()-start,scope='Independent pair-specific mode-Gram subspaces, not optimized global minima. Symmetric-pair compiler provides exact within-subspace product sharing. One residual writer stored once. Width384 costs1340940floats vswidegraph1342028; products1152vs592. Opened448diagnostics, nativez/h stillsupplied; no fresh/adoption claim.')
 torch.save(programs,P/'COST_MATCHED_PAIR_BASELINES_V1.pt');(P/'COST_MATCHED_PAIR_BASELINES_V1.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
