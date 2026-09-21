"""Independent dense/original-coordinate audit of shared-product exports.
Run only after SHARED_PRODUCT_NATIVE_FIT_V1.json and programs are terminal.
"""
from pathlib import Path
import torch,json,argparse
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
parser=argparse.ArgumentParser();parser.add_argument('--family',choices=['square','mixed'],default='square');args=parser.parse_args();prefix='SHARED' if args.family=='square' else 'MIXED'
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True)
result=json.loads((P/f'{prefix}_PRODUCT_NATIVE_FIT_V1.json').read_text())
programs=torch.load(P/f'{prefix}_PRODUCT_NATIVE_PROGRAMS_V1.pt',weights_only=True)
root=torch.linalg.inv(d['inverse_root']);records=[]
for record in result['records']:
 p=programs[record['key']];left=p.get('left_reader',p.get('shared_reader'));right=p.get('right_reader',left)
 raw=torch.einsum('ir,ro,jr->oij',left,p['product_weights'],right);Qhat=.5*(raw+raw.transpose(-1,-2))
 transformed=torch.stack([root@Q@root for Q in Qhat])/d['scales'][:,None,None]
 coef=float((transformed-d['teacher']).norm()/d['teacher'].norm())
 # Direct quadratic matrices replace the factorized product execution.
 reads=torch.einsum('ni,oij,nj->no',d['z'],Qhat,d['z'])+d['z']@p['source_linear']+p['source_bias']
 qa,qb=reads[:,::2],reads[:,1::2]
 phi=((d['h']@p['h_readers']-.5*qa)/d['scale'][:,None]-p['alpha'])*(qb/d['scale'][:,None]-p['beta'])
 errors=[];affine=[];means=[]
 for j,pair in enumerate(d['pairs']):
  truth=pair['truth'][d['indices']];errors.append(float((phi[d['indices'],j]-truth).norm()/(truth-truth.mean()).norm()))
  for k,Q in enumerate(pair['Qs']):
   pos=2*j+k;oldlinear=2*Q@d['mu'];newlinear=p['source_linear'][:,pos]+2*Qhat[pos]@d['mu']
   affine.append(float((oldlinear-newlinear).norm()/oldlinear.norm()))
   oldmean=d['mu']@Q@d['mu']+torch.trace(d['old_covariance']@Q)
   newmean=p['source_bias'][pos]+d['mu']@p['source_linear'][:,pos]+d['mu']@Qhat[pos]@d['mu']+torch.trace(d['old_covariance']@Qhat[pos])
   means.append(float(abs(oldmean-newmean)/(abs(oldmean)+1e-30)))
 scale=(d['h'].square().mean(-1)+torch.finfo(torch.float32).eps).sqrt()
 record_audit=dict(key=record['key'],dense_coefficient_error=coef,coefficient_error_replay=abs(coef-record['coefficient_error']),per_mode_errors=errors,scalar_error_replay=max(abs(a-b) for a,b in zip(errors,record['per_mode_errors'])),centered_linear_replay=max(affine),centered_mean_replay=max(means),stored_scale_recomputed_relative_error=float((scale-d['scale']).norm()/d['scale'].norm()))
 records.append(record_audit)
 assert max(record_audit[k] for k in ['coefficient_error_replay','scalar_error_replay','centered_linear_replay','centered_mean_replay'])<1e-8
out=dict(records=records,all_export_checks_pass=True,scope='Independent dense quadratic evaluation and original-coordinate metric replay. Same opened inputs; not new intervention or OOD evidence.')
(P/f'{prefix}_PRODUCT_NATIVE_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
