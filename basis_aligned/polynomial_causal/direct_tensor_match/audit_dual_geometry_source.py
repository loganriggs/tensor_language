"""Independent native decode and both-metric audit for the two-width fit."""
from pathlib import Path
import json,torch
from compact_source_graph import source_reads,component_scalars
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
def main():
 meta=json.loads((P/'DUAL_GEOMETRY_SOURCE_V1.json').read_text());programs=torch.load(P/'DUAL_GEOMETRY_SOURCE_PROGRAMS_V1.pt',weights_only=True);d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True)
 Q=torch.stack([q for p in d['pairs'] for q in p['Qs']]);energy=Q.square().sum((-1,-2)).reshape(3,2).sum(1);S=torch.linalg.inv(d['inverse_root']);covQ=torch.einsum('ai,oij,jb->oab',S,Q,S);covenergy=covQ.square().sum((-1,-2)).reshape(3,2).sum(1)
 ids=d['indices'];z=d['z'][ids];h=d['h'][ids];truth=torch.stack([p['truth'][ids] for p in d['pairs']],1);rows=[]
 for rec in meta['records']:
  p=programs[rec['key']];raw=torch.einsum('ir,ro,jr->oij',p['left_reader'],p['product_weights'],p['right_reader']);hat=(raw+raw.transpose(-1,-2))/2;hat[5]+=(p['square_reader']*p['square_weights'])@p['square_reader'].T
  E=hat-Q;native=float((E.square().sum((-1,-2)).reshape(3,2).sum(1)/energy).mean().sqrt());covE=torch.einsum('ai,oij,jb->oab',S,E,S);cov=float((covE.square().sum((-1,-2)).reshape(3,2).sum(1)/covenergy).mean().sqrt())
  dense=torch.einsum('ni,oij,nj->no',z,hat,z)+z@p['source_linear']+p['source_bias'];execution=float((dense-source_reads(z,p)).norm()/dense.norm());values=((component_scalars(z,h,p)-truth).norm(dim=0)/(truth-truth.mean(0)).norm(dim=0)).tolist()
  replay=max(execution,abs(native-rec['native_isotropic_equal_pair_error']),abs(cov-rec['calibration_shaped_error']),max(abs(a-b) for a,b in zip(values,rec['per_mode_errors'])))
  price=sum(v.numel() for v in p.values() if v.is_floating_point());products=p['left_reader'].shape[1]+p['square_reader'].shape[1]
  assert replay<1e-8 and price==rec['stored_floats']==2310*rec['width']+48428 and products==rec['source_products']
  rows.append(dict(key=rec['key'],width=rec['width'],alpha=rec['alpha'],maximum_replay_error=replay,native_isotropic_equal_pair_error=native,calibration_shaped_error=cov,per_mode_errors=values,source_products=products,stored_floats=price))
 selected={k:next(r for r in rows if r['key']==v) for k,v in meta['winners'].items()};primary=selected['560_0.5'];control=selected['560_0']
 pred=dict(pred_a_instrument=max(r['maximum_replay_error'] for r in rows)<1e-8,pred_b_values=all(x<=.15 and x<=1.1*y for x,y in zip(primary['per_mode_errors'],meta['plan']['scalar_baseline'])),pred_c_native_gain=primary['native_isotropic_equal_pair_error']<=.8*control['native_isotropic_equal_pair_error'],pred_d_covariance=primary['calibration_shaped_error']<=1.1*control['calibration_shaped_error']);assert pred==meta['predictions']
 out=dict(rows=rows,predictions=pred,scope='Independent dense native-form decode, both coefficient metrics, compact execution and physical price. Opened448notfresh; widergraph notcostmatched to old512baseline. No finalnorm/softcap/OOD/semantic adoption claimed.')
 (P/'DUAL_GEOMETRY_SOURCE_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
