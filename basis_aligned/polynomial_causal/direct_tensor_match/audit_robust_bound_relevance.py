"""Compare global source-error bounds with actual fitting-state component errors."""
import json,torch
from pathlib import Path
from pairwise_reader_graph import source_reads
from pairwise_component_interface import component_scalars
from quadratic_ball_extrema import extrema
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);ids=d['indices'];z,h=d['z'][ids],d['h'][ids];s=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt();pair=d['pairs'][2];q=torch.einsum('ni,oij,nj->no',z,torch.stack(pair['Qs']),z);a=(h@pair['a']-.5*q[:,0])/s-pair['alpha'];b=q[:,1]/s-pair['beta'];truth=a*b
 old=torch.load(P/'FRONTIER_FRESH_GRAPH_V1.pt',weights_only=True);new=torch.load(P/'ROBUST_DIRECTION_DIRECTED_PROGRAM_V1.pt',weights_only=True);worst=json.loads((P/'NATIVE_WORST_READ_V1.json').read_text());epsa=next(r['worst_absolute_error'] for r in worst['rows'] if r['candidate']=='graph' and r['component']==3 and r['read']=='a');epsb_old=next(r['worst_absolute_error'] for r in worst['rows'] if r['candidate']=='graph' and r['component']==3 and r['read']=='b');epsb_new=json.loads((P/'ROBUST_DIRECTION_EXPORT_AUDIT_V1.json').read_text())['worst_read_error'];rows=[]
 for name,program,epsb in [('old',old,epsb_old),('new',new,epsb_new)]:
  errors=source_reads(z,program)[:,4:6]-q;actual=component_scalars(z,h,program)[:,2]-truth;global_bound=.5*b.abs()*epsa/s+a.abs()*epsb/s+.5*epsa*epsb/s.square();local_bound=.5*b.abs()*errors[:,0].abs()/s+a.abs()*errors[:,1].abs()/s+.5*errors[:,0].abs()*errors[:,1].abs()/s.square();assert torch.all(actual.abs()<=local_bound+1e-6);assert torch.all(local_bound<=global_bound+1e-6)
  rows.append(dict(candidate=name,global_bound_norm_over_actual=float(global_bound.norm()/actual.norm()),pointwise_read_triangle_bound_norm_over_actual=float(local_bound.norm()/actual.norm()),bound_norm_over_target_variation=float(global_bound.norm()/(truth-truth.mean()).norm()),actual_component_error=float(actual.norm()/(truth-truth.mean()).norm())))
 v=new['pairs']['2']['private_reader'][:,-1];weight=new['pairs']['2']['product_weights'][-1,1];mu=v@d['mu'];variance=v@d['old_covariance']@v;psi=((z-d['mu'])@v).square()-variance;ext=extrema(v[:,None]*v[None],-2*mu*v,mu.square()-variance,1152**.5);delta=source_reads(z,new)[:,5]-source_reads(z,old)[:,5];added=weight*psi;other=delta-added
 out=dict(rows=rows,new_feature=dict(calibration_variance=float(variance),empirical_projection_variance=float((z@v).var(unbiased=False)),isotropic_unit_direction_variance=float(v.square().sum()),calibration_feature_rms_over_ball_max=float(psi.square().mean().sqrt()/ext['worst_absolute']),new_weight=float(weight),added_feature_rms=float(added.square().mean().sqrt()),other_edit_rms=float(other.square().mean().sqrt()),total_edit_rms=float(delta.square().mean().sqrt()),added_other_uncentered_cosine=float(added@other/(added.norm()*other.norm()))),scope='Original448states only. Bounds are fixed-native-context component bounds, not totalmodel/OOD bounds. Covariance and feature exposure describe relevance, not semantic identity or lack of reachable adversarial examples.')
 (P/'ROBUST_BOUND_RELEVANCE_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
