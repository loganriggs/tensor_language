import json,torch
from pathlib import Path
from joint_quadratic_fit_v1 import product_cross
from sparse_orthogonal_quadratic_core_v1 import orthogonal_core
from sparse_core_stiefel_v1 import project_tangent,retract,selected_score,best_edges,score_gradient,armijo_step

torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(514)
l,r,d=torch.randn(9,8),torch.randn(9,8),torch.randn(5,9);total=((d.T@d)*product_cross(l,r,l,r)).sum();q,_=torch.linalg.qr(torch.randn(8,4));edges,old=best_edges(q,l,r,d,total,5);score,g=score_gradient(q,l,r,d,edges,total)
tangent=project_tangent(q,torch.randn_like(q));eps=1e-6
finite=(selected_score(retract(q,tangent,eps),l,r,d,edges,total)-selected_score(retract(q,tangent,-eps),l,r,d,edges,total))/(2*eps)
analytic=(g*tangent).sum();nextq,step=armijo_step(q,g,g,l,r,d,edges,total);_,actual=best_edges(nextq,l,r,d,total,5)
a=torch.randn(4,4);rotation=torch.matrix_exp(.2*(a-a.T));rot=q@rotation
w,_=orthogonal_core(l,r,d,q.T);rw,_=orthogonal_core(l,r,d,rot.T);_,rotated_sparse=best_edges(rot,l,r,d,total,5)
result=dict(gradient_relative_error=float(abs(finite-analytic)/abs(analytic)),tangent_constraint_error=float((q.T@g+g.T@q).abs().max()),retraction_orthogonality_error=float((nextq.T@nextq-torch.eye(4)).abs().max()),accepted=step['accepted'],old_score=float(old),new_score=float(actual),fixed_support_lower_bound=step['new_score_lower_bound'],full_core_rotation_error=float(abs(w.square().sum()-rw.square().sum())/total),sparse_core_rotation_change=float(abs(rotated_sparse-old)))
result['passed']=result['gradient_relative_error']<=1e-7 and max(result[k] for k in ['tangent_constraint_error','retraction_orthogonality_error','full_core_rotation_error'])<=1e-10 and result['accepted'] and result['new_score']>=result['fixed_support_lower_bound']-1e-12 and result['new_score']>result['old_score'] and result['sparse_core_rotation_change']>1e-6
assert result['passed'],result
Path(__file__).with_name('SPARSE_CORE_STIEFEL_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
