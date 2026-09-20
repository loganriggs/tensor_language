import json
from pathlib import Path
import torch
from core import packed_quadratic,metric
from matrix_free_quadratic import input_operator,input_rhs,conjugate_gradient
P=Path(__file__).resolve().parent;torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(1626);d,o,k,t=6,3,4,5
C=torch.randn(o,k);B=torch.randn(k,d);A=torch.randn(k,d);tc=torch.randn(o,t);ta=torch.randn(t,d);tb=torch.randn(t,d);target=tc@packed_quadratic(ta,tb);chol=torch.linalg.cholesky(metric(d,2,'frobenius'));basis=torch.eye(k*d).reshape(k*d,k,d);design=(torch.einsum('ok,pkm->pom',C,packed_quadratic(basis,B))@chol).reshape(k*d,-1).T;normal=design.T@design;rhs_dense=design.T@(target@chol).flatten();apply,diag=input_operator(C,B);rhs=input_rhs(C,B,tc,ta,tb)
operator=float((apply(A).flatten()-normal@A.flatten()).norm()/(normal@A.flatten()).norm());right=float((rhs.flatten()-rhs_dense).norm()/rhs_dense.norm());diagonal=float((diag.flatten()-normal.diag()).norm()/normal.diag().norm());assert max(operator,right,diagonal)<1e-12
ridge=1e-4*float(diag.mean());b=rhs+ridge*A;h=lambda v:apply(v)+ridge*v;cg,receipt=conjugate_gradient(h,b,A,diag+ridge);dense=torch.linalg.solve(normal+ridge*torch.eye(k*d),rhs_dense+ridge*A.flatten()).reshape(k,d);solution=float((cg-dense).norm()/dense.norm());assert solution<1e-8
out=dict(operator_relative_error=operator,rhs_relative_error=right,diagonal_relative_error=diagonal,cg_solution_relative_error=solution,cg=receipt,scope='Exact implicit symmetric quadratic Frobenius normal equations, tested against explicit weighted canonical design. Proximal ridge preserves current iterate as feasible reference; no global convergence guarantee.')
(P/'MATRIX_FREE_QUADRATIC_CHECK_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
