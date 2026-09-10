"""CPU planted shared-DAG recovery and envelope-gradient control."""
from pathlib import Path
import torch,json,hashlib,time
from joint_quadratic_optimizer_v1 import fit,solve
from joint_quadratic_fit_v1 import product_cross
P=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2);torch.manual_seed(9114007);tic=time.perf_counter();dtype=torch.float64
    eye=torch.eye(5,dtype=dtype);a,b,p,t,n=eye;u=a+b
    L=torch.stack([u,u,a]);R=torch.stack([p,t,n]);D=torch.eye(3,dtype=dtype)
    U=torch.tensor([[1,0,1],[1,0,-1],[0,1,1],[0,1,-1],[1,1,0],[-1,1,0]],dtype=dtype);M=U.T@U
    total=((D.T@M@D)*product_cross(L,R,L,R)).sum()
    a0=torch.randn(3,5,dtype=dtype);b0=torch.randn(3,5,dtype=dtype)
    best,diag=fit(L,R,D,M,a0,b0,total,steps=600,lr=.03)
    assert diag['best_squared_relative_error']<=1e-4,diag
    aa=a0.clone().requires_grad_();bb=b0.clone();cross=product_cross(L,R,aa,bb);G=product_cross(aa,bb,aa,bb)
    with torch.no_grad():w=torch.linalg.solve(G,(D@cross).T).T
    loss=(total+((w.T@M@w)*G).sum()-2*((D.T@M@w)*cross).sum())/total;loss.backward();analytic=float(aa.grad[0,0])
    def f(z):
        w,c,g=solve(L,R,D,z,bb);return float((total+((w.T@M@w)*g).sum()-2*((D.T@M@w)*c).sum())/total)
    plus=a0.clone();minus=a0.clone();plus[0,0]+=1e-5;minus[0,0]-=1e-5;numeric=(f(plus)-f(minus))/2e-5
    assert abs(analytic-numeric)<=1e-7
    result=dict(planted_dag_relative_error=max(0,diag['best_squared_relative_error'])**.5,gradient_absolute_error=abs(analytic-numeric),random_initialization=True,iterations=600,condition=diag['condition'],seconds=time.perf_counter()-tic,optimizer_sha256=hashlib.sha256((P/'joint_quadratic_optimizer_v1.py').read_bytes()).hexdigest(),scope='Small planted numerical recovery and gradient controls; no native tensor fitted.')
    out=P/'JOINT_QUADRATIC_OPTIMIZER_V1_CONTROL.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
