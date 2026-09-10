"""Executable algebra fixtures for two user-directed composition questions."""
from pathlib import Path
import json,hashlib
import torch
from joint_weight_composition_v1 import input_product_gram,full_output_code_grams,nearest_signed_profile,joint_score
P=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2);torch.manual_seed(91831);dtype=torch.float64
    L=torch.tensor([[1,0,0,0],[-2,0,0,0],[0,0,1,0],[0,0,0,1]],dtype=dtype)
    R=torch.tensor([[0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=dtype)
    D=torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,-2]],dtype=dtype)
    U=torch.cat([torch.eye(3,dtype=dtype),-torch.eye(3,dtype=dtype)])
    A=U@D;x=torch.randn(19,4,dtype=dtype);phi=(x@L.T)*(x@R.T)
    fold=float(((phi@D.T)@U.T-phi@A.T).abs().max())
    antipodal=float((((-x)@L.T)*((-x)@R.T)-phi).abs().max())
    GI=input_product_gram(L,R);GO,GC,common=full_output_code_grams(U,D,chunk=2)
    fullgram=float((GO-A.T@A).abs().max());centergram=float((GC-(A-A.mean(0)).T@(A-A.mean(0))).abs().max())
    ip=nearest_signed_profile(GI);op=nearest_signed_profile(GC)
    assert int(ip['partner'][0])==1 and float(ip['error'][0])<1e-7
    assert int(op['partner'][2])==3 and float(op['cosine'][2])<-.999999
    # A and B read disjoint supports, yet BOTH score halves are indispensable.
    xa=torch.tensor([[1.,2.,0.,0.]],dtype=dtype);ya=torch.tensor([[2.,1.,0.,0.]],dtype=dtype)
    xb=torch.tensor([[0.,0.,1.,2.]],dtype=dtype);yb=torch.tensor([[0.,0.,2.,1.]],dtype=dtype)
    T=torch.tensor([1.,3.,2.,4.],dtype=dtype)
    def score(x,y):return joint_score(x,y,T*x,y)
    sa=score(xa,ya);sb=score(xb,yb)
    assert sa.item()!=0 and sb.item()!=0
    maskA=torch.tensor([0.,0.,1.,1.],dtype=dtype)
    assert score(xa*maskA,ya).item()==0 and score(xb*maskA,yb).item()==sb.item()
    for x0,y0 in [(xa,ya),(xb,yb)]:
        assert joint_score(x0*0,y0,T*x0,y0).item()==0
        assert joint_score(x0,y0,T*x0*0,y0).item()==0
        assert torch.equal(joint_score(x0,y0,T*x0,y0),joint_score(T*x0,y0,x0,y0))
    mixed=score(xa+xb,ya+yb)-sa-sb
    assert mixed.item()!=0 # Mixed products cannot be dropped on unrestricted inputs.
    result=dict(full_unembedding_fold_max=fold,product_antipodal_max=antipodal,full_output_gram_max=fullgram,centered_output_gram_max=centergram,shared_input_pair=[0,1],opposite_output_profile_pair=[2,3],joint_routing=dict(task_A=sa.item(),task_B=sb.item(),whole_half_ablation_both_tasks_zero=True,input_A_removal_selective_on_supported_inputs=True,mixed_input_cross_term=mixed.item()),scope='Planted algebra fixtures only. No native task subspaces identified; normalized/mixed native input effects require separate testing.',implementation_sha256=hashlib.sha256((P/'joint_weight_composition_v1.py').read_bytes()).hexdigest())
    assert max(fold,antipodal,fullgram,centergram)<1e-10
    out=P/'JOINT_WEIGHT_COMPOSITION_V1_FIXTURE_RESULT.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
