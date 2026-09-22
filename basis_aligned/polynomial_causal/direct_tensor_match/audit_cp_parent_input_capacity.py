"""Exact coefficient-space rank obstruction for compressing the fitted CP parent.
Not a lower bound on Gaussian/text/native-intervention errors.
"""
import itertools,json,time
from pathlib import Path
import torch
from cp_input_mode_gram import gram
from quartic_cp import cp_entries,cp_gram
P=Path(__file__).resolve().parent

def controls():
    rows=[]
    for seed,family in enumerate(['independent','shared_input','shared_output','squares','cancellation']):
        torch.manual_seed(13100+seed);f=[torch.randn(5,4,dtype=torch.float64) for _ in range(4)];C=torch.randn(2,5,dtype=torch.float64)
        if family=='shared_input':f[1]=f[0].clone()
        if family=='shared_output':C[1]=C[0]
        if family=='squares':f[2:]=[a.clone() for a in f[:2]]
        if family=='cancellation':
            for a in f:a[1]=a[0]
            C[:,1]=-C[:,0]
        idx=torch.tensor(list(itertools.product(range(4),repeat=4)));tensor=cp_entries(C,f,idx).reshape(4,4,4,4,2);unfold=tensor.reshape(4,-1);reference=unfold@unfold.T;K=gram(f,C);error=float((K-reference).norm()/reference.norm());assert error<1e-10
        rows.append(dict(family=family,dense_gram_error=error))
    return rows

def main():
    torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();checks=controls();rows=[]
    for seed in [1001,1002]:
        source=torch.load(P/f'MIXED_CP_FEATURES_SEED{seed}_V1.pt',weights_only=True);f=[a.double() for a in source['factors']];C=source['coefficients'].double()/19054614563.464127;K=gram(f,C);e=torch.linalg.eigvalsh(K);assert e[0]>=-1e-10*e[-1]
        norm=((C.T@C)*cp_gram(f,f)).sum();trace_error=float(abs(K.trace()-norm)/norm);assert trace_error<1e-10
        spectrum=e.clamp_min(0).flip(0);tail=torch.cat([spectrum.flip(0).cumsum(0).flip(0),spectrum.new_zeros(1)])
        bounds=[]
        for terms in [64,128,192,256,288,384,512]:
            rank=min(4*terms,len(e));bounds.append(dict(cp_terms=terms,max_input_rank=rank,relative_coefficient_error_lower_bound=float((tail[rank]/norm).sqrt())))
        necessary={}
        for tolerance in [.1,.05,.01]:
            ranks=torch.where(tail<=tolerance*tolerance*norm)[0];rank=int(ranks[0]);necessary[str(tolerance)]=dict(required_input_rank=rank,necessary_cp_terms=(rank+3)//4)
        rows.append(dict(seed=seed,trace_replay=trace_error,largest_eigenvalue=float(e[-1]),smallest_eigenvalue=float(e[0]),bounds=bounds,necessary=necessary));print(json.dumps(rows[-1]),flush=True)
    result=dict(rows=rows,controls=checks,seconds=time.monotonic()-start,scope='Exact implicit first-input mode Gram of the fully input-symmetric coefficient tensor for fittedmixedCP512parent,16outputs,d1152. A CP-rquartic student usesatmost4rinputdirections, henceunfoldingrank<=4r. Singularvaluetail yieldsnumerical coefficientFrobenius lowerbound, NOT Gaussian/text/nativeeffect bound, NOT whole-native-modelbound, NOT sufficientrecovery condition orintervalcertificate.')
    (P/'CP_PARENT_INPUT_CAPACITY_V1.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
