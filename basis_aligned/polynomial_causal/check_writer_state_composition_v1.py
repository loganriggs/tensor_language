"""Preregistered executor control, not a native-model experiment.

Seed 1405; d=64,k=7,18 blocks,13 readers/block,11 products,96 outputs,
64 inputs at scales .3,1,3. Nonorthogonal writers, unrestricted readers,
signed residual reentry and biases. Four arms: no edit, (5,1), (12,2), both.
A: state/norm/logit relative error <=1e-10 in every arm.
B: signed effect <=1e-8 and nonzero interaction >1e-8 with error <=1e-6.
C: falsely dropping Gram offdiagonals causes >1e-3 output error.
No fitting. Failure rejects this executor or its numerical regime, not structure.
"""
import json
from pathlib import Path
import torch
from writer_state_program_v1 import encode, execute


def main():
    torch.set_num_threads(2);torch.manual_seed(1405);torch.set_default_dtype(torch.float64)
    d,k,v=64,7,96;eps=torch.finfo(torch.float32).eps
    w=torch.randn(d,k)/d**.5;w[:,1]+=0.8*w[:,0]
    u=torch.randn(v,d)/d**.5
    fs=[torch.randn(13,d)/d**.5 for _ in range(18)]
    blocks=[dict(products=torch.randint(13,(11,2)),mixing=.15*torch.randn(k,11),
                 bias=.02*torch.randn(k),reentry=torch.tensor([-.7 if j%4==2 else .9,.2])) for j in range(18)]
    arms=[(),((5,1),),((12,2),),((5,1),(12,2))];rows=[]
    def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
    for scale in (.3,1.,3.):
        x=scale*torch.randn(64,d);enc=encode(x,w,fs,u);pred=[];ref=[];cell=[]
        for edits in arms:
            h=x.clone()
            for j,b in enumerate(blocks):
                lam,mu=b['reentry'];h=lam*h+mu*x
                reads=(h@fs[j].T)/(h.square().mean(-1,keepdim=True)+eps).sqrt()
                pp=b['products'];p=reads[:,pp[:,0]]*reads[:,pp[:,1]]
                for layer,fac in edits:
                    if layer==j:p[:,fac]=0
                h=h+(p@b['mixing'].T+b['bias'])@w.T
            logits=30*torch.tanh((h@u.T)/(h.square().mean(-1,keepdim=True)+eps).sqrt()/30)
            out=execute(enc,blocks,edits);hh=out['alpha']*x+out['coefficients']@w.T
            cell.append(max(rel(hh,h),rel(out['norm2'],h.square().sum(-1)),rel(out['logits'],logits)))
            pred.append(out['logits']);ref.append(logits)
        effect=max(rel(pred[i]-pred[0],ref[i]-ref[0]) for i in (1,2,3))
        ri=ref[3]-ref[2]-ref[1]+ref[0];pi=pred[3]-pred[2]-pred[1]+pred[0]
        broken=dict(enc);broken['gram']=enc['gram'].diag().diag()
        negative=rel(execute(broken,blocks)['logits'],ref[0])
        rows.append(dict(scale=scale,replay=max(cell),effect=effect,interaction=rel(pi,ri),interaction_norm=float(ri.norm()),diagonal_gram_error=negative))
    result={'pred_a':all(r['replay']<=1e-10 for r in rows),
            'pred_b':all(r['effect']<=1e-8 and r['interaction']<=1e-6 and r['interaction_norm']>1e-8 for r in rows),
            'pred_c':all(r['diagonal_gram_error']>1e-3 for r in rows),'rows':rows,
            'scope':'Synthetic repeated composition control; no native circuit or OOD claim.'}
    Path(__file__).with_name('WRITER_STATE_COMPOSITION_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
