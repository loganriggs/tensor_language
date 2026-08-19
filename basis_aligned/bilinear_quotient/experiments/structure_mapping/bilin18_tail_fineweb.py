"""Registered §37 follow-up: are the tail's negative deletion effects shift-specific?
Layers 9, 12, 15 (negative span effects on pile) rescored on the saved fineweb sample.
REGISTERED PREDICTION: all three span-32 deletions cost POSITIVE CE on fineweb, and
layer 15's pred-2 deletion flips positive too."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV, PATCH
D=1152
fine=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/fineweb_eval_tokens.pt')

@torch.no_grad()
def ce_on(tokens):
    tot,n=0.0,0
    for i in range(0,tokens.shape[0],6):
        ce=fwd(tokens[i:i+6].to(DEV))
        tot+=float(ce.sum()); n+=ce.numel()
    return tot/n

@torch.no_grad()
def collect(li, what):
    ins,outs=[],[]
    def hook(mod,inp,o):
        if what!='out': ins.append(inp[0].detach().reshape(-1,D).float())
        if what!='in': outs.append(o.detach().reshape(-1,D).float())
    h=m.transformer.h[li].mlp.register_forward_hook(hook)
    for i in range(0,60,6):
        b=FW[i:i+6,:513].to(DEV)
        m(b[:,:-1].contiguous(), b[:,1:].contiguous())
    h.remove()
    return (torch.cat(ins) if what=='in' else torch.cat(outs))

def main():
    t0=time.time()
    base=ce_on(fine)
    out={'base_fineweb':base,'layers':{}}
    print(f'fineweb base CE {base:.4f}')
    print(f"  {'layer':>5} {'span-32 pile':>13} {'span-32 fineweb':>16} {'pred-2 fineweb':>15}")
    pile={9:-0.0068,12:-0.0025,15:-0.0065}   # recorded
    for li in (9,12,15):
        Y=collect(li,'out'); Ybar=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Ybar).float(), full_matrices=False)
        span=orth(Vh[:32].T)
        X=collect(li,'in'); S=X.T@X/X.shape[0]
        mlp=m.transformer.h[li].mlp
        L=mlp.Left.weight.detach().float(); R=mlp.Right.weight.detach().float()
        Dw=mlp.Down.weight.detach().float()
        G=Dw@((L@S@L.T)*(R@S@R.T))@Dw.T
        ev,U=torch.linalg.eigh(G)
        pred2=orth(U[:,ev.argsort(descending=True)[:2]])
        def val(Q):
            PATCH[li]=(Q,Ybar@Q)
            try: return ce_on(fine)-base
            finally: PATCH.pop(li)
        ds=val(span); dp=val(pred2)
        out['layers'][li]={'span32_fineweb':ds,'pred2_fineweb':dp,'span32_pile':pile[li]}
        print(f"  {li:>5} {pile[li]:>+13.4f} {ds:>+16.4f} {dp:>+15.4f}",flush=True)
    ok=all(v['span32_fineweb']>0 for v in out['layers'].values()) and \
       out['layers'][15]['pred2_fineweb']>0
    out['prediction_held']=bool(ok)
    print(f"\nregistered prediction (all positive in-distribution): "
          f"{'HELD' if ok else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                       'bilin18_tail_fineweb_results.json','w'),indent=1)
    print(f'wrote bilin18_tail_fineweb_results.json ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
