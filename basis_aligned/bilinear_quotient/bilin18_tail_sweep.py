"""Tail sweep, weights-first: the seven still-unprofiled layers in one cheap pass.

Per the adopted protocol (§23/§30): G_lam candidates first, model evaluations only to
verify. For each of layers 5, 6, 8, 10, 12, 14, 15: predict the causal top-2 output
directions from weights + input second moment, then spend exactly three evaluations --
delete predicted-2, delete random-2 (control), delete top-32 span (the layer's scale).
REGISTERED PREDICTIONS: (a) predicted-2 deletion costs >= 5x random-2 at every layer;
(b) at least one tail layer shows the negative span effect seen at layer 9 (§30's
shift-regularisation pattern)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, held, orth, m, FW, DEV, PATCH
D=1152
LAYERS=(5,6,8,10,12,14,15)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_tail_sweep_results.json')

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
    base=held(); out={'layers':{}}
    print(f"  {'layer':>5} {'pred-2':>9} {'rand-2':>9} {'ratio':>7} {'span-32':>9}")
    for li in LAYERS:
        X=collect(li,'in'); S=X.T@X/X.shape[0]
        mlp=m.transformer.h[li].mlp
        L=mlp.Left.weight.detach().float(); R=mlp.Right.weight.detach().float()
        Dw=mlp.Down.weight.detach().float()
        G=Dw@((L@S@L.T)*(R@S@R.T))@Dw.T
        ev,U=torch.linalg.eigh(G)
        pred2=orth(U[:,ev.argsort(descending=True)[:2]])
        Y=collect(li,'out'); Ybar=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Ybar).float(), full_matrices=False)
        span=orth(Vh[:32].T)
        g=torch.Generator(device=DEV).manual_seed(li)
        rnd=orth(torch.randn(D,2,device=DEV,generator=g))
        def val(Q):
            PATCH[li]=(Q,Ybar@Q)
            try: return float((held()-base).mean())
            finally: PATCH.pop(li)
        dp,dr,ds=val(pred2),val(rnd),val(span)
        ratio=dp/max(abs(dr),1e-6)
        out['layers'][li]={'pred2':dp,'rand2':dr,'span32':ds,'ratio':ratio}
        print(f"  {li:>5} {dp:>+9.4f} {dr:>+9.4f} {ratio:>7.1f} {ds:>+9.4f}",flush=True)
    ok=sum(1 for r in out['layers'].values() if r['pred2']>=5*abs(r['rand2']))
    neg=[li for li,r in out['layers'].items() if r['span32']<-0.002]
    out['pred_a_hits']=ok; out['negative_span_layers']=neg
    print(f"\n(a) pred-2 >= 5x random: {ok}/{len(LAYERS)} layers")
    print(f"(b) negative span effects at: {neg if neg else 'none'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
