"""Is the early-determined bus content just token identity? Section 100: the L2
stream predicts L16's bus coordinates at R^2 0.659. At L2 the stream is dominated
by the current token's embedding plus two blocks of early context. If the bus is
largely LEXICAL (a function of the current token), the story flattens; if it is
contextual, the early determination is early CONTEXT.

REGISTERED PREDICTIONS: (a) current-token embedding (wte row, linear ridge)
predicts the bus at held-out R^2 <= 0.4 (well below the L2 stream's 0.659 -- the
early share is context, not identity); (b) the gap L2-stream minus embedding
>= 0.2. Alternative: embedding R^2 ~ 0.6 makes the bus mostly lexical -- also an
answer, and it would demote the 'syntax bus' reading."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_bus_assembly2 import fwd16v
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_bus_lexicality_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    accs=[]
    for i in range(0,60,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=16, acc=acc); accs.append(acc[0])
    Y16=torch.cat(accs)
    _,_,Vh=torch.linalg.svd((Y16-Y16.mean(0)).float(), full_matrices=False)
    BUS=orth(Vh[:8].T)
    def collect(rows):
        bus=[]; emb=[]; in2=[]
        for i in range(0,len(rows),6):
            b=rows[i:i+6].to(DEV)
            mo16,_,_,x2=fwd16v(b)
            bus.append(mo16@BUS)
            emb.append(m.transformer.wte(b).detach().reshape(-1,D).float())
            in2.append(x2)
        return torch.cat(bus),torch.cat(emb),torch.cat(in2)
    yt,Et,S2t=collect(FW[0:60,:257])
    ye,Ee,S2e=collect(FW[300:336,:257])
    out={}
    for tag,Xt,Xe in (('embedding',Et,Ee),('l2stream',S2t,S2e)):
        Xt=Xt-Xt.mean(0); Xe=Xe-Xe.mean(0)
        ytc=yt-yt.mean(0); yec=ye-ye.mean(0)
        lam=1e-2*float((Xt**2).mean())*Xt.shape[1]/Xt.shape[0]
        W=torch.linalg.solve(Xt.T@Xt/Xt.shape[0]+lam*torch.eye(D,device=DEV),
                             Xt.T@ytc/Xt.shape[0])
        r2=1-float(((yec-Xe@W)**2).mean()/(yec**2).mean())
        out[tag]=r2
        print(f'{tag:10s}: held-out R^2 {r2:+.3f}',flush=True)
    pa=out['embedding']<=0.4
    pb=(out['l2stream']-out['embedding'])>=0.2
    out['pred_a']=bool(pa); out['pred_b']=bool(pb)
    print(f"(a) not lexical (emb <=0.4): {'HELD' if pa else 'FAILED'}")
    print(f"(b) context gap >=0.2: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
