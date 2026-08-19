"""CP-TRUNCATION CONTROLS -- make section 261's hidden-unit claim airtight.
261 showed keeping the top-25% of hidden units by ||down||*||l||*||r||
recovers >=80% of every middle MLP, but ran no selection nulls. Here, for
layers 4-9, solo CE cost of:
  top-k    (k = 576, 1152, 2304, 3456)  -- the registered ranking
  rand-k   (k = 1152, random unit subset, fixed seed)
  bot-3456 (drop the top 1152, keep the bottom 3456 -- 3x the parameter
            count of top-1152)
REGISTERED PREDICTIONS:
  (a) rand-1152 cost >= 2x top-1152 cost for >=5/6 layers;
  (b) bot-3456 cost >= top-1152 cost for 6/6 layers (the top quarter
      carries more than the bottom three quarters combined);
  (c) top-576 (12.5%) solo cost <= +0.15 for >=5/6 layers."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'cp_controls_results.json'
R0,R1=120,300
LAYERS=(4,5,6,7,8,9)

@torch.no_grad()
def evalCE():
    ces=[]
    for i in range(R0,R1,4):
        bb=FW[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h:
            x,v1=blk(x,v1,x0)
        lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
        ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                   reduction='none'))
    return float(torch.cat(ces).mean())

@torch.no_grad()
def main():
    t0=time.time()
    base=evalCE()
    print(f'base CE {base:.4f}',flush=True)
    g=torch.Generator(device=DEV).manual_seed(0)
    out={'base':round(base,4),'layers':{}}
    for li in LAYERS:
        mlp=m.transformer.h[li].mlp
        L=mlp.Left.weight.detach().float()
        Rw=mlp.Right.weight.detach().float()
        Dw=mlp.Down.weight.detach().float()
        db=mlp.Down_bias.detach().float()
        imp=Dw.norm(dim=0)*L.norm(dim=1)*Rw.norm(dim=1)
        order=imp.argsort(descending=True)
        H=L.shape[0]
        def cost(keep):
            Lk=L[keep].contiguous(); Rk=Rw[keep].contiguous()
            Dk=Dw[:,keep].contiguous()
            def fh(mo,i_,o_):
                x=i_[0].float()
                return (((x@Lk.T)*(x@Rk.T))@Dk.T+db).to(o_.dtype)
            h=mlp.register_forward_hook(fh)
            c=evalCE()-base
            h.remove()
            return c
        rec={'top':{},'rand':None,'bot':None}
        for k in (576,1152,2304,3456):
            rec['top'][k]=cost(order[:k])
        rec['rand']=cost(torch.randperm(H,device=DEV,generator=g)[:1152])
        rec['bot']=cost(order[1152:])
        print(f'L{li} top576 {rec["top"][576]:+.4f} top1152 '
              f'{rec["top"][1152]:+.4f} top2304 {rec["top"][2304]:+.4f} '
              f'top3456 {rec["top"][3456]:+.4f} | rand1152 '
              f'{rec["rand"]:+.4f} bot3456 {rec["bot"]:+.4f}',flush=True)
        out['layers'][li]=rec
    la=out['layers']
    na=sum(1 for li in LAYERS
           if la[li]['rand']>=2*max(la[li]['top'][1152],1e-4))
    nb=sum(1 for li in LAYERS if la[li]['bot']>=la[li]['top'][1152])
    nc=sum(1 for li in LAYERS if la[li]['top'][576]<=0.15)
    pa=na>=5; pb=nb==6; pc=nc>=5
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['pred_c']=bool(pc)
    print(f"\n(a) rand >= 2x top at k=1152, {na}/6: "
          f"{'HELD' if pa else 'FAILED'}")
    print(f"(b) bottom-3456 worse than top-1152, {nb}/6: "
          f"{'HELD' if pb else 'FAILED'}")
    print(f"(c) top-576 <= +0.15, {nc}/6: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
