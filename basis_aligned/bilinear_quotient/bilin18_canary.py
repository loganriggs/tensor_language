"""Canary: three fast headline checks against frozen reference values, to catch
any future environment/model/instrument drift. Not an experiment -- a standing
regression test. REGISTERED TOLERANCES: (a) score-rank median (L2/9/16, fresh
stats rows) in [3.3, 5.3]; (b) L1 linearization cost in [0.20, 0.37];
(c) dilution ratio at edge 5->6 in [0.22, 0.28] and at 14->15 in [0.03, 0.06]."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
import bilin18_pipe_refit as PR
NH,HD,D=9,128,1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_canary_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    caps={li:[] for li in (2,9,16)}
    ins={5:[],6:[],14:[],15:[]}
    hs=[]
    for li in caps:
        def mk(li=li):
            return lambda mod,inp: caps[li].append(
                F.rms_norm(inp[0].detach().reshape(-1,D).float(),(D,))) or None
        hs.append(m.transformer.h[li].attn.register_forward_pre_hook(mk()))
    mos={5:[],14:[]}
    for li in mos:
        def mko(li=li):
            return lambda mod,i_,o_: mos[li].append(
                o_.detach().reshape(-1,D).float())
        hs.append(m.transformer.h[li].mlp.register_forward_hook(mko()))
    for li in (6,15):
        def mki(li=li):
            return lambda mod,inp: ins[li].append(
                inp[0].detach().reshape(-1,D).float()) or None
        hs.append(m.transformer.h[li].register_forward_pre_hook(mki()))
    for i in range(48,96,6):
        b=FW[i:i+6,:513].to(DEV)
        m(b[:,:-1].contiguous(), b[:,1:].contiguous())
    for h in hs: h.remove()
    ranks=[]
    for li in caps:
        X=torch.cat(caps[li]); Xc=X-X.mean(0)
        C=Xc.T@Xc/Xc.shape[0]
        ev,U=torch.linalg.eigh(C.double())
        Ch=((U*ev.clamp_min(0).sqrt())@U.T).float()
        a=m.transformer.h[li].attn
        for h_ in range(NH):
            for wq,wk in ((a.c_q,a.c_k),(a.c_q2,a.c_k2)):
                Wq=wq.weight.detach().float().view(NH,HD,D)[h_]
                Wk=wk.weight.detach().float().view(NH,HD,D)[h_]
                sv=torch.linalg.svdvals(Ch@Wq.T@Wk@Ch); e=sv**2
                ranks.append(float(e.sum()**2/(e**2).sum()))
    mr=sorted(ranks)[len(ranks)//2]
    rat={}
    for wli,rli in ((5,6),(14,15)):
        Y=torch.cat(mos[wli]); X=torch.cat(ins[rli])
        rat[wli]=float((Y-Y.mean(0)).pow(2).sum(1).mean())/ \
                 float((X-X.mean(0)).pow(2).sum(1).mean())
    PR.LINS={}
    def ce():
        tot,n=0.0,0
        for i in range(300,364,4):
            b=FW[i:i+4,:257].to(DEV)
            lg,_=PR.fwd_lin(b[:,:-1].contiguous())
            c=F.cross_entropy(lg.view(-1,lg.size(-1)), b[:,1:].reshape(-1))
            tot+=float(c)*(b.shape[1]-1)*b.shape[0]; n+=(b.shape[1]-1)*b.shape[0]
        return tot/n
    base=ce(); PR.LINS={1:PR.fit_layer(1)}; cost=ce()-base; PR.LINS={}
    pa=3.3<=mr<=5.3; pb=0.20<=cost<=0.37
    pc=0.22<=rat[5]<=0.28 and 0.03<=rat[14]<=0.06
    out={'score_rank':mr,'l1_cost':cost,'ratio_5_6':rat[5],'ratio_14_15':rat[14],
         'pa':bool(pa),'pb':bool(pb),'pc':bool(pc)}
    print(f'canary: score-rank {mr:.1f} | L1 cost +{cost:.3f} | '
          f'ratios {rat[5]:.3f}/{rat[14]:.3f}')
    print(f"(a) {'OK' if pa else 'DRIFT'} | (b) {'OK' if pb else 'DRIFT'} | "
          f"(c) {'OK' if pc else 'DRIFT'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
