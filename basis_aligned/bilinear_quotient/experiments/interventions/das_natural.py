"""DAS v2 NATURAL -- 358: the learned basis steers (+4.42) but a
shuffled-objective control also steers (+2.70), so the optimized
VALUE may be adversarial. Decisive test: relearn the basis (same
seeds -> same optimum), then transplant NATURAL donor coordinates
(a random pos-wing member's actual coords in the learned basis, no
optimization) at held-out recipients.
REGISTERED PREDICTIONS:
  (a) natural-donor transplant in the learned basis shifts the
      newline logit >= +0.5 held-out -> the basis captures a real
      circuit variable;
  (b) natural transplant >= 30% of the optimized-value effect;
  (c) if (a) fails (<= +0.2), record: the learned direction is an
      adversarial steering direction, not a circuit variable --
      and optimization-based DAS needs naturalness constraints in
      this architecture. Basis saved to das_basis.pt either way."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; NL=198
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'das_natural_results.json'
TAG='r.0.0.1'

def build_positions():
    lf=cl.leaf(TAG); mem=lf['member']
    slm=torch.zeros(54272,dtype=torch.bool); slm[lf['slice']]=True
    mm=torch.zeros(54272,dtype=torch.bool); mm[mem]=True
    tgt=cl.rows()[:,1:257].reshape(-1)
    cand=torch.nonzero(slm&~mm&(tgt!=NL)).squeeze(1)
    g1=torch.Generator().manual_seed(1)
    cand=cand[torch.randperm(len(cand),generator=g1)]
    seen=set(); rec=[]
    for gi in cand.tolist():
        if gi//256 not in seen: seen.add(gi//256); rec.append(gi)
    rec=rec[:96]
    # shuffled-objective control positions: off-slice, non-newline
    c2=torch.nonzero((~slm)&(tgt!=NL)).squeeze(1)
    g5=torch.Generator().manual_seed(5)
    c2=c2[torch.randperm(len(c2),generator=g5)]
    seen=set(); ctl=[]
    for gi in c2.tolist():
        if gi//256 not in seen: seen.add(gi//256); ctl.append(gi)
    return rec,ctl[:96]

def run(positions,label):
    ROWS=cl.rows()
    train=positions[0::2][:48]; hold=positions[1::2][:48]
    W=torch.randn(D,4,device=DEV)*0.02; W.requires_grad_(True)
    val=torch.zeros(4,device=DEV); val.requires_grad_(True)
    opt=torch.optim.Adam([W,val],lr=5e-3)
    g=torch.Generator().manual_seed(7)
    def nl_at(batch_gi,Bx,vv,grad):
        ctx=torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            tot=0
            for gi in batch_gi:
                r_,p_=gi//256,gi%256
                idx=ROWS[r_][None,:257][:,:-1].to(DEV)
                def fh(mo,i_,o_,Bx=Bx,vv=vv,p_=p_):
                    yf=o_.float()
                    co=yf@Bx
                    mask=torch.zeros(1,yf.shape[1],1,device=DEV)
                    mask[0,:p_+1]=1.0
                    yf=yf-((co-vv[None,None,:])*mask)@Bx.T
                    return yf.to(o_.dtype)
                h=cl.MODS['m0'].register_forward_hook(fh)
                x=F.rms_norm(m.transformer.wte(idx),(D,))
                x0=x; v1=None
                for blkm in m.transformer.h: x,v1=blkm(x,v1,x0)
                lg=(30*torch.tanh(m.lm_head(
                    F.rms_norm(x,(D,)))/30)).float()
                h.remove()
                tot=tot+lg[0,p_,NL]
            return tot/len(batch_gi)
    base_h=float(nl_at(hold,torch.zeros(D,4,device=DEV),
                       torch.zeros(4,device=DEV),False))
    for step in range(120):
        bi=[train[int(i)] for i in
            torch.randperm(len(train),generator=g)[:6]]
        Bx=torch.linalg.qr(W)[0]
        loss=-nl_at(bi,Bx,val,True)
        opt.zero_grad(); loss.backward(); opt.step()
        if step%30==0:
            print(f'{label} step {step}: train nl {-float(loss):+.3f}',
                  flush=True)
    with torch.no_grad(): Bx=torch.linalg.qr(W)[0].detach()
    tr=float(nl_at(train,Bx,val.detach(),False))
    base_t=float(nl_at(train,torch.zeros(D,4,device=DEV),
                       torch.zeros(4,device=DEV),False))
    ho=float(nl_at(hold,Bx,val.detach(),False))
    return ({'train_shift':round(tr-base_t,3),
             'holdout_shift':round(ho-base_h,3)},Bx,val)

def main():
    t0=time.time()
    for p in m.parameters(): p.requires_grad_(False)
    rec,ctl=build_positions()
    das,Bx,val=run(rec,'DAS')
    print('DAS relearn:',das,flush=True)
    torch.save({'B':Bx.cpu(),'val':val.detach().cpu()},
               PT+'das_basis.pt')
    # natural transplant: random pos-wing donor's coords in Bx
    lf=cl.leaf(TAG); mem=lf['member']; msc=cl.member_scores(TAG)
    donors=mem[msc>0]
    Y=cl.capture_out('m0').float()
    g4=torch.Generator().manual_seed(4)
    dsel=donors[torch.randperm(len(donors),generator=g4)[:8]]
    hold=rec[1::2][:48]
    ROWS=cl.rows()
    import torch.nn.functional as F2
    @torch.no_grad()
    def nl_hold(vv):
        tot=0
        for gi in hold:
            r_,p_=gi//256,gi%256
            idx=ROWS[r_][None,:257][:,:-1].to(DEV)
            def fh(mo,i_,o_,vv=vv,p_=p_):
                yf=o_.float()
                co=yf@Bx
                mask=torch.zeros(1,yf.shape[1],1,device=DEV)
                mask[0,:p_+1]=1.0
                yf=yf-((co-vv[None,None,:])*mask)@Bx.T
                return yf.to(o_.dtype)
            h=cl.MODS['m0'].register_forward_hook(fh)
            x=F2.rms_norm(m.transformer.wte(idx),(D,))
            x0=x; v1=None
            for blkm in m.transformer.h: x,v1=blkm(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(
                F2.rms_norm(x,(D,)))/30)).float()
            h.remove()
            tot+=float(lg[0,p_,NL])
        return tot/len(hold)
    base_h=nl_hold(torch.zeros(4,device=DEV))
    # base_h with zero patch is NOT unpatched; compute true base:
    @torch.no_grad()
    def nl_base():
        tot=0
        for gi in hold:
            r_,p_=gi//256,gi%256
            idx=ROWS[r_][None,:257][:,:-1].to(DEV)
            x=F2.rms_norm(m.transformer.wte(idx),(D,))
            x0=x; v1=None
            for blkm in m.transformer.h: x,v1=blkm(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(
                F2.rms_norm(x,(D,)))/30)).float()
            tot+=float(lg[0,p_,NL])
        return tot/len(hold)
    b0=nl_base()
    nat=[]
    for dgi in dsel.tolist():
        vv=(Y[dgi]@Bx.cpu()).to(DEV)
        nat.append(nl_hold(vv)-b0)
    natm=sum(nat)/len(nat)
    opt_shift=nl_hold(val.detach())-b0
    zero_shift=base_h-b0
    print(f'natural donors: mean {natm:+.3f} '
          f'({[round(x,2) for x in nat]})')
    print(f'optimized val: {opt_shift:+.3f} | zero-coords: '
          f'{zero_shift:+.3f}',flush=True)
    pa=natm>=0.5
    pb=natm>=0.3*max(opt_shift,1e-3)
    out={'relearn':das,'natural_mean':round(natm,3),
         'natural_each':[round(x,3) for x in nat],
         'optimized_shift':round(opt_shift,3),
         'zero_shift':round(zero_shift,3),
         'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f"(a) natural >=+0.5: {'HELD' if pa else 'FAILED'}")
    print(f"(b) natural >=30% optimized: {'HELD' if pb else 'FAILED'}")
    if not pa:
        print('(c) verdict: adversarial steering direction, not a '
              'circuit variable')
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
