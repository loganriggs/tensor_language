"""INTERCHANGE -- Geiger-style intervention on the line-break
circuit's PUSH channel (b0 = m0 output dirs 12-16 on slice r.0.0).
349 showed b0 pushes line-breaks and mlp3 brakes it. If the channel
CONTAINS the break signal (not just correlates), then SETTING it to
its break-state value at positions where no break occurs must raise
the model's newline logit there, and setting it to its rest-state
value at break-pushing members must lower theirs. Controls: a random
4-dim subspace of m0's output, same donor construction.
REGISTERED PREDICTIONS:
  (a) break-state patch at 128 non-break slice positions raises the
      newline logit by >=+0.5 on average; random-subspace control
      <=40% of that;
  (b) rest-state patch at 96 push-wing members lowers their newline
      logit (mean <=-0.5);
  (c) effect distribution reported both ways (no cherry-pick)."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV, orth
D=1152; NL=198
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'interchange_results.json'
TAG='r.0.0.1'

@torch.no_grad()
def main():
    t0=time.time()
    lf=cl.leaf(TAG); mem=lf['member']; msc=cl.member_scores(TAG)
    B0=cl.pca_block('m0','r.0.0',(12,16)).to(DEV)      # 4 x D
    g=torch.Generator().manual_seed(2)
    R=torch.linalg.qr(torch.randn(D,4,generator=g))[0].T.to(DEV)
    Y=cl.capture_out('m0').float()
    donors=mem[msc>0]
    slm=torch.zeros(54272,dtype=torch.bool); slm[lf['slice']]=True
    mm=torch.zeros(54272,dtype=torch.bool); mm[mem]=True
    tgt=cl.rows()[:,1:257].reshape(-1)
    cand=torch.nonzero(slm&~mm&(tgt!=NL)).squeeze(1)
    g1=torch.Generator().manual_seed(1)
    rec=cand[torch.randperm(len(cand),generator=g1)[:128]]
    rest=torch.nonzero(~slm).squeeze(1)
    g3=torch.Generator().manual_seed(3)
    rest=rest[torch.randperm(len(rest),generator=g3)[:4000]]
    STATES={'break_b0':(B0,(Y[donors]@B0.T.cpu()).mean(0).to(DEV)),
            'rest_b0':(B0,(Y[rest]@B0.T.cpu()).mean(0).to(DEV)),
            'break_rand':(R,(Y[donors]@R.T.cpu()).mean(0).to(DEV)),
            'rest_rand':(R,(Y[rest]@R.T.cpu()).mean(0).to(DEV))}
    ROWS=cl.rows()
    def nl_logits(positions,patch=None):
        """newline logit at flat positions; patch=(basis,val) sets
        the basis-coords of m0's output to val AT those positions."""
        posset={}
        for gi in positions.tolist():
            posset.setdefault(gi//256,[]).append(gi%256)
        out={}
        rws=sorted(posset)
        for i0 in range(0,len(rws),4):
            batch=rws[i0:i0+4]
            bb=torch.stack([ROWS[r_] for r_ in batch])[:,:257].to(DEV)
            idx=bb[:,:-1].contiguous()
            hooks=[]
            if patch is not None:
                Bx,val=patch
                mask=torch.zeros(len(batch),256,dtype=torch.bool)
                for bi,r_ in enumerate(batch):
                    mask[bi,posset[r_]]=True
                mask=mask.to(DEV)
                def fh(mo,i_,o_,Bx=Bx,val=val,mask=mask):
                    yf=o_.float()
                    co=yf@Bx.T
                    yf=yf-(co-val[None,None,:]).masked_fill(
                        ~mask[:,:,None],0.0)@Bx
                    return yf.to(o_.dtype)
                hooks.append(cl.MODS['m0'].register_forward_hook(fh))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blkm in m.transformer.h:
                x,v1=blkm(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            for h in hooks: h.remove()
            for bi,r_ in enumerate(batch):
                for p_ in posset[r_]:
                    out[r_*256+p_]=float(lg[bi,p_,NL])
        return torch.tensor([out[int(gi)] for gi in positions.tolist()])
    baseR=nl_logits(rec)
    upB=nl_logits(rec,STATES['break_b0'])-baseR
    upR=nl_logits(rec,STATES['break_rand'])-baseR
    baseD=nl_logits(donors)
    dnB=nl_logits(donors,STATES['rest_b0'])-baseD
    dnR=nl_logits(donors,STATES['rest_rand'])-baseD
    mu=float(upB.mean()); mur=float(upR.mean())
    md=float(dnB.mean()); mdr=float(dnR.mean())
    print(f'break-state at non-break: b0 {mu:+.3f} | rand {mur:+.3f}')
    print(f'rest-state at push-wing:  b0 {md:+.3f} | rand {mdr:+.3f}')
    qs=lambda v:[round(float(v.quantile(q)),2) for q in (0.1,0.5,0.9)]
    pa=mu>=0.5 and abs(mur)<=0.4*abs(mu)
    pb=md<=-0.5
    out={'up_b0':round(mu,3),'up_rand':round(mur,3),
         'down_b0':round(md,3),'down_rand':round(mdr,3),
         'up_b0_q':qs(upB),'down_b0_q':qs(dnB),
         'n_rec':len(rec),'n_donors':len(donors),
         'frac_up_pos':round(float((upB>0).float().mean()),3),
         'frac_down_neg':round(float((dnB<0).float().mean()),3),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':True}
    print(f"(a) up >=+0.5, rand <=40%: {'HELD' if pa else 'FAILED'}")
    print(f"(b) down <=-0.5: {'HELD' if pb else 'FAILED'}")
    print(f"(c) distributions: up q10/50/90 {out['up_b0_q']} "
          f"down {out['down_b0_q']}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
