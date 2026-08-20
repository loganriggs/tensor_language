"""HEAD 12.6 CAUSAL NAMING -- 491: the structure hypothesis is
confirmed functionally (490). Head 12.6's window damage at match
positions rises monotonically with the text's structural density:
+0.0004, +0.0026, +0.0075, +0.0073 across punctuation-and-newline
quartiles -- an 18.25x gradient bottom to top -- while its
layer-mate 12.3 is flat at 1.26x. Combined with 489 (its distant
reads are enriched 2.33x for punctuation and depleted 0.38x for
prose) the head has a name: a LONG-RANGE STRUCTURE READER.
Close it the way this program closes claims -- by intervening on
the named variable rather than the head. If 12.6's function is
reading distant STRUCTURAL tokens, then blocking its reads to
structural positions specifically should reproduce most of the
window damage, while blocking an equal number of content
positions should not.
Arms at match positions, structured text only (top two quartiles):
  window      : the 4-token window (reference)
  block_struct: 12.6 may read anywhere EXCEPT distant punctuation
                and newline positions
  block_content: it may read anywhere EXCEPT an equal count of
                distant prose positions (matched control)
REGISTERED PREDICTIONS:
  (a) STRUCTURE CARRIES IT: block_struct reproduces >= 50% of the
      window damage;
  (b) CONTENT DOES NOT: block_content reproduces <= 25%;
  (c) the two arms block a comparable number of positions
      (within 20%), so the contrast is not a count artifact."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=12; HD=6; K=4
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'head_12_6_causal_results.json'
NFRESH=48

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    at=m.transformer.h[LJ].attn
    are=sys.modules[type(at).__module__].apply_rotary_emb
    fresh=cl.fineweb_rows(NFRESH)
    def isstruct(t):
        s=cl.d1(int(t)); st=s.strip()
        return (bool(st) and not any(c.isalnum() for c in st)) \
            or (chr(10) in s)
    def iscontent(t):
        s=cl.d1(int(t)); st=s.strip()
        return st.isalpha()
    dens=torch.tensor([sum(isstruct(int(fresh[r,q]))
                           for q in range(T))/T
                       for r in range(NFRESH)])
    keep=(dens>=float(dens.median())).nonzero().squeeze(1).tolist()
    print(f'{len(keep)} structured rows kept',flush=True)
    counts={'struct':0,'content':0}
    def run(mode):
        tot=0.0; n=0
        for i in range(0,len(keep),4):
            rid=torch.tensor(keep[i:i+4])
            bb=fresh[rid,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=len(rid); hs=[]
            smask=torch.zeros(B,T,dtype=torch.bool)
            cmask=torch.zeros(B,T,dtype=torch.bool)
            for b in range(B):
                for q in range(T):
                    t=int(bb[b,q])
                    smask[b,q]=isstruct(t)
                    cmask[b,q]=iscontent(t)
            # match content count to struct count per row
            for b in range(B):
                ns=int(smask[b].sum())
                ci=cmask[b].nonzero().squeeze(1)
                if len(ci)>ns: 
                    drop=ci[ns:]
                    cmask[b,drop]=False
            if mode is not None:
                counts['struct']+=int(smask.sum())
                counts['content']+=int(cmask.sum())
                sm=smask.to(DEV); cm=cmask.to(DEV)
                def fh(mo_,args,o_,mode=mode,sm=sm,cm=cm):
                    y,v1r=o_
                    X=args[0]
                    v1=args[1] if args[1] is not None else v1r
                    v=at.c_v(X).view(B,T,9,128)
                    vm=(1-at.lamb)*v+at.lamb*v1.view_as(v)
                    cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
                    def r2(w):
                        return are(F.rms_norm(
                            w(X).view(B,T,9,128),(128,)),cos,sin)
                    qf,kf=r2(at.c_q),r2(at.c_k)
                    q2,k2=r2(at.c_q2),r2(at.c_k2)
                    sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),
                                    kf.float())/128
                    sc2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),
                                     k2.float())/128
                    tril=torch.tril(torch.ones(T,T,device=DEV))
                    ar=torch.arange(T,device=DEV)
                    far=((ar[:,None]-ar[None,:])>=K).float()
                    mk=tril.clone().expand(B,T,T).clone()
                    if mode=='window':
                        mk=mk*((ar[:,None]-ar[None,:])<K).float()
                    elif mode=='block_struct':
                        blk=(sm[:,None,:].float()*far[None])
                        mk=mk*(1-blk)
                    elif mode=='block_content':
                        blk=(cm[:,None,:].float()*far[None])
                        mk=mk*(1-blk)
                    pat=(sc*sc2)
                    z=torch.einsum('bhqk,bkhd->bhqd',pat*tril,
                                   vm.float())
                    zm=torch.einsum('bqk,bkd->bqd',
                                    pat[:,HD]*mk,vm[:,:,HD].float())
                    z[:,HD]=zm
                    return (at.c_proj(z.transpose(1,2).contiguous()
                            .view(B,T,-1).to(X.dtype)),v1r)
                hs.append(at.register_forward_hook(fh))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x
            v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            ce=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                               reduction='none').view(B,T).cpu()
            for b in range(B):
                toks=fresh[rid[b],:T].tolist(); last={}
                mk2=torch.zeros(T,dtype=torch.bool)
                for q in range(T):
                    t=toks[q]
                    if t in last and last[t]+1<q and q>=8:
                        mk2[q]=True
                    last[t]=q
                tot+=float(ce[b][mk2].sum()); n+=int(mk2.sum())
            for h in hs: h.remove()
        return tot/max(n,1)
    base=run(None)
    res={a:round(run(a)-base,5) for a in
         ('window','block_struct','block_content')}
    print('dCE at match (structured rows):',res,flush=True)
    w=res['window']
    pa=res['block_struct']>=0.5*w
    pb=res['block_content']<=0.25*w
    cs,cc=counts['struct'],counts['content']
    pc=abs(cs-cc)<=0.20*max(cs,cc,1)
    out={'baseline_match_ce':round(base,4),'dce':res,
         'blocked_counts':{'struct':cs,'content':cc},
         'share_struct':round(res['block_struct']/max(w,1e-6),3),
         'share_content':round(res['block_content']/max(w,1e-6),3),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print(f"shares of window damage: struct "
          f"{out['share_struct']}, content {out['share_content']}"
          f" | blocked counts {cs} vs {cc}")
    for nm,v in (('a','blocking structure reproduces >=50%'),
                 ('b','blocking content reproduces <=25%'),
                 ('c','the two arms block comparable counts')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
