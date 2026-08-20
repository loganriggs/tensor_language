"""HEAD ROLE MAP -- 426: at a6.h3 and a4.h7, swapping a head's
READ PATTERN for a sibling's is nearly free while swapping its
VALUES costs as much as deletion (423), and the pattern only needs
the right SHAPE, not the right targets (424). Map that across the
stack: for every head at seven sampled layers, measure dCE under
zero / patswap(sibling) / valswap(sibling), and classify each head
POSITION-SENSITIVE (its read targets matter) or PAYLOAD-DOMINANT
(what it broadcasts matters). Sibling = head (h+1) mod 9, same
layer. Primary metric dCE (422 retired argmax shift).
REGISTERED PREDICTIONS:
  (a) MAJORITY PAYLOAD: >= 60% of tested heads have
      patswap dCE < valswap dCE;
  (b) INDUCTION EXCEPTION: the sampled induction-band heads
      (1.4, 2.5, 8.4 -- their job IS where they read) are
      position-sensitive, patswap >= 0.5 x zero;
  (c) sanity: zero dCE > 0 for >= 80% of heads (deleting a head
      usually costs something);
  (d) report: the layer profile of the payload/position split."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'head_role_map_results.json'
NR=16
LAYERS=[1,2,4,6,8,12,16]
IND=['1.4','2.5','8.4']

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    acc={f'{lj}.{h}':{a:{'s':0.0,'n':0} for a in
         ('zero','patswap','valswap')}
         for lj in LAYERS for h in range(9)}
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        B=4
        def run(lj=None,h1=None,mode=None):
            hs=[]
            if mode is not None:
                h0=(h1+1)%9
                at=m.transformer.h[lj].attn
                def fh(mo_,args,o_,at=at,h1=h1,h0=h0,mode=mode):
                    y,v1r=o_
                    X=args[0]
                    v1=args[1] if args[1] is not None else v1r
                    v=at.c_v(X).view(B,T,9,128)
                    vm=(1-at.lamb)*v+at.lamb*v1.view_as(v)
                    cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
                    qf=F.rms_norm(at.c_q(X).view(B,T,9,128),(128,))
                    kf=F.rms_norm(at.c_k(X).view(B,T,9,128),(128,))
                    qf,kf=are(qf,cos,sin),are(kf,cos,sin)
                    q2=F.rms_norm(at.c_q2(X).view(B,T,9,128),
                                  (128,))
                    k2=F.rms_norm(at.c_k2(X).view(B,T,9,128),
                                  (128,))
                    q2,k2=are(q2,cos,sin),are(k2,cos,sin)
                    sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),
                                    kf.float())/128
                    sc2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),
                                     k2.float())/128
                    pat=(sc*sc2)*torch.tril(
                        torch.ones(T,T,device=DEV))
                    z=torch.einsum('bhqk,bkhd->bhqd',pat,
                                   vm.float())
                    if mode=='zero': z[:,h1]=0
                    elif mode=='patswap':
                        z[:,h1]=torch.einsum('bqk,bkd->bqd',
                            pat[:,h0],vm[:,:,h1].float())
                    elif mode=='valswap':
                        z[:,h1]=torch.einsum('bqk,bkd->bqd',
                            pat[:,h1],vm[:,:,h0].float())
                    ynew=at.c_proj(z.transpose(1,2).contiguous()
                                   .view(B,T,-1).to(X.dtype))
                    return (ynew,v1r)
                hs.append(at.register_forward_hook(fh))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x
            v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            ce=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                               reduction='none')
            for h in hs: h.remove()
            return ce.mean().item(),ce.numel()
        base,_=run()
        for lj in LAYERS:
            for h in range(9):
                k=f'{lj}.{h}'
                for a in ('zero','patswap','valswap'):
                    v,_=run(lj,h,a)
                    acc[k][a]['s']+=(v-base); acc[k][a]['n']+=1
        print(f'batch {i} done ({time.time()-t0:.0f}s)',flush=True)
    out={}
    for k,arms in acc.items():
        out[k]={a:round(v['s']/max(v['n'],1),4)
                for a,v in arms.items()}
        out[k]['role']=('payload' if out[k]['patswap']
                        <out[k]['valswap'] else 'position')
    npay=sum(1 for k in out if out[k]['role']=='payload')
    frac=npay/len(out)
    pa=frac>=0.60
    pb=all(out[k]['patswap']>=0.5*out[k]['zero']
           for k in IND if k in out)
    pc=(sum(1 for k in out if out[k]['zero']>0)/len(out))>=0.80
    bylayer={}
    for lj in LAYERS:
        ks=[f'{lj}.{h}' for h in range(9)]
        bylayer[lj]=round(sum(1 for k in ks
                              if out[k]['role']=='payload')/9,2)
    res={'heads':out,'payload_fraction':round(frac,3),
         'payload_frac_by_layer':bylayer,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print(f'payload-dominant fraction {frac:.2f}; by layer '
          f'{bylayer}')
    for k in IND:
        if k in out: print(f'  induction {k}: {out[k]}')
    for nm,v in (('a','>=60% payload-dominant'),
                 ('b','induction heads position-sensitive'),
                 ('c','deletion costs >0 for >=80%')):
        print(f"({nm}) {v}: "
              f"{'HELD' if res['pred_'+nm] else 'FAILED'}")
    json.dump(res,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({res["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
