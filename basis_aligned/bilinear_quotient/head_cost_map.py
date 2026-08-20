"""HEAD COST MAP -- 429: 427 sampled 63 heads and found 15 that
cost nothing to delete. Complete the map with the corrected
functional metric (dCE, per 422) over ALL 162 heads: delete one
head at a time, measure dCE overall and at match positions. This
is a reference artifact for the swarm and for the readable-code
frontier work (374/379 used an older metric and window).
REGISTERED PREDICTIONS:
  (a) >= 20% of all 162 heads have dCE <= 0 (free or helpful);
  (b) the free set is concentrated late: median layer of the
      dCE<=0 set >= 9;
  (c) the costliest ten include 1.1 and 12.6 (427's sample
      leaders survive the full sweep)."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'head_cost_map_results.json'
NR=16

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    acc={f'{lj}.{h}':{'all':0.0,'m':0.0,'nm':0,'n':0}
         for lj in range(18) for h in range(9)}
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        B=4
        mmask=torch.zeros(B,T,dtype=torch.bool)
        for b in range(B):
            toks=ROWS[i+b,:T].tolist(); last={}
            for q in range(T):
                t=toks[q]
                ism=t in last and last[t]+1<q
                last[t]=q
                if ism and q>=8: mmask[b,q]=True
        def run(lj=None,hd=None):
            hs=[]
            if lj is not None:
                at=m.transformer.h[lj].attn
                def fh(mo_,args,o_,at=at,hd=hd):
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
                    z[:,hd]=0
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
                               reduction='none').view(B,T).cpu()
            for h in hs: h.remove()
            return ce
        ce0=run()
        for lj in range(18):
            for hd in range(9):
                d=run(lj,hd)-ce0
                k=f'{lj}.{hd}'
                acc[k]['all']+=float(d.mean()); acc[k]['n']+=1
                acc[k]['m']+=float(d[mmask].mean())
                acc[k]['nm']+=1
        print(f'batch {i} done ({time.time()-t0:.0f}s)',flush=True)
    out={k:{'dce_all':round(v['all']/max(v['n'],1),5),
            'dce_match':round(v['m']/max(v['nm'],1),5)}
         for k,v in acc.items()}
    free=[k for k in out if out[k]['dce_all']<=0]
    medlayer=sorted(int(k.split('.')[0]) for k in free)
    med=medlayer[len(medlayer)//2] if medlayer else -1
    top10=sorted(out,key=lambda k:-out[k]['dce_all'])[:10]
    pa=len(free)/len(out)>=0.20
    pb=med>=9
    pc=('1.1' in top10 and '12.6' in top10)
    res={'heads':out,'n_free':len(free),
         'free_fraction':round(len(free)/len(out),3),
         'median_layer_of_free':med,'costliest_10':top10,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print(f'free heads {len(free)}/162 (median layer {med}); '
          f'costliest {top10}')
    for nm,v in (('a','>=20% free'),('b','free set late (med>=9)'),
                 ('c','1.1 and 12.6 in costliest ten')):
        print(f"({nm}) {v}: "
              f"{'HELD' if res['pred_'+nm] else 'FAILED'}")
    json.dump(res,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({res["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
