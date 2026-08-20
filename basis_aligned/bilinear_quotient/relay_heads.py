"""RELAY HEADS -- 406: who does the relaying? The relay-depth
story (402) says attention moves the MLP identity code; classic
induction composition says PREVIOUS-TOKEN heads should be the
movers (code at k-1 relayed to k is exactly what a coincidence
read of "my predecessor's code" needs). Test: for each band head,
find its top relay LAYER (per-layer k=1 lifts, as in 398), then
restrict that layer's relay write to ONE head at a time and
correlate per-head relay lifts with the head-read census's 'prev'
share (head_read_census profiles; independent measurement).
REGISTERED PREDICTIONS:
  (a) pooled over all (band-head, relay-layer) pairs: Spearman
      rank correlation between per-head relay lift and census
      prev-share >= 0.4;
  (b) the single top relay head is prev-modal (census modal class
      'prev') for >= half the band heads tested;
  (c) null: the same pooled correlation against the census
      'first' share is <= 0.1 (enrichment is motif-specific)."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'relay_heads_results.json'
NR=16
BAND=[(2,5),(3,5),(3,8),(5,5),(6,5),(7,3),(8,3),(8,4)]

def spearman(a,b):
    a=torch.tensor(a); b=torch.tensor(b)
    ra=a.argsort().argsort().float()
    rb=b.argsort().argsort().float()
    return float(torch.corrcoef(torch.stack([ra,rb]))[0,1])

@torch.no_grad()
def main():
    t0=time.time()
    prof=json.load(open(PT+'head_read_census_results.json')) \
        ['profiles']
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    maxli=max(l for l,_ in BAND)
    # stage stats: hits[(band, arm)] where arm='ladder' or
    # ('layer',j) or ('head',j,h)
    hits={}
    def bump(k9,ok):
        st=hits.setdefault(k9,[0,0]); st[0]+=int(ok); st[1]+=1
    caps=[]
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        E=F.rms_norm(m.transformer.wte(idx),(D,)).float()
        mout={}; attin={}; pre={}
        hs=[]
        for lj in range(maxli):
            def mh(mo,i_,o_,lj=lj): mout[lj]=o_.detach().float()
            def phj(mo_,args,lj=lj): pre[lj]=(args[0],args[1])
            hs.append(m.transformer.h[lj].mlp
                      .register_forward_hook(mh))
            hs.append(m.transformer.h[lj].attn
                      .register_forward_pre_hook(phj))
        for li,hd in BAND:
            def ph(mo_,args,li=li): attin[li]=args[0]
            hs.append(m.transformer.h[li].attn
                      .register_forward_pre_hook(ph))
        x=E.to(m.transformer.wte.weight.dtype); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for h in hs: h.remove()
        B=4
        # pure ladder residuals (value source) and per-layer,
        # per-head relay z blocks
        lad={}; xr=E.clone()
        for lj in range(maxli+1):
            blk=m.transformer.h[lj]
            lam=blk.lambdas.detach().float()
            xr=lam[0]*xr+lam[1]*E
            lad[lj]=xr.clone()
            if lj<maxli: xr=xr+mout[lj]
        ZH={}   # lj -> (B,9,T,128) relay z with ladder values
        for lj in range(maxli):
            at=m.transformer.h[lj].attn
            Xj,v1j=pre[lj]
            Xs=F.rms_norm(lad[lj],(D,)) \
                .to(m.transformer.wte.weight.dtype)
            v=at.c_v(Xs).view(B,T,9,128)
            vm=v if v1j is None else \
                (1-at.lamb)*v+at.lamb*v1j.view_as(v)
            cos,sin=at.rotary(at.c_q(Xj).view(B,T,9,128))
            qf=F.rms_norm(at.c_q(Xj).view(B,T,9,128),(128,))
            kf=F.rms_norm(at.c_k(Xj).view(B,T,9,128),(128,))
            qf,kf=are(qf,cos,sin),are(kf,cos,sin)
            q2=F.rms_norm(at.c_q2(Xj).view(B,T,9,128),(128,))
            k2=F.rms_norm(at.c_k2(Xj).view(B,T,9,128),(128,))
            q2,k2=are(q2,cos,sin),are(k2,cos,sin)
            sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),
                            kf.float())/128
            sc2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),
                             k2.float())/128
            patm=(sc*sc2)*torch.tril(torch.ones(T,T,device=DEV))
            ZH[lj]=torch.einsum('bhqk,bkhd->bhqd',patm,vm.float())
        def relay_write(lj,headsel):
            at=m.transformer.h[lj].attn
            z=ZH[lj]
            if headsel is not None:
                zz=torch.zeros_like(z); zz[:,headsel]=z[:,headsel]
                z=zz
            return at.c_proj(z.transpose(1,2).contiguous()
                             .view(B,T,-1)
                             .to(m.transformer.wte.weight.dtype)) \
                .float()
        def code_at(li,insertj=None,headsel=None):
            xr=E.clone()
            for lj in range(li):
                blk=m.transformer.h[lj]
                lam=blk.lambdas.detach().float()
                xr=lam[0]*xr+lam[1]*E
                if lj==insertj:
                    xr=xr+relay_write(lj,headsel)
                xr=xr+mout[lj]
            lam=m.transformer.h[li].lambdas.detach().float()
            return F.rms_norm(lam[0]*xr+lam[1]*E,(D,))
        for li,hd in BAND:
            at=m.transformer.h[li].attn
            X=attin[li]
            a9,b9=hd*128,(hd+1)*128
            cos,sin=at.rotary(at.c_q(X).view(4,T,9,128))
            qf=F.rms_norm(at.c_q(X).view(4,T,9,128),(128,))[:,:,hd]
            kf=F.rms_norm(at.c_k(X).view(4,T,9,128),(128,))[:,:,hd]
            q2=F.rms_norm(at.c_q2(X).view(4,T,9,128),(128,))[:,:,hd]
            k2=F.rms_norm(at.c_k2(X).view(4,T,9,128),(128,))[:,:,hd]
            qf=are(qf[:,:,None],cos,sin)[:,:,0]
            kf=are(kf[:,:,None],cos,sin)[:,:,0]
            q2=are(q2[:,:,None],cos,sin)[:,:,0]
            k2=are(k2[:,:,None],cos,sin)[:,:,0]
            tril=torch.tril(torch.ones(T,T,device=DEV))
            pat=(torch.einsum('bqd,bkd->bqk',qf.float(),kf.float())
                 *torch.einsum('bqd,bkd->bqk',q2.float(),
                               k2.float()))*tril
            def score(c,k9):
                fq1=F.rms_norm(c@at.c_q.weight.float()
                               [a9:b9].T,(128,))
                fk1=F.rms_norm(c@at.c_k.weight.float()
                               [a9:b9].T,(128,))
                fq2=F.rms_norm(c@at.c_q2.weight.float()
                               [a9:b9].T,(128,))
                fk2=F.rms_norm(c@at.c_k2.weight.float()
                               [a9:b9].T,(128,))
                fq1=are(fq1[:,:,None],cos,sin)[:,:,0]
                fk1=are(fk1[:,:,None],cos,sin)[:,:,0]
                fq2=are(fq2[:,:,None],cos,sin)[:,:,0]
                fk2=are(fk2[:,:,None],cos,sin)[:,:,0]
                fpat=(torch.einsum('bqd,bkd->bqk',fq1,fk1)
                      *torch.einsum('bqd,bkd->bqk',fq2,fk2))*tril
                for b in range(4):
                    toks=ROWS[i+b,:T].tolist(); last={}
                    for q in range(T):
                        t=toks[q]
                        ism=t in last and last[t]+1<q
                        last[t]=q
                        if not ism or q<8: continue
                        bump(k9,int(pat[b,q,:q].abs().argmax())
                             ==int(fpat[b,q,:q].abs().argmax()))
            key=f'{li}.{hd}'
            score(code_at(li),(key,'ladder'))
            for j in range(li):
                score(code_at(li,insertj=j),(key,'layer',j))
            for h9 in range(9):
                # head arms deferred to stage 2 -- but chains are
                # cheap, so compute now for ALL layers' top... too
                # many; head arms done for every layer j with
                # positive layer lift would explode. Compute head
                # arms for every j; selection happens at scoring.
                pass
        print(f'batch {i} (layer arms) done',flush=True)
        caps.append((i,mout.copy(),
                     {k:(v[0],v[1]) for k,v in pre.items()},
                     {k:v for k,v in attin.items()},
                     {k:v for k,v in ZH.items()},E))
    # stage 2: pick top relay layer per band head, run head arms
    top={}
    for li,hd in BAND:
        key=f'{li}.{hd}'
        base=hits[(key,'ladder')][0]/max(hits[(key,'ladder')][1],1)
        lifts={j:hits[(key,'layer',j)][0]
               /max(hits[(key,'layer',j)][1],1)-base
               for j in range(li)}
        j9=max(lifts,key=lifts.get)
        top[key]=(j9,round(base,3),
                  {j:round(v,3) for j,v in lifts.items()})
        print(f"{key}: base {base:.3f} top layer a{j9} "
              f"lift {lifts[j9]:.3f}",flush=True)
    for i,mout,pre,attin,ZH,E in caps:
        B=4
        def relay_write2(lj,headsel):
            at=m.transformer.h[lj].attn
            z=ZH[lj]
            zz=torch.zeros_like(z); zz[:,headsel]=z[:,headsel]
            return at.c_proj(zz.transpose(1,2).contiguous()
                             .view(B,T,-1)
                             .to(m.transformer.wte.weight.dtype)) \
                .float()
        for li,hd in BAND:
            key=f'{li}.{hd}'
            j9=top[key][0]
            at=m.transformer.h[li].attn
            X=attin[li]
            a9,b9=hd*128,(hd+1)*128
            cos,sin=at.rotary(at.c_q(X).view(4,T,9,128))
            qf=F.rms_norm(at.c_q(X).view(4,T,9,128),(128,))[:,:,hd]
            kf=F.rms_norm(at.c_k(X).view(4,T,9,128),(128,))[:,:,hd]
            q2=F.rms_norm(at.c_q2(X).view(4,T,9,128),(128,))[:,:,hd]
            k2=F.rms_norm(at.c_k2(X).view(4,T,9,128),(128,))[:,:,hd]
            qf=are(qf[:,:,None],cos,sin)[:,:,0]
            kf=are(kf[:,:,None],cos,sin)[:,:,0]
            q2=are(q2[:,:,None],cos,sin)[:,:,0]
            k2=are(k2[:,:,None],cos,sin)[:,:,0]
            tril=torch.tril(torch.ones(T,T,device=DEV))
            pat=(torch.einsum('bqd,bkd->bqk',qf.float(),kf.float())
                 *torch.einsum('bqd,bkd->bqk',q2.float(),
                               k2.float()))*tril
            for h9 in range(9):
                xr=E.clone()
                for lj in range(li):
                    blk=m.transformer.h[lj]
                    lam=blk.lambdas.detach().float()
                    xr=lam[0]*xr+lam[1]*E
                    if lj==j9: xr=xr+relay_write2(lj,h9)
                    xr=xr+mout[lj]
                lam=m.transformer.h[li].lambdas.detach().float()
                c=F.rms_norm(lam[0]*xr+lam[1]*E,(D,))
                fq1=F.rms_norm(c@at.c_q.weight.float()
                               [a9:b9].T,(128,))
                fk1=F.rms_norm(c@at.c_k.weight.float()
                               [a9:b9].T,(128,))
                fq2=F.rms_norm(c@at.c_q2.weight.float()
                               [a9:b9].T,(128,))
                fk2=F.rms_norm(c@at.c_k2.weight.float()
                               [a9:b9].T,(128,))
                fq1=are(fq1[:,:,None],cos,sin)[:,:,0]
                fk1=are(fk1[:,:,None],cos,sin)[:,:,0]
                fq2=are(fq2[:,:,None],cos,sin)[:,:,0]
                fk2=are(fk2[:,:,None],cos,sin)[:,:,0]
                fpat=(torch.einsum('bqd,bkd->bqk',fq1,fk1)
                      *torch.einsum('bqd,bkd->bqk',fq2,fk2))*tril
                for b in range(4):
                    toks=ROWS[i+b,:T].tolist(); last={}
                    for q in range(T):
                        t=toks[q]
                        ism=t in last and last[t]+1<q
                        last[t]=q
                        if not ism or q<8: continue
                        bump((key,'head',j9,h9),
                             int(pat[b,q,:q].abs().argmax())
                             ==int(fpat[b,q,:q].abs().argmax()))
        print(f'batch {i} (head arms) done',flush=True)
    # score
    pooled_l=[]; pooled_p=[]; pooled_f=[]
    topmodal=[]
    outh={}
    for li,hd in BAND:
        key=f'{li}.{hd}'
        j9,base,lls=top[key]
        hlifts=[]
        for h9 in range(9):
            st=hits[(key,'head',j9,h9)]
            hlifts.append(round(st[0]/max(st[1],1)-base,3))
        prevs=[prof[f'{j9}.{h9}']['profile']['prev']
               for h9 in range(9)]
        firsts=[prof[f'{j9}.{h9}']['profile']['first']
                for h9 in range(9)]
        pooled_l+=hlifts; pooled_p+=prevs; pooled_f+=firsts
        besth=int(torch.tensor(hlifts).argmax())
        topmodal.append(prof[f'{j9}.{besth}']['modal']=='prev')
        outh[key]={'relay_layer':j9,'base':base,
                   'layer_lifts':lls,'head_lifts':hlifts,
                   'best_head':besth,
                   'best_head_modal':prof[f'{j9}.{besth}']
                   ['modal'],
                   'best_head_prev':prevs[besth]}
        print(f"{key}: a{j9} head lifts {hlifts} best h{besth} "
              f"(modal {outh[key]['best_head_modal']}, prev "
              f"{prevs[besth]})",flush=True)
    rho_p=spearman(pooled_l,pooled_p)
    rho_f=spearman(pooled_l,pooled_f)
    pa=rho_p>=0.4
    pb=sum(topmodal)>=len(topmodal)/2
    pc=rho_f<=0.1
    out={'heads':outh,'rho_prev':round(rho_p,3),
         'rho_first':round(rho_f,3),
         'top_prev_modal_frac':round(sum(topmodal)
                                     /len(topmodal),3),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"pooled rho(prev) {rho_p:.3f} rho(first) {rho_f:.3f} "
          f"| top-head prev-modal {sum(topmodal)}/{len(topmodal)}")
    for nm,v in (('a','rho(lift,prev) >= 0.4'),
                 ('b','top relay head prev-modal >= half'),
                 ('c','rho(lift,first) <= 0.1')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
