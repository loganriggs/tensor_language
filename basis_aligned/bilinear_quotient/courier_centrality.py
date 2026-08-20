"""COURIER CENTRALITY -- 417: a6.h3 is the deep induction trio's
code courier (407/408: deleting it shifts 7.3/8.3/8.4's reads
19-29%, early band 0%). How central is it to the REST of the
stack? Delete a6.h3 and measure every downstream head's top-read
shift rate (layers 7-17, all 9 heads each = 99 heads); control:
a6.h0 (same layer, no relay role).
REGISTERED PREDICTIONS:
  (a) SPECIFIC COURIER: >=80% of downstream heads shift < 5% of
      their top reads;
  (b) the strongly-shifted set (>10%) is enriched in match-class
      heads: >=half of them have census profile match-share
      >=0.1 (head_read_census), vs base rate < 0.2 across
      downstream heads;
  (c) control: a6.h0 deletion shifts <5% for >=95% of heads."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'courier_centrality_results.json'
NR=16
KILLS={'a6h3':(6,3),'a6h0':(6,0)}
DOWN=[(li,hd) for li in range(7,18) for hd in range(9)]

@torch.no_grad()
def main():
    t0=time.time()
    prof=json.load(open(PT+'head_read_census_results.json')) \
        ['profiles']
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    shifts={a:{f'{li}.{hd}':[0,0] for li,hd in DOWN}
            for a in KILLS}
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        B=4
        def run(kill=None):
            caps={}
            hs=[]
            for li in range(7,18):
                def ph(mo_,args,li=li): caps[li]=args[0]
                hs.append(m.transformer.h[li].attn
                          .register_forward_pre_hook(ph))
            if kill is not None:
                kl,kh=kill
                at=m.transformer.h[kl].attn
                def fh(mo_,args,o_,at=at,kh=kh):
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
                    sc2=torch.einsum('bqhd,bkhd->bhqk',
                                     q2.float(),k2.float())/128
                    pat=(sc*sc2)*torch.tril(
                        torch.ones(T,T,device=DEV))
                    z=torch.einsum('bhqk,bkhd->bhqd',pat,
                                   vm.float())
                    z[:,kh]=0
                    ynew=at.c_proj(z.transpose(1,2).contiguous()
                                   .view(B,T,-1).to(X.dtype))
                    return (ynew,v1r)
                hs.append(at.register_forward_hook(fh))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x
            v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            argm={}
            for li in range(7,18):
                at=m.transformer.h[li].attn
                X=caps[li]
                cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
                qf=F.rms_norm(at.c_q(X).view(B,T,9,128),(128,))
                kf=F.rms_norm(at.c_k(X).view(B,T,9,128),(128,))
                qf,kf=are(qf,cos,sin),are(kf,cos,sin)
                q2=F.rms_norm(at.c_q2(X).view(B,T,9,128),(128,))
                k2=F.rms_norm(at.c_k2(X).view(B,T,9,128),(128,))
                q2,k2=are(q2,cos,sin),are(k2,cos,sin)
                sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),
                                kf.float())
                sc2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),
                                 k2.float())
                pat=(sc*sc2)*torch.tril(torch.ones(T,T,device=DEV))
                mask=torch.tril(torch.ones(T,T,device=DEV))==0
                pat=pat.abs().masked_fill(mask[None,None],-1)
                argm[li]=pat.argmax(-1).cpu()   # B,9,T
            for h in hs: h.remove()
            return argm
        a0=run(None)
        for a,kill in KILLS.items():
            a1=run(kill)
            for li,hd in DOWN:
                sh=shifts[a][f'{li}.{hd}']
                d0=a0[li][:,hd,8:]; d1=a1[li][:,hd,8:]
                sh[0]+=int((d0!=d1).sum()); sh[1]+=d0.numel()
        print(f'batch {i} done',flush=True)
    out={}
    for a in KILLS:
        rates={k:round(v[0]/max(v[1],1),4)
               for k,v in shifts[a].items()}
        out[a]={'rates':rates,
                'frac_under_5pct':round(
                    sum(r<0.05 for r in rates.values())
                    /len(rates),3),
                'strong':{k:r for k,r in rates.items() if r>0.10}}
        print(f"{a}: under5% {out[a]['frac_under_5pct']} strong "
              f"{out[a]['strong']}",flush=True)
    strong=list(out['a6h3']['strong'])
    def mshare(k):
        return prof.get(k,{}).get('profile',{}).get('match',0)
    enr=(sum(mshare(k)>=0.1 for k in strong)/len(strong)
         if strong else 0.0)
    base=sum(mshare(f'{li}.{hd}')>=0.1 for li,hd in DOWN)/len(DOWN)
    pa=out['a6h3']['frac_under_5pct']>=0.8
    pb=enr>=0.5 and base<0.2
    pc=out['a6h0']['frac_under_5pct']>=0.95
    out.update({'strong_match_enrichment':round(enr,3),
                'base_match_rate':round(base,3),
                'pred_a':bool(pa),'pred_b':bool(pb),
                'pred_c':bool(pc)})
    for nm,v in (('a','>=80% of heads shift <5%'),
                 ('b','strong set match-enriched'),
                 ('c','control >=95% under 5%')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
