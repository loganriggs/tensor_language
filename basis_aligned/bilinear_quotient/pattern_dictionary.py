"""PATTERN DICTIONARY -- user direction: decompose attention as
small combinations of shared archetypes. The motif census assigned
each head ONE label (self/prev/ind/first/diffuse); this measures
whether realized patterns are sparse LINEAR COMBINATIONS of a
6-archetype dictionary: self (k=q), prev (k=q-1), first (k<=1),
match (k=j+1 after the previous occurrence of the query token),
nl-anchor (k at newlines), uniform (1/q causal). Per head, least
squares of the realized pattern on the archetype masks over 24
census rows; R^2 per head.
REGISTERED PREDICTIONS:
  (a) >=120/162 heads reach R^2 >= 0.7 with the 6-atom dictionary;
  (b) for heads the motif census labeled self/prev/ind/first, the
      dominant atom agrees with the census label >=80% of the time;
  (c) the lowest-R^2 heads are listed with their top contexts --
      the target list for NEW archetypes."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'pattern_dictionary_results.json'
NR=24

@torch.no_grad()
def main():
    t0=time.time()
    mod2=sys.modules[type(m.transformer.h[0].attn).__module__]
    are=mod2.apply_rotary_emb
    ROWS=cl.rows()[:NR]
    tril=torch.tril(torch.ones(T,T,device=DEV))
    # archetype masks per row
    def masks(row):
        toks=row[:T].tolist()
        A=torch.zeros(6,T,T,device=DEV)
        A[0]=torch.eye(T,device=DEV)                     # self
        A[1,1:]=torch.eye(T,device=DEV)[:-1]             # prev
        A[2,:,0]=1; A[2,1:,1]=1                          # first
        last={}
        for q in range(T):
            t=toks[q]
            if t in last and last[t]+1<=q: A[3,q,last[t]+1]=1   # match
            last[t]=q
        nl=[k for k in range(T) if chr(10) in cl.d1(toks[k])]
        for q in range(T):
            ks=[k for k in nl if k<=q]
            if ks: A[4,q,ks]=1.0/len(ks)                 # nl-anchor
            A[5,q,:q+1]=1.0/(q+1)                        # uniform
        return A*tril[None]
    G=torch.zeros(162,6,6,device=DEV); C=torch.zeros(162,6,device=DEV)
    YY=torch.zeros(162,device=DEV)
    cap={}
    def mkpre(li):
        def h(mo_,args):
            cap['x']=args[0]; cap['v1']=args[1]
        return h
    for ri in range(NR):
        bb=ROWS[ri][None,:257].to(DEV)
        idx=bb[:,:-1].contiguous()
        A=masks(ROWS[ri])
        Af=A.reshape(6,-1)
        Gm=Af@Af.T
        hs=[m.transformer.h[li].attn.register_forward_pre_hook(
            mkpre(li)) for li in range(18)]
        # run and recompute patterns per layer
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for li,blkm in enumerate(m.transformer.h):
            x,v1=blkm(x,v1,x0)
            at=m.transformer.h[li].attn
            X=cap['x']; v1i=cap['v1']
            cos,sin=at.rotary(at.c_q(X).view(1,T,9,128))
            qf=F.rms_norm(at.c_q(X).view(1,T,9,128),(128,))
            kf=F.rms_norm(at.c_k(X).view(1,T,9,128),(128,))
            qf,kf=are(qf,cos,sin),are(kf,cos,sin)
            q2=F.rms_norm(at.c_q2(X).view(1,T,9,128),(128,))
            k2=F.rms_norm(at.c_k2(X).view(1,T,9,128),(128,))
            q2,k2=are(q2,cos,sin),are(k2,cos,sin)
            sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),kf.float())/128
            s2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),k2.float())/128
            pat=(sc*s2)[0]*tril[None]
            for hd in range(9):
                gi9=li*9+hd
                pf=pat[hd].reshape(-1)
                G[gi9]+=Gm; C[gi9]+=Af@pf; YY[gi9]+=pf@pf
        for h in hs: h.remove()
        if ri%6==0: print(f'row {ri}/{NR}',flush=True)
    R2={}; DOM={}
    for gi9 in range(162):
        li,hd=gi9//9,gi9%9
        Gm=G[gi9]+1e-4*torch.eye(6,device=DEV)
        w=torch.linalg.solve(Gm,C[gi9])
        r2=float((C[gi9]@w)/YY[gi9].clamp_min(1e-8))
        R2[f'{li}.{hd}']=round(r2,3)
        DOM[f'{li}.{hd}']=int((w.abs()*torch.diag(G[gi9]).sqrt())
                              .argmax())
    ATOMS=['self','prev','first','match','nl','uniform']
    n07=sum(1 for v in R2.values() if v>=0.7)
    pa=n07>=120
    mt=json.load(open(PT+'attn_motifs3_results.json'))['motif_table']
    lab={'self':'self','prev':'prev','ind':'match','first':'first'}
    agree=0; tot=0
    for li,hd,mo,fr in mt:
        if mo in lab:
            tot+=1
            if ATOMS[DOM[f'{li}.{hd}']]==lab[mo]: agree+=1
    pb=tot>0 and agree/tot>=0.8
    worst=sorted(R2.items(),key=lambda kv:kv[1])[:10]
    out={'n_r2_ge_07':n07,'r2':R2,
         'dominant':{k:ATOMS[v] for k,v in DOM.items()},
         'motif_agree':f'{agree}/{tot}','worst':worst,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':True}
    print(f'R2>=0.7: {n07}/162 | motif agreement {agree}/{tot}')
    print('worst heads:',worst)
    print(f"(a) >=120 heads R2>=0.7: {'HELD' if pa else 'FAILED'}")
    print(f"(b) motif agreement >=80%: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
