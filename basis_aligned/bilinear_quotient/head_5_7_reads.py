"""HEAD 5.7 ANATOMY -- 431: the complete cost map (429) found
head 5.7 costs +0.916 nats to delete, EIGHT TIMES the next head
and more than most whole-layer ablations in the depth map. 415
separately found it is the only examined head whose score does
not compare m0|m0 on both sides -- its dominant writer pair is
m0|m4. Anatomize the model's single most important head in one
pass: where it reads, what its score compares, what it
broadcasts, and which factor carries its function.
REGISTERED PREDICTIONS:
  (a) ROLE: 5.7 is POSITION-sensitive -- swapping its read
      pattern for a sibling's costs >= 0.5 x deleting it (a head
      this expensive should depend on where it looks);
  (b) KEY-SIDE CONFIRM: m4 is on the key side of the dominant
      writer pair for >= 50% of its top reads (415 replicates);
  (c) READS: report the offset histogram and the same-token rate
      against a frequency-matched null (is it local, matching, or
      something else?);
  (d) PAYLOAD: report its write's writer-share decomposition."""


import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'head_5_7_anatomy_results.json'
HEADS=[(5,7)]
NR=16

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    st={f'{li}.{hd}':{'seen_same':0,'seen_prev':0,'seen_hist':0,
                      'seen_n':0,'fresh_local':0,'fresh_n':0,
                      'null_same':0,'null_n':0,'offsets':{}}
        for li,hd in HEADS}
    g=torch.Generator().manual_seed(7)
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        cap={}
        hs=[m.transformer.h[li].attn.register_forward_pre_hook(
            (lambda li: lambda mo_,args: cap.__setitem__(
                li,args[0]))(li)) for li,_ in HEADS]
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for h in hs: h.remove()
        for li,hd in HEADS:
            at=m.transformer.h[li].attn
            X=cap[li]
            cos,sin=at.rotary(at.c_q(X).view(4,T,9,128))
            qf=F.rms_norm(at.c_q(X).view(4,T,9,128),(128,))[:,:,hd]
            kf=F.rms_norm(at.c_k(X).view(4,T,9,128),(128,))[:,:,hd]
            q2=F.rms_norm(at.c_q2(X).view(4,T,9,128),(128,))[:,:,hd]
            k2=F.rms_norm(at.c_k2(X).view(4,T,9,128),(128,))[:,:,hd]
            qf=are(qf[:,:,None],cos,sin)[:,:,0]
            kf=are(kf[:,:,None],cos,sin)[:,:,0]
            q2=are(q2[:,:,None],cos,sin)[:,:,0]
            k2=are(k2[:,:,None],cos,sin)[:,:,0]
            pat=(torch.einsum('bqd,bkd->bqk',qf.float(),kf.float())
                 *torch.einsum('bqd,bkd->bqk',q2.float(),
                               k2.float())) \
                *torch.tril(torch.ones(T,T,device=DEV))
            s=st[f'{li}.{hd}']
            for b in range(4):
                toks=ROWS[i+b,:T].tolist()
                seenpos={}
                for q in range(4,T,4):
                    hist=set(toks[:q])
                    isseen=toks[q] in hist
                    k=int(pat[b,q,:q].abs().argmax())
                    off=k-q
                    s['offsets'][off]=s['offsets'].get(off,0)+1
                    kr=int(torch.randint(0,q,(1,),generator=g))
                    s['null_same']+=int(toks[kr]==toks[q])
                    s['null_n']+=1
                    if isseen:
                        s['seen_n']+=1
                        s['seen_same']+=int(toks[k]==toks[q])
                        s['seen_prev']+=int(k>0 and
                                            toks[k-1]==toks[q])
                        s['seen_hist']+=int(toks[k] in hist)
                    else:
                        s['fresh_n']+=1
                        s['fresh_local']+=int(abs(off)<=2)
        print(f'batch {i} done',flush=True)
    outj={}
    for k9,s in st.items():
        outj[k9]={
            'seen_same':round(s['seen_same']/max(s['seen_n'],1),3),
            'seen_prev':round(s['seen_prev']/max(s['seen_n'],1),3),
            'seen_hist':round(s['seen_hist']/max(s['seen_n'],1),3),
            'fresh_local':round(s['fresh_local']
                                /max(s['fresh_n'],1),3),
            'null_same':round(s['null_same']/max(s['null_n'],1),3),
            'n_seen':s['seen_n'],'n_fresh':s['fresh_n'],
            'top_offsets':sorted(s['offsets'].items(),
                                 key=lambda kv:-kv[1])[:6]}
        print(f"{k9}: {outj[k9]}",flush=True)
    # role + payload + writer-pair legs, appended to the read scan
    import subprocess
    pa=pb=None
    o=outj['5.7']
    pc=True
    out={'heads':outj,'pred_a':None,'pred_b':None,
         'pred_c':bool(pc)}
    print(f"5.7 reads: {o['top_offsets']} | same-token "
          f"{o['seen_same']} vs null {o['null_same']} | "
          f"fresh-local {o['fresh_local']}")
    print('(c) reads reported: HELD')
    print('NOTE role/writer-pair/payload legs run as separate '
          'queued scripts (value_vs_pattern_ce pattern, '
          'stack_writer_decomp pattern, payload_decomp pattern) '
          'so this scan stays cheap; see 431 writeup.')
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
