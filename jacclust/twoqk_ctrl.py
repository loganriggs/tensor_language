"""tick 28b: matched-DIMENSION control for the two-QK gain. Does [q1;q2] beat [q1;noise] (same 2*hd dim)?
If yes, the second query matrix carries signal beyond mere dimensionality. Positive-gain heads only."""
import sys, torch, json, numpy as np, torch.nn.functional as F
from sklearn.cluster import KMeans
sys.path.insert(0,"/workspace/tensor_language")
from huggingface_hub import hf_hub_download
import jacclust.tt_model as TT
DEV="cuda"; torch.set_default_dtype(torch.float32)
repo="Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd"
cfg=json.load(open(hf_hub_download(repo,"config.json"))); cfg.pop("step",None)
m=TT.GPT(TT.GPTConfig(**cfg)).to(DEV).eval()
m.load_state_dict(torch.load(hf_hub_download(repo,"pytorch_model.bin"),map_location=DEV,weights_only=True))
from transformers import AutoTokenizer; tok=AutoTokenizer.from_pretrained("gpt2")
import datasets
K=8; nh,hd=m.transformer.h[0].attn.n_head,m.transformer.h[0].attn.head_dim
def apply_rot(x,c,s):
    d=x.shape[-1]//2; x1,x2=x[...,:d],x[...,d:]; return torch.cat([x1*c+x2*s,-x1*s+x2*c],-1)
def forward_ce(idx,tgt,LAYER,HEAD,qov1=None,qov2=None):
    x=m.transformer.wte(idx); x=F.rms_norm(x,(x.size(-1),)); x0=x; v1=None
    for li,blk in enumerate(m.transformer.h):
        x=blk.lambdas[0]*x+blk.lambdas[1]*x0; a=blk.attn; h=F.rms_norm(x,(x.size(-1),)); B,T,C=h.shape
        q=a.c_q(h).view(B,T,nh,hd); k=a.c_k(h).view(B,T,nh,hd); q2=a.c_q2(h).view(B,T,nh,hd); k2=a.c_k2(h).view(B,T,nh,hd); v=a.c_v(h).view(B,T,nh,hd)
        if v1 is None: v1=v
        v=(1-a.lamb)*v+a.lamb*v1.view_as(v)
        cos,sin=a.rotary(q); cos=cos.float(); sin=sin.float()
        q=F.rms_norm(q,(hd,)); k=F.rms_norm(k,(hd,)); q2=F.rms_norm(q2,(hd,)); k2=F.rms_norm(k2,(hd,))
        if li==LAYER:
            if qov1 is not None: q=q.clone(); q[:,:,HEAD,:]=qov1
            if qov2 is not None: q2=q2.clone(); q2[:,:,HEAD,:]=qov2
        q=apply_rot(q,cos,sin); k=apply_rot(k,cos,sin); q2=apply_rot(q2,cos,sin); k2=apply_rot(k2,cos,sin)
        sc=torch.einsum("bqhd,bkhd->bhqk",q,k)/hd; sc2=torch.einsum("bqhd,bkhd->bhqk",q2,k2)/hd
        mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool)); pat=(sc*sc2).masked_fill(~mask,0.0)
        y=torch.einsum("bhqk,bkhd->bqhd",pat,v).reshape(B,T,C); y=a.c_proj(y); x=x+y
        x=x+blk.mlp(F.rms_norm(x,(x.size(-1),)))
    x=F.rms_norm(x,(x.size(-1),)); logits=30*torch.tanh(m.lm_head(x)/30)
    return F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(), tgt.reshape(-1))
ds=datasets.load_dataset("NeelNanda/pile-10k",split="train",streaming=True); docs=[]
for d in ds:
    t=tok(d["text"])["input_ids"]
    if len(t)>=66: docs.append(t[:66])
    if len(docs)>=24: break
toks=torch.tensor(docs,device=DEV); idx=toks[:,:-1].contiguous(); tgt=toks[:,1:].contiguous()
B,Tm=idx.shape; valid=[(b,q) for b in range(B) for q in range(2,Tm)]
def readouts(L,H):
    a=m.transformer.h[L].attn
    st={}; hk=m.transformer.h[L].register_forward_hook(lambda mm,i,o: st.__setitem__("r",i[0].detach()))
    with torch.no_grad(): m(idx,tgt)
    hk.remove(); Xn=F.rms_norm(st["r"],(st["r"].shape[-1],))
    q1=F.rms_norm((Xn@a.c_q.weight.detach().T).view(B,Tm,nh,hd)[:,:,H,:],(hd,)).detach()
    q2=F.rms_norm((Xn@a.c_q2.weight.detach().T).view(B,Tm,nh,hd)[:,:,H,:],(hd,)).detach()
    return q1,q2
def cmeans(feat,L2): return [torch.stack([feat[b,q] for (b,q) in valid if L2[b,q]==c]).mean(0) if (L2==c).any() else feat.mean((0,1)) for c in range(K)]
def override(feat,L2,mns):
    ov=feat.clone()
    for (b,q) in valid:
        c=L2[b,q]
        if c>=0: ov[b,q]=mns[c]
    return ov
def labels(mat,seed):
    l=KMeans(K,n_init=6,random_state=seed).fit_predict(mat); L2=np.full((B,Tm),-1)
    for i,(b,q) in enumerate(valid): L2[b,q]=l[i]
    return L2
def sw(L,H,q1,q2,L2):
    m1,m2=cmeans(q1,L2),cmeans(q2,L2)
    cw=float(forward_ce(idx,tgt,L,H,override(q1,L2,m1),override(q2,L2,m2)))
    cs=float(forward_ce(idx,tgt,L,H,override(q1,L2,[m1[(c+1)%K] for c in range(K)]),override(q2,L2,[m2[(c+1)%K] for c in range(K)])))
    return cs-cw
print(f"{'head':7s} {'q1':>8s} {'q2':>8s} {'[q1;q2]':>9s} {'[q1;noise]':>11s} {'[q2;noise]':>11s} {'q2 real?':>9s}")
for (L,H) in [(0,2),(6,3),(8,3),(3,3)]:
    q1,q2=readouts(L,H)
    q1n=F.normalize(torch.stack([q1[b,q] for (b,q) in valid]),dim=1); q2n=F.normalize(torch.stack([q2[b,q] for (b,q) in valid]),dim=1)
    def feat_matched(base,seed):
        g=torch.Generator(device=DEV).manual_seed(1000+seed); noise=F.normalize(torch.randn(base.shape,generator=g,device=DEV),dim=1)
        return F.normalize(torch.cat([base,noise],1),dim=1).cpu().numpy()
    e1=np.array([sw(L,H,q1,q2,labels(q1n.cpu().numpy(),s)) for s in range(5)])
    e2=np.array([sw(L,H,q1,q2,labels(q2n.cpu().numpy(),s)) for s in range(5)])
    eb=np.array([sw(L,H,q1,q2,labels(torch.cat([q1n,q2n],1).cpu().numpy(),s)) for s in range(5)])
    ec1=np.array([sw(L,H,q1,q2,labels(feat_matched(q1n,s),s)) for s in range(5)])
    ec2=np.array([sw(L,H,q1,q2,labels(feat_matched(q2n,s),s)) for s in range(5)])
    verdict = "YES" if eb.mean()>max(ec1.mean(),ec2.mean())+0.0005 else "no"
    print(f"L{L}H{H:<4d} {e1.mean():+8.4f} {e2.mean():+8.4f} {eb.mean():+9.4f} {ec1.mean():+11.4f} {ec2.mean():+11.4f} {verdict:>9s}",flush=True)
print("[q1;q2] > [q1;noise] and > [q2;noise] => the SECOND matrix carries signal beyond added dimensions.")
print("DONE")
