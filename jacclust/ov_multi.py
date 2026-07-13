"""tick 26b: confirm the OV/value-pathway causal cluster effect across HEADS (not just L6H0).
Value-swap intervention only, 5 seeds, controls = residual-x clusters + random. Reuse ov_squared setup."""
import sys, torch, json, numpy as np, torch.nn.functional as F
from sklearn.cluster import KMeans
sys.path.insert(0, "/workspace/tensor_language")
from huggingface_hub import hf_hub_download
import jacclust.tt_model as TT
DEV="cuda"; torch.set_default_dtype(torch.float32)
repo="Elriggs/gpt2-bilinear-sqrd-attn-12l-6h-768embd"
cfg=json.load(open(hf_hub_download(repo,"config.json"))); cfg.pop("step",None)
m=TT.GPT(TT.GPTConfig(**cfg)).to(DEV).eval()
m.load_state_dict(torch.load(hf_hub_download(repo,"pytorch_model.bin"),map_location=DEV,weights_only=True))
from transformers import AutoTokenizer; tok=AutoTokenizer.from_pretrained("gpt2")
import datasets
K=8
def apply_rot(x,c,s):
    d=x.shape[-1]//2; x1,x2=x[...,:d],x[...,d:]; return torch.cat([x1*c+x2*s,-x1*s+x2*c],-1)
def forward_ce(idx,tgt,LAYER,HEAD,vov=None):
    x=m.transformer.wte(idx); x=F.rms_norm(x,(x.size(-1),)); x0=x; v1=None
    for li,blk in enumerate(m.transformer.h):
        x=blk.lambdas[0]*x+blk.lambdas[1]*x0; a=blk.attn; h=F.rms_norm(x,(x.size(-1),)); B,T,C=h.shape; nh,hd=a.n_head,a.head_dim
        q=a.c_q(h).view(B,T,nh,hd); k=a.c_k(h).view(B,T,nh,hd); v=a.c_v(h).view(B,T,nh,hd)
        if v1 is None: v1=v
        v=(1-a.lamb)*v+a.lamb*v1.view_as(v)
        if li==LAYER and vov is not None: v=v.clone(); v[:,:,HEAD,:]=vov
        cos,sin=a.rotary(q); cos=cos.float(); sin=sin.float(); q=F.rms_norm(q,(hd,)); k=F.rms_norm(k,(hd,))
        q=apply_rot(q,cos,sin); k=apply_rot(k,cos,sin)
        sc=torch.einsum("bqhd,bkhd->bhqk",q,k)/hd; mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
        pat=sc.square().masked_fill(~mask,0.0); pat=pat/pat.sum(-1,keepdim=True).clamp_min(1e-9)
        y=torch.einsum("bhqk,bkhd->bqhd",pat,v).reshape(B,T,C); y=a.c_proj(y); x=x+y
        x=x+blk.mlp(F.rms_norm(x,(x.size(-1),)))
    x=F.rms_norm(x,(x.size(-1),)); logits=30*torch.tanh(m.lm_head(x)/30)
    return F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(), tgt.reshape(-1))
ds=datasets.load_dataset("NeelNanda/pile-10k",split="train",streaming=True); docs=[]
for d in ds:
    t=tok(d["text"])["input_ids"]
    if len(t)>=66: docs.append(t[:66])
    if len(docs)>=32: break
toks=torch.tensor(docs,device=DEV); idx=toks[:,:-1].contiguous(); tgt=toks[:,1:].contiguous()
Wv0=m.transformer.h[0].attn.c_v.weight.detach()
st0={}; hk0=m.transformer.h[0].register_forward_hook(lambda mm,i,o: st0.__setitem__("v0",F.rms_norm(i[0].detach(),(i[0].shape[-1],))))
with torch.no_grad(): m(idx,tgt)
hk0.remove()
def value_at(LAYER,HEAD):
    a=m.transformer.h[LAYER].attn; nh,hd=a.n_head,a.head_dim; dm=a.n_embd
    st={}; hk=m.transformer.h[LAYER].register_forward_hook(lambda mm,i,o: st.__setitem__("r",i[0].detach()))
    with torch.no_grad(): m(idx,tgt)
    hk.remove(); Xn=F.rms_norm(st["r"],(dm,))
    vh=(Xn@a.c_v.weight.detach().T).view(-1,st["r"].shape[1],nh,hd)[:,:,HEAD,:]
    v1h=(st0["v0"]@Wv0.T).view(-1,st["r"].shape[1],nh,hd)[:,:,HEAD,:]
    Vmix=((1-float(a.lamb))*vh+float(a.lamb)*v1h).detach(); return Xn.detach(), Vmix
B,Tm=idx.shape; valid=[(b,q) for b in range(B) for q in range(2,Tm)]
def override(feat,L2,mns):
    ov=feat.clone()
    for (b,q) in valid:
        c=L2[b,q]
        if c>=0: ov[b,q]=mns[c]
    return ov
def means(feat,L2):
    return [torch.stack([feat[b,q] for (b,q) in valid if L2[b,q]==c]).mean(0) if (L2==c).any() else feat.mean((0,1)) for c in range(K)]
def lab_from(mat,seed):
    l=KMeans(K,n_init=6,random_state=seed).fit_predict(mat); L2=np.full((B,Tm),-1)
    for i,(b,q) in enumerate(valid): L2[b,q]=l[i]
    return L2
def sw(LAYER,HEAD,feat,L2):
    mns=means(feat,L2); cw=float(forward_ce(idx,tgt,LAYER,HEAD,override(feat,L2,mns)))
    cs=float(forward_ce(idx,tgt,LAYER,HEAD,override(feat,L2,[mns[(c+1)%K] for c in range(K)]))); return cs-cw
print(f"{'head':7s} {'value-cos':>18s} {'resid-x':>14s} {'random':>12s}")
for (LAYER,HEAD) in [(2,4),(4,2),(6,0),(9,0),(11,3)]:
    Xn,Vmix=value_at(LAYER,HEAD)
    Vcos=F.normalize(torch.stack([Vmix[b,q] for (b,q) in valid]),dim=1).cpu().numpy()
    Xm=F.normalize(torch.stack([Xn[b,q] for (b,q) in valid]),dim=1).cpu().numpy()
    vv=[];xx=[];rr=[]
    for s in range(5):
        vv.append(sw(LAYER,HEAD,Vmix,lab_from(Vcos,s)))
        xx.append(sw(LAYER,HEAD,Vmix,lab_from(Xm,s)))
        rl=np.random.RandomState(s).randint(0,K,len(valid)); L2=np.full((B,Tm),-1)
        for i,(b,q) in enumerate(valid): L2[b,q]=rl[i]
        rr.append(sw(LAYER,HEAD,Vmix,L2))
    vv,xx,rr=map(np.array,(vv,xx,rr))
    print(f"L{LAYER}H{HEAD:<4d} {vv.mean():+.4f}+-{vv.std():.4f} {xx.mean():+.4f}+-{xx.std():.4f} {rr.mean():+.4f}+-{rr.std():.4f}",flush=True)
print("DONE")
