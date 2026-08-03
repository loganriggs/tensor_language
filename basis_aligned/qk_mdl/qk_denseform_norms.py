"""Norm-corrected DenseFormer mix matrix (Logan's scale point): effective share
w[l][j]*mean||s_j|| / row-sum, using held-data stream norms."""
import sys, json, numpy as np, torch, torch.nn.functional as F
import importlib.util
spec=importlib.util.spec_from_file_location('qdf','/workspace/tensor_language/basis_aligned/qk_mdl/qk_denseform.py')
qdf=importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(qdf)
except SystemExit: pass
DEV='cuda'; DEPTH=12; D=384
model=qdf.DenseMini(DEPTH).to(DEV)
sd=torch.load(f'/workspace/tensor_language/basis_aligned/qk_mdl/qk_denseform_{DEPTH}.pt',map_location=DEV)
model.load_state_dict(sd['state_dict'] if isinstance(sd,dict) and 'state_dict' in sd else sd)
model.eval()
FW=torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
HELD=FW[5500:5560,:512].to(DEV)
sums=np.zeros(1+2*DEPTH); cnt=0
with torch.no_grad():
    for s in range(0,HELD.shape[0],4):
        idx=HELD[s:s+4]
        col={'entry_norm':[],'attn_write':[],'mlp_write':[]}
        model(idx,collect=col)
        e=F.rms_norm(model.wte(idx),(D,)).float().norm(dim=-1).mean().item()
        row=[e]
        for l in range(DEPTH):
            row.append(col['attn_write'][l].float().norm(dim=-1).mean().item())
            row.append(col['mlp_write'][l].float().norm(dim=-1).mean().item())
        sums+=np.array(row); cnt+=1
norms=(sums/cnt)
print("stream mean norms [emb, a0, m0, a1, m1, ...]:")
print([round(float(x),1) for x in norms])
mm=[model.mix[l].detach().float().cpu().numpy() for l in range(DEPTH)]
mm.append(model.w_out.detach().float().cpu().numpy() if hasattr(model,'w_out') else None)
if mm[-1] is None:
    # find the readout row parameter name
    names=[n for n,_ in model.named_parameters() if 'out' in n or 'final' in n]
    print("readout param names:", names)
    p=dict(model.named_parameters())[names[0]]
    mm[-1]=p.detach().float().cpu().numpy()
eff=[]
for i,row in enumerate(mm):
    k=len(row)
    e=row*norms[:k]
    eff.append((e/e.sum()).round(4).tolist())
print("effective share rows (w*norm / rowsum):")
for i,r in enumerate(eff):
    lbl='out' if i==DEPTH else f'b{i}'
    print(lbl,[round(x,3) for x in r])
json.dump({'stream_norms':[float(x) for x in norms],'effective_share':eff,
           'raw_mix':[r.tolist() for r in mm]},
          open('/workspace/tensor_language/basis_aligned/qk_mdl/qk_denseform_norms.json','w'),indent=1)
print("saved qk_denseform_norms.json")
