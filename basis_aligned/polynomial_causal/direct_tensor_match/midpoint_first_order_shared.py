"""Separate diagnostic for sharing linear terms around fixed calibration means."""
from pathlib import Path
import json,torch,time
p=Path(__file__).resolve().parent;out=p/'MIDPOINT_FIRST_ORDER_SHARED_V1.json';assert not out.exists();torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.perf_counter()
state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,mmap=True,map_location='cpu')
ru=torch.linalg.qr(state['lm_head.weight'].double(),mode='r').R;L=state['transformer.h.17.mlp.Left.weight'].double();R=state['transformer.h.17.mlp.Right.weight'].double();C=ru@state['transformer.h.17.mlp.Down.weight'].double()
rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);n=rows['n'].flatten(0,1).double();m=rows['m'].flatten(0,1).double();nb=n.mean(0);mb=m.mean(0)
Jn=(L.T*(mb@R.T))@C.T+(R.T*(mb@L.T))@C.T;Jm=(L.T*(nb@R.T))@C.T+(R.T*(nb@L.T))@C.T;fbar=((nb@L.T)*(mb@R.T)+(nb@R.T)*(mb@L.T))@C.T
S=torch.load(p/'MIDPOINT_CENTERED_V1.pt',weights_only=True)['metric_sqrt'].double();target=(n-nb)@Jn+(m-mb)@Jm
_,s,Vh=torch.linalg.svd(target@S,full_matrices=False);records=[];exports={}
for rank in [8,16,32,64,128,256]:
 Q=Vh[:rank].T;Hn=Jn@S@Q;Hm=Jm@S@Q;W=torch.linalg.solve(S,Q);offset=nb@Hn+mb@Hm
 pred=(n@Hn+m@Hm-offset)@W.T;err=float(((target-pred)@S).norm()/(target@S).norm());tail=float(s[rank:].norm()/s.norm());assert abs(err-tail)<1e-10
 records.append(dict(rank=rank,linear_variation_error=err,weight_coefficients=3*1152*rank,extra_state=rank+1152))
 if rank in [32,64,128]:exports[f'rank{rank}']=dict(left_reader=Hn,right_reader=Hm,writer=W,offset=offset,constant=fbar)
artifact=p/'MIDPOINT_FIRST_ORDER_SHARED_V1.pt';assert not artifact.exists();torch.save(exports,artifact)
out.write_text(json.dumps(dict(records=records,exact_linear_weight_coefficients=2*1152**2,seconds=time.perf_counter()-start,scope='Joint paired-calibration linear-output metric, exact original linear terms around fixedcalmeans. Centered interaction excluded and unchanged. This is calibration projection error of linear variation only, not native intervention or wholefunction error. Sharedlinearfeatures combine n/m reads before one outputwrite; constant retained.'),indent=2)+'\n');print(out.read_text())
