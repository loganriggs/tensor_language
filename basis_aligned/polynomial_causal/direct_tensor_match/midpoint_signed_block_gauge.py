from pathlib import Path
import torch,json,math
p=Path('/workspace/tensor_language/basis_aligned/polynomial_causal/direct_tensor_match');torch.set_num_threads(2);torch.set_grad_enabled(False);program=torch.load(p/'MIDPOINT_SIGN_BLOCK3_V1.pt',weights_only=True)['program'];V=program['directions'].double();rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);n=rows['n'].flatten(0,1).double();m=rows['m'].flatten(0,1).double();records=[]
for t in [.3,.7]:
 c=math.cosh(t);s=math.sinh(t)
 for g in range(4):
  P=V[:,6*g:6*g+3];N=V[:,6*g+3:6*g+6];Pnew=c*P+s*N;Nnew=s*P+c*N;oldp=((n@P)*(m@P)).sum(1);oldn=((n@N)*(m@N)).sum(1);newp=((n@Pnew)*(m@Pnew)).sum(1);newn=((n@Nnew)*(m@Nnew)).sum(1);net=oldp-oldn;replay=float(((newp-newn)-net).norm()/net.norm());assert replay<1e-12;changes=[]
  for old,new in [(oldp,newp),(oldn,newn)]:
   a=old-old.mean();b=new-new.mean();changes.append(dict(relative_activation_change=float((b-a).norm()/a.norm()),activation_cosine=float((a@b)/(a.norm()*b.norm()))))
  records.append(dict(hyperbolic_parameter=t,feature=g,net_replay=replay,positive_block_change=changes[0],negative_block_change=changes[1]))
out=p/'MIDPOINT_SIGNED_BLOCK_GAUGE_V1.json';assert not out.exists();out.write_text(json.dumps(dict(records=records,scope='Exact rank-preserving positive/negative block mixing. Same products, same netfunction, different blockfunctions. Fixedmetric spectralorthogonality chooses one gauge but is not implied by functionmatch.'),indent=2)+'\n');print(json.dumps(records,indent=2))
