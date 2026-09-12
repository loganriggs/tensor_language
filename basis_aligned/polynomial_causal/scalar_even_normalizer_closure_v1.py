"""Normalizer closure of frozen joint-key reflection, no subspace refit."""
import torch,json,time
import torch.nn.functional as F
from pathlib import Path
from scalar_joint_key_even_v1 import even_sectors
P=Path(__file__).resolve().parent

def defect(K,B):
 inside=K@B;G=K@K.T;Gi=inside@inside.T;cross=torch.trace(Gi@(G-Gi)).clamp_min(0)
 return float(2*(2*cross).sqrt()/G.norm())

def main():
 torch.set_num_threads(2);tic=time.perf_counter();out=P/'SCALAR_EVEN_NORMALIZER_CLOSURE_V1_RESULT.json';assert not out.exists();g=torch.Generator().manual_seed(195401)
 K=torch.randn(7,13,dtype=torch.float64,generator=g);B=torch.linalg.qr(torch.randn(13,5,dtype=torch.float64,generator=g)).Q;G=K.T@K;R=torch.eye(13,dtype=torch.float64)-2*B@B.T;control=abs(defect(K,B)-float((R@G@R-G).norm()/G.norm()))
 p=torch.load(P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt',weights_only=True,map_location='cpu');bands=torch.load(P/'SCALAR_JOINT_KEY_BANK_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['bands'];geometry=[dict(head=h,key=k,gram_reflection_defect=defect(p[k][h].double(),bands[h,0])) for h in range(2) for k in ('k1','k2')]
 cache=torch.load(P/'SCALAR_PRODUCER_MLP_BRIDGE_CACHE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu');rows=json.loads((P/'STRUCTURED_PRODUCER_CONFIRMATION_V1_ROWS.json').read_text())['rows'];old=torch.zeros(2,48,22,dtype=torch.float64);new=torch.zeros_like(old)
 for i,row in enumerate(rows):
  n=len(row['ids']);x=F.rms_norm(cache['r9'][:,i,:n],(1152,),eps=torch.finfo(torch.float32).eps);ids=torch.tensor([row['ids']]).expand(2,-1);basis=bands[1,0];inv=1/(10000**(torch.arange(0,128,2,dtype=torch.float32)/128));angle=torch.outer(torch.arange(n,dtype=torch.float32),inv);co=angle.cos().bfloat16();si=angle.sin().bfloat16()
  def rot(t):
   a,b=t.chunk(2,-1);return torch.cat((a*co+b*si,-a*si+b*co),-1)
  orig=[];reflect=[]
  for qn,kn in (('q1','k1'),('q2','k2')):
   q=rot(F.rms_norm(F.linear(x,p[qn][1]),(128,),eps=torch.finfo(torch.float32).eps)).double();native=F.linear(x,p[kn][1]);den=(native.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt().double();num=x.double()@p[kn][1].double().T;other=num-2*(x.double()@basis)@(p[kn][1].double()@basis).T;denother=(other.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();orig.append(q@rot(num/den).transpose(-1,-2)/128);reflect.append(q@rot(other/denother).transpose(-1,-2)/128)
  gamma=((orig[0]*orig[1]+reflect[0]*reflect[1])/2).masked_fill(~torch.ones(n,n,dtype=torch.bool).tril(),0);v=x.double()@p['current_value_readers'][1];new[:,i,:n]=(gamma@v[...,None])[...,0];old[:,i,:n]=even_sectors(x,ids,p,1,basis)[...,0]
 cells=[]
 for family in range(2):
  ix=[i for i,r in enumerate(rows) if r['family']==family];cells.append(dict(family=family,pristine_field_error=float((new[0,ix]-old[0,ix]).norm()/old[0,ix].norm()),changed_field_error=float((new[1,ix]-old[1,ix]).norm()/old[1,ix].norm()),serial_change_error=float(((new[1,ix]-new[0,ix])-(old[1,ix]-old[0,ix])).norm()/(old[1,ix]-old[0,ix]).norm())))
 A=control<=1e-10;result=dict(pred_a=A,pred_b=A and all(r['gram_reflection_defect']<=.01 for r in geometry),pred_c=A and all(max(c['pristine_field_error'],c['changed_field_error'])<=.01 for c in cells),dense_control_error=control,geometry=geometry,cells=cells,seconds=time.perf_counter()-tic,scope='Exact normalizer Gram reflection condition and native9 comparison with branch-specific reflected keynormalizers. Originalquery/values unchanged; not fullrawstate reflection. Originalconfirmed candidate remains frozen with originalfullkeydenominators. No endbehavior or newOOD claim.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert A
if __name__=='__main__':main()
