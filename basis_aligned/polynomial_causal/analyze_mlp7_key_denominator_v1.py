"""CPU exact nested-RMS identity and price; omission is diagnostic only."""
from pathlib import Path
import hashlib,json
import torch
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2)
 path=P/'extracted_circuits/odd_attention8h2_mlp7_donor_v1/program.pt'
 p=torch.load(path,weights_only=True,map_location='cpu');d={k:v.double() if v.is_floating_point() else v for k,v in p['donor'].items()}
 rows=torch.load(P/'TYPED_FACE_MLP7_DONOR_V1_FIXTURES.pt',weights_only=True,map_location='cpu');eps=torch.finfo(torch.float32).eps
 residual=[]
 for row in rows:
  g=row['g7'].double();e=d['initial_table'][d['token_ids'].tolist().index(row['donor_token'])][None]
  z=g/(g.square().mean(-1,keepdim=True)+eps).sqrt();h=(z@d['left'].T)*(z@d['right'].T)
  residual.append(d['lambda8'][0]*(g+h@d['down'].T+d['bias'])+d['lambda8'][1]*e)
 raw=torch.cat(residual);rho2=raw.square().mean(-1,keepdim=True)+eps
 records={}
 for name in ['k1','k2']:
  key=p['head'][name].double();u=raw@key.T;q=u.square().mean(-1,keepdim=True)
  native=(u/rho2.sqrt())/(q/rho2+eps).sqrt()
  exact=u/(q+eps*rho2).sqrt()
  omitted=u/(q+eps).sqrt()
  error=float((exact-native).norm()/native.norm())
  records[name]={'exact_identity_relative_error':error,'omitted_residual_scale_relative_error':float((omitted-exact).norm()/exact.norm()),'max_abs_relative_denominator_squared_change':float((eps*(rho2-1)/(q+eps*rho2)).abs().max())}
 assert all(v['exact_identity_relative_error']<=1e-12 for v in records.values())
 result={'pred_a':True,'keys':records,'identity':'RMS_eps(K RMS_eps(r)) = Kr / sqrt(mean((Kr)^2) + eps*(mean(r^2)+eps))',
  'retained_D_scalars':1152*4608,'dense_DtD_scalars':4608**2,'symmetric_DtD_scalars':4608*4609//2,
  'folded_two_key_D_scalars':2*128*4608,'scope':'Exact FP64 algebra with native FP32 epsilon on opened states. Small epsilon diagnostic is not behavioral omission certification. Dense Gram is larger than D, and cross terms with g/token remain required; no exact storage win shown.',
  'program_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 (P/'MLP7_KEY_DENOMINATOR_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
