"""Fold learned weights and verify native channel inputs; no fitted selection."""
from pathlib import Path
import sys,json,torch,torch.nn.functional as F
from attention8_context_channels_v1 import channels
from head2_mlp8_bilinear_kernel_v1 import cross,quadratic
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);sys.path.insert(0,str(P/'extracted_circuits/typed_face_mlp8_coupled_v1'));import head8
 local=torch.load(P/'extracted_circuits/typed_face_mlp8_coupled_v1/program.pt',weights_only=True);h=local['head8'];context=torch.load(P/'ATTENTION8_CONTEXT_GENERATOR_V1_PROGRAM.pt',weights_only=True);L,R,D=[local['mlp8'][k].double() for k in ['left','right','down']];W=h['output'].double();p={'left_writer':L@W,'right_writer':R@W}
 fixtures=torch.load(P/'TYPED_FACE_MLP8_COUPLED_V1_ARTIFACT.pt',weights_only=True)['fixtures'];sys.path.insert(0,str(P.parent/'bilinear_quotient/ops'));from regional_endpoint_batching_v1 import group_rows
 groups,_=group_rows(json.loads((P/'TYPED_FACE_NATIVE8_FRESH_V1_ROWS.json').read_text())['rows']);errors=[];symmetry=[];difference=[];native_errors=[]
 for row,f in zip(groups,fixtures):
  x=f['inputs'];cur=x['current8'];city=x['city'];a0=head8.routing(h,cur,cur[:,city],city);a1=head8.routing(h,cur,x['donor_city8'],city);c=F.linear(cur,h['current_value'])[:,city];i0=head8.inherited(h,x['recipient_token']);i1=head8.inherited(h,x['donor_token']);lam=h['mixture'];v0=(1-lam)*c+lam*i0;v1=(1-lam)*c+lam*i1
  u=(.5*(a1[...,None]*v1[:,None]-a0[...,None]*v0[:,None])*x['destination'][None,:,None]).double();z=channels(context,cur,torch.tensor([row['ids']]))[:,:,2].double();delta=u@W.T;background=z@W.T
  direct=((delta@L.T)*(background@R.T)+(background@L.T)*(delta@R.T))@D.T;folded=cross(p,D,u,z);den=direct.norm().clamp_min(1e-30);errors.append(float((folded-direct).norm()/den));symmetry.append(float((folded-cross(p,D,z,u)).norm()/den));difference.append(float((quadratic(p,D,z+u)-quadratic(p,D,z)-quadratic(p,D,u)-folded).norm()/den))
  native_delta=(u.float()@h['output'].T).double();native_bg=(z.float()@h['output'].T).double();native=((native_delta@L.T)*(native_bg@R.T)+(native_bg@L.T)*(native_delta@R.T))@D.T;native_errors.append(float((folded-native).norm()/native.norm()))
 result={'pred_a':max(errors+symmetry+difference)<=1e-10,'pred_b':max(native_errors)<=1e-4,'fixtures':40,'max_fold_error':max(errors),'max_symmetry_error':max(symmetry),'max_quadratic_identity_error':max(difference),'max_fp32_writer_cross_error':max(native_errors),'derived_adapter_scalars':sum(v.numel() for v in p.values()),'down_scalars_shared':D.numel(),'factored_kernel_scalars_if_standalone':sum(v.numel() for v in p.values())+D.numel(),'dense_symmetric_coefficients':1152*128*129//2,'scope':'Exact shared-writer symmetric bilinear NUMERATOR on native channel inputs. Complete RMS denominator, normalization correction and residual/initial crosses remain outside this kernel. No new effect confirmation or overall parameter saving.'}
 torch.save(p,P/'HEAD2_MLP8_BILINEAR_V1_PROGRAM.pt');(P/'HEAD2_MLP8_BILINEAR_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
