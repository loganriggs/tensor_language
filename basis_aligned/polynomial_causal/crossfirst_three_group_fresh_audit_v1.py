from pathlib import Path
import json,torch
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 a=torch.load(P/'CROSSFIRST_THREE_GROUP_FRESH_V1_ARTIFACT.pt',weights_only=True);cells=[]
 for g in range(1,5):
  lo,hi=24*g,24*(g+1);z=a['readouts'][lo:hi,:,0];m=a['measures'][lo:hi,:,0];nat=z[:,1]-z[:,0];pred=z[:,3]-z[:,0];total=m[:,2]-m[:,1]-m[:,3]+m[:,0];local=m[:,4]-z[:,0];contrast=m[::2,0]-m[1::2,0];removed=m[::2,1]-m[1::2,1];zero=(nat==0)
  cells.append(dict(group=g,prediction_error_vs_native_head=float((pred-nat).norm()/nat.norm()),signs_matching_native_head=int(((pred*nat)>0).sum()),native_head_zero_count=int(zero.sum()),maxabs_prediction_on_native_zero=float(pred[zero].abs().max()) if zero.any() else None,candidate_norm_over_original_nonadditivity=float(pred.norm()/total.norm()),candidate_aligned_over_original_nonadditivity=float((pred*total).sum()/total.square().sum()),candidate_norm_over_full_local_block17=float(pred.norm()/local.norm()),child_removal_change_in_regional_contrast=float((removed-contrast).mean()/contrast.mean()),meanabs_candidate_margin_change=float(pred.abs().mean())))
 r=dict(cells=cells,scope='Executed fresh outcome counter-review against independent nativeheadwrite, notonlyportformula. Norm/alignment fractions are conditional additive-background response, notwholecircuit coverage. No filtering or refitting.')
 (P/'CROSSFIRST_THREE_GROUP_FRESH_V1_AUDIT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
