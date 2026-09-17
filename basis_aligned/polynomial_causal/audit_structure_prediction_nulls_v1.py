"""Posthoc base-trained null audit; freeze code for a future prospective screen."""
from pathlib import Path
import hashlib,json
import torch
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2)
 rowpath=P/'TYPED_FACE_STRUCTURE_V1_ROWS.json';rows=json.loads(rowpath.read_text())['rows']
 path=P/'TYPED_FACE_STRUCTURE_V1_ARTIFACT.pt';v=torch.load(path,weights_only=True,map_location='cpu')['values'];y=v[1,:,0]-v[0,:,0];face=v[2,:,0]-v[0,:,0]
 # One fixed low-dimensional design, no hyperparameter/feature selection.
 # Input features only; no activations, face effects, or test targets in the fit.
 features=[]
 for r in rows:
  sign=1 if r['cue']=='British' else -1
  features.append([1.,float(sign),len(r['ids'])/32.,r['city_position']/32.,float(r['pair'])]+[float(r['endpoint']==i) for i in range(1,6)])
 X=torch.tensor(features,dtype=torch.float64);train=torch.tensor([i for i,r in enumerate(rows) if r['variant']=='base'])
 coeff=torch.linalg.lstsq(X[train],y[train],driver='gelsd').solution;fit=X@coeff
 mean={cue:float(y[[i for i in train.tolist() if rows[i]['cue']==cue]].mean()) for cue in ['British','American']}
 constant=torch.tensor([mean[r['cue']] for r in rows],dtype=torch.float64)
 records={}
 for variant in ['base','no_colon','no_quote','neither','short_frame']:
  idx=[i for i,r in enumerate(rows) if r['variant']==variant];target=y[idx];den=target.norm()
  records[variant]={name:float((pred[idx]-target).norm()/den) for name,pred in [('frozen_face',face),('cue_constant',constant),('text_ols',fit)]}
 result={'status':'posthoc diagnostic; not prospective evidence','training':'base only: four context/city cells, eight sequences; endpoint rows correlated','features':['intercept','cue_sign','length/32','city_position/32','pair_index']+[f'endpoint_{i}' for i in range(1,6)],
 'coefficients':coeff.tolist(),'design_rank':int(torch.linalg.matrix_rank(X[train])),'design_columns':X.shape[1],'cue_constants':mean,'errors':records,
 'scope':'OLS and constants do not use test targets for fitting; feature choice was written after structural outcomes, so evaluation is opened and cannot supply the missing preregistered prediction-null gate. Rank deficiency reported, no tuning. Freeze this recipe for the next independent panel.',
 'row_sha256':hashlib.sha256(rowpath.read_bytes()).hexdigest(),'artifact_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 (P/'STRUCTURE_PREDICTION_NULLS_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
