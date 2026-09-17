"""Freeze simple predictors on opened native8 effects for a later fresh panel."""
from pathlib import Path
import json,hashlib,torch
P=Path(__file__).resolve().parent

def features(r):
 sign=1. if r['cue']=='British' else -1.
 endpoints=[float(r['endpoint']==i) for i in range(1,6)]
 return [1.,sign,len(r['ids'])/32.,r['city_position']/32.]+endpoints+[sign*x for x in endpoints]

def main():
 torch.set_num_threads(2);rp=P/'TYPED_FACE_NATIVE8_FRESH_V1_ROWS.json';ap=P/'TYPED_FACE_NATIVE8_FRESH_V1_ARTIFACT.pt';rows=json.loads(rp.read_text())['rows'];v=torch.load(ap,weights_only=True,map_location='cpu')['values'];y=v[1,:,0]-v[0,:,0]
 X=torch.tensor([features(r) for r in rows],dtype=torch.float64);coef=torch.linalg.lstsq(X,y,driver='gelsd').solution
 means={cue:float(y[[i for i,r in enumerate(rows) if r['cue']==cue]].mean()) for cue in ['British','American']}
 result={'status':'Frozen training artifact, no held-out result','training_cells':20,'training_sequences':40,'training_rows':240,'target':'full-city native attention8 parent effect, target spelling margin','features':['intercept','cue_sign','length/32','city_position/32']+[f'endpoint_{i}' for i in range(1,6)]+[f'cue_sign*endpoint_{i}' for i in range(1,6)],'coefficients':coef.tolist(),'cue_constants':means,'design_rank':int(torch.linalg.matrix_rank(X)),'design_columns':X.shape[1],'future_gate':'On each fresh family, retained-face relativeL2 parent-effect error <=.8 times EACH cue-constant and fixed-OLS error, alongside unchanged native capability/selectivity gates. Same six endpoints; no pair-index feature. No fitting on future rows.','source_shas':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [rp,ap,Path(__file__)]},'scope':'Opened training only. Fixed unregularized least squares with no feature/hyperparameter search. Does not retroactively fill the missing null gate on earlier fresh results.'}
 out=P/'NATIVE8_PREDICTION_NULLS_V1_FROZEN.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
