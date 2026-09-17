"""Freeze null predictors of the full native half-write for fresh cross validation."""
from pathlib import Path
import json,torch,hashlib
from freeze_native8_prediction_nulls_v1 import features
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);rp=P/'TYPED_FACE_NATIVE8_FRESH_V1_ROWS.json';ap=P/'TYPED_FACE_NATIVE8_FRESH_V1_ARTIFACT.pt';rows=json.loads(rp.read_text())['rows'];v=torch.load(ap,weights_only=True)['values'];y=v[3,:,0]-v[0,:,0];X=torch.tensor([features(r) for r in rows],dtype=torch.float64)
 result={'training':'20opened cells,40sequences,240endpoint rows from earlier native8 fresh screen','target':'full native8 retained half-write effect; differs from full-city parent target of previous baselines','coefficients':torch.linalg.lstsq(X,y,driver='gelsd').solution.tolist(),'design_rank':int(torch.linalg.matrix_rank(X)),'cue_constants':{cue:float(y[[i for i,r in enumerate(rows) if r['cue']==cue]].mean()) for cue in ['British','American']},'feature_code_sha256':hashlib.sha256((P/'freeze_native8_prediction_nulls_v1.py').read_bytes()).hexdigest(),'training_artifact_sha256':hashlib.sha256(ap.read_bytes()).hexdigest(),'scope':'Frozen before fresh effects. Same fixed14feature recipe, no search or future targets. No heldout result yet.'}
 out=P/'HEAD2_MLP8_CROSS_FRESH_V1_BASELINES.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print('Frozen rank',result['design_rank'])
if __name__=='__main__':main()
