"""Freeze endpoint-agnostic nulls from opened donor-free removal effects."""
from pathlib import Path
import torch,json,hashlib
P=Path(__file__).resolve().parent
def features(r):return [1.,1. if r['cue']=='British' else -1.,len(r['ids'])/32.,r['city_position']/32.]
def main():
 torch.set_num_threads(2);rp=P/'SINGLE_HEAD_FRESH_V1_ROWS.json';ap=P/'CITY_INHERITED_REMOVAL_V1_ARTIFACT.pt';rows=json.loads(rp.read_text())['rows'];v=torch.load(ap,weights_only=True)['values'];y=v[1,:,0]-v[0,:,0];X=torch.tensor([features(r) for r in rows],dtype=torch.float64)
 d={'target':'native attention8 inherited-city half-removal effect','training':'20opened cells/40sequences/240rows; six old endpoints','features':['intercept','cue_sign','length/32','city_position/32'],'coefficients':torch.linalg.lstsq(X,y,driver='gelsd').solution.tolist(),'design_rank':int(torch.linalg.matrix_rank(X)),'cue_constants':{cue:float(y[[i for i,r in enumerate(rows) if r['cue']==cue]].mean()) for cue in ['British','American']},'source_shas':{str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in [rp,ap,Path(__file__)]},'scope':'No endpoint indicators; fixed four text features allow unseen endpoint application without assigning new words to old endpoint categories. No model outcomes on the new panel.'}
 out=P/'CITY_REMOVAL_ENDPOINT_FRESH_V1_BASELINES.json';assert not out.exists();out.write_text(json.dumps(d,indent=2)+'\n');print('Frozen baseline design rank',d['design_rank'])
if __name__=='__main__':main()
