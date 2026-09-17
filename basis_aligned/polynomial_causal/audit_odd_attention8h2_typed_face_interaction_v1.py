"""Opened-panel composition diagnostic against the stronger user rubric."""
from pathlib import Path
import hashlib
import json
import torch

P = Path(__file__).resolve().parent
STEM = 'ODD_ATTENTION8H2_TYPED_FACE_REPLICATION_V1'

def main():
    artifact = P/(STEM+'_ARTIFACT.pt')
    data = torch.load(artifact, map_location='cpu', weights_only=True)
    v = data['values'].double()[:,48:,0]
    rows = json.loads((P/(STEM+'_ROWS.json')).read_text())['rows']
    a,b = v[1]-v[0],v[4]-v[0]
    interaction = v[5]-v[1]-v[4]+v[0]
    full = v[7]-v[0]
    cells = {'all':list(range(len(rows)))}
    cells.update({cue:[i for i,r in enumerate(rows) if r['cue']==cue] for cue in ('British','American')})
    cells.update({f'endpoint_{e}':[i for i,r in enumerate(rows) if r['endpoint']==e] for e in range(6)})
    metrics = {}
    for name, ids in cells.items():
        x,y,z,parent = a[ids],b[ids],interaction[ids],full[ids]
        denom = min(float(x.norm()),float(y.norm()))
        metrics[name] = {
            'rows':len(ids),
            'interaction_over_smallest_single':float(z.norm())/max(denom,1e-30),
            'interaction_aligned_fraction_of_full':float(z@parent)/max(float(parent@parent),1e-30),
            'interaction_cosine_to_full':float(z@parent)/max(float(z.norm()*parent.norm()),1e-30),
        }
    result = {'panel_status':'opened','pred_a':list(data['selected_masks'])==[1,4,5] and bool(torch.isfinite(v).all()),
              'pred_b':all(x['interaction_over_smallest_single']<=.35 for x in metrics.values()),
              'cells':metrics,'random_split_null':'missing; output-only artifact cannot reconstruct same-write random splits',
              'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
              'scope':'Diagnostic under newly registered threshold, not a retrospective promotion or fresh composition test.'}
    (P/'ODD_ATTENTION8H2_TYPED_FACE_INTERACTION_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
