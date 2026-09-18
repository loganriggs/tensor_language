"""Audit one frozen regional panel against the repository's other *_ROWS panels."""
import argparse,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent

def audit(stem):
    path=P/(stem+'_ROWS.json');d=json.loads(path.read_text());old_docs=set();old_inputs=set()
    for f in P.glob('*_ROWS.json'):
        if f==path:continue
        other=json.loads(f.read_text())
        if not isinstance(other,dict):continue
        old_docs.update(c['document_sha256'] for c in other.get('contexts',[]) if isinstance(c,dict) and 'document_sha256' in c)
        old_inputs.update(tuple(r['ids']) for r in other.get('rows',[]) if isinstance(r,dict) and isinstance(r.get('ids'),list))
    docs={c['document_sha256'] for c in d['contexts']};inputs={tuple(r['ids']) for r in d['rows']}
    r={'document_overlap':len(docs & old_docs),'input_overlap':len(inputs & old_inputs),'documents':len(docs),'sequences':len(inputs),'rows':len(d['rows']),'all_masks_after_city':all(all(r['city_position']<j<len(r['ids']) for j in r['destination_positions']) for r in d['rows']),'row_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'model_calls':0}
    r['passes']=r['document_overlap']==r['input_overlap']==0 and r['documents']==20 and r['sequences']==40 and r['rows']==240 and r['all_masks_after_city']
    return r

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stem');ap.add_argument('--output',type=Path);args=ap.parse_args()
    out=args.output or P/(args.stem+'_ROW_AUDIT.json')
    if out.exists():raise FileExistsError('Preserve existing frozen audits')
    result=audit(args.stem);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
