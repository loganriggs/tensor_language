"""New-to-this-path ground-truth newline prefixes; no model-score selection."""
from pathlib import Path
import json,hashlib,random,torch
P=Path(__file__).resolve().parent

def main():
 old=json.loads((P/'SCALAR_PRODUCERS_NEWLINE_NATURAL_V2_ROWS.json').read_text());source=Path(old['source']);assert hashlib.sha256(source.read_bytes()).hexdigest()==old['source_sha'];data=torch.load(source,weights_only=True,map_location='cpu');used={r['source_row'] for r in old['rows'] if r['pool']=='fineweb'};seen=set()
 for f in P.glob('*ROWS.json'):
  d=json.loads(f.read_text());groups=[d] if isinstance(d,list) else [v for v in d.values() if isinstance(v,list)] if isinstance(d,dict) else []
  for group in groups:
   for row in group:
    if isinstance(row,dict) and isinstance(row.get('ids'),list) and all(isinstance(x,int) for x in row['ids']):seen.add(tuple(row['ids']))
 candidates=[]
 for i in range(24,192):
  if i in used:continue
  positions=torch.where(data[i,64:257]==198)[0]
  if not len(positions):continue
  pos=64+int(positions[0]);ids=data[i,:pos].tolist()
  if tuple(ids) in seen:continue
  candidates.append((i,pos,ids))
 random.Random(241316).shuffle(candidates);chosen=candidates[:32];assert len(chosen)==32
 rows=[dict(r,row_id=i) for i,r in enumerate(old['rows'][16:])]
 for n,(i,pos,ids) in enumerate(chosen):rows.append(dict(row_id=len(rows),pool='fineweb',family=2+n//16,source_row=i,target_position=pos,ids=ids,newline_id=198,comma_id=11))
 out=P/'FIRST_TOKEN_REMOVAL_NEWLINE_FRESH_V1_ROWS.json';assert not out.exists();out.write_text(json.dumps(dict(rows=rows,source=str(source),source_sha=old['source_sha'],new_candidates=len(candidates),seed=241316,scope='32oldanchor and32new-to-path FineWeb next-newline prefixes. New cacheindices excludeoldcontrols and24meanrows; exactfullprefixes absentfromknownROWS files. Documentindependence notestablished; samecache mayhavebeenused byotherresearch. No modelscorefilter.'),indent=2)+'\n');print(dict(rows=len(rows),new=32,max_tokens=max(len(r['ids']) for r in rows),candidates=len(candidates)))
if __name__=='__main__':main()
