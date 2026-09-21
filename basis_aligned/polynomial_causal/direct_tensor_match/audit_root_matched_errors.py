"""Post-result prefix exclusion and scale diagnostic; not new validation."""
import json,math
from pathlib import Path
P=Path(__file__).resolve().parent

def metrics(rows):
 ref=sum(r['reference']**2 for r in rows);err=sum((r['predicted']-r['reference'])**2 for r in rows)
 dot=sum(r['reference']*r['predicted'] for r in rows);est=sum(r['predicted']**2 for r in rows)
 scale=dot/est;scaled=sum((scale*r['predicted']-r['reference'])**2 for r in rows)
 return dict(pairs=len(rows),error=math.sqrt(err/ref),oracle_scale=scale,oracle_scaled_error=math.sqrt(scaled/ref),remaining_squared_error=scaled/err)

def main():
 source=json.loads((P/'ROOT_MATCHED_READER_V1.json').read_text());result=[]
 for panel in source['rows']:
  rows=panel['pair_rows'];prefixes=sorted({r[k]//64 for r in rows for k in ['recipient','donor']});excluded=[]
  for prefix in prefixes:
   subset=[r for r in rows if r['recipient']//64!=prefix and r['donor']//64!=prefix]
   if subset:excluded.append(dict(prefix=prefix,**metrics(subset)))
  result.append(dict(panel=panel['panel'],pooled=metrics(rows),exclude_prefix=excluded,all_exclusion_errors_above_10pct=all(r['error']>.1 for r in excluded),exclusion_error_range=[min(r['error'] for r in excluded),max(r['error'] for r in excluded)],unique_unordered_pairs=len({tuple(sorted((r['recipient'],r['donor']))) for r in rows})))
 (P/'ROOT_MATCHED_ERROR_AUDIT_V1.json').write_text(json.dumps(result,indent=2)+'\n')
 for row in result:print(json.dumps({k:v for k,v in row.items() if k!='exclude_prefix'},indent=2))
if __name__=='__main__':main()
