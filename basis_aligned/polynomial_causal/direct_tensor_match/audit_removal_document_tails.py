"""Document concentration in already-completed native removal failures."""
import json,math
from pathlib import Path
P=Path(__file__).resolve().parent

def main():
 records=[]
 for name in ['MIXED_CP_REMOVAL_V1','CONDITIONAL_CP_REMOVAL_V1','LEAN_CONDITIONAL_CP_REMOVAL_V1']:
  d=json.loads((P/(name+'.json')).read_text())
  for cell in d['summary']:
   key=('seed','domain','alpha','condition');rows=[r for r in d['rows'] if all(r[k]==cell[k] for k in key)];e=sum(r['error_energy'] for r in rows);n=sum(r['reference_energy'] for r in rows);assert abs(math.sqrt(e/n)-cell['error'])<1e-12
   ordered=sorted(rows,key=lambda r:r['error_energy'],reverse=True);errs=sorted(math.sqrt(r['error_energy']/r['reference_energy']) for r in rows if r['reference_energy']>0);top=ordered[:2]
   records.append(dict(program=name,**{k:cell[k] for k in key},pooled_error=cell['error'],documents=len(rows),documents_over_10pct=sum(v>.1 for v in errs),median_document_error=(errs[(len(errs)-1)//2]+errs[len(errs)//2])/2,maximum_document_error=max(errs),worst_two_error_share=sum(r['error_energy'] for r in top)/e,worst_two_reference_share=sum(r['reference_energy'] for r in top)/n,worst_two_documents=[r['document'] for r in top]))
 (P/'REMOVAL_DOCUMENT_TAILS_V1.json').write_text(json.dumps(dict(rows=records,scope='Openednativeinterventionpanels; perdocumentaggregateerrors not independenttokenstats. Top2 chosenbyabsoluteerrorenergy, not percentage. Zero-reference doc cells excludedfromrelativequantiles only.'),indent=2)+'\n')
 for r in records:
  if r['program']=='LEAN_CONDITIONAL_CP_REMOVAL_V1' and r['pooled_error']>.1:print(r)
if __name__=='__main__':main()
