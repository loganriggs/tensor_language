"""Attribute failed relative cells without changing their registered verdicts."""
import json
from pathlib import Path
P=Path(__file__).parent;d=json.loads((P/'FRONTIER_FRESH_NATIVE_V1.json').read_text());rows=[]
for cell in d['all_candidate_comparisons']:
 for baseline,flag in [('separate','covariance_relative_pass'),('isotropic_baseline','isotropic_relative_pass')]:
  if cell[flag]:continue
  docs=d['plan']['recipient_documents'][cell['domain']];errors={doc:{name:0. for name in ('graph',baseline)} for doc in docs}
  for r in d['records']:
   if all(r[k]==cell[k] for k in ('domain','selection','family','cohort')) and r['candidate'] in ('graph',baseline):errors[r['document']][r['candidate']]+=r['error_energy']
  contrib=[dict(document=doc,graph_error_energy=e['graph'],baseline_error_energy=e[baseline],excess=e['graph']-1.21*e[baseline]) for doc,e in errors.items()];total=sum(r['excess'] for r in contrib);assert total>0
  rows.append(dict(**{k:cell[k] for k in ('domain','selection','family','cohort')},baseline=baseline,total_excess_error_energy=total,documents_with_positive_excess=sum(r['excess']>0 for r in contrib),documents=len(docs),leave_one_document_out_failures=sum(total-r['excess']>0 for r in contrib),largest_positive_documents=sorted(contrib,key=lambda r:r['excess'],reverse=True)[:3]))
(P/'FRONTIER_FAILURE_DOCUMENTS_V1.json').write_text(json.dumps(dict(records=rows,scope='Opened fresh-result diagnostic only. Positive excess means graph squared-error energy exceeds1.21times baseline squared-error energy; common reference-energy denominator cancels. Leave-one-out is descriptive and does not replace all-document registered verdict. No removed documents or candidate refitting.'),indent=2)+'\n');print(json.dumps(rows,indent=2))
