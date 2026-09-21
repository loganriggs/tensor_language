"""Recipient-document paired bootstrap; fixed donor map, not donor uncertainty."""
from pathlib import Path
import json
import numpy as np
P=Path(__file__).resolve().parent
data=json.loads((P/'SHARED_MODE3_FRESH_NATIVE_V1.json').read_text());records=data['records'];rng=np.random.default_rng(612)
rows=[]
for domain in ['fineweb','stdlib']:
 docs=list(dict.fromkeys(r['document'] for r in records if r['domain']==domain))
 samples=rng.integers(0,len(docs),size=(4000,len(docs)))
 for family in ['natural','hybrid','change']:
  for cohort in ['all','continuation','spaced_word']:
   columns=[]
   for name in ['independent128','shared_affine256']:
    values=[]
    for doc in docs:
     rr=[r for r in records if r['domain']==domain and r['document']==doc and r['candidate']==name and r['family']==family and r['cohort']==cohort]
     values.append([sum(r['error_energy'] for r in rr),sum(r['reference_energy'] for r in rr)])
    columns.append(np.asarray(values))
   a,b=columns
   ratio=np.sqrt(b[:,0].sum()/a[:,0].sum())
   boot_a=a[samples].sum(axis=1);boot_b=b[samples].sum(axis=1)
   boot_ratio=np.sqrt(boot_b[:,0]/boot_a[:,0])
   boot_error=np.sqrt(boot_b[:,0]/boot_b[:,1])
   rows.append(dict(domain=domain,family=family,cohort=cohort,recipient_documents=len(docs),relative_error_ratio=float(ratio),ratio_interval95=np.quantile(boot_ratio,[.025,.975]).tolist(),primary_error_interval95=np.quantile(boot_error,[.025,.975]).tolist()))
out=dict(records=rows,replicates=4000,scope='Paired bootstrap over recipient document identities for FW and source-file identities for reusedcode. Donors and candidate selection frozen; intervals do not include donor-map uncertainty or fullmodel/semantic uncertainty.')
(P/'SHARED_MODE3_FRESH_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n')
for row in rows:
 if row['cohort']=='all':print(row)
