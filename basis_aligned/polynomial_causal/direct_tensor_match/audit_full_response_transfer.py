"""Paired fixed-program comparison; no independent-site significance claim."""
from pathlib import Path
import json
p=Path(__file__).resolve().parent;a=json.loads((p/'FULL_CHANNEL_DIRECT_SOURCE_SWAP_V1.json').read_text());b=json.loads((p/'FULL_CHANNEL_RESPONSE_NATIVE_V1.json').read_text());keys=('domain','family','kind','cohort');old={tuple(r[k] for k in keys):r for r in a['summary']};oldrows={tuple(r[k] for k in keys)+(r['document_index'],):r for r in a['records']};out=[]
for r in b['summary']:
 key=tuple(r[k] for k in keys);before=old[key];rows=[v for v in b['records'] if tuple(v[k] for k in keys)==key]
 assert all(abs(v['reference_energy']-oldrows[key+(v['document_index'],)]['reference_energy'])<1e-8*max(1,v['reference_energy']) for v in rows)
 out.append(dict(zip(keys,key),old_error=before['effect_error'],new_error=r['effect_error'],relative_error_reduction=1-r['effect_error']/before['effect_error'],documents_improved=sum(v['error_energy']<oldrows[key+(v['document_index'],)]['error_energy'] for v in rows),documents=len(rows)))
result=dict(records=out,scope='Same teacher and frozen donor maps; response candidate chosen before native results. Descriptive document comparisons, not independent validation or significance under crossed donor-recipient dependence. Natural-input fidelity of new program not yet revalidated.')
(p/'FULL_CHANNEL_RESPONSE_TRANSFER_AUDIT_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps([r for r in out if r['family']=='same_cohort'],indent=2))
