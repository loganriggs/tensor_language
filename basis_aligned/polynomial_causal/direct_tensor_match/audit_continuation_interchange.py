"""Preserve all failures and audit token/donor support without iid uncertainty."""
from pathlib import Path
from collections import Counter
import json,torch
p=Path(__file__).resolve().parent;d=json.loads((p/'FULL_CHANNEL_CONTINUATION_INTERCHANGE_V1.json').read_text());maps=json.loads((p/'FULL_CHANNEL_CONTINUATION_INTERCHANGE_DONORS_V1.json').read_text());tokens=torch.load(p/'FULL_CHANNEL_FRESH_TOKENS_V1.pt',weights_only=True);support=[]
for domain,data in maps['domains'].items():
 flat=tokens[domain][:,16:255].flatten().tolist()
 for family,mapping in data['maps'].items():
  for cohort,label in [('continuation',1),('spaced_word',2)]:
   ids=[i for i,j in enumerate(mapping) if j>=0 and data['cohort_labels'][i]==label];counts=Counter(mapping[i] for i in ids);tc=Counter(flat[i] for i in ids)
   support.append(dict(domain=domain,family=family,cohort=cohort,sites=len(ids),distinct_current_tokens=len(tc),distinct_donor_sites=len(counts),distinct_recipient_documents=len(set(i//239 for i in ids)),distinct_donor_documents=len(set(mapping[i]//239 for i in ids)),largest_token_fraction=max(tc.values())/len(ids),largest_donor_fraction=max(counts.values())/len(ids),donor_weight_effective_count=len(ids)**2/sum(v*v for v in counts.values())))
primary=[r for r in d['summary'] if r['family']=='same_cohort'];result=dict(primary_failed_cells=[r for r in primary if r['effect_error'] is None or r['effect_error']>=.1],primary_mode_error_range=[min(r['effect_error'] for r in primary if r['kind']=='mode'),max(r['effect_error'] for r in primary if r['kind']=='mode')],support=support,scope='Descriptive support and inherited repeated-donor dependence. Kish donor weight count is not independent sample size; no iid or document-only confidence interval claimed for crossed donor-recipient outcomes.')
(p/'FULL_CHANNEL_CONTINUATION_INTERCHANGE_AUDIT_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(failed_cells=len(result['primary_failed_cells']),mode_error_range=result['primary_mode_error_range'],support=support),indent=2))
