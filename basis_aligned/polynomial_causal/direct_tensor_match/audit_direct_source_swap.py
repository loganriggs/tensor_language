"""Retain native-path failures and quantify normalizer shifts without iid claims."""
from pathlib import Path
import json
p=Path(__file__).resolve().parent;d=json.loads((p/'FULL_CHANNEL_DIRECT_SOURCE_SWAP_V1.json').read_text());rows=[]
for s in d['summary']:
 if s['family']!='same_cohort':continue
 rr=[r for r in d['records'] if all(r[k]==s[k] for k in ('domain','family','kind','cohort')) and r['reference_energy']>0]
 rows.append(dict(domain=s['domain'],kind=s['kind'],cohort=s['cohort'],aggregate_error=s['effect_error'],worst_document_error=max((r['error_energy']/r['reference_energy'])**.5 for r in rr),documents_above_10pct=sum(r['error_energy']>.01*r['reference_energy'] for r in rr),documents=len(rr)))
 scales=[]
for domain in sorted(set(r['domain'] for r in d['rms_ratios'])):
 for family in ('same_cohort','opposite_cohort'):
  rr=[r for r in d['rms_ratios'] if r['domain']==domain and r['family']==family];scales.append(dict(domain=domain,family=family,minimum_rms_ratio=min(r['rms_ratio_min'] for r in rr),maximum_rms_ratio=max(r['rms_ratio_max'] for r in rr)))
result=dict(primary=rows,normalizer_ranges=scales,scope='Descriptive paired path intervention audit. Donor-recipient dependencies inherited; no iid uncertainty. RMS ratios quantify shifts but do not identify the cause of the reconstruction gap.')
(p/'FULL_CHANNEL_DIRECT_SOURCE_SWAP_AUDIT_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
