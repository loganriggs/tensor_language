"""Prefix-level sensitivity of root-case removal; post-screen diagnostic."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent

def main():
 a=json.loads((P/'ROOT_CASE_ABLATION_V1.json').read_text());out=[]
 for domain in ['fineweb','stdlib']:
  for alpha in [.25,1.]:
   rows=[r for r in a['rows'] if r['domain']==domain and r['alpha']==alpha];new=[r for r in rows if r['condition']=='newline'];other=[r for r in rows if r['condition']=='other']
   def mean(rr,key):return sum(r[key] for r in rr)/sum(r['n'] for r in rr)
   ratios=[]
   for doc in sorted({r['document'] for r in rows}):
    n=[r for r in new if r['document']!=doc];o=[r for r in other if r['document']!=doc]
    ratios.append(dict(excluded_prefix=doc,ratio=mean(n,'native_case_abs_sum')/mean(o,'native_case_abs_sum'),newline_mean=mean(n,'native_case_sum'),other_mean=mean(o,'native_case_sum')))
   out.append(dict(domain=domain,alpha=alpha,newline_negative_prefixes=sum(r['native_case_sum']<0 for r in new),other_positive_prefixes=sum(r['native_case_sum']>0 for r in other),newline_prefixes=len(new),other_prefixes=len(other),pooled_selectivity_ratio=mean(new,'native_case_abs_sum')/mean(other,'native_case_abs_sum'),exclude_prefix=ratios))
 (P/'ROOT_CASE_DOCUMENT_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n')
 for r in out:print(json.dumps({k:v for k,v in r.items() if k!='exclude_prefix'}), 'exclusion_ratio_range',min(x['ratio'] for x in r['exclude_prefix']),max(x['ratio'] for x in r['exclude_prefix']))
if __name__=='__main__':main()
