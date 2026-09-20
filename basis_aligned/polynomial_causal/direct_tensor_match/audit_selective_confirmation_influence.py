"""Recipient-document influence, not IID confidence intervals for linked donors."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent

def main():
 old=json.loads((P/'SELECTIVE_CONFIRMATION_SWAP_ORIGINAL_V1.json').read_text());new=json.loads((P/'SELECTIVE_CONFIRMATION_SWAP_SELECTIVE_V1.json').read_text());sources=json.loads((P/'SELECTIVE_CONFIRMATION_PANELS_V1.json').read_text())['code_sources'];out=[]
 for domain in ['fineweb','code']:
  a=[r for r in old['records'] if r['domain']==domain and r['family']=='same_token' and r['mode']==1];b=[r for r in new['records'] if r['domain']==domain and r['family']=='same_token' and r['mode']==1];assert len(a)==len(b)
  def metrics(rows):
   native=sum(r['native_effect_energy'] for r in rows);pred=sum(r['predicted_effect_energy'] for r in rows);error=sum(r['effect_error_energy'] for r in rows);dot=sum(r['effect_dot'] for r in rows)
   return dict(relative_error=(error/native)**.5,cosine=dot/(native*pred)**.5,sse=error)
  doc=[]
  for i,(o,n) in enumerate(zip(a,b)):
   assert o['document']==n['document'];om=metrics(a[:i]+a[i+1:]);nm=metrics(b[:i]+b[i+1:]);doc.append(dict(document=n['document'],source=sources[n['document']]['path'] if domain=='code' else None,native_energy=n['native_effect_energy'],old_error_energy=o['effect_error_energy'],new_error_energy=n['effect_error_energy'],leave_one_out_old=om,leave_one_out_new=nm))
  out.append(dict(domain=domain,aggregate_old=metrics(a),aggregate_new=metrics(b),documents_improved=sum(n['effect_error_energy']<o['effect_error_energy'] for o,n in zip(a,b)),documents=len(a),maximum_new_error_fraction=max(r['effect_error_energy'] for r in b)/sum(r['effect_error_energy'] for r in b),leave_one_out_new_error_range=[min(r['leave_one_out_new']['relative_error'] for r in doc),max(r['leave_one_out_new']['relative_error'] for r in doc)],leave_one_out_new_cosine_range=[min(r['leave_one_out_new']['cosine'] for r in doc),max(r['leave_one_out_new']['cosine'] for r in doc)],gain_positive_all_leave_one_out=all(r['leave_one_out_new']['sse']<r['leave_one_out_old']['sse'] for r in doc),documents_detail=doc))
 result=dict(records=out,scope='Descriptive leave-one-recipient-document influence. Donor maps remain fixed and connect documents; this is not an independent-document bootstrap or calibrated confidence interval. No refitting or selection.')
 (P/'SELECTIVE_CONFIRMATION_INFLUENCE_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items() if k!='documents_detail'} for r in out],indent=2))
if __name__=='__main__':main()
