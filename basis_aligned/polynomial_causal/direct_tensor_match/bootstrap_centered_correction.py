from pathlib import Path
import torch,json
p=Path(__file__).resolve().parent;torch.set_num_threads(2);r=json.loads((p/'MIDPOINT_CENTERED_CORRECTION_NATIVE_RAW_V1.json').read_text())['records'];gen=torch.Generator().manual_seed(26206);records=[]
for domain,count in [('fineweb',32),('code',16)]:
 draw=torch.randint(count,(10000,count),generator=gen)
 for family in ['replacement','removal','same_token']:
  a=[v for v in r if v['domain']==domain and v['family']==family and v['candidate']=='product_baseline'];b=[v for v in r if v['domain']==domain and v['family']==family and v['candidate']=='product_rank32'];assert len(a)==len(b)==count;assert [v['sites'] for v in a]==[v['sites'] for v in b]
  field=lambda rr,key:torch.tensor([v[key] for v in rr],dtype=torch.float64)
  def interval(x):return torch.quantile(x,torch.tensor([.025,.975],dtype=x.dtype)).tolist()
  if family=='replacement':
   sites=field(a,'sites');aa=field(a,'ce_added');bb=field(b,'ce_added');diff=(bb[draw].sum(1)-aa[draw].sum(1))/sites[draw].sum(1);absolute=bb[draw].sum(1)/sites[draw].sum(1);records.append(dict(domain=domain,family=family,corrected_ce=float(bb.sum()/sites.sum()),corrected_ce_interval=interval(absolute),ce_difference=float((bb-aa).sum()/sites.sum()),ce_difference_interval=interval(diff)))
  else:
   en=field(a,'native_centered_effect_energy');assert torch.equal(en,field(b,'native_centered_effect_energy'));ea=field(a,'centered_effect_error_energy');eb=field(b,'centered_effect_error_energy');ratio=(eb[draw].sum(1)/ea[draw].sum(1)).sqrt();error=(eb[draw].sum(1)/en[draw].sum(1)).sqrt();records.append(dict(domain=domain,family=family,corrected_error=float((eb.sum()/en.sum()).sqrt()),corrected_error_interval=interval(error),error_ratio=float((eb.sum()/ea.sum()).sqrt()),error_ratio_interval=interval(ratio),worst_document_error=float((eb/en).sqrt().max())))
out=p/'MIDPOINT_CENTERED_CORRECTION_BOOTSTRAP_V1.json';assert not out.exists();out.write_text(json.dumps(dict(records=records,bootstrap_draws=10000,scope='Paired documentbootstrap on reused diagnostic panel. Records in deterministic documentloop order; token/site and referenceenergy agreement asserted. Descriptive95%intervals, notfamilywisebounds or proof ofpopulationrepresentativeness. Doesnot change preregisteredpoint-estimate gates.'),indent=2)+'\n');print(out.read_text())
