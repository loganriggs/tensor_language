"""Fit only three scalar constants on original448 fitting states; frozen programs intact."""
import json,hashlib
from pathlib import Path
import torch
from pairwise_component_interface import component_scalars
from source_interface import residual_write
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False)
 data=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);ids=data['indices'];z,h=data['z'],data['h'];truth=torch.stack([p['truth'] for p in data['pairs']],-1)
 programs={'graph':'FRONTIER_FRESH_GRAPH_V1.pt','separate':'FRONTIER_FRESH_BASELINE_CALIBRATION_SHAPED_V1.pt','isotropic_baseline':'FRONTIER_FRESH_BASELINE_NATIVE_ISOTROPIC_V1.pt'}
 out=dict(calibration_rows=ids.tolist(),hashes={},programs={},scope='Original448 fitting states only. Remaining cached states already examined in research, not claimed fresh. Constants change component levels, not scalar derivatives or donor differences. Coefficient-tensor numerator scores do not measure this additional normalized-component constant.')
 for name,file in programs.items():
  p=torch.load(P/file,weights_only=True);out['hashes'][file]=hashlib.sha256((P/file).read_bytes()).hexdigest()
  if name=='graph':values=component_scalars(z,h,p)
  else:
   writer=data['residual_writer'];values=torch.stack([residual_write(z,h,p[str(j)])@writer/writer.square().sum() for j in range(3)],-1)
  error=values-truth;offset=error[ids].mean(0);corrected=error-offset
  outside=torch.ones(len(z),dtype=torch.bool);outside[ids]=False
  result=dict(offset=offset.tolist(),extra_stored_floats=3,extra_additions_per_site=3,additional_source_products=0,metrics={})
  for label,select in (('fit',ids),('other_cached',outside)):
   denominator=(truth[select]-truth[select].mean(0)).norm(dim=0)
   result['metrics'][label]=dict(before=(error[select].norm(dim=0)/denominator).tolist(),after=(corrected[select].norm(dim=0)/denominator).tolist(),mean_error=error[select].mean(0).tolist())
  assert torch.all(corrected[ids].square().sum(0)<=error[ids].square().sum(0)+1e-8)
  out['programs'][name]=result
 records=json.loads((P/'FRONTIER_READ_ERROR_V1.json').read_text())['records'];cells=[]
 for panel in (1,2):
  for family in ('natural','hybrid','change'):
   for cohort in ('all','continuation','spaced_word'):
    c=dict(panel=panel,family=family,cohort=cohort,candidates={})
    for name in ('graph','separate'):
     rr=[r for r in records if r['panel']==panel and r['family']==family and r['cohort']==cohort and r['candidate']==name]
     n=sum(r['sites'] for r in rr);e=sum(r['error_energy'] for r in rr);s=sum(sum(r['term_sums']) for r in rr);offset=out['programs'][name]['offset'][2] if family!='change' else 0
     c['candidates'][name]=dict(before_energy=e,after_energy=e-2*offset*s+n*offset**2)
    g,b=c['candidates']['graph'],c['candidates']['separate'];c['before_ratio']=(g['before_energy']/b['before_energy'])**.5;c['after_ratio']=(g['after_energy']/b['after_energy'])**.5;cells.append(c)
 out['opened_code_scalar_cells']=cells
 out['predictions']=dict(pred_a_fit_monotone=True,pred_b_code_spaced_word=all(c['after_ratio']<=1.1 for c in cells if c['cohort']=='spaced_word'),pred_c_no_new_relative_failures=all(c['after_ratio']<=1.1 for c in cells if c['before_ratio']<=1.1))
 (P/'COMPONENT_CONSTANT_CALIBRATION_V1.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(dict(programs=out['programs'],predictions=out['predictions'],spaced_word=[c for c in cells if c['cohort']=='spaced_word']),indent=2))
if __name__=='__main__':main()
