"""Pool preregistered unchanged replication by raw energies, retaining panel outcomes."""
from pathlib import Path
import json,copy,math
P=Path(__file__).parent;panels=[json.loads((P/f'FRONTIER_FRESH_NATIVE_V{i}.json').read_text()) for i in (1,2)];out=copy.deepcopy(panels[0]);out['records']=panels[0]['records']+panels[1]['records'];out['panel_predictions']=[p['predictions'] for p in panels]
for domain in out['summary']:
 docs=sum([p['plan']['recipient_documents'][domain] for p in panels],[]);assert len(docs)==len(set(docs));out['plan']['recipient_documents'][domain]=docs
 for selection,candidates in out['summary'][domain].items():
  for candidate,families in candidates.items():
   for family,cohorts in families.items():
    for cohort in cohorts:
     rows=[r for r in out['records'] if all(r[k]==v for k,v in dict(domain=domain,selection=selection,candidate=candidate,family=family,cohort=cohort).items())];sites=sum(r['sites'] for r in rows);reference=sum(r['reference_energy'] for r in rows);assert sites>0 and reference>0
     cohorts[cohort]=dict(sites=sites,effect_relative_error=math.sqrt(sum(r['error_energy'] for r in rows)/reference),reference_logit_rms=math.sqrt(reference/(sites*50304)),ce_added=sum(r['ce_added_sum'] for r in rows)/sites,ce_disagreement=sum(r['ce_error_sum'] for r in rows)/sites)
out['cells']=[];out['all_candidate_comparisons']=[]
for domain in out['summary']:
 for selection,candidates in out['summary'][domain].items():
  for family,limit in [('natural',.15),('hybrid',.15),('change',.20)]:
   for cohort in ('all','continuation','spaced_word'):
    a,b,c=[candidates[n][family][cohort]['effect_relative_error'] for n in ('graph','separate','isotropic_baseline')];base=dict(domain=domain,selection=selection,family=family,cohort=cohort)
    out['cells'].append(dict(**base,graph_error=a,baseline_error=b,ratio=a/b,absolute_pass=a<=limit,relative_pass=a<=1.1*b));out['all_candidate_comparisons'].append(dict(**base,candidate='graph',error=a,absolute_pass=a<=limit,covariance_baseline_error=b,isotropic_baseline_error=c,covariance_relative_pass=a<=1.1*b,isotropic_relative_pass=a<=1.1*c))
rows=out['all_candidate_comparisons'];out['predictions']=dict(pred_a_instrument=all(p['predictions']['pred_a_instrument'] for p in panels),pred_b_absolute=all(r['absolute_pass'] for r in rows),pred_c_relative=all(r['covariance_relative_pass'] for r in rows),pred_d_both_baselines=all(r['covariance_relative_pass'] and r['isotropic_relative_pass'] for r in rows));out['plan']['scope']='Pooled unchanged two-panel evaluation:64FWdocuments/32codefiles. Raw error/reference energies pooled, each panel original donor map retained. Both panel verdicts remain; pooled pass cannot erase a panel failure.';out['seconds']=sum(p['seconds'] for p in panels);out['final_state_replay']=max(p['final_state_replay'] for p in panels);out['source_replay']=max(p['source_replay'] for p in panels)
(P/'FRONTIER_FRESH_NATIVE_POOLED_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out['predictions'])
