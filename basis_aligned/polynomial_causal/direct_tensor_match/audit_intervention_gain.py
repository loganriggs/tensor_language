"""Post-hoc best scalar correction in effect space; diagnostic, never exported."""
import json,math
from pathlib import Path
P=Path(__file__).resolve().parent

def main():
 rows=[]
 for domain,file in [('fineweb','NATIVE_MODE_INTERVENTION_V1.json'),('code','CODE_SHIFT_MODES_V1.json')]:
  r=json.loads((P/file).read_text())
  for mode in r['plan']['modes']:
   rs=[x for x in r['records'] if x['mode']==mode]
   for prefix in ['effect','ce_effect']:
    native=sum(x['native_'+prefix+'_energy'] for x in rs);pred=sum(x['predicted_'+prefix+'_energy'] for x in rs);dot=sum(x[prefix+'_dot'] for x in rs)
    rows.append(dict(domain=domain,mode=mode,metric=prefix,oracle_gain=dot/pred,best_scalar_relative_error=math.sqrt(max(0,1-dot*dot/(native*pred))),original_relative_error=math.sqrt(sum(x[prefix+'_error_energy'] for x in rs)/native)))
 result=dict(records=rows,scope='Same-panel unconstrained scalar least-squares oracle on post-softcap effect vectors. Lower bound for a scalar rescaling of those vectors, not for nonlinear model edits at changed amplitudes; no candidate refit/export.')
 (P/'INTERVENTION_GAIN_ORACLE_V1.json').write_text(json.dumps(result,indent=2)+'\n')
 for x in rows:
  if x['metric']=='effect':print(x)
if __name__=='__main__':main()
