"""CPU paired receipt analysis and precision diagnosis; no checkpoint or fitting."""
import json,hashlib,time
from pathlib import Path
import torch
from paired_panel_bootstrap_v1 import PairedPanelBootstrap
P=Path(__file__).resolve().parent

def main():
    tic=time.perf_counter();torch.set_num_threads(2)
    r=json.loads((P/'TERMINAL_COMPLEMENT_SOURCE_V2_RESULT.json').read_text())
    a=P/'TERMINAL_COMPLEMENT_SOURCE_V2_STATES.pt';assert hashlib.sha256(a.read_bytes()).hexdigest()==r['artifact_sha256']
    old=torch.load(P/'TERMINAL_COMPLEMENT_SOURCE_V1_STATES.pt',map_location='cpu',weights_only=True)
    new=torch.load(a,map_location='cpu',weights_only=True)
    result={'panels':{},'state_replay_bitwise':True,'source_predictions':r['predictions'],'rounding_diagnostics':{},'scope':'Post-result paired intervals on reused sixteen-row panels, not OOD or independent confirmation. Original V1 instrument failure retained.'}
    for j,n in enumerate(['A1','A2']):
        boot=PairedPanelBootstrap(16,9112000+j);result['panels'][n]={};result['rounding_diagnostics'][n]=new[n]['rounding_diagnostics']
        for label,cells in r['reports'][n].items():
            result['panels'][n][label]={}
            for k,v in cells.items():
                result['panels'][n][label][k]={'relative_l2':v['relative_l2'],'paired_95_interval':boot.relative_l2(v['error_squared_per_row'],v['reference_squared_per_row'])}
        for label,state in old[n]['states'].items():
            for field,value in state.items():
                result['state_replay_bitwise'] &= torch.equal(value,new[n]['states'][label][field])
    result['cpu_seconds']=time.perf_counter()-tic
    result['artifact_sha256']=r['artifact_sha256'];result['audit_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out=P/'TERMINAL_COMPLEMENT_SOURCE_V2_AUDIT_RESULT.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
