"""Rowwise audit of registered recursive removal results; no refitting or exclusions."""
import json,torch
from pathlib import Path
P=Path(__file__).resolve().parent
def main():
 a=torch.load(P/'SCALAR_PRODUCERS_RECURSIVE_REMOVAL_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
 rows=[r for r in json.loads((P/'SCALAR_PRODUCERS_NEWLINE_NATURAL_V2_ROWS.json').read_text())['rows'] if r['pool']=='fineweb']
 delta=a['newline_ce']-a['newline_ce'][:,:1]
 failures=[dict(index=i,source_row=rows[i]['source_row'],family=rows[i]['family'],baseline_ce=float(a['newline_ce'][i,0]),ce_changes=delta[i].tolist()) for i in range(32) if float(delta[i,1:4].abs().max())>.1]
 p=torch.load(P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt',weights_only=True,map_location='cpu')
 writes=torch.load(P/'SCALAR_PRODUCERS_NATIVE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['contributions']
 w=p['output_coefficients'][:,None]*p['shared_output'][None,:]
 scalars=(writes*w[:,None,None,:]).sum(-1)/w.square().sum(-1)[:,None,None]
 replay=float((scalars[...,None]*w[:,None,None,:]-writes).norm()/writes.norm())
 result=dict(newline_failure_rows=failures,all_rows_retained=True,scalar_reencoding_error=replay,pred_a=replay<=1e-12,scope='Outcome audit and exact output de-encoding only. One-row max preservation miss remains. Native scalar validation pending serial test.')
 out=P/'SCALAR_PRODUCERS_RECURSIVE_AUDIT_V1_RESULT.json';assert not out.exists()
 out.write_text(json.dumps(result,indent=2)+'\n');torch.save(dict(scalars=scalars),P/'SCALAR_PRODUCERS_SERIAL_NATIVE_SCALARS_V1_ARTIFACT.pt');print(json.dumps(result))
if __name__=='__main__':main()
