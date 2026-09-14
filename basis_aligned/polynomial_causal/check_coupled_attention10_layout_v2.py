"""Reproduce native layout boundary and compare every frozen grid strength."""
from pathlib import Path
from datetime import datetime,timezone
import json,torch
from coupled_attention10_ports_v1 import execute as flat_execute
from coupled_attention10_ports_v2 import execute


@torch.no_grad()
def main():
    p=Path(__file__).resolve().parent;out=p/'COUPLED_ATTENTION10_LAYOUT_V2_CONTROL.json'
    assert not out.exists();torch.set_num_threads(2)
    headed=torch.load(p/'COUPLED_ATTENTION10_NATIVE_V1_EXAMPLE_PROGRAM.pt',weights_only=True)['attention']
    assert headed['first_values'].ndim==4
    flat=dict(headed);flat['first_values']=headed['first_values'].flatten(-2)
    errors=[]
    for a in [-1,0,.5,1,2]:
        for b in [-1,0,.5,1,2]:
            reference=flat_execute(flat,a,b)
            errors.append(float((execute(headed,a,b)-reference).abs().max()))
            errors.append(float((execute(flat,a,b)-reference).abs().max()))
    malformed=dict(flat);malformed['first_values']=flat['first_values'][...,:128]
    try:execute(malformed,0,0)
    except ValueError:rejected=True
    else:rejected=False
    result=dict(utc=datetime.now(timezone.utc).isoformat(),pred_a=max(errors)==0 and rejected,
                max_layout_difference=max(errors),cases=len(errors),malformed_rejected=rejected,
                scope='Layout-only recovery, fixed saved native program; no weights or mathematical criteria changed. Native suffix rerun still required.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
