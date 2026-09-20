"""Partition native directional cubic response using an exact readout root."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from final_readout_field_program import evaluate_fields, evaluate_jet

torch.set_num_threads(2)
P=Path(__file__).resolve().parent
A=P.parent/'bilinear_quotient/circuits/followups'


def third_along_fields(jet, third_fields=None):
    t=torch.zeros(len(jet),dtype=torch.float64,requires_grad=True)
    fields=jet[:,0]+t[:,None]*jet[:,1]+t[:,None].square()*jet[:,2]
    if third_fields is not None:
        fields=fields+t[:,None]**3*third_fields/6
    values=evaluate_fields(fields);result=[]
    for o in range(values.shape[1]):
        first=torch.autograd.grad(values[:,o].sum(),t,create_graph=True,retain_graph=True)[0]
        second=torch.autograd.grad(first.sum(),t,create_graph=True,retain_graph=True)[0]
        third=torch.autograd.grad(second.sum(),t,retain_graph=o<values.shape[1]-1)[0]
        result.append(third.detach())
    return torch.stack(result,dim=1)


def main():
    field_path=A/'source_ood_v2_readout_fields_result.json'
    cubic_path=A/'source_ood_v2_cubic_radius_result.json'
    fields=json.loads(field_path.read_text());cubic=json.loads(cubic_path.read_text())
    assert fields['predictions']['pred_a_instrument'] and cubic['predictions']['pred_a_instrument']
    native={(c['panel'],c['role'],c['template']):c for c in cubic['contexts']}
    targets={(c['panel'],c['role'],c['family'],c['base_arm'],c['radius']):c for c in fields['records']}
    records=[];partitions=[];replays=[];control=None
    for c in fields['contexts']:
        prior=native[c['panel'],c['role'],c['template']]
        rows=json.loads((P/f"SOURCE_OOD_V2_{c['panel'].upper()}_ROWS.json").read_text())
        rows=[r for r in rows if r['template']==c['template']]
        for arm,rawjet in c['field_jets'].items():
            jet=torch.tensor(rawjet,dtype=torch.float64)
            root=third_along_fields(jet).numpy()
            total=np.array(prior['third'][arm]);upstream=total-root
            if control is None:
                torch.manual_seed(99);z3=.1*torch.randn_like(jet[:,0])
                true=third_along_fields(jet,z3)
                _,transported=torch.autograd.functional.jvp(evaluate_fields,jet[:,0],z3)
                control=float((true-torch.tensor(root)-transported).abs().max())
                assert control<1e-10
            baseline=evaluate_fields(jet[:,0]).numpy()
            for family in dict.fromkeys(r['family'] for r in rows):
                ids=[i for i,r in enumerate(rows) if r['family']==family]
                den=max(np.linalg.norm(total[ids,0]),1e-30)
                partitions.append(dict(panel=c['panel'],role=c['role'],family=family,arm=arm,
                                       readout_third_ratio=float(np.linalg.norm(root[ids,0])/den),
                                       upstream_third_ratio=float(np.linalg.norm(upstream[ids,0])/den)))
            for radius in [1.,.5,.25]:
                field_effect=baseline-evaluate_jet(jet,radius).numpy()
                corrected=field_effect-radius**3*upstream/6
                for family in dict.fromkeys(r['family'] for r in rows):
                    ids=[i for i,r in enumerate(rows) if r['family']==family]
                    target=targets[c['panel'],c['role'],family,arm,radius]
                    y=np.array(target['target']);den=max(np.linalg.norm(y[:,0]),1e-30)
                    replay=float(np.linalg.norm(field_effect[ids,0]-y[:,0])/den)
                    replays.append(abs(replay-target['field_number_error']))
                    records.append(dict(panel=c['panel'],role=c['role'],family=family,arm=arm,radius=radius,
                                        number_error=float(np.linalg.norm(corrected[ids,0]-y[:,0])/den),
                                        modal_error=float(max(np.linalg.norm(corrected[ids,1:]-y[:,1:],axis=0))/den)))
    assert max(replays)<1e-10
    full=[r for r in records if r['radius']==1.]
    out=dict(source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [field_path,cubic_path]},
             planted_chain_rule_error=control,max_field_error_replay=max(replays),partitions=partitions,
             records=records,full_pass=all(r['number_error']<=.1 and r['modal_error']<=.05 for r in full),
             full_number_max=max(r['number_error'] for r in full),full_modal_max=max(r['modal_error'] for r in full),
             scope='Opened diagnostic. Upstream term inferred as total minus root third derivative, not separately measured. Requires native third derivatives plus fields;31coefficients/ray before direction and generators.')
    (P/'READOUT_CUBIC_PARTITION_V1_RESULT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['records','partitions','source_sha256']},indent=2))
    print(json.dumps([r for r in partitions if r['panel']=='opposite' and r['role']=='subject' and r['family']=='beside_subject|singular'],indent=2))


if __name__=='__main__':main()
