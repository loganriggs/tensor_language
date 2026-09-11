"""Descriptive parent/private energy accounting and durable executable export."""
import hashlib
import json
from pathlib import Path
import torch
from structured_branch_amplitudes_v1 import inner


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    root=Path(__file__).parent;rows=[]
    for label in ('spectral','native'):
        source=Path(f'/dev/shm/bilin18_ll1_shared_parent_optimize_v2_{label}.pt')
        state=torch.load(source,weights_only=True,map_location='cpu')
        parent=state['parent'];groups=state['groups']
        aa=[parent];bb=[parent];ww=[sum(g['alpha']*g['writer'] for g in groups)]
        private_a=[];private_w=[]
        for g in groups:
            aa.append(parent);bb.append(g['private']@g['beta']);ww.append(2*g['writer'])
            private_a.append(g['private'].T);private_w.append(g['writer'][:,None]*g['lam'])
        shared=(torch.stack(aa),torch.stack(bb),torch.stack(ww,1))
        a=torch.cat(private_a);private=(a,a,torch.cat(private_w,1))
        whole=(torch.cat((shared[0],a)),torch.cat((shared[1],a)),torch.cat((shared[2],private[2]),1))
        es=float(inner(shared,shared));ep=float(inner(private,private));cross=2*float(inner(shared,private));total=float(inner(whole,whole))
        error=abs(es+ep+cross-total)/total
        executable=dict(parent=parent,groups=[{k:g[k] for k in ('private','beta','lam','alpha','writer')} for g in groups])
        floats=parent.numel()+sum(v.numel() for g in executable['groups'] for v in g.values())
        path=root/f'LL1_SHARED_PARENT_OPTIMIZE_V2_{label.upper()}_GRAPH.pt'
        torch.save(executable,path)
        row=dict(label=label,shared_parent_energy_over_pair=es/total,private_energy_over_pair=ep/total,
                 signed_cross_energy_over_pair=cross/total,energy_identity_error=error,
                 pair_energy_over_native=total/99245061353.47293,executable_floats=floats,
                 artifact=path.name,artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 pred_a=error<=1e-10 and floats==38078)
        rows.append(row)
    result=dict(rows=rows,scope='Exact coefficient accounting of algebraic shared-parent deletion in fitted pairs. Ratios include signed cross-group interactions; not a native behavioral intervention or identified semantic hierarchy.')
    (root/'LL1_SHARED_PARENT_CONTRIBUTION_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
