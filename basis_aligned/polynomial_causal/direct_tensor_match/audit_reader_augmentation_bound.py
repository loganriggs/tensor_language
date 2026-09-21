"""Executable consequence of the measured fixed-reader sensitivity spectrum."""
import json,math
from pathlib import Path
import torch
P=Path(__file__).resolve().parent


def relative_tail(eigenvalues,rank):
    e=eigenvalues.sort(descending=True).values.clamp_min(0)
    if not 0<=rank<=len(e):raise ValueError('invalid rank')
    return (e[rank:].sum()/e.sum()).sqrt()


def main():
    torch.set_num_threads(2);torch.manual_seed(4900);j=torch.randn(17,11,dtype=torch.float64);_,s,vh=torch.linalg.svd(j,full_matrices=False);controls=[]
    for rank in [0,1,4,11]:
        q=vh[:rank].T;actual=(j-(j@q)@q.T).norm()/j.norm();bound=relative_tail(s.square(),rank);err=float(abs(actual-bound));assert err<1e-12;controls.append(dict(rank=rank,relative_tail=float(bound),oracle_error=err))
    source=json.loads((P/'QUARTIC_READER_SPAN_V1.json').read_text());fraction=source['aggregate_unread_jacobian_fraction'];capture=source['top32_unread_energy_fraction'];residual32=fraction*math.sqrt(1-capture);capture_needed_for10=1-(.10/fraction)**2
    result=dict(controls=controls,existing_reader_rank=source['reader_rank'],best_possible_remaining_fraction_after32_added_readers=residual32,required_unread_energy_capture_for10percent_total=capture_needed_for10,remaining_fraction_at_exact90percent_capture=fraction*math.sqrt(.1),scope='Local derivative error relative to total teacher Jacobian norm across fixed16anchors. Existing256-reader span held fixed; optimally add32directions and allow an otherwise unrestricted differentiable function. Bound need not be attainable by the constrained quartic graph. Rotating existing readers or measuring other input geometries changes the question.')
    (P/'READER_AUGMENTATION_BOUND_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
