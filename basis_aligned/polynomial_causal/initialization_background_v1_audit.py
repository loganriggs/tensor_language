"""Check the literal initialization-background hypothesis, with provenance limits."""
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from native_reader_msp_generalization_v1 import P,CK

def main():
    root=P.parents[1]
    sys.path.insert(0,str(root/'jacclust'))
    from tt_model import Bilinear
    torch.set_num_threads(2)
    tiny=Bilinear(SimpleNamespace(n_embd=8,expansion_factor=4,gated=False))
    x=torch.arange(40,dtype=torch.float32).reshape(5,8)/17
    output=tiny(x)
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    rows=[]
    for name in ('Left','Right'):
        w=sd[f'transformer.h.17.mlp.{name}.weight'].double()
        bound=1/math.sqrt(w.shape[1])
        maximum_initial_norm=math.sqrt(w.numel())*bound
        norm=float(w.norm())
        rows.append(dict(name=name,shape=list(w.shape),initial_entry_bound=bound,
            maximum_initial_frobenius_norm=maximum_initial_norm,native_frobenius_norm=norm,
            native_over_maximum_initial_norm=norm/maximum_initial_norm,
            entries_outside_initial_bound_fraction=float((w.abs()>bound).double().mean()),
            minimum_coordinate_update_norm_fraction=max(0,1-maximum_initial_norm/norm)))
    source=Path(root/'jacclust/tt_model.py')
    result=dict(predictions=dict(pred_a_constructor_zero=bool((output==0).all()) and bool((tiny.Down.weight==0).all()),
        pred_b_norm=all(r['native_over_maximum_initial_norm']>=5 for r in rows),
        pred_c_entries=all(r['entries_outside_initial_bound_fraction']>=.5 for r in rows)),
        readers=rows,native_down_norm=float(sd['transformer.h.17.mlp.Down.weight'].double().norm()),
        native_quadratic_energy=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())['native_total'],
        constructor_output_max=float(output.abs().max()),torch_version=torch.__version__,
        constructor_source=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
        linear_reset_source=inspect.getsource(torch.nn.Linear.reset_parameters),
        provenance='Checked-in constructor plus installed Linear reset. No saved training initialization found in the inspected local model snapshot; external training-time overrides are not reconstructed.',
        scope='Rules out an unchanged small constructor initialization dominating current reader entries under this initializer. Does not identify signal/noise, defeat gauge rescaling, or justify removing spectral bulk.')
    (P/'INITIALIZATION_BACKGROUND_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='linear_reset_source'}))

if __name__=='__main__':main()
