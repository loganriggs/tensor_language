"""Reproduce the CPU-only observable-memory compilation of the original pilot.

No trained-model claim. An existing result is immutable: this script refuses overwrite.
"""
import hashlib
import json
from pathlib import Path
import time
import sympy as s
import bilinear_reconstruction_reference as R


def compile_memory(updates, readers):
    """Same quotient identity as v1; domain arithmetic and square pivot solve.

    The original generic Matrix columnspace path was stopped after >120 CPU seconds
    on independent_routers. No scientific threshold or coefficient changed.
    """
    joined=s.Matrix.hstack(*updates.values())
    columns=list(joined.to_DM().rref()[1])
    reachable=joined[:,columns]
    observed=readers*reachable
    selected=list(observed.T.to_DM().rref()[1])
    encoder=readers[selected,:]
    reduced=encoder*reachable
    pivot=list(reduced.to_DM().rref()[1])
    decoder=observed[:,pivot]*reduced[:,pivot].inv(method='DM')
    assert decoder*reduced==observed
    compiled={name:encoder*u for name,u in updates.items()}
    for name,u in updates.items():assert decoder*compiled[name]==readers*u
    return dict(width=len(selected),reachable_width=reachable.cols,
                updates=compiled,decoder=decoder)


def main():
    out={}
    for kind in ('planted','perturbed','independent_routers'):
        start=time.monotonic()
        f=R.exact_fixture(kind)
        p=compile_memory({'head0':f['head_updates'][0],'head1':f['head_updates'][1]},f['readers'])
        out[kind]=dict(reachable_width=p['reachable_width'],observable_memory_width=p['width'],
                       coefficient_identities_exact=True,seconds=time.monotonic()-start,
                       compiled_update_constants=sum(u.rows*u.cols for u in p['updates'].values()),
                       compiled_reader_constants=p['decoder'].rows*p['decoder'].cols)
        print(kind,out[kind],flush=True)
    root=Path(__file__).parent
    record=dict(experiment='edit_policy_original_pilot_v1',fixtures=out,model_forwards=0,
                semantic_discovery=False,bindings={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [root/'bilinear_reconstruction_reference.py',root/'edit_policy_memory_compiler_v1.py',Path(__file__)]},
                first_attempt='Generic Matrix elimination stopped after >120 CPU seconds; two fixtures completed, third did not. Domain arithmetic and pivot solve used for this receipt.')
    with (root/'EDIT_POLICY_ORIGINAL_PILOT_V1_RESULT.json').open('x') as f:
        json.dump(record,f,indent=2);f.write('\n')


if __name__=='__main__':main()
