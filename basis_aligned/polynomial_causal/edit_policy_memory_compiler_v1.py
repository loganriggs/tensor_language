"""Exact linear memory quotient for an additive, externally driven feature stream.

This does not compile contextual feature producers or discover semantic circuits.
Allowed updates and the complete query-reader coefficient family are explicit inputs.
"""
import hashlib
import json
from pathlib import Path
import sympy as s


def compile_memory(updates, readers):
    """updates: name -> n x r; readers: p x n, all exact rational coefficients.

    Runtime updates z += compiled[name] phi; reader coefficients = decoder z.
    No original-state reconstruction appears in the runtime.
    """
    joined = s.Matrix.hstack(*updates.values())
    reachable = s.Matrix.hstack(*joined.columnspace())
    observed = readers * reachable
    selected = list(observed.T.rref()[1])
    if not selected:
        return dict(width=0, updates={k:s.zeros(0,u.cols) for k,u in updates.items()},
                    decoder=s.zeros(readers.rows,0), encoder=s.zeros(0,readers.cols),
                    reachable_width=reachable.cols)
    encoder = readers[selected, :]
    reduced = encoder * reachable
    decoder = observed * reduced.T * (reduced * reduced.T).inv()
    assert decoder * reduced == observed
    compiled = {k:encoder*u for k,u in updates.items()}
    for k,u in updates.items():
        assert decoder * compiled[k] == readers * u
    return dict(width=len(selected), updates=compiled, decoder=decoder,
                encoder=encoder, reachable_width=reachable.cols)


def feature(x, y):
    return s.Matrix([x**3, x*x*y, x*y*y, y**3])


def execute(program, tokens, masks):
    z = s.zeros(program['width'], 1)
    outputs = []
    for (x,y),mask in zip(tokens,masks,strict=True):
        # Unknown masks raise KeyError: the declared intervention domain is binding.
        z += program['updates'][mask] * feature(x,y)
        outputs.append(program['decoder']*z)
    return outputs


def fixture(masks, readers=None, gauge=None):
    readers = s.eye(12) if readers is None else readers
    updates={name:s.kronecker_product(s.Matrix(mask),s.eye(4))
             for name,mask in masks.items()}
    if gauge is not None:
        updates={name:gauge*u for name,u in updates.items()}
        readers=readers*gauge.inv()
    return updates,readers,compile_memory(updates,readers)


def main():
    policies={
        'natural':{'all':(1,1,1)},
        'a_b_tied':{'all':(1,1,1),'ab':(1,1,0),'c':(0,0,1),'off':(0,0,0)},
        'independent':{''.join(map(str,(a,b,c))):(a,b,c)
                       for a in (0,1) for b in (0,1) for c in (0,1)},
    }
    tokens=[(s.Integer(i%5-2),s.Integer((3*i)%7-3)) for i in range(19)]
    checks={}; widths={}; prefixes=0
    for name,masks in policies.items():
        updates,readers,p=fixture(masks)
        widths[name]=p['width']
        names=list(masks)
        schedule=[names[(3*i+1)%len(names)] for i in range(len(tokens))]
        outputs=execute(p,tokens,schedule)
        state=s.zeros(12,1)
        for i,((x,y),mask) in enumerate(zip(tokens,schedule)):
            state+=updates[mask]*feature(x,y)
            assert outputs[i]==readers*state
            prefixes+=1
        checks[name+'_all_prefixes_exact']=True
    checks['policy_changes_required_state']=widths=={'natural':4,'a_b_tied':8,'independent':12}

    # A legitimate smaller quotient when all query readers ignore the last two features.
    reader=s.zeros(6,12)
    for branch in range(3):
        for j in range(2):reader[2*branch+j,4*branch+j]=1
    _,_,blind=fixture(policies['independent'],reader)
    checks['reader_blind_state_removed']=blind['width']==6
    widths['independent_partial_readers']=blind['width']

    # Dense invertible change of physical state coordinates does not alter the decision.
    gauge=s.eye(12)+s.ones(12)
    _,_,plain=fixture(policies['a_b_tied'])
    _,_,gauged=fixture(policies['a_b_tied'],gauge=gauge)
    schedule=['all','ab','c','off']*4+['all','c','ab']
    checks['dense_gauge_exact']=execute(plain,tokens,schedule)==execute(gauged,tokens,schedule)
    checks['dense_gauge_width']=gauged['width']==plain['width']==8
    try:execute(plain,[(1,1)],['a_only'])
    except KeyError:checks['unregistered_edit_rejected']=True
    else:checks['unregistered_edit_rejected']=False

    # Producer removal affects every consumer; branch removals have independent histories.
    _,_,ind=fixture(policies['independent'])
    base=['111']*len(tokens)
    a_cut=['011' if i%2==0 else '111' for i in range(len(tokens))]
    b_cut=['101' if i%3==0 else '111' for i in range(len(tokens))]
    joint=[''.join(str(int(a)&int(b)) for a,b in zip(am,bm))
           for am,bm in zip(a_cut,b_cut)]
    def direct(schedule):
        state=s.zeros(12,1)
        for (x,y),mask in zip(tokens,schedule):
            state+=s.kronecker_product(s.Matrix(list(map(int,mask))),feature(x,y))
        return state
    for name,sch in [('base',base),('a_removal',a_cut),('b_removal',b_cut),('joint',joint)]:
        checks[name+'_replay']=execute(ind,tokens,sch)[-1]==direct(sch)
    checks['branch_effects_compose_at_this_linear_reader']=(
        direct(joint)-direct(base)==direct(a_cut)+direct(b_cut)-2*direct(base))
    checks['shared_producer_removal']=execute(ind,[(0,0)]*len(tokens),base)[-1]==s.zeros(12,1)
    checks['distinct_histories_observable']=direct(a_cut)[:4]!=direct(a_cut)[4:8]

    # Global, fixed gates known to the decoder need only one feature history plus metadata.
    common=sum((feature(x,y) for x,y in tokens),s.zeros(4,1))
    checks['global_gate_side_information_exception']=all(
        direct([name]*len(tokens))==s.kronecker_product(s.Matrix(mask),common)
        for name,mask in policies['independent'].items())
    # Four cubic features span R^4 over real inputs: Vandermonde evaluations certify it.
    checks['feature_span_exact']=s.Matrix.hstack(*(feature(1,t) for t in range(4))).det()!=0
    assert all(checks.values()),checks
    out=dict(experiment='edit_policy_memory_compiler_v1',passed=True,checks=checks,
             state_widths=widths,exact_prefix_replays=prefixes,shared_feature_count=4,
             model_forwards=0,native_parameters_removed=0,
             scope='Exact rational planted additive feature stream; linear memory encoders; no trained circuit discovery.',
             source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=Path(__file__).with_name('EDIT_POLICY_MEMORY_COMPILER_V1_CONTROLS.json')
    with path.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
