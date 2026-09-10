"""Causal source support and linear-reader bookkeeping on saved native edges."""
import hashlib,json
from pathlib import Path
import numpy as np
import mixed_state_projector_v1 as Q


def main():
    p=Path(__file__).resolve().parent;source=p/'THIRD_NOUN_L9_ROUTE_VALUE_V1_RESULT.json'
    d=json.loads(source.read_text());assert d['predictions']['pred_a_instrument'] and d['predictions']['pred_b_materiality']
    rows=json.loads((p/'THIRD_NOUN_ANIMACY_V1_ROWS.json').read_text())
    assert all(len(r['ids'])==10 for w in rows['worlds'] for r in w['rows'])
    # Every h edge changes only position7; every o edge changes only position4.
    for world in rows['worlds']:
        lookup={tuple(r['factors']):r['ids'] for r in world['rows']}
        for c,ids in lookup.items():
            for factor,position in ((2,4),(4,7)):
                other=list(c);other[factor]*=-1
                assert [j for j,(x,y) in enumerate(zip(ids,lookup[tuple(other)])) if x!=y]==[position]
    numerators={k:np.array(v) for k,v in d['branch_edge_reader_numerators'].items()}
    rms=np.array(d['native_final_rms']);q=Q.projector(d['corners'],2,4)
    # Exact linear normalized-reader difference with the native RMS frozen per row.
    edges={k:(v[...,0]-v[...,1])/rms[:,:,None,None] for k,v in numerators.items()}
    project=lambda x:np.einsum('ij,wj...->wi...',q,x)
    all_edges=sum(edges.values());target=project(all_edges.sum((2,3)))
    value_early=float(np.linalg.norm(numerators['value'][:,:,:,:7])/max(1e-30,np.linalg.norm(numerators['value'])))
    cross_early=float(np.linalg.norm(numerators['cross'][:,:,:,:4])/max(1e-30,np.linalg.norm(numerators['cross'])))
    assert value_early<1e-6 and cross_early<1e-6
    reports=[]
    for w,ref in enumerate(d['reports']):
        t=target[w];den=float(t@t)
        def summary(vector):
            return {'signed_projection_on_local_linear_effect':float(vector@t/den),
                    'norm_ratio':float(np.linalg.norm(vector)/np.sqrt(den))}
        groups={}
        for name,array in edges.items():
            for label,positions in [('before_object',slice(0,4)),('earlier_number',slice(4,7)),('later_context',slice(7,10))]:
                groups[f'{name}:{label}']=summary(project(array[:,:,:,positions].sum((2,3)))[w])
        heads={f'head:{h:02d}':summary(project(all_edges[:,:,h,:].sum(-1))[w]) for h in range(9)}
        assert abs(sum(x['signed_projection_on_local_linear_effect'] for x in groups.values())-1)<1e-10
        assert abs(sum(x['signed_projection_on_local_linear_effect'] for x in heads.values())-1)<1e-10
        reports.append({'world_id':ref['world_id'],'source_branches':groups,'heads':heads,
                        'local_linear_effect_rms':float(np.sqrt(den/32))})
    result={'passed':True,'model_forwards':0,'value_branch_before_category_relative':value_early,
        'cross_branch_before_object_relative':cross_early,'reports':reports,
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'rows_sha256':hashlib.sha256((p/'THIRD_NOUN_ANIMACY_V1_ROWS.json').read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Exact causal availability and linear normalized-reader attribution with per-row native RMS fixed, before softcap. Head/source projections are not independent physical ablations, full-vocabulary fidelity, or held-out circuit identification. First-value token-local payload cannot alone supply the inherited oh value term.'}
    with (p/'THIRD_NOUN_CAUSAL_SOURCE_SPLIT_V1_RESULT.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'passed':True,'value_early':value_early,'cross_early':cross_early,
        'worlds':[{'world_id':r['world_id'],
            'earlier_number_cross':r['source_branches']['cross:earlier_number'],
            'later_value':r['source_branches']['value:later_context'],
            'later_cross':r['source_branches']['cross:later_context'],
            'largest_head':max(r['heads'],key=lambda h:r['heads'][h]['signed_projection_on_local_linear_effect'])} for r in reports]},indent=2))


if __name__=='__main__':main()
