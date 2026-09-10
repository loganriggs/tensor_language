"""Paired outcome uncertainty and exact native-state product-rule diagnostics."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent


def main():
    src=P/'CORRELATIVE_ROUTE_VALUE_V1_RESULT.json';r=json.loads(src.read_text());assert r['predictions']['pred_a_instrument']
    maps=torch.load(P/'CORRELATIVE_WEIGHT_PORTS_V1_MAPS.pt',weights_only=True,map_location='cpu')
    rng=np.random.default_rng(9111304);reports={}
    for name,data in r['reports'].items():
        ix=rng.integers(0,16,size=(4000,16));base=np.array(data['native_base_margin']);den=base+np.array(data['native_donor_margin'])
        joint=np.array(data['effects']['joint']['effect_squared_per_row']);arms={}
        for arm,e in data['effects'].items():
            error=np.array(e['full_error_squared_per_row'])
            ratios=np.sqrt(error[ix].sum(1)/joint[ix].sum(1))
            m=np.array(e['margin']);recovery=(base-m)/den
            arms[arm]={'full_effect_error_ci95':np.quantile(ratios,[.025,.975]).tolist(),
                'raw_recovery_ci95':np.quantile(recovery[ix].mean(1),[.025,.975]).tolist() if name in ('A1','A2') else None}
        interaction=np.array(data['interaction_squared_per_row']);ival=np.sqrt(interaction[ix].sum(1)/joint[ix].sum(1))
        vectors={k:[] for k in ('route','value','cross','joint')}
        for layer,terms in sorted(data['natural_scalar_product_terms'].items(),key=lambda kv:int(kv[0])):
            wnorm=float(maps[int(layer)]['writer'].double().norm())
            for key in vectors:vectors[key].append(np.array(terms[key])*wnorm)
        vectors={k:np.stack(v,axis=1) for k,v in vectors.items()}
        jnorm=float(np.linalg.norm(vectors['joint']));assert jnorm>0
        identity=float(np.linalg.norm(vectors['route']+vectors['value']+vectors['cross']-vectors['joint'])/jnorm)
        reports[name]={'arms':arms,'interaction_ci95':np.quantile(ival,[.025,.975]).tolist(),
            'native_product_identity_relative':identity,
            'native_scalar_term_relative_norm':{k:float(np.linalg.norm(v)/jnorm) for k,v in vectors.items()},
            'native_cross_omission_relative_error':float(np.linalg.norm(vectors['cross'])/jnorm),
            'native_route_only_scalar_error':float(np.linalg.norm(vectors['route']-vectors['joint'])/jnorm),
            'native_value_only_scalar_error':float(np.linalg.norm(vectors['value']-vectors['joint'])/jnorm)}
    out={'schema':'correlative.route_value.audit.v1','source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
        'seed':9111304,'bootstrap_draws':4000,'reports':reports,
        'scope':'Descriptive paired bootstrap over16authored groups/panel. Native scalar product terms use unchanged recipient; they do not predict live multi-block endpoint changes. Stacked write norm is not final-logit norm. No fitted gain, selection or new threshold.'}
    with (P/'CORRELATIVE_ROUTE_VALUE_AUDIT_V1_RESULT.json').open('x') as f:json.dump(out,f,sort_keys=True);f.write('\n')
    print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
