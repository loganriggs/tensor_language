"""Opened interchange: signed producer/reader change attribution, no rule rescue."""
import hashlib,json
from pathlib import Path
import numpy as np


def main():
    p=Path(__file__).resolve().parent;source=p/'THIRD_NOUN_COMMON_INTERCHANGE_V1_RESULT.json'
    data=json.loads(source.read_text());assert data['predictions']['pred_a_instrument']
    reports=[]
    for row in data['reports']:
        report={'world_id':row['world_id']}
        for name,score in row['scores'].items():
            d=np.asarray(score['structural_effect_change']);a=np.asarray(score['symmetric_producer_change']);b=np.asarray(score['symmetric_reader_change'])
            den=float(np.sum(d*d));assert den>0
            pa=float(np.sum(a*d)/den);pb=float(np.sum(b*d)/den)
            assert abs(pa+pb-1)<1e-12
            report[name]={'producer_signed_projection':pa,'reader_signed_projection':pb,
                          'reader_change_relative_norm':float(np.linalg.norm(b)/np.linalg.norm(d)),
                          'producer_change_relative_norm':float(np.linalg.norm(a)/np.linalg.norm(d))}
        reports.append(report)
    result={'reports':reports,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'Exact symmetric attribution of the observed 2x2 effect difference. Signed projections permit cancellation; smaller reader contribution does not overturn failed producer-only fidelity.'}
    with (p/'COMMON_INTERCHANGE_CHANGE_ATTRIBUTION_V1_RESULT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({name:{key:[min(r[name][key] for r in reports),max(r[name][key] for r in reports)]
                           for key in reports[0][name]} for name in ('margin','readers')},indent=2))


if __name__=='__main__':main()
