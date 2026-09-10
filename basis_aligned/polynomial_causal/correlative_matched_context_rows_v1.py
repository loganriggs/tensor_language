"""Same token positions, one either/only substitution, frozen correlative interface."""
import json
from dataclasses import replace
from pathlib import Path
import circuit_fast_screen_behaviour_spec as bs
import circuit_fast_screen_candidate_correlative_open_rec as old

SEED = 9111290


def build():
    panels = {}
    for context in ('either', 'only'):
        transform = lambda text: text.replace('holding either', 'holding '+context)
        spec = replace(old.SPEC, task_id='correlative_matched_context_v1.'+context,
            a1=bs.Family('open_frame', 'open_frame_cue_swap',
                lambda i, pos: transform(old.SPEC.a1.text(i, pos))),
            a2=bs.Family('noted_frame', 'noted_frame_cue_swap',
                lambda i, pos: transform(old.SPEC.a2.text(i, pos))),
            p_donor=lambda i, pos: transform(old.SPEC.p_donor(i, pos)),
            a1_suffix=lambda i: transform(old.SPEC.a1_suffix(i)),
            a2_suffix=lambda i: transform(old.SPEC.a2_suffix(i)))
        build_rows, validate, _ = spec.api()
        rows = build_rows(groups=16, seed=SEED)
        validate(rows, groups=16, seed=SEED)
        panels[context] = {f: [r for r in rows if r['family']==f] for f in ('A1','A2')}
    checks = []
    for frame in ('A1','A2'):
        for e, o in zip(panels['either'][frame], panels['only'][frame]):
            assert e['reporter']==o['reporter'] and e['base_answer']==o['base_answer']
            for side in ('base','donor'):
                a,b=e[side+'_ids'],o[side+'_ids']
                difference=[i for i,(u,v) in enumerate(zip(a,b)) if u!=v]
                assert len(a)==len(b) and len(difference)==1 and a[-1]==b[-1]
                assert e[side+'_text'].replace('holding either','holding only')==o[side+'_text']
                cue=[i for i,(u,v) in enumerate(zip(e['base_ids'],e['donor_ids'])) if u!=v]
                cue_o=[i for i,(u,v) in enumerate(zip(o['base_ids'],o['donor_ids'])) if u!=v]
                assert len(cue)==1 and cue==cue_o and difference[0]>cue[0]
                checks.append({'frame':frame,'group':e['group_number'],'side':side,
                               'length':len(a),'cue_position':cue[0],'context_position':difference[0],
                               'final_token':a[-1]})
    return {'seed':SEED,'panels':panels,'matching_checks':checks,
            'scope':'Matched word substitution; discharged template is prior art v493/v495/v497. No training OOD claim, no model-based row filtering.'}


if __name__=='__main__':
    data=build()
    with Path(__file__).with_name('CORRELATIVE_MATCHED_CONTEXT_ROWS_V1.json').open('x') as f:
        json.dump(data,f,sort_keys=True);f.write('\n')
    print(json.dumps({'matched_pairs':len(data['matching_checks']),'panels':4,'rows_per_panel':16}))
