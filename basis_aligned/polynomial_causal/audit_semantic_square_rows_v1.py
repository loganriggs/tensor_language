"""Build disjoint cue-by-lexical-background squares from existing paired rows.

No model outputs, feature scores, or result-dependent choices are consulted.
The context change can include multiple lexical fields: never label it a
single noun/subject variable without an additional controlled dataset.
"""
from collections import defaultdict
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
SOURCE=P/'BILIN18_L9_SHARED_QUERY_ROUTER_V1_ROWS.json'
OUT=P/'SEMANTIC_SQUARE_ROWS_V1_AUDIT.json'


def build():
    data=json.loads(SOURCE.read_text());panels={}
    for panel,rows in data['splits'].items():
        groups=defaultdict(list);rejected=[]
        for row in rows:
            side0,side1=sorted(('base','donor'),key=lambda s:row[s+'_answer'])
            t0,t1=row[side0+'_ids'],row[side1+'_ids']
            if len(t0)!=len(t1):rejected.append({'row_id':row['row_id'],'reason':'unequal_lengths'});continue
            cue=tuple(i for i,(x,y) in enumerate(zip(t0,t1)) if x!=y)
            if not cue:rejected.append({'row_id':row['row_id'],'reason':'no_changed_tokens'});continue
            key=(row[side0+'_answer'],row[side1+'_answer'],len(t0),cue,tuple(t0[i] for i in cue),tuple(t1[i] for i in cue))
            groups[key].append({'row_id':row['row_id'],'side0':side0,'side1':side1,'tokens0':t0,'tokens1':t1,'text0':row[side0+'_text'],'text1':row[side1+'_text']})
        squares=[];unused=[]
        for key,group in sorted(groups.items()):
            group.sort(key=lambda r:r['row_id'])
            while len(group)>1:
                a=group.pop(0);j=next((j for j,b in enumerate(group) if a['tokens0']!=b['tokens0']),None)
                if j is None:unused.append(a['row_id']);continue
                b=group.pop(j);cue=key[3]
                context=tuple(i for i,(x,y) in enumerate(zip(a['tokens0'],b['tokens0'])) if x!=y)
                assert context and not set(cue)&set(context)
                assert all(a['tokens1'][i]==b['tokens1'][i] for i in cue)
                assert all(a['tokens0'][i]==a['tokens1'][i] and b['tokens0'][i]==b['tokens1'][i] for i in context)
                squares.append({'context0':a,'context1':b,'cue_positions':cue,'context_positions':context,'answer0':key[0],'answer1':key[1],'token_length':key[2]})
            unused.extend(r['row_id'] for r in group)
        used=[r[k]['row_id'] for r in squares for k in ('context0','context1')]
        assert len(used)==len(set(used));assert len(used)+len(unused)+len(rejected)==len(rows)
        panels[panel]={'squares':squares,'square_count':len(squares),'used_pair_count':len(used),'unused_rows':unused,'rejected':rejected}
    return {'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'panels':panels,
            'model_forwards':0,'selection_uses_model_outcomes':False,
            'scope':'Previously opened rows; context is a lexical-background bundle. Native normalized-input additivity not assumed.'}


if __name__=='__main__':
    r=build()
    with OUT.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps({k:{'squares':v['square_count'],'unused':len(v['unused_rows']),'rejected':len(v['rejected'])} for k,v in r['panels'].items()}))
