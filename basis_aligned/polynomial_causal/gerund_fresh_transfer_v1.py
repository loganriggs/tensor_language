"""New lexical/construction authority; tokenization checks never filter on model outcomes."""
import hashlib

PAIRS=[('run','running','runs'),('ride','riding','rides'),('shop','shopping','shops'),
       ('travel','traveling','travels'),('laugh','laughing','laughs'),('cry','crying','cries'),
       ('smile','smiling','smiles'),('shout','shouting','shouts'),('wash','washing','washes'),
       ('train','training','trains'),('visit','visiting','visits'),('search','searching','searches'),
       ('sell','selling','sells'),('buy','buying','buys'),('fight','fighting','fights'),('hide','hiding','hides')]

def build(tokenizer,old):
    encode=lambda s:tokenizer.encode(s,add_special_tokens=False)
    previous={v for pairs in (old['fit_pairs'],old['test_pairs']) for pair in pairs for v in pair}
    assert not previous&{x for triple in PAIRS for x in triple[:2]}
    def one(s):
        ids=encode(' '+s);assert len(ids)==1,(s,ids)
        return ids[0]
    panels={k:[] for k in ['A1','A2','P','G']}
    for i,(bare,ing,third) in enumerate(PAIRS):
        lead=f'The workers often {bare}. '
        specs={'A1':(lead+'The workers might already',lead+'The workers were already',bare,ing),
               'A2':(lead+'The report says the worker would perhaps',lead+'The report says the worker was perhaps',bare,ing),
               'P':(lead+'The workers might already',lead+'The workers could already',bare,bare),
               'G':(lead+'In the village he always',lead+'In the village they always','runs','run')}
        for name,(bt,dt,ba,da) in specs.items():
            bf=ing if name=='P' else da;df=ing if name=='P' else ba
            bi,di=encode(bt),encode(dt);assert len(bi)==len(di) and bi[-1]==di[-1]
            ba,bf,da,df=map(one,[ba,bf,da,df]);assert ba!=bf and da!=df
            for prompt,ids,target in [(bt,bi,ba),(bt,bi,bf),(dt,di,da),(dt,di,df)]:
                assert encode(prompt+tokenizer.decode([target]))==ids+[target]
            rid=hashlib.sha256((name+'|'+bt+'|'+dt).encode()).hexdigest()
            panels[name].append({'row_id':rid,'group_id':str(i),'base_text':bt,'donor_text':dt,'base_ids':bi,'donor_ids':di,
                'base_answer_id':ba,'base_foil_id':bf,'donor_answer_id':da,'donor_foil_id':df,
                'base_semantic_position':len(bi)-1,'donor_semantic_position':len(di)-1,'bare':bare,'gerund':ing})
    panels['C']=old['panels']['C'];panels['R']=old['panels']['A1']
    return {'new_lexical_pairs':[list(t[:2]) for t in PAIRS],'agreement_control_answers':['runs','run'],'panels':panels,'scope':'New lexical items and cues for fixed intervention; G fixed runs/run agreement under the same varied contexts; C and R reused. No training-distribution OOD claim.'}
