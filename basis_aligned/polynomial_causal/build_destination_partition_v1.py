"""Freeze semantic and cardinality-matched source partitions, without scoring."""
from pathlib import Path
import hashlib,itertools,json,math,random
import tiktoken
from regional_endpoint_batching_v1 import group_rows
P=Path(__file__).resolve().parent

def main():
 path=P/'TYPED_FACE_PROSPECTIVE_V1_ROWS.json';doc=json.loads(path.read_text());groups,_=group_rows(doc['rows']);enc=tiktoken.get_encoding('gpt2');records=[]
 for r in groups:
  text=r['text'].encode();needle=b'we invited our new';offset=text.lower().rfind(needle);assert offset>=0
  pieces=[enc.decode_single_token_bytes(t) for t in r['ids']];assert b''.join(pieces)==text
  ends=list(itertools.accumulate(map(len,pieces)));clause=next(i for i,end in enumerate(ends) if end>offset)
  full=tuple(r['destination_positions']);a=tuple(i for i in full if i<clause);b=tuple(i for i in full if i>=clause);assert a and b and not set(a)&set(b) and set(a)|set(b)==set(full)
  n,k=len(full),len(a)
  def canonical(s):
   s=tuple(sorted(s));other=tuple(i for i in full if i not in s)
   return min(s,other) if k==n-k else s
  own=canonical(a);available=math.comb(n,k)//(2 if k==n-k else 1)-1;assert available>0
  generator=random.Random(17092230+r['context_id']);nulls=set();null_order=[]
  while len(nulls)<min(16,available):
   candidate=canonical(generator.sample(full,k))
   if candidate!=own and candidate not in nulls:
    nulls.add(candidate);null_order.append(candidate)
  records.append({'context_id':r['context_id'],'cue':r['cue'],'variant':r['variant'],'city_position':r['city_position'],'clause_position':clause,'full':full,'framing':a,'clause':b,'available_unique_unlabeled_partitions':available,'nulls':null_order})
 count=min(len(r['nulls']) for r in records)
 for r in records:r['nulls']=r['nulls'][:count]
 paired=all(records[i]['framing']==records[i+1]['framing'] and records[i]['nulls']==records[i+1]['nulls'] for i in range(0,len(records),2));assert paired
 result={'pred_a':True,'row_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'records':records,'uniform_null_count':count,'contexts':20,'sequences':40,'panel_status':'opened prospective panel; new partition test not yet scored',
 'scope':'Clause begins at token containing the final we/We invited our new span. Framing/clause partition covers frozen destinations. Random partitions match framing cardinality; exclude semantic partition and label-swapped duplicates. Finite-support null count is explicit. No causal composition or selective source conclusion.',
 'prior_art':['ODD_FRAMING_EQUAL_HALVES_NATIVE_V1_PREREGISTRATION.md','ODD_FRAMING_ROLE_SPLIT_NATIVE_V2_PREREGISTRATION.md','TYPED_FACE_WRITE_COMPOSITION_V1_RESULT.json'],
 'novelty':'Old role/equal-half work tests removal/localization; physical-write random split failed. Proposed next test concerns destination partitions of the complete retained write against matched-cardinality source partitions, not those older objects.'}
 (P/'DESTINATION_PARTITION_V1_CPU_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()
