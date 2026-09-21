"""Freeze balanced same-token donors before evaluating any interchange outcomes."""
from pathlib import Path
from collections import defaultdict
import json,hashlib,torch,tiktoken
from token_boundary_conditions import annotate
p=Path(__file__).resolve().parent;f=p/'FULL_CHANNEL_FRESH_TOKENS_V1.pt';tokens=torch.load(f,weights_only=True);enc=tiktoken.get_encoding('gpt2');domains={}
for domain,documents in tokens.items():
 flat=documents[:,16:255].flatten().tolist();labels=[]
 for row in documents:
  labels.extend([1 if r['continuation'] else 2 if r['spaced_word'] else 0 for r in annotate(row.tolist(),enc)[16:255]])
 strata=defaultdict(list)
 for i,(token,label) in enumerate(zip(flat,labels)):
  if label:strata[token,label].append(i)
 maps={};stats={}
 for family in ('same_cohort','opposite_cohort'):
  mapping=[-1]*len(flat);uses=defaultdict(int)
  for i,(token,label) in enumerate(zip(flat,labels)):
   if not label:continue
   wanted=label if family=='same_cohort' else 3-label
   candidates=[j for j in strata[token,wanted] if j//239!=i//239]
   if candidates:
    j=min(candidates,key=lambda j:(uses[j],abs(j%239-i%239),j));mapping[i]=j;uses[j]+=1
  for i,j in enumerate(mapping):
   if j>=0:assert flat[i]==flat[j] and i//239!=j//239 and ((labels[i]==labels[j])==(family=='same_cohort'))
  maps[family]=mapping;valid=[j for j in mapping if j>=0];stats[family]=dict(pairs=len(valid),distinct_donors=len(set(valid)),maximum_donor_reuse=max(uses.values(),default=0),continuation_pairs=sum(j>=0 and labels[i]==1 for i,j in enumerate(mapping)),spaced_word_pairs=sum(j>=0 and labels[i]==2 for i,j in enumerate(mapping)))
 domains[domain]=dict(maps=maps,cohort_labels=labels,stats=stats)
result=dict(token_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),domains=domains,rule='Same current token, other document, balanced reuse with nearest-position then flat-index ties. Same or opposite continuation/spaced boundary cohort. Token/label selection only, no activations or outcomes.',scope='Opened panel diagnostic. Same cohort preserves a token-boundary label, not necessarily next-token answer or semantic task. Swaps operate on centered normalized midpoint ports, not whole upstream state.')
(p/'FULL_CHANNEL_CONTINUATION_INTERCHANGE_DONORS_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v['stats'] for k,v in domains.items()},indent=2))
