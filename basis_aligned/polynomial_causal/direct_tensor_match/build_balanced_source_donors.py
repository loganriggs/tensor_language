"""Donor-family sensitivity: greedily balance reuse, same token/different document."""
from pathlib import Path
import torch,json,hashlib
p=Path(__file__).resolve().parent;tokens=torch.load(p/'MIDPOINT_SHARED_SOURCE_TOKENS_V1.pt',weights_only=True);donors={};stats={}
for domain,t in tokens.items():
 flat=t[:,16:256].flatten();groups={}
 for i,v in enumerate(flat.tolist()):groups.setdefault(v,[]).append(i)
 mapping=torch.full_like(flat,-1)
 for indices in groups.values():
  uses={j:0 for j in indices}
  for i in indices:
   candidates=[j for j in indices if j//240!=i//240]
   if candidates:
    j=min(candidates,key=lambda j:(uses[j],abs(j%240-i%240),j));mapping[i]=j;uses[j]+=1
 valid=mapping>=0;assert torch.all(flat[valid]==flat[mapping[valid]]);assert torch.all(torch.arange(len(flat))[valid]//240!=mapping[valid]//240)
 donors[domain]=mapping;counts=torch.bincount(mapping[valid]);docs=torch.bincount(mapping[valid]//240);stats[domain]=dict(pairs=int(valid.sum()),distinct_donors=int((counts>0).sum()),max_site_fraction=float(counts.max()/valid.sum()),max_document_fraction=float(docs.max()/valid.sum()))
f=p/'MIDPOINT_BALANCED_SOURCE_DONORS_V1.pt';torch.save(donors,f);plan=json.loads((p/'MIDPOINT_SHARED_SOURCE_PLAN_V1.json').read_text());plan['donor_sha256']=hashlib.sha256(f.read_bytes()).hexdigest();plan['donor_rule']='Within each current-token stratum, greedily choose least-used donor in another document, ties nearest sequence position then flat index; recipient order fixed. No outcomes used.';plan['donor_concentration']=stats;plan['scope']='Donor-family sensitivity on reused shared-source panel; programs unchanged. Same absolute and relative gates. No fresh-data claim.';(p/'MIDPOINT_BALANCED_SOURCE_PLAN_V1.json').write_text(json.dumps(plan,indent=2)+'\n');print(stats)
