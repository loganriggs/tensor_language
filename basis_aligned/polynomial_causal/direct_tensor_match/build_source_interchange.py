"""Freeze same-token, different-document donors before source-interchange outcomes."""
from pathlib import Path
import torch,json,hashlib
p=Path(__file__).resolve().parent;tokens=torch.load(p/'MIDPOINT_SOURCE_NATIVE_TOKENS_V1.pt',weights_only=True);donors={};counts={}
for domain,t in tokens.items():
 flat=t[:,16:256].flatten();groups={}
 for i,v in enumerate(flat.tolist()):groups.setdefault(v,[]).append(i)
 mapping=torch.full_like(flat,-1)
 for indices in groups.values():
  for i in indices:
   options=[j for j in indices if j//240!=i//240]
   if options:mapping[i]=options[0]
 valid=mapping>=0;assert torch.all(flat[valid]==flat[mapping[valid]]);assert torch.all(torch.arange(len(flat))[valid]//240!=mapping[valid]//240)
 donors[domain]=mapping;counts[domain]=int(valid.sum())
file=p/'MIDPOINT_SOURCE_INTERCHANGE_DONORS_V1.pt';torch.save(donors,file)
plan=dict(donor_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),program_sha256=hashlib.sha256((p/'MIDPOINT_SOURCE_INTERFACE_V1.pt').read_bytes()).hexdigest(),tokens_sha256={k:hashlib.sha256(t.numpy().tobytes()).hexdigest() for k,t in tokens.items()},valid_pairs=counts,recipient_documents=json.loads((p/'MIDPOINT_SOURCE_NATIVE_PLAN_V1.json').read_text())['documents'],scope='Reused source-native panels. Deterministic first same-token donor in different document; repeated donors allowed. Recipient background h-m fixed, donor m inserted, recipient RMS and lastMLP recomputed; attention17 held fixed. This is recipient input-port interchange, not full upstream MLP ablation. Compare leading-feature removal and change in removal between base and hybrid states, not claim to explain full source-swap effect.')
(p/'MIDPOINT_SOURCE_INTERCHANGE_PLAN_V1.json').write_text(json.dumps(plan,indent=2)+'\n');print(counts)
