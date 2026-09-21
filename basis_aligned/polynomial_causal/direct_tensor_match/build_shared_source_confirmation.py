"""Freeze shared-source programs and unused FineWeb rows before native outcomes."""
from pathlib import Path
import torch,json,hashlib
from source_interface import residual_write
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False);base=torch.load(p/'MIDPOINT_SOURCE_INTERFACE_V1.pt',weights_only=True)['covariance_16'];shared=torch.load(p/'MIDPOINT_SOURCE_SHARED_DICTIONARY_V1.pt',weights_only=True);programs={'covariance_16':base,**{'shared'+k:shared[k] for k in ['8','16','24','32']}}
# Full32 dictionary must reproduce the original program, including output mapping.
torch.manual_seed(219);z=torch.randn(2,3,1152,dtype=torch.float64);h=torch.randn_like(z);ref=residual_write(z,h,base);pred=residual_write(z,h,shared['32']);replay=float((ref-pred).norm()/ref.norm());assert replay<1e-8
file=p/'MIDPOINT_SHARED_SOURCE_PROGRAMS_V1.pt';torch.save(programs,file)
fw=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/.rowcache/fineweb_n192_skip11000.pt',weights_only=True)[176:192,:257].clone();code=torch.load(p/'MIDPOINT_STDLIB_CONTINUATION_TOKENS_V1.pt',weights_only=True);tokens={'fineweb':fw,'stdlib':code};torch.save(tokens,p/'MIDPOINT_SHARED_SOURCE_TOKENS_V1.pt');donors={}
for domain,t in tokens.items():
 flat=t[:,16:256].flatten();groups={}
 for i,v in enumerate(flat.tolist()):groups.setdefault(v,[]).append(i)
 mapping=torch.full_like(flat,-1)
 for indices in groups.values():
  for i in indices:
   options=[j for j in indices if j//240!=i//240]
   if options:mapping[i]=options[0]
 donors[domain]=mapping
f=p/'MIDPOINT_SHARED_SOURCE_DONORS_V1.pt';torch.save(donors,f)
plan=dict(program_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),donor_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),tokens_sha256={k:hashlib.sha256(t.numpy().tobytes()).hexdigest() for k,t in tokens.items()},recipient_documents={'fineweb':list(range(176,192)),'stdlib':list(range(16))},valid_pairs={k:int((v>=0).sum()) for k,v in donors.items()},full_dictionary_synthetic_replay=replay,scope='Frozen shared dictionary on unusedFW176:192 and reusedstdlib. Same-token source interchange, fixed recipient background and attention; recompute RMS/lastMLP. Shared16 main candidate: <=1.10 baseline error and absolute<=.15 hybrid/<=.20 change in every cohort/domain. No fitting; native leading-feature removal reference, not whole source effect.')
(p/'MIDPOINT_SHARED_SOURCE_PLAN_V1.json').write_text(json.dumps(plan,indent=2)+'\n');print(plan)
