"""Extend frozen token tables algebraically, without model execution or fitting."""
from pathlib import Path
import json,hashlib,torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);rows=json.loads((P/'CITY_FULL_PILE_V1_ROWS.json').read_text())['rows'];ids=torch.tensor(sorted({t for r in rows for t in r['ids']}))
 binding=json.loads((P/'PHI4_PROVENANCE_V1_BINDING.json').read_text())['files'];ckpt=next(x for x in binding if x.endswith('pytorch_model.bin'));sd=torch.load(ckpt,weights_only=True,mmap=True,map_location='cpu')
 initial=F.rms_norm(sd['transformer.wte.weight'][ids].float(),(1152,));l=sd['transformer.h.0.lambdas'];first=F.linear(F.rms_norm(l[0]*initial+l[1]*initial,(1152,)),sd['transformer.h.0.attn.c_v.weight'])
 out=P/'CITY_FULL_PILE_V1_TABLES.pt';assert not out.exists();torch.save({'token_ids':ids,'initial_table':initial,'first_table':first,'lambdas8':sd['transformer.h.8.lambdas'].clone()},out)
 receipt={'tokens':len(ids),'floating_scalars':initial.numel()+first.numel()+2,'checkpoint_sha256':binding[ckpt],'tables_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'scope':'Exact weight-derived tables only, full first-value width retained for reference instrumentation; no native model effects inspected.'};(P/'CITY_FULL_PILE_V1_TABLES_RESULT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt)
if __name__=='__main__':main()
