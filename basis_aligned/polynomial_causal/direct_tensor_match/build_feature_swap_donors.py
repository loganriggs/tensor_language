"""Token-only donor maps, fixed before any swap outputs are evaluated."""
import json,hashlib
from pathlib import Path
import torch
P=Path(__file__).resolve().parent

PANEL_PREFIX='BLEND_CONFIRMATION'
OUTPUT_PREFIX='FEATURE_SWAP_DONORS'
SCOPE='Previously used confirmation panels, new intervention definition; no independent OOD claim.'

def main():
 torch.set_num_threads(2);maps={};records=[]
 for domain in ['fineweb','code']:
  tokens=torch.load(P/f'{PANEL_PREFIX}_{domain.upper()}_V1.pt',weights_only=True);n,t=tokens.shape[0],256;flat=tokens[:,:t].reshape(-1);doc=torch.arange(n).repeat_interleave(t);pos=torch.arange(t).repeat(n);eligible=pos>=16;aligned=((doc+1)%n)*t+pos;aligned[~eligible]=-1;same=torch.full_like(aligned,-1)
  for token in torch.unique(flat[eligible]):
   ids=torch.where((flat==token)&eligible)[0];distance=(pos[ids,None]-pos[ids][None,:]).abs();distance[doc[ids,None]==doc[ids][None,:]]=10000;best=distance.argmin(1);good=distance[torch.arange(len(ids)),best]<10000;same[ids[good]]=ids[best[good]]
  valid=same>=0;assert torch.all(doc[valid]!=doc[same[valid]]) and torch.all(flat[valid]==flat[same[valid]])
  maps[domain]=dict(aligned=aligned,same_token=same,documents=n,context=t);records.append(dict(domain=domain,aligned_pairs=int((aligned>=0).sum()),same_token_pairs=int(valid.sum()),same_token_coverage=float(valid.sum()/eligible.sum()),token_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest()))
 torch.save(maps,P/f'{OUTPUT_PREFIX}_V1.pt');(P/f'{OUTPUT_PREFIX}_V1.json').write_text(json.dumps(dict(records=records,selection='Recipient positions16:256. Aligned:next document at same position. Same-token:closest-position occurrence in another document, ties by flattened index. No activation/output selection.',scope=SCOPE),indent=2)+'\n');print(records)
if __name__=='__main__':main()
