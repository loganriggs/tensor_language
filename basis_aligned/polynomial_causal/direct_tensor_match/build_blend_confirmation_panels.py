"""Fresh fixed panels selected before evaluating the frozen primary blend."""
import hashlib,json,subprocess
from pathlib import Path
import torch,tiktoken
ROOT=Path(__file__).resolve().parents[3];P=Path(__file__).resolve().parent

def main():
 excluded={r['path'] for r in json.loads((P/'CODE_SHIFT_PANEL_V1.json').read_text())['sources']};enc=tiktoken.get_encoding('gpt2');rows=[];sources=[]
 for name in sorted(subprocess.check_output(['git','ls-files','jacclust/*.py'],cwd=ROOT,text=True).splitlines()):
  if name in excluded:continue
  text=(ROOT/name).read_text();tokens=enc.encode(text)
  if len(tokens)<257:continue
  rows.append(tokens[:257]);sources.append(dict(path=name,sha256=hashlib.sha256(text.encode()).hexdigest()))
  if len(rows)==16:break
 assert len(rows)==16
 panels={'code':torch.tensor(rows,dtype=torch.long),'fineweb':torch.load(ROOT/'basis_aligned/bilinear_quotient/.rowcache/fineweb_n192_skip7000.pt',weights_only=True)[96:128,:257].clone()}
 metadata=dict(context=256,code_sources=sources,excluded_code_sources=sorted(excluded),fineweb_cache='fineweb_n192_skip7000.pt',fineweb_documents=list(range(96,128)),panels={})
 for name,tokens in panels.items():
  torch.save(tokens,P/f'BLEND_CONFIRMATION_{name.upper()}_V1.pt');metadata['panels'][name]=dict(shape=list(tokens.shape),token_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),unique_rows=len(torch.unique(tokens,dim=0)))
 metadata['program_sha256']=hashlib.sha256((P/'SKIP_METRIC_BLEND_PROGRAM_V1.pt').read_bytes()).hexdigest();(P/'BLEND_CONFIRMATION_PANELS_V1.json').write_text(json.dumps(metadata,indent=2)+'\n');print(metadata['panels'])
if __name__=='__main__':main()
