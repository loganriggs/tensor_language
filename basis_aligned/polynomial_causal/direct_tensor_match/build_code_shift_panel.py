"""Deterministic local-code shift panel, selected without model outputs."""
import hashlib,json,subprocess
from pathlib import Path
import torch,tiktoken
ROOT=Path(__file__).resolve().parents[3];P=Path(__file__).resolve().parent

def main():
 enc=tiktoken.get_encoding('gpt2');files=sorted(subprocess.check_output(['git','ls-files','jacclust/*.py'],cwd=ROOT,text=True).splitlines());rows=[];sources=[]
 for name in files:
  content=(ROOT/name).read_text();ids=enc.encode(content)
  if len(ids)<129:continue
  rows.append(ids[:129]);sources.append(dict(path=name,sha256=hashlib.sha256(content.encode()).hexdigest(),offset=0))
  if len(rows)==16:break
 assert len(rows)==16
 tokens=torch.tensor(rows,dtype=torch.long)
 torch.save(tokens,P/'CODE_SHIFT_PANEL_V1.pt')
 (P/'CODE_SHIFT_PANEL_V1.json').write_text(json.dumps(dict(selection='First16 lexicographically sorted tracked jacclust Python files with >=129 GPT2 tokens; first129 tokens each. No model-output selection.',sources=sources,token_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),scope='Local Python-code domain shift relative to FineWeb calibration. Files may be correlated and are not a representative code benchmark; model pretraining overlap unknown.'),indent=2)+'\n')
 print('built',tuple(tokens.shape),'distinct sources',len(sources))
if __name__=='__main__':main()
