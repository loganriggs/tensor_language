"""Unused deterministic panels for the frozen, diagnostic-selected scalar program."""
import hashlib,json,subprocess
from pathlib import Path
import torch,tiktoken
import build_feature_swap_donors as donors
ROOT=Path(__file__).resolve().parents[3];P=Path(__file__).resolve().parent

def main():
 assert not (P/"MIDPOINT_CONFIRMATION_PANELS_V1.json").exists()
 excluded={r['path'] for r in json.loads((P/'CODE_SHIFT_PANEL_V1.json').read_text())['sources']}
 excluded|={r['path'] for r in json.loads((P/'BLEND_CONFIRMATION_PANELS_V1.json').read_text())['code_sources']}
 excluded|={r['path'] for r in json.loads((P/'SELECTIVE_CONFIRMATION_PANELS_V1.json').read_text())['code_sources']}
 enc=tiktoken.get_encoding('gpt2');rows=[];sources=[]
 oldcode=[torch.load(P/n,weights_only=True) for n in ['CODE_SHIFT_PANEL_V1.pt','BLEND_CONFIRMATION_CODE_V1.pt','SELECTIVE_CONFIRMATION_CODE_V1.pt']]
 for name in sorted(subprocess.check_output(['git','ls-files','archive/*.py'],cwd=ROOT,text=True).splitlines()):
  if name in excluded:continue
  text=(ROOT/name).read_text();tokens=enc.encode(text)
  if len(tokens)<257:continue
  row=torch.tensor(tokens[:257])
  # Reject exact prefixes matching earlier panels (initial panel had128context).
  if any(any(torch.equal(row[:len(other)],other) for other in panel) for panel in oldcode):continue
  if any(torch.equal(row,other) for other in rows):continue
  rows.append(row);sources.append(dict(path=name,sha256=hashlib.sha256(text.encode()).hexdigest()))
  if len(rows)==16:break
 assert len(rows)==16
 panels=dict(code=torch.stack(rows),fineweb=torch.load(ROOT/'basis_aligned/bilinear_quotient/.rowcache/fineweb_n192_skip7000.pt',weights_only=True)[160:192,:257].clone())
 metadata=dict(context=256,code_source_glob='archive/*.py',selection_note='Next lexicographic16 eligible archive files, excluding all three prior code panels by source and exact prefix; selected before midpoint confirmation outputs.',code_sources=sources,excluded_code_sources=sorted(excluded),fineweb_documents=list(range(160,192)),panels={},program_sha256=hashlib.sha256((P/'MIDPOINT_EXTRACTED_PROGRAM_V1.pt').read_bytes()).hexdigest(),scope='Unused in this decomposition study at selection time. Related local source files, not broad code representativeness; pretrained overlap unknown.')
 for name,tokens in panels.items():
  torch.save(tokens,P/f'MIDPOINT_CONFIRMATION_{name.upper()}_V1.pt');metadata['panels'][name]=dict(shape=list(tokens.shape),token_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest())
 (P/'MIDPOINT_CONFIRMATION_PANELS_V1.json').write_text(json.dumps(metadata,indent=2)+'\n')
 donors.PANEL_PREFIX='MIDPOINT_CONFIRMATION';donors.OUTPUT_PREFIX='MIDPOINT_CONFIRMATION_DONORS';donors.SCOPE=metadata['scope'];donors.main();print(metadata['panels'])
if __name__=='__main__':main()
