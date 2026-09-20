import json
from pathlib import Path
import numpy as np
from shared_selective_source_lp import choose
ref=np.array([1.,0.]);g=np.zeros((4,3,2));g[:,0,0]=-1;g[:,1,1]=1;g[1::2,0,0]=1
weights,good=choose(g,ref);assert abs(good['retention']-1)<1e-10
g[:,1,0]=1;g[:,1,1]=0;weights,bad=choose(g,ref);assert abs(bad['retention']-.08)<1e-10
out=dict(planted_independent=good,planted_coupled=bad,scope='LP correctness and feasibility ceiling only, not a native circuit guarantee.')
(Path(__file__).parent/'SHARED_SELECTIVE_SOURCE_LP_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
