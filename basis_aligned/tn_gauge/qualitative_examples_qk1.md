# Layer-1 QK equivalence classes — qualitative examples (bilin18)

139 tokens with >=20 occurrences, clustered into 40 classes by their layer-1 QK input signature (F30: ~82% of QK-1 is current-token-determined).

Each class = tokens that make layer-1 attention select the same way.

- **class 1** (10 tokens): ' the', ' a', '�', ' my', ' an', ' this', ' your', ' its', ' their', ' his'
- **class 31** (10 tokens): ' of', ' to', ' in', ' for', ' on', ' as', ' at', ' about', ' into', ' like'
- **class 26** (9 tokens): '.', '>', ':', ')', '!', '..', ' =', '">', ').'
- **class 27** (8 tokens): ' time', ' do', ' work', ' states', ' used', ' way', ' women', ' pain'
- **class 38** (8 tokens): ' is', ' was', ' are', ' be', ' have', ' had', ' were', ' been'
- **class 21** (6 tokens): 'field', 'dat', 'layout', ' android', ' game', 'PD'
- **class 28** (6 tokens): ' that', ' which', ' what', ' how', ' because', ' when'
- **class 0** (5 tokens): ' I', ' me', ' It', 'I', ' In'
- **class 24** (5 tokens): '"', '?', '�', '),', ';'
- **class 30** (5 tokens): '\r', ' quantum', ' state', ' wave', ' counter'
- **class 39** (5 tokens): 'at', 'rist', 'M', 'A', 'a'
- **class 23** (4 tokens): '\n', '="', ' "', '�'
- **class 37** (4 tokens): ' some', ' more', ' other', ' something'
- **class 6** (3 tokens): ' would', ' has', ' will'
- **class 8** (3 tokens): ' not', 't', ' also'
- **class 10** (3 tokens): ' but', ' so', ' But'
- **class 14** (3 tokens): '-', '/', '_'
- **class 19** (3 tokens): '0', '1', ' one'
- **class 20** (3 tokens): ',', ' and', ' or'
- **class 29** (3 tokens): ' ', ' <', ' </'
- **class 35** (3 tokens): ' by', ' through', ' using'
- **class 3** (2 tokens): 's', "'s"
- **class 4** (2 tokens): '</', 'Occ'
- **class 5** (2 tokens): ' up', ' over'

## Real attention co-occurrence (data-validated) — layer-1, top-attended pairs

For sampled positions, the token that layer-1 attends to most (summed over heads), showing only pairs that actually occur in the data.

- q=' min'  →  attends to  ' 30'  (offset 1)
- q=' in'  →  attends to  ' investments'  (offset 1)
- q=' Al'  →  attends to  ' Al'  (offset 0)
- q='urs'  →  attends to  'Occ'  (offset 1)
- q=' results'  →  attends to  ' simulation'  (offset 1)
- q='\n'  →  attends to  '\n'  (offset 0)
- q=' used'  →  attends to  ' used'  (offset 0)
- q=' you'  →  attends to  ' that'  (offset 1)
- q='con'  →  attends to  'con'  (offset 0)
- q=' while'  →  attends to  ' while'  (offset 0)
- q='.'  →  attends to  ' photons'  (offset 1)
- q=' are'  →  attends to  ' those'  (offset 5)
- q=' than'  →  attends to  ' differently'  (offset 3)
- q=' M'  →  attends to  ' M'  (offset 0)
- q='.'  →  attends to  ' argument'  (offset 1)
- q=' years'  →  attends to  ' years'  (offset 0)
- q=' really'  →  attends to  ' some'  (offset 1)
- q=' '  →  attends to  ' '  (offset 0)
- q=','  →  attends to  ' time'  (offset 1)
- q='�'  →  attends to  ' didn'  (offset 1)
- q='ctions'  →  attends to  'fun'  (offset 1)
- q='"'  →  attends to  '"'  (offset 0)
- q=' me'  →  attends to  ' made'  (offset 1)
- q=' number'  →  attends to  ' endless'  (offset 1)
- q=' extension'  →  attends to  ' knee'  (offset 1)
- q=' Rowling'  →  attends to  ' Rowling'  (offset 0)
- q='at'  →  attends to  'dat'  (offset 1)
- q=' jobs'  →  attends to  ' jobs'  (offset 0)
- q=' and'  →  attends to  'rology'  (offset 1)
- q=' Hampshire'  →  attends to  ' Hampshire'  (offset 0)
