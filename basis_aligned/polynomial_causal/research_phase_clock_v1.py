"""Append UTC phase boundaries for a bounded research block; never infer old phases."""
import json
from datetime import datetime,timezone
from pathlib import Path


def mark(path,phase,category):
    if category not in ('science','implementation','validation','review','publication'):
        raise ValueError('Explicit phase category required')
    record={'utc':datetime.now(timezone.utc).isoformat(),'phase':phase,'category':category}
    with Path(path).open('a') as handle:handle.write(json.dumps(record)+'\n')
    return record


if __name__=='__main__':
    import sys
    print(json.dumps(mark(*sys.argv[1:])))
