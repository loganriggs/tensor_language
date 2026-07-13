# Archived projects (pre-2026-07-08 reorg)

Completed experiment code, frozen: cycles (`data.py`, `train.py`, `analysis.py`, `mech.py`),
geometry walks (`geodata.py`, `train_geo.py`, `analysis_geo.py`, `probe.py`, `structure3d.py`,
`icl_reps.py`), graph-family generalists (`graphs.py`, `train_general.py`, `analysis_general.py`,
`mech_general.py`, `geometry.py`, `compare3d.py`, `toy_*.py`), and GPT-2/LLM representation
studies (`gpt2_*.py`, `llm_*.py`). Findings live in the root `results*.md` and `LOG.md`.

These scripts were written to run from the repo root (they import `model.py` and read
`runs*/` there). To re-run one:

```bash
PYTHONPATH=.:archive python archive/train_geo.py ...
```

Root `*.log` / `*.err` from these runs are in `logs/`.
