#!/usr/bin/env python3
"""Surface the traceback that `execution_invalid` throws away.

`circuit_fast_screen_producer.run_science` wraps its whole body in a bare `except Exception`
that returns `finish("invalid", "execution_invalid")`. The receipt then records the outcome and
nothing about the cause, which makes this failure class expensive to diagnose -- it cost this
lane eight minutes of guessing on `animacy_state.who_vs_which` before the cause was found.

This runs the same science through a COPY of the producer with the swallow removed, so the
exception propagates. The producer is not modified; the copy lives in a temporary directory and
is rebuilt on every run, so it cannot drift from the original.

    python ops/screen_repro.py --runner run_circuit_fast_screen_<name>

Found this way on 2026-09-05: animacy raised `InvalidEvidenceError: donor denominator must be
positive and greater than 1e-6` from `kernel.signed_pairwise_donor_recovery`. That is a per-row
requirement -- the native donor-versus-base separation must be strictly positive on the
donor-oriented axis -- and it is STRICTER than the capability gate, which only asks for 0.85
accuracy per cell. A weakly encoded behaviour can pass capability and still die in the
normalizer.
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

OPS = Path(__file__).resolve().parent
SWALLOW = '''    except Exception:
        return finish("invalid", "execution_invalid")'''


def build_debug_producer(into: Path) -> Path:
    """Copy the producer with the bare except turned into a re-raise."""
    source = (OPS / "circuit_fast_screen_producer.py").read_text()
    if SWALLOW not in source:
        raise SystemExit(
            "producer no longer contains the expected bare except; "
            "re-read circuit_fast_screen_producer.py before trusting this tool"
        )
    target = into / "producer_debug.py"
    target.write_text(source.replace(SWALLOW, "    except Exception:\n        raise"))
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runner", required=True, help="runner module name (no .py)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        build_debug_producer(tmpdir)
        sys.path.insert(0, str(tmpdir))
        sys.path.insert(0, str(OPS))
        import producer_debug  # noqa: E402  (path is set above)

        runner = importlib.import_module(args.runner.removesuffix(".py"))
        candidate = runner.candidate
        rows = candidate.build_rows()
        spec = runner.build_spec(rows)
        print(f"{args.runner}: {len(rows)} rows, spec built")
        try:
            run = producer_debug.run_science(spec, rows, device=args.device)
        except Exception:
            print("\nEXCEPTION the producer would have swallowed:\n")
            traceback.print_exc()
            return 1
        print(f"terminal={run.terminal} reason={run.reason}")
        print("no exception: this screen does not fail the way execution_invalid implies")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
