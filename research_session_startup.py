#!/usr/bin/env python3
"""Restore/check a bilin18 research session. Default is read-only, no GPU work.

Run at /workspace/tensor_language on the replacement instance. See
CODEX_RESEARCH_SESSION_STARTUP.md before selecting research work.
"""
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
P = ROOT / 'basis_aligned/polynomial_causal'
BQ = ROOT / 'basis_aligned/bilinear_quotient'
SNAP = Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240')
CHECKPOINT_SHA = '680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(args):
    return subprocess.run(args, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=30)


def review_due(prefix, hours):
    candidates = []
    for f in P.glob(prefix + '*.md'):
        m = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})\.md$', f.name)
        if m:
            stamp = dt.datetime.fromisoformat(f'{m[1]}T{m[2]}:{m[3]}:00+00:00')
            candidates.append((stamp, f))
    if not candidates:
        print(prefix, 'MISSING: review due before choosing work')
        return
    stamp, f = max(candidates)
    due = stamp + dt.timedelta(hours=hours)
    print(f'{f.name}: next {due.isoformat()} — '
          f'{"DUE at first safe boundary" if dt.datetime.now(dt.timezone.utc) >= due else "not yet due"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install-runners', action='store_true', help='Install missing Supervisor configs; preserve differing existing ones')
    parser.add_argument('--start-runners', action='store_true', help='Start stopped managed services after dependency preflight; never restart live services')
    parser.add_argument('--install-skill', action='store_true', help='Install saved research skill when absent; preserve an existing different version')
    parser.add_argument('--verify-checkpoint', action='store_true', help='Read and SHA256-check the full checkpoint')
    args = parser.parse_args()
    print('UTC:', dt.datetime.now(dt.timezone.utc).isoformat())
    print('Read:', ROOT / 'CODEX_RESEARCH_SESSION_STARTUP.md')
    print('Latest full report:', P / 'explanations/for_logan/LATEST.md')
    print('Next-session prompt:', ROOT / 'NEXT_CODEX_PROMPT.md')
    print('Research remains active; completed local compression is not goal completion.')
    failures = []
    if ROOT != Path('/workspace/tensor_language'):
        failures.append('Existing executors have absolute paths: restore checkout at /workspace/tensor_language.')
    print('Python:', sys.executable)
    for pkg in ['torch', 'numpy', 'scipy', 'transformers', 'datasets', 'huggingface-hub', 'safetensors', 'einops', 'tiktoken']:
        try:
            print(pkg, importlib.metadata.version(pkg))
        except importlib.metadata.PackageNotFoundError:
            failures.append('Missing Python package: ' + pkg)
    if not Path('/venv/main/bin/python').exists():
        failures.append('Runners require /venv/main/bin/python and activate.')
    checkpoint = SNAP / 'pytorch_model.bin'
    for name in ['pytorch_model.bin', 'config.json']:
        if not (SNAP / name).exists():
            failures.append('Missing model snapshot file: ' + str(SNAP / name))
    if args.verify_checkpoint and checkpoint.exists():
        actual = digest(checkpoint)
        print('Checkpoint SHA256:', actual)
        if actual != CHECKPOINT_SHA:
            failures.append('Checkpoint SHA256 mismatch')
    package = P / 'extracted_circuits/sparse_even_key_producers_8_2_9_8_v1'
    if (package / 'manifest.json').exists():
        manifest = json.loads((package / 'manifest.json').read_text())
        for filename, key in [('program.pt', 'program_sha256'), ('execute.py', 'runtime_sha256')]:
            f = package / filename
            if not f.exists() or digest(f) != manifest[key]:
                failures.append('Portable package missing/hash mismatch: ' + str(f))
            else:
                print('Portable package verified:', filename)
    else:
        failures.append('Portable package manifest missing')
    review_due('HOURLY_STRATEGIC_REVIEW_', 1)
    review_due('THREE_HOURLY_MATHEMATICAL_REVIEW_', 3)
    print('Reviews are active-agent duties, not unattended cron-generated research. Do not backfill missed hours.')
    for queue in ['queue.txt', 'queue2.txt']:
        f = BQ / queue
        lines = f.read_text().splitlines() if f.exists() else []
        print(queue, 'entries:', len([line for line in lines if line.strip()]))
        for line in lines[:5]:
            print(' ', line)
    free = shutil.disk_usage(ROOT).free
    print('Free disk GiB:', round(free / 2**30, 3))
    if args.install_skill:
        source = ROOT / 'session_recovery/bilin18-research-driver/SKILL.md'
        target = Path('/root/.agents/skills/bilin18-research-driver/SKILL.md')
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            print('Installed research skill:', target)
        elif digest(source) != digest(target):
            print('Preserved existing different skill; compare with', source)
        else:
            print('Research skill already matches.')
    if args.install_runners:
        if ROOT != Path('/workspace/tensor_language'):
            failures.append('Refusing runner installation at incompatible root.')
        else:
            config_dir = Path('/etc/supervisor/conf.d')
            config_dir.mkdir(parents=True, exist_ok=True)
            for service in ['bqrunner', 'bqrunner2']:
                source = ROOT / 'session_recovery' / (service + '.conf')
                target = config_dir / source.name
                if target.exists() and target.read_bytes() != source.read_bytes():
                    print('Preserved differing Supervisor config:', target)
                elif not target.exists():
                    target.write_bytes(source.read_bytes())
                    print('Installed:', target)
                (BQ / 'ops' / (service + '.sh')).chmod(0o755)
            # Do not load/start implicitly: require --start-runners after preflight.
    if shutil.which('supervisorctl'):
        if args.start_runners and not failures:
            for command in [['supervisorctl', 'reread'], ['supervisorctl', 'update', 'bqrunner', 'bqrunner2']]:
                result = run(command)
                print(result.stdout.strip())
                if result.returncode:
                    failures.append('Supervisor reload failed')
                    break
            if not failures:
                for service in ['bqrunner', 'bqrunner2']:
                    status = run(['supervisorctl', 'status', service])
                    if 'RUNNING' not in status.stdout and 'STARTING' not in status.stdout:
                        result = run(['supervisorctl', 'start', service])
                        print(result.stdout.strip())
                        if result.returncode:
                            failures.append('Failed to start ' + service)
        result = run(['supervisorctl', 'status', 'bqrunner', 'bqrunner2'])
        print(result.stdout.strip())
    else:
        print('supervisorctl unavailable; runner setup pending.')
        if args.install_runners or args.start_runners:
            failures.append('Install/start Supervisor before loading the runner services.')
    for reason in failures:
        print('NEEDS RESTORATION:', reason)
    if args.start_runners and failures:
        print('Runner start withheld: resolve preflight failures, then rerun.')
    print('GPU jobs: lane1, reviewed hash through ops/enqueue.sh. Lane2: CPU-only.')
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
