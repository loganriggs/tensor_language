#!/usr/bin/env python3
"""Robust receipt walker (ops lane, 13:06 review).

Three audits today stalled on nested receipt structures: a shallow walk()
printed nothing (485, 486 twice) and cost a GPU-free but wall-clock round
trip each time.  This is the one tool for the job:

  python3 ops/receipt_dump.py <receipt.json> [--depth N] [--grep SUBSTR]

Prints every leaf as `dotted.path = value` (floats rounded to 5 s.f.,
long lists summarized), never crashes on None/mixed nesting, and --grep
filters paths case-insensitively (e.g. --grep pred, --grep cosine).
"""
import argparse
import json


def walk(node, path, out, depth, maxdepth):
    if maxdepth is not None and depth > maxdepth:
        out.append((path, f"<pruned {type(node).__name__}>"))
        return
    if isinstance(node, dict):
        if not node:
            out.append((path, "{}"))
        for key, value in node.items():
            walk(value, f"{path}.{key}" if path else str(key), out, depth + 1, maxdepth)
    elif isinstance(node, list):
        if len(node) <= 8 and all(not isinstance(x, (dict, list)) for x in node):
            out.append((path, [round(x, 5) if isinstance(x, float) else x for x in node]))
        elif len(node) <= 4:
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]", out, depth + 1, maxdepth)
        else:
            out.append((path, f"<list len {len(node)}>"))
    elif isinstance(node, float):
        out.append((path, float(f"{node:.5g}")))
    else:
        out.append((path, node))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt")
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--grep", default=None)
    args = parser.parse_args()
    with open(args.receipt) as handle:
        data = json.load(handle)
    rows = []
    walk(data, "", rows, 0, args.depth)
    needle = args.grep.lower() if args.grep else None
    for path, value in rows:
        if needle is None or needle in path.lower():
            print(f"{path} = {value}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        import os, sys
        sys.stderr.close()
        os._exit(0)   # clean exit when piped into head
