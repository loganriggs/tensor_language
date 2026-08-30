# Causal-response factorization v1 — prospective amendment 11

Status: frozen after exact Amendment-10 GO and a production launch that failed before
tensor deserialization. It authorizes no retry until independently audited.

## Preserved launch result

The first direct command failed at import because the repository root was absent from
Python's module path. The corrected audited entrypoint reached the outcome-blind FIT
parent binder and failed closed before deserialization with
`FIT receipt-bound bundle artifact changed`. No authority, input, manifest, terminal,
failure, or live lock was left behind.

The receipt-bound and observed bundle records have identical path, presence, SHA-256,
byte count, device, inode, and mtime. Only `ctime_ns` differs. The observed SHA-256 is
`f0b23bcb9ce926f19bc680aaccc4cf8c7b2694e6a9f97a46c2e9af57e887218a` and the
size is 55,475,273 bytes.

## Why ctime is not content identity

POSIX ctime changes when inode metadata changes, including hardlink count. It cannot
be restored directly and does not imply that file bytes changed. Synthetic integrity
tests had temporarily hardlinked the real production bundle, then removed the link;
that preserved SHA-256, size, inode, and mtime while changing ctime.

The parent binder now compares every receipt-bound artifact on all recorded fields
except ctime. It still requires exact SHA-256, bytes, path, presence, device, inode,
and mtime. Its second full replay requires the complete first observed record,
including the new ctime, to remain unchanged during validation. Thus historical
ctime-only drift is admitted, but concurrent metadata or content drift still fails.

## Test isolation

Loader integrity tests now monkeypatch a temporary stand-in production bundle. They
never create links to the real production inode. A dedicated parent-binding test
creates and removes a link to a synthetic receipt-bound bundle and verifies that
ctime-only drift is admitted while byte mutation remains rejected.

This is a prospective provenance amendment, not a silent retry. It requires an exact-
source independent GO before the production lifecycle may run again.
