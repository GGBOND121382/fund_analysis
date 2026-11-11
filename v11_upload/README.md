# v1.1 B6 — Research-only source transport

Base: `12e4fdc3cc6a5ffe2a8235db59b886151c605efa`

This folder contains four binary parts of one pinned **tar.xz source bundle**.
They were staged via GitHub's authorized Git-object API because the execution
container cannot access GitHub directly. The original bundle checksum is
`9228c6fb0f350399f2cd20104f45e879ee3ea8bc2d672653b325bdcf50d1b05d`.

GitHub Actions runs `python scripts/v11_apply_staged.py`, installs B6 into
the current branch, runs all unit tests, then commits human-readable source
files to **this feature branch only** if the tests pass.

Manual equivalent:

```bash
python scripts/v11_apply_staged.py
python -m unittest discover -s tests -p 'test_*.py' -q
```

No trading is performed. Do not merge unless CI is green and the diff is reviewed.
The algorithm remains research-only and has no validated 5% CAGR/4% MDD claim.
