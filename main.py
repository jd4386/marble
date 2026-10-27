"""Deprecated root entrypoint. Prefer ``python -m marble`` or ``marble-train``."""

from __future__ import annotations

import warnings

warnings.warn(
    "Root main.py is deprecated; use `python -m marble` or the `marble-train` console script.",
    DeprecationWarning,
    stacklevel=2,
)

from marble.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
