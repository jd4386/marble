"""Allow ``python -m marble`` as the training entrypoint."""

from marble.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
