"""Entry point: ``uv run main.py [viz|recon|config] ...`` (same as ``uv run rerun-viz ...``).

The original invocation ``main.py --config cfg.yaml --bags a.mcap b.mcap`` still works
and runs ``viz``.
"""

import sys

from rerun_viz.cli import main


def _legacy_argv(argv: list[str]) -> list[str]:
    if not argv or not argv[0].startswith("-") or argv[0] in ("-h", "--help"):
        return argv
    out, in_bags = ["viz"], False
    for arg in argv:
        if arg == "--bags":
            in_bags = True
        elif in_bags and not arg.startswith("-"):
            out += ["--bag", arg]
        else:
            in_bags = False
            out.append(arg)
    return out


if __name__ == "__main__":
    sys.argv[1:] = _legacy_argv(sys.argv[1:])
    main()
