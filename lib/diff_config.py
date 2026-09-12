#!/usr/bin/env python3
"""
Print the difference between the saved config snapshot and the pending one,
paired per key.

Usage: diff_config.py <snapshot> <pending>

Colour is applied only when stdout is a terminal.
"""

import sys

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = DIM = RESET = ""


def load(path):
    values = {}
    try:
        with open(path) as fh:
            for line in fh:
                if "=" in line:
                    key, value = line.rstrip("\n").split("=", 1)
                    values[key] = value
    except OSError:
        pass
    return values


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2

    before = load(sys.argv[1])
    after = load(sys.argv[2])

    changed = 0
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key), after.get(key)
        if old == new:
            continue
        changed += 1
        print(f"    {key:<22} {RED}{old if old is not None else '(unset)'}{RESET}"
              f"  {DIM}->{RESET}  {GREEN}{new if new is not None else '(unset)'}{RESET}")
    return 0 if changed else 1


if __name__ == "__main__":
    sys.exit(main())
