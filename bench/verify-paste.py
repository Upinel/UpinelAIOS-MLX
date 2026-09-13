#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  Nova Upinel Chow, MSc, LLM, BBA, MENSA  ·  dev@upinel.com  ·  upinel.com
#  Copyright (c) 2026 Nova Upinel Chow. All rights reserved.
#
#  Upinel Personal Free License: free for personal use, and free for creators
#  (YouTubers, KOLs) to make content with - just email dev@upinel.com to say so.
#  Other commercial use needs written permission. Derivatives must credit the
#  author. Covers this project's own code only. See LICENSE.
#
#  "Make it work, make it right, make it fast - then measure it, because
#   the third one is only a claim until the numbers agree."
# ─────────────────────────────────────────────────────────────────────────────
"""Regression tests for multi-line paste handling in lib/chat.py.

The bug these cover: input() returns only the FIRST line of a paste and leaves
the rest queued, so pasting a code block was sent as one message per line - and
a pasted line beginning with "/" was executed as a command, which meant pasting
a file that contained "/exit" quit the session and silently dropped the rest.

read_message() fixes that by collecting the remainder. Its grouping decision is
"keep taking lines as long as they arrive at once", which is what these tests
drive directly: chat.next_line is replaced with a scripted feed, where a string
means "this line was already buffered" and a pause means "nothing pending, the
message is over".

That keeps the tests offline and instant - no terminal, no server, no model.

Run:  python3 bench/verify-paste.py
"""

import importlib.util
import io
import os
import sys
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

_spec = importlib.util.spec_from_file_location(
    "chat", os.path.join(REPO, "lib", "chat.py"))
chat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chat)

PAUSE = object()          # nothing pending: end of message

PASSED = 0
FAILED = 0


def check(label, got, want):
    global PASSED, FAILED
    if got == want:
        print(f"  [PASS] {label}")
        PASSED += 1
    else:
        print(f"  [FAIL] {label}\n         got  {got!r}\n         want {want!r}")
        FAILED += 1


def read(*items):
    """Run read_message() against a scripted sequence of lines."""
    feed = list(items)

    def fake_next_line(timeout=None):
        if not feed:
            return chat._NOT_PENDING
        item = feed.pop(0)
        return chat._NOT_PENDING if item is PAUSE else item

    chat.next_line = fake_next_line
    with redirect_stdout(io.StringIO()):        # swallow the prompt
        return chat.read_message("p> ")


def main():
    print("\n  Multi-line paste regression tests\n")

    # The reported bug: a three-line paste is ONE message.
    check("three lines arriving at once become one message",
          read("alpha", "bravo", "charlie", PAUSE), ("alpha\nbravo\ncharlie", 3))

    # Separate messages must stay separate.
    check("a single line is one message", read("hello", PAUSE), ("hello", 1))
    check("a second message is not folded into the first",
          read("one", PAUSE), ("one", 1))

    # Two lines, then the user pauses, then two more.
    check("a later message starts fresh",
          read("a", "b", PAUSE), ("a\nb", 2))

    # Blank lines the user pasted on purpose survive.
    check("a blank line inside a paste is kept",
          read("one", "", "two", PAUSE), ("one\n\ntwo", 3))

    # A paste normally ends with a newline; that must not add a blank line.
    check("a trailing blank line is dropped",
          read("one", "two", "", PAUSE), ("one\ntwo", 2))
    check("several trailing blank lines are dropped too",
          read("one", "", "", PAUSE), ("one", 1))

    # Terminals that bracket a paste may put the whole block in one line, or
    # hand the markers through - either way they must not reach the model.
    check("markers around a whole block in one line are stripped",
          read("\x1b[200~a\nb\x1b[201~", PAUSE), ("a\nb", 2))
    check("markers on their own lines are stripped",
          read("\x1b[200~a", "b", "\x1b[201~", PAUSE), ("a\nb", 2))
    check("a marker inside a line is stripped",
          read("code\x1b[201~", PAUSE), ("code", 1))

    # main() only treats a SINGLE line as a command, so a pasted block that
    # starts with "/" is text. These assert the line count that decision uses.
    check("a pasted command block reports more than one line",
          read("/reset", "/help", "/exit", PAUSE)[1] > 1, True)
    check("a lone pasted command reports one line",
          read("/help", PAUSE)[1], 1)

    # Content with trailing spaces is not silently trimmed here: main() does
    # that, and this function must not mangle pasted code indentation.
    check("indentation is preserved",
          read("def f():", "    return 1", PAUSE),
          ("def f():\n    return 1", 2))

    print(f"\n  {PASSED} passed, {FAILED} failed\n")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
