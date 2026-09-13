#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  Nova Upinel Chow, MSc, LLM, BBA, MENSA  ·  upinel@me.com  ·  upinel.com
#  Copyright (c) 2026 Nova Upinel Chow. All rights reserved.
#
#  Upinel Personal Free License: free for personal use, commercial use by
#  written permission, and anything built from this must credit the author.
#  See LICENSE.
#
#  "Make it work, make it right, make it fast - then measure it, because
#   the third one is only a claim until the numbers agree."
# ─────────────────────────────────────────────────────────────────────────────
"""Regression tests for the on-disk model scan behind the start/restart picker.

(Differs from the GGUF project only in that MLX models are .safetensors.)

The picker offers whatever `model_dirs_on_disk` returns, and start.sh loads the
first thing the user picks. So a directory that is not actually loadable - left
behind by an interrupted download, or emptied by hand - must never appear in
that list, or the picker becomes a way to crash the server on purpose.

Runs offline against a throwaway models/ tree of empty files.

Run:  python3 bench/verify-model-picker.py
"""

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

SHIM = r'''
source "$1/lib/common.sh"
load_config
MODELS_DIR="$2"
model_dirs_on_disk | while IFS= read -r d; do basename "$d"; done
echo "---"
model_repo_from_dir "$2/owner--name"
'''

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


def scan(root):
    r = subprocess.run(["bash", "-c", SHIM, "shim", REPO, root],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    dirs, _, repo = r.stdout.partition("---")
    return sorted(d for d in dirs.strip().split("\n") if d), repo.strip()


def main():
    print("\n  On-disk model scan regression tests\n")
    root = tempfile.mkdtemp(prefix="pick-")
    try:
        os.makedirs(os.path.join(root, "good--model-a"))
        open(os.path.join(root, "good--model-a", "model-00001-of-00002.safetensors"), "w").close()
        os.makedirs(os.path.join(root, "good--model-b"))
        open(os.path.join(root, "good--model-b", "model-00002-of-00002.safetensors"), "w").close()

        # An interrupted download: the directory exists, the weights do not.
        os.makedirs(os.path.join(root, "half--downloaded"), exist_ok=True)
        # A directory holding only side files is not a model.
        os.makedirs(os.path.join(root, "side--only"))
        open(os.path.join(root, "side--only", "tokenizer.json"), "w").close()
        # A stray file where a directory should be.
        open(os.path.join(root, "not-a-dir"), "w").close()

        # A download that got some shards but not all. Safetensors files are
        # present, so "does it have weights" says yes; the index names every
        # shard, and that is what catches it.
        partial = os.path.join(root, "partial--model")
        os.makedirs(partial)
        open(os.path.join(partial, "model-00001-of-00002.safetensors"), "w").close()
        open(os.path.join(partial, "model.safetensors.index.json"), "w").write(
            '{"weight_map": {"a": "model-00001-of-00002.safetensors",'
            ' "b": "model-00002-of-00002.safetensors"}}')

        dirs, repo = scan(root)
        check("only loadable model directories are offered",
              dirs, ["good--model-a", "good--model-b"])
        check("an interrupted download is not offered",
              "half--downloaded" in dirs, False)
        check("a partially downloaded multi-shard pack is not offered",
              "partial--model" in dirs, False)
        check("the directory name round-trips to a repo id",
              repo, "owner/name")

        empty = tempfile.mkdtemp(prefix="pick-empty-")
        try:
            check("an empty models directory offers nothing", scan(empty)[0], [])
        finally:
            shutil.rmtree(empty, ignore_errors=True)

        print(f"\n  {PASSED} passed, {FAILED} failed\n")
        return 1 if FAILED else 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
