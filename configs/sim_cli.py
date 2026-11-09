"""
sim_cli.py - what the sim CLIs share: their exit codes, and running many
cases where a refusal is reported and the run goes on.

Exit codes, one meaning in main.py, main_HW.py, main_ALG.py and
hw_vs_alg_check.py:
    0  every case (section) ran clean
    1  a failure (an error that is not a refusal) or a mismatch
    2  something was refused or skipped, and nothing failed
       (an argparse usage error also exits 2)
A refusal is never a pass: it must not read as 0 to a caller of the code.
(main.py adds 3: everything ran, only float taps differ.)
"""

import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import sim_config as SC

EXIT_CLEAN, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2


def exit_code(failed, refused):
    """The code for a run: failed / refused are truthy when any case was."""
    return EXIT_FAILED if failed else EXIT_REFUSED if refused else EXIT_CLEAN


def load_config(path, layer=None, backend=None):
    """sim_config.load_config for a CLI: a file the sim refuses prints
    `refused: <why>` and exits EXIT_REFUSED (not 1: nothing failed)."""
    try:
        return SC.load_config(path, layer, backend)
    except SC.REFUSED as e:
        print(f"refused: {e}")
        sys.exit(EXIT_REFUSED)


def not_run(header, e, side=""):
    """A case that raised: a documented refusal (sim_config.REFUSED, its
    message names what and why) prints `[side ]refused: <why>` and gives
    "refused"; anything else prints `[side ]FAILED: <type>: <msg>` with its
    traceback and gives "failed". The one rule every CLI reports cases by."""
    who = f"{side} " if side else ""
    if isinstance(e, SC.REFUSED):
        print(f"\n=== {header} ===\n  {who}refused: {e}")
        return "refused"
    print(f"\n=== {header} ===\n  {who}FAILED: {type(e).__name__}: {e}")
    traceback.print_exception(e)
    return "failed"


def run_each(names, side, run_one):
    """run_one(name) for every case name, in order. A documented refusal
    or any other error is reported by not_run; either way the run goes on.
    Prints the summary line `<n>/<total> ran[, refused: ...][, FAILED: ...]`
    and returns the exit code."""
    if not names:
        raise ValueError("no cases to run")        # 0/0 ran must never read as clean
    refused, failed = [], []
    for name in names:
        try:
            run_one(name)
        except (Exception, SystemExit) as e:
            kind = not_run(f"{name}  {side}", e)
            (refused if kind == "refused" else failed).append(name)
    ran = len(names) - len(refused) - len(failed)
    print(f"\n{ran}/{len(names)} ran"
          + (f", refused: {', '.join(refused)}" if refused else "")
          + (f", FAILED: {', '.join(failed)}" if failed else ""))
    return exit_code(failed, refused)
