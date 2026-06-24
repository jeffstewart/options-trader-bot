#!/usr/bin/env python3
"""
spawn_daemon.py PIDFILE LOGFILE CMD...

Launch CMD as a TRUE daemon: double-fork + setsid so it detaches from the
controlling terminal, becomes its own session leader, and is reparented to
init (ppid 1) — surviving independent of any Claude/harness session or shell.
stdout/stderr → LOGFILE (append), stdin → /dev/null. Writes the daemon PID to
PIDFILE so the caller can verify it.

macOS has no `setsid` binary; this is the portable Python equivalent.
"""
import os
import sys
from pathlib import Path

PIDFILE = sys.argv[1]
LOGFILE = sys.argv[2]
CMD = sys.argv[3:]
# Daemons run with CWD=data/ so the codebase's bare data-file refs (open("foo.json")) resolve into
# data/. Code is imported via the _trader_paths.pth (core/ + research/ on sys.path), so CWD doesn't
# affect imports. Pass PIDFILE/LOGFILE as ABSOLUTE paths (logs/ lives outside data/).
# WORKDIR is derived (no hardcoded host path) and matches config.DATA_DIR's TRADER_DATA_DIR override.
WORKDIR = os.environ.get("TRADER_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))

# first fork — parent returns to the caller immediately
if os.fork() > 0:
    os._exit(0)
os.setsid()                     # new session, detach from controlling terminal
# second fork — prevent re-acquiring a terminal; reparent to init
if os.fork() > 0:
    os._exit(0)

os.chdir(WORKDIR)
with open(PIDFILE, "w") as f:
    f.write(str(os.getpid()))   # PID is preserved across execv below

fd = os.open(LOGFILE, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(fd, 1)
os.dup2(fd, 2)
dn = os.open(os.devnull, os.O_RDONLY)
os.dup2(dn, 0)

os.execv(CMD[0], CMD)           # replace image; PID/ session unchanged
