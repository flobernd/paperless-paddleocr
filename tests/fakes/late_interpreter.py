"""Stand-in for the worker interpreter that starts the worker after its client died.

The client starts this script in place of the interpreter, through a shell
script that runs it with ``exec``. It waits until the client's request sits
in its stdin, without reading it, writes its pid to the file that
``FAKE_INTERPRETER_PID_FILE`` names, and waits until its parent changes, as
when a test kills the client. Then it replaces itself with this Python and
the worker's arguments, so the worker keeps the pid and finds the request.
"""

import os
import select
import sys
import time

client = os.getppid()
ready, _, _ = select.select([sys.stdin.fileno()], [], [], 30)
if not ready:
    sys.exit("no request arrived within 30 s")
pid_file = os.environ["FAKE_INTERPRETER_PID_FILE"]
# The rename means a reader never finds the file empty.
with open(pid_file + ".tmp", "w", encoding="utf-8") as stream:
    stream.write(str(os.getpid()))
os.replace(pid_file + ".tmp", pid_file)
deadline = time.monotonic() + 30
while os.getppid() == client:
    if time.monotonic() > deadline:
        sys.exit("the client was not killed within 30 s")
    time.sleep(0.02)
# The dead client held the only reader of the worker's stderr, and the first
# write there would end the worker at once, with or without its parent check.
os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
