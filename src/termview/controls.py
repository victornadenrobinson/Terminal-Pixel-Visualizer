"""Nonblocking keyboard and SGR mouse controls for a terminal."""

import os
import re
import select
import sys

try:
    import termios
    import tty
except ImportError:
    termios = None  # type: ignore[assignment]
    tty = None  # type: ignore[assignment]


class Controls:
    """Context-managed keyboard and mouse controls; inert on unsupported terminals."""

    def __init__(self, mouse=True, stdin=None, stdout=None):
        self.mouse = mouse
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout
        self.saved = None
        self.buffer = ""
        self.drag = None

    def __enter__(self):
        if self.stdin.isatty() and termios is not None and tty is not None:
            self.saved = termios.tcgetattr(self.stdin.fileno())
            tty.setcbreak(self.stdin.fileno())
        if self.stdout.isatty() and os.name != "nt":
            self.stdout.write("\x1b[?1049h\x1b[?25l")
            if self.mouse:
                self.stdout.write("\x1b[?1000h\x1b[?1002h\x1b[?1006h")
            self.stdout.flush()
        return self

    def __exit__(self, *exc):
        try:
            if self.saved is not None:
                termios.tcsetattr(self.stdin.fileno(), termios.TCSADRAIN, self.saved)
        finally:
            if self.stdout.isatty() and os.name != "nt":
                self.stdout.write(
                    "\x1b[0m\x1b[?1000l\x1b[?1002l\x1b[?1006l\x1b[?25h\x1b[?1049l"
                )
                self.stdout.flush()

    def poll(self):
        if self.saved is None:
            return []
        while select.select([self.stdin], [], [], 0)[0]:
            data = os.read(self.stdin.fileno(), 4096)
            if not data:
                break
            self.buffer += data.decode("utf-8", "replace")
        events = []
        while self.buffer:
            if self.buffer.startswith("\x1b[<"):
                match = re.match(r"\x1b\[<(\d+);(\d+);(\d+)([Mm])", self.buffer)
                if not match:
                    break
                button, x, y = map(int, match.group(1, 2, 3))
                release = match.group(4) == "m"
                self.buffer = self.buffer[match.end() :]
                if button & 64:
                    events.append(("zoom", 1 if button & 1 == 0 else -1))
                elif release:
                    self.drag = None
                elif button & 32:
                    if self.drag is not None:
                        events.append(("rot", x - self.drag[0], y - self.drag[1]))
                    self.drag = (x, y)
                else:
                    self.drag = (x, y)
                    events.append(("click", x, y))
            elif self.buffer.startswith("\x1b["):
                if len(self.buffer) < 3:
                    break
                key = {"A": "up", "B": "down", "C": "right", "D": "left"}.get(
                    self.buffer[2]
                )
                self.buffer = self.buffer[3:]
                if key:
                    events.append(("arrow", key))
            elif self.buffer == "\x1b":
                break
            else:
                events.append(("key", self.buffer[0]))
                self.buffer = self.buffer[1:]
        return events
