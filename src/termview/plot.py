"""Braille scatter plots, histograms and four-panel reports for numeric tables."""

import argparse
import math
import sys

import numpy as np

from .config import RenderConfig
from .pack import BITS


def nice(lo: float, hi: float, count: int = 5) -> np.ndarray:
    """Return human-friendly tick locations covering a finite range."""
    if not np.isfinite([lo, hi]).all() or hi < lo or count < 2:
        raise ValueError("invalid tick range")
    if hi == lo:
        delta = abs(lo) * 0.05 or 1.0
        lo, hi = lo - delta, hi + delta
    raw = (hi - lo) / (count - 1)
    power = 10 ** math.floor(math.log10(raw))
    step = next(n * power for n in (1, 2, 5, 10) if n * power >= raw)
    return np.arange(math.ceil(lo / step), math.floor(hi / step) + 1) * step


class Canvas:
    """Braille dot canvas with a configurable terminal palette."""

    def __init__(self, width: int, height: int, config: RenderConfig | None = None):
        if width < 1 or height < 1:
            raise ValueError("canvas dimensions must be positive")
        self.width, self.height = width, height
        self.palette = (config or RenderConfig()).palette
        self.masks = np.zeros((height, width), dtype=np.uint8)
        self.colors = np.full((height, width), -1, dtype=int)

    def dots(self, x, y, color: int = 0) -> None:
        """Set one or more pixel coordinates, ignoring coordinates outside the canvas."""
        x, y = np.broadcast_arrays(np.asarray(x, dtype=int), np.asarray(y, dtype=int))
        x, y = x.ravel(), y.ravel()
        valid = (x >= 0) & (x < self.width * 2) & (y >= 0) & (y < self.height * 4)
        x, y = x[valid], y[valid]
        np.bitwise_or.at(self.masks, (y // 4, x // 2), BITS[y % 4, x % 2])
        self.colors[y // 4, x // 2] = int(color) % len(self.palette)

    def line(self, x0: int, y0: int, x1: int, y1: int, color: int = 0) -> None:
        """Draw a line between pixel coordinates."""
        n = max(abs(int(x1) - int(x0)), abs(int(y1) - int(y0))) + 1
        self.dots(
            np.rint(np.linspace(x0, x1, n)), np.rint(np.linspace(y0, y1, n)), color
        )

    def rows_text(self, color: bool = True) -> list[str]:
        """Return packed terminal rows, optionally including palette escapes."""
        rows = []
        for masks, colors in zip(self.masks, self.colors):
            row = ""
            previous = -1
            for mask, index in zip(masks, colors):
                if color and index != previous:
                    row += f"\x1b[{self.palette[index]}m" if index >= 0 else "\x1b[0m"
                    previous = index
                row += chr(0x2800 + int(mask)) if mask else " "
            rows.append(row + ("\x1b[0m" if color else ""))
        return rows


def _range(values):
    lo, hi = float(np.min(values)), float(np.max(values))
    if lo == hi:
        pad = abs(lo) * 0.05 or 1
        return lo - pad, hi + pad
    return lo, hi


def plot(
    x,
    y,
    width: int,
    height: int,
    xlabel: str = "x",
    ylabel: str = "y",
    groups=None,
    connect: bool = False,
) -> list[str]:
    """Create labelled braille plot rows for numeric data."""
    if width < 12 or height < 5:
        raise ValueError("plots require width >= 12 and height >= 5")
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if groups is not None:
        groups = np.asarray(groups)[valid]
    if not len(x):
        raise ValueError("no finite data to plot")
    xmin, xmax = _range(x)
    ymin, ymax = _range(y)
    canvas = Canvas(width - 9, height - 3)
    xd = np.rint((x - xmin) / (xmax - xmin) * (canvas.width * 2 - 1)).astype(int)
    yd = np.rint((ymax - y) / (ymax - ymin) * (canvas.height * 4 - 1)).astype(int)
    if groups is None:
        groups = np.zeros(len(x), dtype=int)
    _, indices = np.unique(groups, return_inverse=True)
    for group in np.unique(indices):
        mask = indices == group
        if connect:
            xx, yy = xd[mask], yd[mask]
            for i in range(len(xx) - 1):
                canvas.line(xx[i], yy[i], xx[i + 1], yy[i + 1], int(group))
        canvas.dots(xd[mask], yd[mask], int(group))
    ticks = nice(ymin, ymax, min(5, canvas.height))
    labels = [" " * 7 for _ in range(canvas.height)]
    for tick in ticks:
        row = round((ymax - tick) / (ymax - ymin) * (canvas.height - 1))
        labels[row] = f"{tick:7.3g}"[-7:]
    lines = [f"{ylabel[:7]:>7} │" + " " * canvas.width]
    lines += [label + " │" + row for label, row in zip(labels, canvas.rows_text())]
    lines.append(" " * 8 + "└" + "─" * canvas.width)
    xticks = nice(xmin, xmax)
    tickline = [" "] * canvas.width
    for tick in xticks:
        text = f"{tick:.3g}"[: canvas.width]
        pos = round((tick - xmin) / (xmax - xmin) * (canvas.width - 1))
        start = min(max(0, pos - len(text) // 2), max(0, canvas.width - len(text)))
        tickline[start : start + len(text)] = text
    lines.append(" " * 9 + "".join(tickline))
    lines.append(" " * 9 + xlabel[: canvas.width])
    return lines


def _column(data, spec):
    names = data.dtype.names
    if names:
        if spec in names:
            return np.asarray(data[spec])
        try:
            return np.asarray(data[names[int(spec)]])
        except (ValueError, IndexError):
            raise ValueError(f"unknown column {spec!r}") from None
    try:
        return np.asarray(data[:, int(spec)])
    except (ValueError, IndexError):
        raise ValueError(f"unknown column {spec!r}") from None


def main(argv: list[str] | None = None) -> None:
    """Run the standalone plotting CLI."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("scatter", "hist", "report"):
        s = sub.add_parser(name)
        s.add_argument(
            "file", help="whitespace or comma-separated numeric table; '-' for stdin"
        )
        s.add_argument("--x", default="0", help="column index (or name with --header)")
        s.add_argument("--y", default="1", help="column index (or name with --header)")
        s.add_argument("--color", help="column used to group points by color")
        s.add_argument("--clip", type=float, nargs=2, metavar=("LO", "HI"))
        s.add_argument("--width", type=int, default=80)
        s.add_argument("--height", type=int, default=20)
        s.add_argument(
            "--header", action="store_true", help="first row contains column names"
        )
        if name in ("hist", "report"):
            s.add_argument("--bins", type=int, default=20)
            s.add_argument(
                "--centre",
                action="store_true",
                help="center histogram data on its mean",
            )
    a = p.parse_args(argv)
    if a.width < 12 or a.height < 5 or (hasattr(a, "bins") and a.bins < 1):
        p.error("width >= 12, height >= 5 and bins >= 1 are required")
    if a.clip and a.clip[0] >= a.clip[1]:
        p.error("--clip requires LO < HI")
    try:
        if a.file == "-":
            text = sys.stdin.read()
        else:
            with open(a.file, encoding="utf-8") as source:
                text = source.read()
        from io import StringIO

        first = next(
            (
                line
                for line in text.splitlines()
                if line.strip() and not line.startswith("#")
            ),
            "",
        )
        data = np.genfromtxt(
            StringIO(text),
            delimiter="," if "," in first else None,
            names=True if a.header else None,
            dtype=None,
            encoding="utf-8",
            ndmin=1 if a.header else 2,
        )
        x = _column(data, a.x).astype(float)
        y = _column(data, a.y).astype(float) if a.command != "hist" else None
        groups = _column(data, a.color) if a.color else None
        valid = np.isfinite(x)
        if y is not None:
            valid &= np.isfinite(y)
        if a.clip:
            valid &= (x >= a.clip[0]) & (x <= a.clip[1])
        x = x[valid]
        if y is not None:
            y = y[valid]
        if groups is not None:
            groups = groups[valid]
        if not len(x):
            raise ValueError("no finite data after filtering")

        def hist(values, width, height, label):
            if a.centre:
                values = values - values.mean()
            counts, edges = np.histogram(values, bins=a.bins)
            centres = (edges[:-1] + edges[1:]) / 2
            return plot(centres, counts, width, height, label, "count", connect=True)

        if a.command == "scatter":
            lines = plot(x, y, a.width, a.height, a.x, a.y, groups)
        elif a.command == "hist":
            lines = hist(x, a.width, a.height, a.x)
        else:
            if a.width < 25 or a.height < 10:
                raise ValueError("report requires width >= 25 and height >= 10")
            left, right = (a.width - 1) // 2, a.width // 2
            top, bottom = a.height // 2, a.height - a.height // 2
            panels = [
                plot(x, y, left, top, a.x, a.y, groups),
                hist(x, right, top, a.x),
                plot(np.arange(len(x)), x, left, bottom, "sample", a.x, connect=True),
                hist(y, right, bottom, a.y),
            ]
            lines = [
                left_row + " " + right_row
                for left_row, right_row in zip(panels[0], panels[1])
            ]
            lines += [
                left_row + " " + right_row
                for left_row, right_row in zip(panels[2], panels[3])
            ]
        print("\n".join(lines))
    except (ValueError, OSError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
