"""Automatically view structures, trajectories, or numeric tables."""

import argparse
import contextlib
import io
import itertools
import json
import os
import re
import shutil
import sys
import time

import numpy as np

from . import plot
from .controls import Controls
from .raster import render_lit

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_RGB = re.compile(r"(38|48);2;(\d+);(\d+);(\d+)")


def _xyz_header(lines):
    if len(lines) < 3:
        return False
    try:
        fields = lines[2].split()
        return (
            int(lines[0].strip()) > 0
            and len(fields) >= 4
            and bool(re.fullmatch(r"[A-Z][a-z]?", fields[0]))
            and all(np.isfinite(float(value)) for value in fields[1:4])
        )
    except ValueError:
        return False


def _prepare(atoms):
    if not len(atoms):
        raise ValueError("the structure contains no atoms")
    if np.any(atoms.cell.lengths() <= 0) or abs(atoms.cell.volume) < 1e-9:
        atoms = atoms.copy()
        lo, hi = atoms.positions.min(axis=0), atoms.positions.max(axis=0)
        atoms.positions -= lo - 2
        atoms.set_cell(np.maximum(hi - lo + 4, 4))
        atoms.pbc = False
    return atoms


def color_support(stream=None) -> str:
    """Detect truecolor, 256-colour, or plain output from the terminal environment."""
    stream = stream or sys.stdout
    if (
        not stream.isatty()
        or "NO_COLOR" in os.environ
        or os.environ.get("TERM") == "dumb"
    ):
        return "plain"
    if os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
        return "truecolor"
    return "256" if "256color" in os.environ.get("TERM", "") else "plain"


def _output_color(text, mode):
    if mode == "plain":
        return _ANSI.sub("", text)
    if mode == "256":

        def convert(match):
            values = [round(int(v) / 255 * 5) for v in match.group(2, 3, 4)]
            return (
                f"{match.group(1)};5;{16 + 36 * values[0] + 6 * values[1] + values[2]}"
            )

        return _RGB.sub(convert, text)
    return text


def _structures(path, fmt, width, height, args, color):
    try:
        from ase.io import iread
    except ImportError as exc:
        raise ValueError(
            "molecular formats require: pip install 'terminal-pixel-visualizer[ase]'"
        ) from exc
    frames = iter(iread(path, format=fmt, index=":"))
    first = next(frames, None)
    if first is None:
        raise ValueError("the structure file contains no frames")
    playback = sys.stdout.isatty() or bool(args.record) or args.follow
    stream = itertools.chain((first,), frames) if playback else iter((first,))
    yaw, tilt, count = 0.0, 18.0, 0
    started = time.monotonic()
    with contextlib.ExitStack() as stack:
        cast = (
            stack.enter_context(open(args.record, "w", encoding="utf-8"))
            if args.record
            else None
        )
        if cast:
            cast.write(
                json.dumps({"version": 2, "width": width + 2, "height": height + 3})
                + "\n"
            )
        controls = (
            stack.enter_context(
                Controls(
                    mouse=not args.no_mouse and color != "plain",
                    screen=color != "plain",
                )
            )
            if sys.stdout.isatty()
            else None
        )

        def show(atoms):
            text = render_lit(
                _prepare(atoms),
                yaw=yaw,
                tilt=tilt,
                width=width,
                height=height,
                mode=args.mode,
                ascii_only=color == "plain",
                note=str(path),
            )
            text = _output_color(text, color) + "\n"
            sys.stdout.write(text)
            sys.stdout.flush()
            if cast:
                cast.write(
                    json.dumps([round(time.monotonic() - started, 6), "o", text]) + "\n"
                )
                cast.flush()

        def poll():
            nonlocal yaw, tilt
            for event in controls.poll() if controls else ():
                if event[0] == "key" and event[1] in ("q", "\x03"):
                    return False
                if event[0] == "rot":
                    yaw += event[1] * 2.5
                    tilt = np.clip(tilt - event[2] * 3, -89, 89)
                elif event[0] == "arrow":
                    yaw += {"left": -6, "right": 6}.get(event[1], 0)
                    tilt = np.clip(
                        tilt + {"up": 6, "down": -6}.get(event[1], 0), -89, 89
                    )
            return True

        for atoms in stream:
            start = time.monotonic()
            if not poll():
                return
            show(atoms)
            count += 1
            if playback:
                time.sleep(max(0.0, 1 / args.fps - (time.monotonic() - start)))
        while args.follow and poll():
            time.sleep(0.2)
            try:
                current = list(iread(path, format=fmt, index=":"))
            except (OSError, ValueError, EOFError):
                continue
            if len(current) < count:
                count = 0
            for atoms in current[count:]:
                show(atoms)
            count = len(current)
    if controls and color != "plain":
        cast = None
        show(atoms)


def main(argv: list[str] | None = None) -> None:
    """Entry point for automatic format detection and playback."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", help="molecular structure, trajectory, or numeric table")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--lit", action="store_true", help="use lit spheres (the default)")
    p.add_argument("--mode", choices=("half", "quad", "braille"), default="half")
    p.add_argument(
        "--color", choices=("auto", "truecolor", "256", "plain"), default="auto"
    )
    p.add_argument("--fps", type=float, default=10)
    p.add_argument(
        "--follow", action="store_true", help="watch for appended trajectory frames"
    )
    p.add_argument("--record", help="write an asciinema v2 cast")
    p.add_argument("--no-mouse", action="store_true")
    args = p.parse_args(argv)
    terminal = shutil.get_terminal_size((100, 40))
    width = args.width if args.width is not None else terminal.columns
    height = args.height if args.height is not None else terminal.lines
    if width < 14 or height < 8 or args.fps <= 0:
        p.error("width >= 14, height >= 8 and fps > 0 are required")
    if args.record:
        same_path = os.path.realpath(args.record) == os.path.realpath(args.dataset)
        if (
            not same_path
            and os.path.exists(args.record)
            and os.path.exists(args.dataset)
        ):
            same_path = os.path.samefile(args.record, args.dataset)
        if same_path:
            p.error("--record must not overwrite the input dataset")
    color = color_support() if args.color == "auto" else args.color
    try:
        with open(args.dataset, "rb") as source:
            sample = source.read(8192)
        try:
            text = sample.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = ""
        raw_lines = text.splitlines()
        lines = [line for line in raw_lines if line.strip()]
        if _xyz_header(raw_lines):
            _structures(args.dataset, "extxyz", width - 2, height - 3, args, color)
        elif args.dataset.lower().endswith(
            (".traj", ".json", ".cif", ".pdb", ".vasp", ".xyz", ".extxyz")
        ) or os.path.basename(args.dataset).upper() in ("POSCAR", "CONTCAR"):
            _structures(args.dataset, None, width - 2, height - 3, args, color)
        elif lines:
            if args.follow or args.record:
                raise ValueError("--follow and --record require a molecular trajectory")
            first = next(
                (line for line in lines if not line.lstrip().startswith("#")), ""
            )
            fields = first.split(",") if "," in first else first.split()
            try:
                float(fields[0])
                header = False
            except (ValueError, IndexError):
                header = True
            options = [
                "hist" if len(fields) == 1 else "scatter",
                args.dataset,
                "--width",
                str(width),
                "--height",
                str(height - 1),
            ]
            if header:
                options.append("--header")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                plot.main(options)
            print(_output_color(output.getvalue(), color), end="")
        else:
            raise ValueError("empty or unsupported dataset")
    except (OSError, ValueError, IndexError, KeyError) as exc:
        p.error(str(exc))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
