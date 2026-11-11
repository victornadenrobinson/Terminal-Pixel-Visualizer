#!/usr/bin/env python3
"""Automatically view a molecular structure, trajectory, or numeric table."""
import argparse
import itertools
import shutil
import sys
import time

import numpy as np
from ase.data import atomic_numbers
from ase.io import iread

from mdview_lit import render_lit
from termctl import Controls
import termplot


def _xyz_header(lines):
    if len(lines) < 3:
        return False
    try:
        count = int(lines[0].strip())
        fields = lines[2].split()
        return count > 0 and len(fields) >= 4 and fields[0] in atomic_numbers and all(
            np.isfinite(float(value)) for value in fields[1:4])
    except ValueError:
        return False


def _prepare(atoms):
    if not len(atoms):
        raise ValueError("the structure contains no atoms")
    if np.any(atoms.cell.lengths() <= 0) or abs(atoms.cell.volume) < 1e-9:
        atoms = atoms.copy()
        lo = atoms.positions.min(axis=0)
        hi = atoms.positions.max(axis=0)
        atoms.positions -= lo - 2
        atoms.set_cell(np.maximum(hi - lo + 4, 4))
        atoms.pbc = False
    return atoms


def _structures(path, fmt, width, height):
    frames = iter(iread(path, format=fmt, index=":"))
    first = next(frames, None)
    if first is None:
        raise ValueError("the structure file contains no frames")
    second = next(frames, None) if sys.stdout.isatty() else None
    if second is None:
        print(render_lit(_prepare(first), width=width, height=height, note=str(path)))
        return
    last = first
    yaw, tilt = 0., 18.
    with Controls() as controls:
        for last in itertools.chain((first, second), frames):
            start = time.monotonic()
            quit_ = False
            for event in controls.poll():
                if event[0] == "key" and event[1] in ("q", "\x03"):
                    quit_ = True
                elif event[0] == "rot":
                    yaw += event[1] * 2.5
                    tilt = np.clip(tilt - event[2] * 3, -89, 89)
                elif event[0] == "arrow":
                    yaw += {"left": -6, "right": 6}.get(event[1], 0)
                    tilt = np.clip(tilt + {"up": 6, "down": -6}.get(event[1], 0), -89, 89)
            if quit_:
                break
            print(render_lit(_prepare(last), yaw=yaw, tilt=tilt, width=width,
                             height=height, note=str(path)), flush=True)
            time.sleep(max(0, .1 - (time.monotonic() - start)))
    print(render_lit(_prepare(last), yaw=yaw, tilt=tilt, width=width, height=height, note=str(path)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", help="XYZ, ASE trajectory, or numeric CSV/whitespace table")
    p.add_argument("--width", type=int, help="maximum output columns")
    p.add_argument("--height", type=int, help="maximum output rows")
    args = p.parse_args(argv)
    terminal = shutil.get_terminal_size((100, 40))
    width = args.width if args.width is not None else terminal.columns
    height = args.height if args.height is not None else terminal.lines
    if width < 14 or height < 8:
        p.error("width must be at least 14 and height at least 8")
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
            _structures(args.dataset, "extxyz", width - 2, height - 3)
        elif args.dataset.lower().endswith((".traj", ".json", ".cif", ".pdb")):
            _structures(args.dataset, None, width - 2, height - 3)
        elif lines:
            first = next((line for line in lines if not line.lstrip().startswith("#")), "")
            fields = first.split(",") if "," in first else first.split()
            try:
                float(fields[0])
                header = False
            except (ValueError, IndexError):
                header = True
            command = "hist" if len(fields) == 1 else "scatter"
            options = [command, args.dataset, "--width", str(width), "--height", str(height - 1)]
            if header:
                options.append("--header")
            termplot.main(options)
        else:
            raise ValueError("empty or unsupported dataset")
    except (OSError, ValueError, IndexError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
