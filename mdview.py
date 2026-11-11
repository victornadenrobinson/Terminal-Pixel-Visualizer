"""Terminal rendering primitives and the unlit molecular views."""
import itertools
import sys

import numpy as np

LAST = {"head": "", "cells": []}
O_COLOR = np.array([255, 70, 60], dtype=np.uint8)
H_COLOR = np.array([90, 230, 255], dtype=np.uint8)


def _header(atoms, step, note=""):
    lengths = atoms.cell.lengths()
    return (f"{len(atoms)} atoms  box {lengths[0]:.2f}×{lengths[1]:.2f}×{lengths[2]:.2f} Å"
            f"  step {step}  T {atoms.info.get('T_inst', atoms.info.get('T', 0)):.0f} K"
            f"  P {atoms.info.get('P', atoms.info.get('P_inst', 0)):.3g}  {note}")


def _rgb(c):
    return f"{int(c[0])};{int(c[1])};{int(c[2])}"


def pack_half(pixels):
    """Pack a (2*height, width, 3) RGB image into terminal cells."""
    top, bottom = pixels[::2], pixels[1::2]
    return [[f"\x1b[38;2;{_rgb(a)};48;2;{_rgb(b)}m▀" for a, b in zip(ar, br)]
            for ar, br in zip(top, bottom)]


def frame(head, cells):
    """Record a frame for DiffPrinter and return a full redraw."""
    width = len(cells[0]) if cells else 0
    head = head[:width + 2].replace("\n", " ").replace("\x1b", "")
    rows = [["│", *row, "\x1b[0m│"] for row in cells]
    LAST["head"] = head
    LAST["cells"] = [["┌" + "─" * width + "┐"], *rows, ["└" + "─" * width + "┘"]]
    return ("\x1b[H\x1b[0m" + head + "\x1b[K\n"
            + "\n".join("".join(row) + "\x1b[0m" for row in LAST["cells"]) + "\x1b[0m")


class DiffPrinter:
    """Write only cells changed since the previous draw."""
    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.head = None
        self.cells = []

    def draw(self):
        head, cells = LAST["head"], LAST["cells"]
        out = []
        if head != self.head:
            out.append(f"\x1b[1;1H\x1b[0m{head}\x1b[K")
        for y, row in enumerate(cells):
            for x, cell in enumerate(row):
                if y >= len(self.cells) or x >= len(self.cells[y]) or cell != self.cells[y][x]:
                    out.append(f"\x1b[{y + 2};{x + 1}H{cell}")
        if len(cells) != len(self.cells) or any(
                len(a) != len(b) for a, b in zip(cells, self.cells)):
            # Clear stale content when the frame shrinks.
            if self.cells:
                out = ["\x1b[2J\x1b[H\x1b[0m" + head + "\n" +
                       "\n".join("".join(row) + "\x1b[0m" for row in cells)]
        if out:
            self.stream.write("".join(out) + "\x1b[0m")
            self.stream.flush()
        self.head, self.cells = head, [row[:] for row in cells]


def render(atoms, step=0, width=96, height=30, slab=2.2, sweep=0.02,
           note="", atom_scale=None):
    if width < 1 or height < 1 or slab < 0:
        raise ValueError("width and height must be positive and slab non-negative")
    lengths = atoms.cell.lengths()
    if np.any(lengths <= 0):
        raise ValueError("atoms must have a nonzero cell")
    pixels = np.zeros((height * 2, width, 3), dtype=np.uint8)
    depth = np.full((height * 2, width), -np.inf)
    positions = atoms.get_scaled_positions(wrap=True) * lengths
    zplane = (step * sweep) % lengths[2]
    scale = max(1, int(atom_scale or 1))
    for pos, number in zip(positions, atoms.numbers):
        if number not in (1, 8):
            continue
        x = min(width - 1, int(pos[0] / lengths[0] * width))
        y = min(height * 2 - 1, int((1 - pos[1] / lengths[1]) * height * 2))
        dz = abs((pos[2] - zplane + lengths[2] / 2) % lengths[2] - lengths[2] / 2)
        color = O_COLOR if number == 8 else H_COLOR
        color = color if dz <= slab / 2 else (color * 0.28).astype(np.uint8)
        size = scale * (2 if number == 8 else 1)
        x0, y0 = x - size // 2, y - size // 2
        xs, ys = slice(max(0, x0), min(width, x0 + size)), slice(max(0, y0), min(height * 2, y0 + size))
        mask = pos[2] > depth[ys, xs]
        pixels[ys, xs][mask] = color
        depth[ys, xs][mask] = pos[2]
    return frame(_header(atoms, step, f"slab z {zplane:.2f} Å  {note}"), pack_half(pixels))


def camera(atoms, yaw=0, tilt=18, view_dir=None):
    if view_dir is not None:
        forward = np.asarray(view_dir, dtype=float)
        if forward.shape != (3,) or not np.any(forward):
            raise ValueError("view direction must contain three nonzero-direction components")
        forward /= np.linalg.norm(forward)
        up = np.array([0., 0., 1.])
        if abs(forward @ up) > .99:
            up = np.array([0., 1., 0.])
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
    else:
        y, t = np.radians([yaw, tilt])
        forward = np.array([np.sin(y) * np.cos(t), -np.cos(y) * np.cos(t), np.sin(t)])
        right = np.array([np.cos(y), np.sin(y), 0.])
        up = np.cross(forward, right)
    return np.stack((right, up, forward))


def box_corners(atoms):
    return np.array(list(itertools.product((0., 1.), repeat=3))) @ np.asarray(atoms.cell)


def bonds_for(atoms):
    """Return O-H pairs within a generous covalent-bond cutoff."""
    oi = np.flatnonzero(atoms.numbers == 8)
    hi = np.flatnonzero(atoms.numbers == 1)
    if not len(oi) or not len(hi):
        return []
    positions = atoms.positions
    delta = positions[oi, None, :] - positions[hi, :]
    distances = np.linalg.norm(delta, axis=2)
    return [(int(oi[i]), int(hi[j])) for i, j in zip(*np.where(distances < 1.25))]


def render3d(atoms, yaw=0, tilt=18, step=0, width=96, height=30,
             ascii_only=False, note="", bonds=True, box=True, zoom=2.0,
             clip=0.0, hi=1.0, view_dir=None, ortho=False, braille=False, **kwargs):
    from mdview_lit import render_lit
    return render_lit(atoms, yaw=yaw, tilt=tilt, step=step, width=width,
                      height=height, note=note, bonds=bonds, box=box, zoom=zoom,
                      clip=clip, hi=hi, view_dir=view_dir, ortho=ortho,
                      mode="braille" if braille else "half", light=False,
                      ascii_only=ascii_only, **kwargs)
