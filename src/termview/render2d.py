"""Terminal rendering primitives and the unlit molecular views."""

import itertools
import sys

import numpy as np

from .config import RenderConfig
from .pack import pack_half


def _header(atoms, step, note=""):
    lengths = atoms.cell.lengths()
    return (
        f"{len(atoms)} atoms  box {lengths[0]:.2f}×{lengths[1]:.2f}×{lengths[2]:.2f} Å"
        f"  step {step}  T {atoms.info.get('T_inst', atoms.info.get('T', 0)):.0f} K"
        f"  P {atoms.info.get('P', atoms.info.get('P_inst', 0)):.3g}  {note}"
    )


def frame(head, cells, renderer=None):
    """Record a frame for DiffPrinter and return a full redraw."""
    width = len(cells[0]) if cells else 0
    head = "".join(c if c.isprintable() else " " for c in head)[: width + 2]
    rows = [["│", *row, "\x1b[0m│"] for row in cells]
    framed_cells = [["┌" + "─" * width + "┐"], *rows, ["└" + "─" * width + "┘"]]
    if renderer is not None:
        renderer.head, renderer.cells = head, framed_cells
    return (
        "\x1b[H\x1b[0m"
        + head
        + "\x1b[K\n"
        + "\n".join("".join(row) + "\x1b[0m" for row in framed_cells)
        + "\x1b[0m"
    )


class Renderer:
    """Own frame state and configuration for independent render sessions."""

    def __init__(self, config: RenderConfig | None = None):
        self.config = config or RenderConfig()
        self.head = ""
        self.cells: list[list[str]] = []

    def render(self, atoms, **kwargs) -> str:
        """Render a slab view and update this renderer's frame."""
        return render(atoms, renderer=self, **kwargs)

    def render3d(self, atoms, **kwargs) -> str:
        """Render an unlit 3D view and update this renderer's frame."""
        return render3d(atoms, renderer=self, **kwargs)

    def render_lit(self, atoms, **kwargs) -> str:
        """Render shaded impostors and update this renderer's frame."""
        from .raster import render_lit

        return render_lit(atoms, renderer=self, **kwargs)


class DiffPrinter:
    """Write only cells changed since the previous draw."""

    def __init__(self, renderer: Renderer, stream=None):
        self.renderer = renderer
        self.stream = stream or sys.stdout
        self.head: str | None = None
        self.cells: list[list[str]] = []

    def draw(self) -> None:
        """Write changed cells from the associated renderer."""
        head, cells = self.renderer.head, self.renderer.cells
        out = []
        if head != self.head:
            out.append(f"\x1b[1;1H\x1b[0m{head}\x1b[K")
        for y, row in enumerate(cells):
            for x, cell in enumerate(row):
                if (
                    y >= len(self.cells)
                    or x >= len(self.cells[y])
                    or cell != self.cells[y][x]
                ):
                    out.append(f"\x1b[{y + 2};{x + 1}H\x1b[0m{cell}")
        if len(cells) != len(self.cells) or any(
            len(a) != len(b) for a, b in zip(cells, self.cells)
        ):
            # Clear stale content when the frame shrinks.
            if self.cells:
                out = [
                    "\x1b[2J\x1b[H\x1b[0m"
                    + head
                    + "\n"
                    + "\n".join("".join(row) + "\x1b[0m" for row in cells)
                ]
        if out:
            self.stream.write("".join(out) + "\x1b[0m")
            self.stream.flush()
        self.head, self.cells = head, [row[:] for row in cells]


def render(
    atoms,
    step=0,
    width=96,
    height=30,
    slab=2.2,
    sweep=0.02,
    note="",
    atom_scale=None,
    renderer=None,
) -> str:
    """Render an x–y slice of any ASE-compatible atomic structure."""
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
    config = renderer.config if renderer is not None else RenderConfig()
    for pos, number in zip(positions, atoms.numbers):
        x = min(width - 1, int(pos[0] / lengths[0] * width))
        y = min(height * 2 - 1, int((1 - pos[1] / lengths[1]) * height * 2))
        dz = abs((pos[2] - zplane + lengths[2] / 2) % lengths[2] - lengths[2] / 2)
        style = config.element(number)
        color = np.array(style.color, dtype=np.uint8)
        color = color if dz <= slab / 2 else (color * 0.28).astype(np.uint8)
        size = scale * style.slab_size
        x0, y0 = x - size // 2, y - size // 2
        xs, ys = (
            slice(max(0, x0), min(width, x0 + size)),
            slice(max(0, y0), min(height * 2, y0 + size)),
        )
        mask = pos[2] > depth[ys, xs]
        pixels[ys, xs][mask] = color
        depth[ys, xs][mask] = pos[2]
    return frame(
        _header(atoms, step, f"slab z {zplane:.2f} Å  {note}"),
        pack_half(pixels),
        renderer,
    )


def camera(atoms, yaw: float = 0, tilt: float = 18, view_dir=None) -> np.ndarray:
    """Construct an orthonormal camera basis for the selected direction."""
    if view_dir is not None:
        forward = np.asarray(view_dir, dtype=float)
        if forward.shape != (3,) or not np.any(forward):
            raise ValueError(
                "view direction must contain three nonzero-direction components"
            )
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, 0.0, 1.0])
        if abs(forward @ up) > 0.99:
            up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
    else:
        y, t = np.radians([yaw, tilt])
        forward = np.array([np.sin(y) * np.cos(t), -np.cos(y) * np.cos(t), np.sin(t)])
        right = np.array([np.cos(y), np.sin(y), 0.0])
        up = np.cross(forward, right)
    return np.stack((right, up, forward))


def box_corners(atoms) -> np.ndarray:
    """Return the eight corners of the structure's cell."""
    return np.array(list(itertools.product((0.0, 1.0), repeat=3))) @ np.asarray(
        atoms.cell
    )


def bonds_for(atoms, config: RenderConfig | None = None) -> list[tuple[int, int]]:
    """Estimate bonds from interatomic distances and configured radii."""
    config = config or RenderConfig()
    positions = atoms.positions
    i, j = np.triu_indices(len(atoms), 1)
    distances = np.linalg.norm(positions[i] - positions[j], axis=1)
    radii = np.array([config.element(n).radius for n in atoms.numbers])
    mask = (distances > 0.1) & (distances < 2.5 * (radii[i] + radii[j]))
    return list(zip(i[mask], j[mask]))


def render3d(
    atoms,
    yaw=0,
    tilt=18,
    step=0,
    width=96,
    height=30,
    ascii_only=False,
    note="",
    bonds=True,
    box=True,
    zoom=2.0,
    clip=0.0,
    hi=1.0,
    view_dir=None,
    ortho=False,
    braille=False,
    **kwargs,
) -> str:
    """Render an unlit perspective or orthographic structure."""
    from .raster import render_lit

    return render_lit(
        atoms,
        yaw=yaw,
        tilt=tilt,
        step=step,
        width=width,
        height=height,
        note=note,
        bonds=bonds,
        box=box,
        zoom=zoom,
        clip=clip,
        hi=hi,
        view_dir=view_dir,
        ortho=ortho,
        mode="braille" if braille else "half",
        light=False,
        ascii_only=ascii_only,
        **kwargs,
    )
