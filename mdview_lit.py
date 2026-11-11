"""Vectorized sphere and cylinder impostors for terminal molecular views."""
import numpy as np

from mdview import (LAST, O_COLOR, H_COLOR, _header, _rgb, bonds_for,
                    box_corners, camera, frame, pack_half)

_BITS = np.array([[1, 8], [2, 16], [4, 32], [64, 128]], dtype=np.uint16)
_QUAD = np.array(list(" ▘▝▀▖▌▞▛▗▚▐▜▄▙▟█"))
_LIGHT = np.array([-0.4, 0.6, 0.7])
_LIGHT /= np.linalg.norm(_LIGHT)


def _pack(pixels, mode, ascii_only=False):
    if mode == "half" and not ascii_only:
        return pack_half(pixels)
    if mode == "half":
        occupied = pixels.reshape(pixels.shape[0] // 2, 2, pixels.shape[1], 3).any(axis=(1, 3))
        return [["#" if v else " " for v in row] for row in occupied]
    h, w = pixels.shape[:2]
    rows = 4 if mode == "braille" else 2
    blocks = pixels.reshape(h // rows, rows, w // 2, 2, 3).transpose(0, 2, 1, 3, 4)
    dots = blocks.any(axis=-1)
    bits = _BITS if mode == "braille" else np.array([[1, 2], [4, 8]])
    masks = (dots * bits).sum(axis=(2, 3))
    colors = blocks.reshape(h // rows, w // 2, rows * 2, 3)
    # Choose the most frequent non-black RGB value in each cell.
    eq = (colors[:, :, :, None, :] == colors[:, :, None, :, :]).all(axis=-1)
    counts = (eq & dots.reshape(h // rows, w // 2, 1, rows * 2)).sum(axis=-1)
    counts = np.where(dots.reshape(h // rows, w // 2, rows * 2), counts, -1)
    choice = counts.argmax(axis=-1)
    selected = np.take_along_axis(colors, choice[:, :, None, None], axis=2)[:, :, 0, :]
    chars = np.vectorize(lambda m: chr(0x2800 + int(m)) if mode == "braille" else _QUAD[int(m)])(masks)
    return [[f"\x1b[38;2;{_rgb(c)}m{ch}" if mask else " "
             for c, ch, mask in zip(cr, hr, mr)] for cr, hr, mr in zip(selected, chars, masks)]


def render_lit(atoms, yaw=0, tilt=18, step=0, width=96, height=30, note="",
               bonds=True, box=True, clip=0.0, hi=1.0, view_dir=None,
               ortho=False, zoom=1.0, mode="half", show_O=True, show_H=True,
               light=True, depth_fade=True, ascii_only=False):
    if width < 1 or height < 1 or zoom <= 0 or mode not in ("half", "quad", "braille"):
        raise ValueError("invalid dimensions, zoom, or packing mode")
    if not 0 <= clip < 1 or not 0 < hi <= 1:
        raise ValueError("clip must be in [0,1) and hi in (0,1]")
    if np.any(atoms.cell.lengths() <= 0):
        raise ValueError("atoms must have a nonzero cell")
    px, py = (1, 2) if mode == "half" else (2, 4 if mode == "braille" else 2)
    w, h = width * px, height * py
    image = np.zeros((h, w, 3), dtype=np.uint8)
    zbuf = np.full((h, w), -np.inf)
    basis = camera(atoms, yaw, tilt, view_dir)
    corners = box_corners(atoms)
    center = np.asarray(atoms.cell).sum(axis=0) / 2
    points = (atoms.positions - center) @ basis.T
    box_points = (corners - center) @ basis.T
    extent = max(np.linalg.norm(corners - center, axis=1).max(), 1e-6)
    scale = min(w, h) * .45 * zoom / extent
    distance = 4 * extent

    def project(p):
        factor = np.ones(p.shape[:-1]) if ortho else distance / np.maximum(distance - p[..., 2], extent * .2)
        return np.stack((w / 2 + p[..., 0] * scale * factor,
                         h / 2 - p[..., 1] * scale * factor), axis=-1), factor

    screen, factors = project(points)
    box_screen, _ = project(box_points)
    zmin, zmax = box_points[:, 2].min(), box_points[:, 2].max()
    span = max(zmax - zmin, 1e-9)
    low, high = zmin + clip * span, zmin + hi * span
    visible = (points[:, 2] >= low) & (points[:, 2] <= high)
    visible &= ((atoms.numbers == 8) & show_O) | ((atoms.numbers == 1) & show_H)

    def raster(bounds, compute, color):
        x0, x1, y0, y1 = bounds
        x0, x1 = max(0, int(np.floor(x0))), min(w, int(np.ceil(x1)) + 1)
        y0, y1 = max(0, int(np.floor(y0))), min(h, int(np.ceil(y1)) + 1)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        mask, z, normal = compute(xx + .5, yy + .5)
        mask &= z > zbuf[y0:y1, x0:x1]
        if not np.any(mask):
            return
        if light:
            diffuse = np.maximum(normal @ _LIGHT, 0)
            halfvec = _LIGHT + np.array([0., 0., 1.])
            halfvec /= np.linalg.norm(halfvec)
            spec = np.maximum(normal @ halfvec, 0) ** 24
            intensity = .22 + .65 * diffuse
            rgb = np.clip(intensity[..., None] * color + 125 * spec[..., None], 0, 255)
        else:
            rgb = np.broadcast_to(color, (*mask.shape, 3)).copy()
        if depth_fade:
            rgb = rgb * np.clip(.55 + .45 * (z - zmin) / span, .4, 1)[..., None]
        image[y0:y1, x0:x1][mask] = rgb[mask].astype(np.uint8)
        zbuf[y0:y1, x0:x1][mask] = z[mask]

    def cylinder(a, b, za, zb, radius, color):
        v = b - a
        length2 = max(v @ v, 1e-12)
        def compute(x, y):
            t = np.clip(((x - a[0]) * v[0] + (y - a[1]) * v[1]) / length2, 0, 1)
            dx, dy = (x - (a[0] + t * v[0])) / radius, -(y - (a[1] + t * v[1])) / radius
            d2 = dx * dx + dy * dy
            nz = np.sqrt(np.maximum(0, 1 - d2))
            return d2 <= 1, za + t * (zb - za) + nz * radius / scale, np.stack((dx, dy, nz), axis=-1)
        raster((min(a[0], b[0]) - radius, max(a[0], b[0]) + radius,
                min(a[1], b[1]) - radius, max(a[1], b[1]) + radius), compute, color)

    if box:
        for i in range(8):
            for bit in (1, 2, 4):
                j = i ^ bit
                if i < j:
                    cylinder(box_screen[i], box_screen[j], box_points[i, 2], box_points[j, 2],
                             max(.5, px * .35), np.array([100, 110, 120]))
    if bonds:
        for i, j in bonds_for(atoms):
            if visible[i] and visible[j]:
                cylinder(screen[i], screen[j], points[i, 2], points[j, 2],
                         max(.6, scale * .075 * (factors[i] + factors[j]) / 2),
                         np.array([160, 165, 165]))
    for i in np.flatnonzero(visible):
        radius_a = .30 if atoms.numbers[i] == 8 else .17
        radius = max(.65, radius_a * scale * factors[i])
        cx, cy = screen[i]
        def sphere(x, y):
            dx, dy = (x - cx) / radius, -(y - cy) / radius
            d2 = dx * dx + dy * dy
            nz = np.sqrt(np.maximum(0, 1 - d2))
            return d2 <= 1, points[i, 2] + radius_a * nz, np.stack((dx, dy, nz), axis=-1)
        raster((cx - radius, cx + radius, cy - radius, cy + radius), sphere,
               O_COLOR if atoms.numbers[i] == 8 else H_COLOR)
    return frame(_header(atoms, step, note), _pack(image, mode, ascii_only))
