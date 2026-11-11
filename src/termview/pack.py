"""Pack RGB pixels into terminal half blocks, quadrants, or braille."""

import numpy as np

BITS = np.array([[1, 8], [2, 16], [4, 32], [64, 128]], dtype=np.uint16)
QUAD = np.array(list(" ▘▝▀▖▌▞▛▗▚▐▜▄▙▟█"))


def rgb(c: np.ndarray) -> str:
    """Format an RGB value for a 24-bit terminal escape."""
    return f"{int(c[0])};{int(c[1])};{int(c[2])}"


def pack_half(pixels: np.ndarray) -> list[list[str]]:
    """Pack a (2*height, width, 3) RGB image into terminal cells."""
    top, bottom = pixels[::2], pixels[1::2]
    return [
        [f"\x1b[38;2;{rgb(a)};48;2;{rgb(b)}m▀" for a, b in zip(ar, br)]
        for ar, br in zip(top, bottom)
    ]


def pack(pixels: np.ndarray, mode: str, ascii_only: bool = False) -> list[list[str]]:
    """Pack a pixel array; braille and quadrants use one colour per cell."""
    if mode == "half" and not ascii_only:
        return pack_half(pixels)
    if mode == "half":
        occupied = pixels.reshape(pixels.shape[0] // 2, 2, pixels.shape[1], 3).any(
            axis=(1, 3)
        )
        return [["#" if v else " " for v in row] for row in occupied]  # type: ignore[union-attr]
    if mode not in ("braille", "quad"):
        raise ValueError("invalid packing mode")
    h, w = pixels.shape[:2]
    rows = 4 if mode == "braille" else 2
    blocks = pixels.reshape(h // rows, rows, w // 2, 2, 3).transpose(0, 2, 1, 3, 4)
    dots = blocks.any(axis=-1)
    bits = BITS if mode == "braille" else np.array([[1, 2], [4, 8]])
    masks = (dots * bits).sum(axis=(2, 3))
    if ascii_only:
        return [["#" if mask else " " for mask in row] for row in masks]
    colors = blocks.reshape(h // rows, w // 2, rows * 2, 3)
    eq = (colors[:, :, :, None, :] == colors[:, :, None, :, :]).all(axis=-1)
    counts = (eq & dots.reshape(h // rows, w // 2, 1, rows * 2)).sum(axis=-1)
    counts = np.where(dots.reshape(h // rows, w // 2, rows * 2), counts, -1)
    choice = counts.argmax(axis=-1)
    selected = np.take_along_axis(colors, choice[:, :, None, None], axis=2)[:, :, 0, :]
    chars = np.vectorize(
        lambda m: chr(0x2800 + int(m)) if mode == "braille" else QUAD[int(m)]
    )(masks)
    return [
        [f"\x1b[38;2;{rgb(c)}m{ch}" if mask else " " for c, ch, mask in zip(cr, hr, mr)]
        for cr, hr, mr in zip(selected, chars, masks)
    ]
