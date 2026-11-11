# Terminal Pixel Visualizer

View molecular structures, trajectories and numeric data over SSH in a
24-bit-colour terminal (100×40 or larger recommended). The application uses
NumPy and ASE; pip installs ASE's own dependencies as needed.

```sh
python -m pip install .
termview XYZ.dataset
```

`termview` detects XYZ by its contents, including files with arbitrary
extensions, and displays a lit molecular view. ASE `.traj`, `.json`, `.cif`
and `.pdb` files are also supported. Multi-frame structures play in a terminal;
redirected output contains the first frame. CSV and whitespace-separated
numeric tables are shown as a scatter plot (or histogram for one column);
the first row may contain column names. Use `--width` and `--height` to
override the terminal size. Unsupported or malformed data produces an error.

## Live toy MD

```sh
toy-live --phase bcc --nrep 3 --T 3000 --vpa 2.4 --fps 12
toy-live --lit --lit-mode braille --view 111 --ortho
toy-live --view3d --spin 1 --delta
```

The default view is an x–y slab sweeping through the periodic box. `--view3d`
enables the unlit 3D view and `--lit` enables shaded spheres and bonds.
`--view 001`, `101` or `111` selects a crystallographic direction; `--clip`
removes the back fraction, and `--hi` keeps the front fraction. See
`toy-live --help` for the other controls. Drag or use arrow keys to rotate,
wheel or +/- to zoom, `s` to toggle spin, space to pause, `r` to reset and
`q` or Ctrl-C to quit. `--no-mouse` disables mouse reporting. The terminal
state is restored on exit.

This is a qualitative toy pair potential for observing proton motion and
disorder, **not** a model for quantitative results.

## Plots

```sh
termplot scatter data.csv --header --x time --y energy --color group
termplot hist data.txt --x 0 --centre --bins 20
termplot report data.csv --header --x time --y energy --width 100 --height 40
```

Input may be comma- or whitespace-separated; columns are zero-based unless
`--header` supplies names. `--clip LO HI` filters the x column. For stdin use
`-` as the filename. Plot colors are grouped into a 16-colour terminal palette.

The lit renderer supports `half`, `quad` and `braille` packing. Braille has
only **one colour per cell**, chosen as the most common colour among its lit
dots, so neighbouring differently coloured atoms can lose colour detail.
Quadrants also share one foreground colour. A terminal font with the relevant
block and braille glyphs is required.

## Python API and tests

`mdview.render`, `mdview.render3d` and `mdview_lit.render_lit` return ANSI
frames and update the shared `mdview.LAST`; `mdview.DiffPrinter.draw()`
prints only changed cells. `termplot.Canvas` supplies dot, line and text
packing primitives. Run focused tests with `python -m unittest discover -s tests`.
