# Terminal-Pixel-Visualizer
Software tool for viewing graphs, trajectories, images, animations as a pixelated terminal output.

Goal:
termmol/                    # pick a name
├── pyproject.toml
├── README.md               # GIF/screenshot at the top
├── LICENSE                 # MIT or Apache-2.0
├── CHANGELOG.md
├── src/termmol/
│   ├── __init__.py
│   ├── raster.py           # sphere impostors, z-buffer
│   ├── pack.py             # half-block / quad / braille packing
│   ├── render2d.py         # slab view (from mdview.render)
│   ├── render3d.py
│   ├── plot.py             # from termplot
│   ├── controls.py         # from termctl
│   └── cli.py              # from toy_live: `termmol traj.extxyz --lit`
├── examples/
├── tests/
└── .github/workflows/ci.yml

To do:
--Example images.
