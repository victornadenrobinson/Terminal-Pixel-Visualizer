#!/usr/bin/env python3
"""Live toy molecular dynamics in a 24-bit-color terminal; not for quantitative science."""

import argparse
import time

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.neighborlist import neighbor_list

from termview import DiffPrinter, Renderer
from termview.controls import Controls
from termview.render2d import camera


class ToyPotential(Calculator):
    """Short-range pair potential for qualitative structural motion."""

    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        i, j, d = neighbor_list("ijD", self.atoms, 3.0)
        r = np.maximum(np.linalg.norm(d, axis=1), 0.1)
        ni, nj = self.atoms.numbers[i], self.atoms.numbers[j]
        oo, oh = (ni == 8) & (nj == 8), ni != nj
        sigma = np.where(oo, 2.2, np.where(oh, 0.9, 0.7))
        sr6 = (sigma / r) ** 6
        energy = 0.04 * (sr6 * sr6 - sr6)
        derivative = 0.24 * (sr6 - 2 * sr6 * sr6) / r
        exp = np.exp(np.clip(-2 * (r - 0.98), -30, 30))
        energy += np.where(oh, 0.3 * ((1 - exp) ** 2 - 1), 0)
        derivative += np.where(oh, 1.2 * (1 - exp) * exp, 0)
        # Smoothly turn all pair interactions off at the cutoff.
        switch = np.clip((3.0 - r) / 0.5, 0, 1)
        ds = np.where(r > 2.5, -2.0, 0.0)
        derivative = derivative * switch + energy * ds
        energy *= switch
        pair_force = derivative[:, None] * d / r[:, None]
        forces = np.zeros((len(self.atoms), 3))
        np.add.at(forces, i, pair_force)
        virial = np.einsum("ni,nj->ij", d, pair_force) / 2
        self.results = {
            "energy": energy.sum() / 2,
            "forces": forces,
            "stress": np.array(
                [
                    virial[0, 0],
                    virial[1, 1],
                    virial[2, 2],
                    virial[1, 2],
                    virial[0, 2],
                    virial[0, 1],
                ]
            )
            / self.atoms.get_volume(),
        }


def build(phase, nrep, vpa, seed=1):
    bases = {
        "bcc": [[0, 0, 0], [0.5, 0.5, 0.5]],
        "fcc": [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    }
    if phase not in bases or nrep < 1 or vpa <= 0:
        raise ValueError("phase must be bcc or fcc, nrep and vpa must be positive")
    rng = np.random.default_rng(seed)
    n_o = len(bases[phase]) * nrep**3
    side = (3 * n_o * vpa) ** (1 / 3)
    positions = []
    for cell in np.ndindex(nrep, nrep, nrep):
        for base in bases[phase]:
            oxygen = (np.array(cell) + base) * side / nrep
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            other = rng.normal(size=3)
            other -= (other @ direction) * direction
            other /= np.linalg.norm(other)
            positions.extend(
                (
                    oxygen,
                    oxygen + 0.97 * direction,
                    oxygen + 0.97 * (-0.25 * direction + np.sqrt(1 - 0.25**2) * other),
                )
            )
    atoms = Atoms("OH2" * n_o, positions=positions, cell=[side] * 3, pbc=True)
    atoms.wrap()
    return atoms


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("bcc", "fcc"), default="bcc")
    p.add_argument("--nrep", type=int, default=3)
    p.add_argument("--vpa", type=float, default=2.12)
    p.add_argument("--T", type=float, default=5000.0)
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--every", type=int, default=2)
    p.add_argument("--slab", type=float, default=2.2)
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--height", type=int, default=30)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--view3d", action="store_true")
    p.add_argument("--spin", type=float, default=0.0)
    p.add_argument("--tilt", type=float, default=18.0)
    p.add_argument("--lit", action="store_true")
    p.add_argument("--lit-mode", choices=("half", "quad", "braille"), default="half")
    p.add_argument("--braille", action="store_true")
    p.add_argument("--delta", action="store_true")
    p.add_argument("--atom-scale", type=int, default=0)
    p.add_argument(
        "--view", default="", help="look along a direction such as 001, 101 or 111"
    )
    p.add_argument("--ortho", action="store_true")
    p.add_argument("--zoom", type=float, default=1.0)
    p.add_argument("--clip", type=float, default=0.0)
    p.add_argument("--hi", type=float, default=1.0)
    p.add_argument("--nobonds", action="store_true")
    p.add_argument("--nobox", action="store_true")
    p.add_argument("--ascii", action="store_true")
    p.add_argument("--no-mouse", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    return p


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    if (
        a.nrep < 1
        or a.vpa <= 0
        or a.T < 0
        or a.fps <= 0
        or a.every < 1
        or a.width < 1
        or a.height < 1
        or a.steps < 0
        or a.slab < 0
        or a.atom_scale < 0
        or a.zoom <= 0
        or not 0 <= a.clip < a.hi <= 1
    ):
        p.error("invalid simulation or display parameter")
    if a.view and (
        len(a.view) != 3
        or not a.view.isascii()
        or not a.view.isdigit()
        or a.view == "000"
    ):
        p.error("--view must be a nonzero three-digit direction such as 001")
    view = [int(c) for c in a.view] if a.view else None
    atoms = build(a.phase, a.nrep, a.vpa, a.seed)
    atoms.calc = ToyPotential()
    rng = np.random.default_rng(a.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=a.T, rng=rng)
    Stationary(atoms)
    dyn = Langevin(
        atoms, 1 * units.fs, temperature_K=a.T, friction=1 / (100 * units.fs), rng=rng
    )
    renderer = Renderer()
    dp = DiffPrinter(renderer)
    yaw, tilt, zoom, spin = 0.0, a.tilt, 1.0, a.spin
    paused = False
    fancy = bool(a.lit or a.view3d or a.view)
    try:
        with Controls(mouse=not a.no_mouse) as ctl:
            quit_ = False
            while not quit_ and (not a.steps or dyn.nsteps < a.steps):
                start = time.monotonic()
                for event in ctl.poll():
                    if event[0] == "rot":
                        yaw += event[1] * 2.5
                        tilt = np.clip(tilt - event[2] * 3, -89, 89)
                    elif event[0] == "arrow":
                        yaw += {"left": -6, "right": 6}.get(event[1], 0)
                        tilt = np.clip(
                            tilt + {"up": 6, "down": -6}.get(event[1], 0), -89, 89
                        )
                    elif event[0] == "zoom":
                        zoom = np.clip(
                            zoom * (1.12 if event[1] > 0 else 1 / 1.12), 0.4, 3
                        )
                    elif event[0] == "key":
                        k = event[1]
                        if k in ("q", "\x03"):
                            quit_ = True
                        elif k == "s":
                            spin = 0.0 if spin else 1.5
                        elif k == " ":
                            paused = not paused
                        elif k in ("+", "="):
                            zoom = min(3.0, zoom * 1.12)
                        elif k in ("-", "_"):
                            zoom = max(0.4, zoom / 1.12)
                        elif k == "r":
                            yaw, tilt, zoom = 0.0, a.tilt, 1.0
                if quit_:
                    break
                if not paused:
                    dyn.run(min(a.every, a.steps - dyn.nsteps) if a.steps else a.every)
                yaw += spin
                atoms.wrap()
                atoms.info["T_inst"] = atoms.get_temperature()
                atoms.info["P"] = (
                    -np.mean(atoms.get_stress(include_ideal_gas=True)[:3]) / units.GPa
                )
                note = f"{a.phase}  target {a.T:.0f} K  P in GPa"
                if fancy:
                    direction = view
                    if view is not None:
                        right, up, forward = camera(atoms, view_dir=view)
                        y, t = np.radians([yaw, tilt - a.tilt])
                        direction = (
                            np.cos(t) * (np.cos(y) * forward + np.sin(y) * right)
                            + np.sin(t) * up
                        )
                    kw = dict(
                        yaw=yaw,
                        tilt=tilt,
                        step=dyn.nsteps,
                        width=a.width,
                        height=a.height,
                        note=note,
                        bonds=not a.nobonds,
                        box=not a.nobox,
                        zoom=a.zoom * zoom,
                        clip=a.clip,
                        hi=a.hi,
                        view_dir=direction,
                        ortho=a.ortho or bool(view),
                    )
                    text = (
                        renderer.render_lit(
                            atoms, mode=a.lit_mode, ascii_only=a.ascii, **kw
                        )
                        if a.lit
                        else renderer.render3d(
                            atoms, ascii_only=a.ascii, braille=a.braille, **kw
                        )
                    )
                else:
                    text = renderer.render(
                        atoms,
                        step=dyn.nsteps,
                        width=a.width,
                        height=a.height,
                        slab=a.slab,
                        note=note,
                        atom_scale=a.atom_scale or None,
                    )
                if a.delta:
                    dp.draw()
                else:
                    print(text, flush=True)
                time.sleep(max(0.0, 1 / a.fps - (time.monotonic() - start)))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
