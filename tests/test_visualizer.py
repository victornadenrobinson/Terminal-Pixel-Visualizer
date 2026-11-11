import contextlib
import io
import os
import select
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms

try:
    import pty
    import termios
except ImportError:
    pty = termios = None

from examples.toy_live import ToyPotential, build
from termview import ElementStyle, RenderConfig, Renderer
from termview import cli, pack, plot, render2d
from termview.controls import Controls


class RenderingTests(unittest.TestCase):
    def setUp(self):
        self.atoms = Atoms(
            "OH", positions=[[1, 1, 1], [1.9, 1, 1]], cell=[4, 4, 4], pbc=True
        )
        config = RenderConfig(
            elements={
                8: ElementStyle((255, 70, 60), 0.30, 2),
                1: ElementStyle((90, 230, 255), 0.17),
            }
        )
        self.renderer = Renderer(config)

    def test_slab_and_depth(self):
        atoms = Atoms("OH", positions=[[1, 1, 1], [1, 1, 2]], cell=[4, 4, 4])
        self.renderer.render(atoms, width=8, height=4, slab=0.5, sweep=0)
        cells = "".join("".join(row) for row in self.renderer.cells)
        self.assertIn("25;64;71", cells)
        self.assertIn("71;19;16", cells)
        self.assertIn("25;64;71", self.renderer.cells[4][3])

    def test_all_modes_share_last_and_diff(self):
        stream = io.StringIO()
        printer = render2d.DiffPrinter(self.renderer, stream)
        for mode in ("half", "quad", "braille"):
            text = self.renderer.render_lit(self.atoms, width=16, height=8, mode=mode)
            self.assertTrue(text.startswith("\x1b[H"))
            self.assertEqual(len(self.renderer.cells), 10)
            printer.draw()
            stream.seek(0)
            stream.truncate()
            printer.draw()
            self.assertEqual(stream.getvalue(), "")
        self.renderer.render3d(self.atoms, width=16, height=8, view_dir=[1, 1, 1])
        printer.draw()
        self.assertIn("\x1b[", stream.getvalue())

    def test_braille_bits_and_majority_color(self):
        image = np.zeros((4, 2, 3), dtype=np.uint8)
        image[0, 0] = image[1, 1] = [255, 0, 0]
        image[3, 0] = [0, 255, 0]
        cell = pack.pack(image, "braille")[0][0]
        self.assertIn("\x1b[38;2;255;0;0m", cell)
        self.assertTrue(cell.endswith(chr(0x2800 + 1 + 16 + 64)))

    def test_clip_keeps_front(self):
        atoms = Atoms("OH", positions=[[2, 1, 2], [2, 3, 2]], cell=[4, 4, 4])
        self.renderer.render_lit(
            atoms,
            width=20,
            height=10,
            view_dir=[0, 1, 0],
            ortho=True,
            box=False,
            bonds=False,
            hi=0.5,
            light=False,
            depth_fade=False,
        )
        cells = "".join("".join(row) for row in self.renderer.cells)
        self.assertNotIn("255;70;60", cells)
        self.assertIn("90;230;255", cells)

    def test_ascii_in_all_modes(self):
        for mode in ("half", "quad", "braille"):
            image = np.full(
                (4 if mode == "braille" else 2, 2 if mode != "half" else 1, 3),
                255,
                dtype=np.uint8,
            )
            self.assertEqual(pack.pack(image, mode, ascii_only=True), [["#"]])

    def test_independent_renderers_and_snapshot(self):
        other = Renderer()
        frame = self.renderer.render(self.atoms, width=8, height=4, step=1)
        other.render(self.atoms, width=5, height=3)
        self.assertNotEqual(self.renderer.cells, other.cells)
        golden = Path(__file__).parent / "golden" / "slab.ansi"
        self.assertEqual(frame, golden.read_text(encoding="utf-8"))

    def test_zbuffer_front_sphere_wins(self):
        atoms = Atoms("OH", positions=[[2, 1, 2], [2, 3, 2]], cell=[4, 4, 4])
        for indices in ([0, 1], [1, 0]):
            self.renderer.render_lit(
                atoms[indices],
                width=30,
                height=15,
                view_dir=[0, 1, 0],
                ortho=True,
                box=False,
                bonds=False,
                light=False,
                depth_fade=False,
            )
            cell = self.renderer.cells[8][16]
            self.assertIn("90;230;255", cell)

    def test_invalid_light_direction(self):
        renderer = Renderer(RenderConfig(light_direction=(0, 0, 0)))
        with self.assertRaisesRegex(ValueError, "light direction"):
            renderer.render_lit(self.atoms)


class PlotTests(unittest.TestCase):
    def test_canvas_and_ticks(self):
        canvas = plot.Canvas(1, 1)
        canvas.dots([0, 1, 0], [0, 1, 3])
        self.assertEqual(canvas.rows_text(False), [chr(0x2800 + 1 + 16 + 64)])
        np.testing.assert_array_equal(plot.nice(0, 10), [0, 5, 10])

    def test_categorical_table_and_auto_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "values.dataset")
            with open(path, "w", encoding="utf-8") as file:
                file.write("x,y,group\n0,1,A\n1,2,B\n")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                plot.main(
                    [
                        "scatter",
                        path,
                        "--header",
                        "--x",
                        "x",
                        "--y",
                        "y",
                        "--color",
                        "group",
                        "--width",
                        "30",
                        "--height",
                        "8",
                    ]
                )
            self.assertIn("\x1b[31m", output.getvalue())
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                cli.main([path, "--width", "30", "--height", "10"])
            self.assertIn("⠀", output.getvalue().replace(" ", "⠀"))

    def test_xyz_dataset_blank_comment_and_other_element(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "molecule.dataset")
            with open(path, "w", encoding="utf-8") as file:
                file.write("2\n\nC 0 0 0\nH 0 0 1\n")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                cli.main(
                    [path, "--width", "30", "--height", "10", "--color", "truecolor"]
                )
            self.assertIn("2 atoms", output.getvalue())
            self.assertIn("▀", output.getvalue())


class DynamicsTests(unittest.TestCase):
    def test_build_and_force_gradient(self):
        atoms = build("bcc", 1, 2.12)
        self.assertEqual(len(atoms), 6)
        atoms = Atoms(
            "OH", positions=[[1, 1, 1], [2.1, 1, 1]], cell=[10, 10, 10], pbc=True
        )
        atoms.calc = ToyPotential()
        force = atoms.get_forces()[1, 0]
        epsilon = 1e-5
        atoms.positions[1, 0] += epsilon
        plus = atoms.get_potential_energy()
        atoms.positions[1, 0] -= 2 * epsilon
        minus = atoms.get_potential_energy()
        self.assertAlmostEqual(force, -(plus - minus) / (2 * epsilon), places=5)

    def test_fcc_counts(self):
        self.assertEqual(len(build("fcc", 2, 2.12)), 96)


@unittest.skipIf(pty is None, "pseudo-terminals unavailable")
class ControlTests(unittest.TestCase):
    def test_mouse_keyboard_and_terminal_restore(self):
        master, slave = pty.openpty()
        try:
            before = termios.tcgetattr(slave)
            with (
                os.fdopen(os.dup(slave), "r", encoding="utf-8") as stdin,
                os.fdopen(os.dup(slave), "w", encoding="utf-8") as stdout,
            ):
                with Controls(stdin=stdin, stdout=stdout) as controls:
                    os.write(master, b"\x1b[C\x1b[<0;4;5M\x1b[<32;6;7M")
                    events = controls.poll()
                    self.assertIn(("arrow", "right"), events)
                    self.assertIn(("click", 4, 5), events)
                    self.assertIn(("rot", 2, 2), events)
                self.assertEqual(termios.tcgetattr(slave), before)
            output = b""
            while select.select([master], [], [], 0.1)[0]:
                output += os.read(master, 4096)
            self.assertIn(b"\x1b[?25h", output)
            self.assertIn(b"\x1b[?1049l", output)
        finally:
            os.close(master)
            os.close(slave)


class DatasetTests(unittest.TestCase):
    def test_all_example_datasets(self):
        directory = Path(__file__).parents[1] / "examples" / "datasets"
        files = sorted(directory.iterdir())
        self.assertGreaterEqual(len(files), 16)
        for path in files:
            with (
                self.subTest(path=path.name),
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                cli.main([str(path), "--width", "30", "--height", "10"])
                self.assertTrue(output.getvalue())

    def test_record_and_color_fallback(self):
        path = (
            Path(__file__).parents[1] / "examples" / "datasets" / "water-motion.extxyz"
        )
        with tempfile.TemporaryDirectory() as directory:
            cast = Path(directory) / "movie.cast"
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main(
                    [
                        str(path),
                        "--width",
                        "30",
                        "--height",
                        "10",
                        "--fps",
                        "1000",
                        "--record",
                        str(cast),
                        "--color",
                        "256",
                    ]
                )
            import json

            lines = [json.loads(line) for line in cast.read_text().splitlines()]
            self.assertEqual(lines[0]["version"], 2)
            self.assertEqual(len(lines), 5)
            self.assertIn("38;5;", lines[1][2])

    def test_record_cannot_overwrite_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.xyz"
            original = "1\n\nH 0 0 0\n"
            path.write_text(original)
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                cli.main([str(path), "--record", str(path)])
            self.assertEqual(path.read_text(), original)

    @unittest.skipIf(os.name == "nt", "POSIX signal test")
    def test_follow_new_frames(self):
        frame = "2\n\nO 0 0 0\nH 0 0 1\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "growing.xyz"
            path.write_text(frame)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "termview.cli",
                    str(path),
                    "--width",
                    "30",
                    "--height",
                    "10",
                    "--fps",
                    "1000",
                    "--follow",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                output = b""

                def wait_for_frames(count):
                    nonlocal output
                    deadline = time.monotonic() + 10
                    while (
                        output.count(b"2 atoms") < count and time.monotonic() < deadline
                    ):
                        if select.select([process.stdout], [], [], 0.1)[0]:
                            output += os.read(process.stdout.fileno(), 65536)
                    self.assertGreaterEqual(output.count(b"2 atoms"), count)

                wait_for_frames(1)
                with path.open("a") as source:
                    source.write(frame)
                wait_for_frames(2)
                process.send_signal(signal.SIGINT)
                remaining, error = process.communicate(timeout=10)
                output += remaining
                self.assertEqual(process.returncode, 0, error)
                self.assertGreaterEqual(output.count(b"2 atoms"), 2)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()


if __name__ == "__main__":
    unittest.main()
