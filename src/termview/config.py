"""Display configuration independent of the simulation or input format."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ElementStyle:
    """RGB colour, display radius in Å, and slab pixel width for an element."""

    color: tuple[int, int, int]
    radius: float
    slab_size: int = 1


def _elements() -> dict[int, ElementStyle]:
    return {
        1: ElementStyle((255, 255, 255), 0.17),
        6: ElementStyle((144, 144, 144), 0.35, 2),
        7: ElementStyle((48, 80, 248), 0.32, 2),
        8: ElementStyle((255, 13, 13), 0.30, 2),
        9: ElementStyle((144, 224, 80), 0.30, 2),
        15: ElementStyle((255, 128, 0), 0.40, 2),
        16: ElementStyle((255, 255, 48), 0.40, 2),
        17: ElementStyle((31, 240, 31), 0.40, 2),
    }


@dataclass
class RenderConfig:
    """Configurable element styles and lighting for a renderer."""

    elements: dict[int, ElementStyle] = field(default_factory=_elements)
    default_element: ElementStyle = field(
        default_factory=lambda: ElementStyle((180, 180, 180), 0.35, 2)
    )
    light_direction: tuple[float, float, float] = (-0.4, 0.6, 0.7)
    ambient: float = 0.22
    diffuse: float = 0.65
    specular: float = 125
    shininess: int = 24
    palette: tuple[int, ...] = (
        31,
        32,
        33,
        34,
        35,
        36,
        91,
        92,
        93,
        94,
        95,
        96,
        37,
        90,
        97,
        30,
    )

    def element(self, number: int) -> ElementStyle:
        """Return the configured style or the neutral fallback."""
        return self.elements.get(int(number), self.default_element)
