"""
ArrowETC Demonstration Script

This module provides a visual demonstration of the `ArrowETC` class for generating
custom, stylized arrows with segmented or Bezier-shaped shafts. Each example produces
a PNG image that highlights a different geometric or styling feature of ArrowETC.

Examples included
-----------------
- Basic arrow with a straight shaft and arrowhead
- Multi-segmented arrows with right angles or sharp bends
- Arrows with obtuse or acute interior angles
- Shaft-only arrows (no arrowhead) for bracket or flow-line usage
- Curved arrows using smooth Bezier paths with variable control points
- Complex compound arrows with many segments and dual arrowheads

Key Features Demonstrated
-------------------------
- Shaft width and arrowhead sizing control
- Edge (`ec`) and face (`fc`) color styling
- Bezier vs. straight-line interpolation
- Support for acute/obtuse angles with proper miter handling
- High-DPI rendering with consistent arrow geometry

Output
------
Each example is saved as a PNG file in the `resources/arrow_examples/` directory,
with filenames indicating the arrow style used.

Notes
-----
The x and y limits of the axes need to be set before constructing your arrows. Otherwise, they
may look skewed when plotting. This is a by-product of the way arrow vertices are determined
using the matplotlib.axes.Axes aspect ratio.
"""

from pathlib import Path
import os
import sys

from matplotlib.pyplot import close, rcParams, subplots

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from logictree import ArrowETC  # noqa: E402


def main():
    # set style presets
    font_defaults = {
        "axes.facecolor": "black",
        "figure.facecolor": "black",
    }
    rcParams.update(font_defaults)

    base_path = Path(__file__).resolve().parent.parent / "resources/arrow_examples"

    # basic arrow with head
    _, ax = subplots()
    path = [(0, 0), (0, 4)]
    ax.set_xlim(min([p[0] for p in path]) - 1, max([p[0] for p in path]) + 1)
    ax.set_ylim(min([p[1] for p in path]) - 1, max([p[1] for p in path]) + 1)
    arrow = ArrowETC(ax, path, shaft_width=20, arrow_head=True, ec="white", fc="cyan")
    arrow.save_arrow(base_path / "basic_arrow_with_head.png")
    close()

    # multi segmented arrows
    _, ax = subplots()
    path = [(0, 0), (0, 4), (5, 4), (5, 0)]
    ax.set_xlim(min([p[0] for p in path]) - 1, max([p[0] for p in path]) + 1)
    ax.set_ylim(min([p[1] for p in path]) - 1, max([p[1] for p in path]) + 1)
    arrow = ArrowETC(
        ax, path, shaft_width=20, arrow_head=True, ec="white", fc="magenta", lw=2
    )
    arrow.save_arrow(base_path / "multi_segment_arrow_with_head.png")
    close()

    # obtuse angles
    _, ax = subplots()
    path = [(0, 0), (4, 0), (8, 2)]
    ax.set_xlim(min([p[0] for p in path]) - 1, max([p[0] for p in path]) + 1)
    ax.set_ylim(min([p[1] for p in path]) - 1, max([p[1] for p in path]) + 1)
    arrow = ArrowETC(ax, path, shaft_width=20, arrow_head=True, ec="white", fc="orange")
    arrow.save_arrow(base_path / "obtuse_arrow_with_head.png")
    close()

    # acute angles
    _, ax = subplots()
    path = [(0, 0), (4, 0), (1, 4)]
    ax.set_xlim(min([p[0] for p in path]) - 1, max([p[0] for p in path]) + 1)
    ax.set_ylim(min([p[1] for p in path]) - 1, max([p[1] for p in path]) + 1)
    arrow = ArrowETC(ax, path, shaft_width=20, arrow_head=True, ec="white", fc="cyan")
    arrow.save_arrow(base_path / "acute_arrow_with_head.png")
    close()

    # basic segments without head
    _, ax = subplots()
    path = [(0, 0), (0, -10), (10, -10), (10, 0)]
    ax.set_xlim(min([p[0] for p in path]) - 1, max([p[0] for p in path]) + 1)
    ax.set_ylim(min([p[1] for p in path]) - 1, max([p[1] for p in path]) + 1)
    arrow = ArrowETC(ax, path, shaft_width=20, arrow_head=False, ec="white", fc="cyan")
    arrow.save_arrow(base_path / "multi_segment_no_head.png")
    close()

    # basic bezier
    _, ax = subplots()
    path = [(0, 0), (4, 0), (8, 2)]
    ax.set_xlim(min([p[0] for p in path]) - 1, max([p[0] for p in path]) + 1)
    ax.set_ylim(min([p[1] for p in path]) - 1, max([p[1] for p in path]) + 1)
    arrow = ArrowETC(
        ax, path, shaft_width=20, arrow_head=True, bezier=True, ec="white", fc="orange"
    )
    arrow.save_arrow(base_path / "basic_bezier_with_head.png")
    close()

    # crazier bezier
    _, ax = subplots()
    path = [(0, 0), (4, -5), (8, 2), (14, -8)]
    ax.set_xlim(min([p[0] for p in path]) - 1, max([p[0] for p in path]) + 1)
    ax.set_ylim(min([p[1] for p in path]) - 1, max([p[1] for p in path]) + 3)
    arrow = ArrowETC(
        ax,
        path,
        shaft_width=20,
        arrow_head=True,
        bezier=True,
        bezier_n=800,
        ec="white",
        fc="cyan",
    )
    arrow.save_arrow(base_path / "crazier_bezier_with_head.png")
    close()

    # many segments
    _, ax = subplots()
    path = [(0, 0), (1, 2), (2, -1), (4, -2), (5, 0), (7, 0)]
    ax.set_xlim(min([p[0] for p in path]) - 1, max([p[0] for p in path]) + 1)
    ax.set_ylim(min([p[1] for p in path]) - 1, max([p[1] for p in path]) + 1)
    arrow = ArrowETC(
        ax,
        path,
        arrow_head=True,
        arrow_head_at_tail=True,
        shaft_width=20,
        ec="white",
        fc="cyan",
        arrow_head_length_multiplier=2,
        arrow_head_width_multiplier=8 / 3,
    )
    arrow.save_arrow(base_path / "many_segments_with_head.png")
    close()


if __name__ == "__main__":
    main()
