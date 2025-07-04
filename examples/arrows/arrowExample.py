"""
Example ArrowETC Arrows: Multi-Segment Arrow Shapes

This example demonstrates the ArrowETC class by generating individual PNGs of arrows
with different paths, including single-segment, multi-segment, straight, and jointed arrows.
Each arrow is saved as a separate PNG file, showcasing the flexibility of ArrowETC for
creating arrows with arbitrary segments and orientations.

It shows:
- Creating straight arrows pointing in different directions (up, left, diagonal).
- Building arrows with one or multiple right-angle joints.
- Saving arrow shapes as PNGs with clear visualization of the arrow shaft and head.
- Using ArrowETC with minimal setup to generate complex arrows.
- Creating segmented rectangles (arrows with no head).

This script produces six PNG images illustrating different arrow styles.

Usage:
------
Run this script directly. It will save:
- up_arrow.png
- left_arrow.png
- right_down_arrow.png
- single_joint_arrow.png
- multi_joint_arrow-acute.png
- multi_joint_arrow-obtuse.png
- multi_joint_rect.png
in the same directory as this script.

Dependencies:
-------------
- Python packages: matplotlib, numpy, your installed logictree package.
"""
from pathlib import Path
import sys
import os

# Compute absolute path to the parent directory of examples/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from logictree.ArrowETC import ArrowETC

def main():
    arrow = ArrowETC([(0, 0), (0, 10)], 2, True)
    arrow.save_arrow(name=Path(__file__).resolve().parent / "up_arrow.png")

    arrow = ArrowETC([(0, 0), (-10, 0)], 2, True)
    arrow.save_arrow(name=Path(__file__).resolve().parent / "left_arrow.png")

    arrow = ArrowETC([(0, 0), (10, -8)], 2, True)
    arrow.save_arrow(name=Path(__file__).resolve().parent / "right_down_arrow.png")

    arrow = ArrowETC([(0, 0), (-10, 0), (-10, -10)], 2, True)
    arrow.save_arrow(name=Path(__file__).resolve().parent / "single_joint_arrow.png")

    arrow = ArrowETC([(0, 0), (-10, 0), (-10, -10), (-2, -5)], 2, True)
    arrow.save_arrow(name=Path(__file__).resolve().parent / "multi_joint_arrow-acute.png")

    arrow = ArrowETC([(0, 0), (-10, 0), (-10, -10), (-2, -15)], 2, True)
    arrow.save_arrow(name=Path(__file__).resolve().parent / "multi_joint_arrow-obtuse.png")

    arrow = ArrowETC([(0, 0), (-10, 0), (-10, -10), (-2, -15)], 2, False)
    arrow.save_arrow(name=Path(__file__).resolve().parent / "multi_joint_rect.png")


if __name__ == "__main__":
    main()
