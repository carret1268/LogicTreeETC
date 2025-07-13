# LogicTreeETC

[![Documentation Status](https://readthedocs.org/projects/logictreeetc/badge/?version=latest)](https://logictreeetc.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/logictreeetc.svg)](https://pypi.org/project/logictreeetc/)
[![codecov](https://codecov.io/gh/carret1268/LogicTreeETC/branch/main/graph/badge.svg)](https://codecov.io/gh/carret1268/LogicTreeETC)
[![CI](https://github.com/carret1268/LogicTreeETC/actions/workflows/ci.yml/badge.svg)](https://github.com/carret1268/LogicTreeETC/actions/workflows/ci.yml)

**Flexible, publication-ready logic trees, arrow connectors, and annotated diagrams - all in Python.**

---

## Overview

`LogicTreeETC` is a Python package built on `matplotlib` for creating structured visual diagrams using logic boxes, stylized multi-segment arrows, and feature-detected anchors from underlying images via OpenCV. It enables clean, programmatic layouts for flowcharts, decision trees, image annotations, and more.

The package includes:

- `LogicTree`: A canvas manager for diagram elements and titles  
- `ArrowETC`: A highly customizable arrow-drawing engine with explicit path control  
- `VectorDetector`: A utility for detecting and labeling image vertices using OpenCV  

Together, these tools help you build annotated, reusable diagrams for scientific or analytical workflows.

---

## Installation

Install from PyPI:

```bash
pip install logictreeetc
```

Upgrade to the latest version:

```bash
pip install --upgrade logictreeetc
```

Then import the tools:

```python
from logictree import ArrowETC, LogicTree, VectorDetector
```

---

## Why Use LogicTreeETC?

Matplotlib's default arrows are inflexible and opaque. LogicTreeETC is designed for users who need:

- Explicit vertex control of arrows  
- Clean logic box layouts with flexible styling  
- Integration with images for analytical or anatomical diagrams  
- Full access to metadata (segment angles, vertices, offsets) for debugging or alignment  

### Highlights

- **Precise geometry control**  
  Define the explicit path of your arrow in data coordinates, whether straight, segmented, or curved. Access arrow metadata (e.g., the coordinates of every vertex, the angles each line segment makes with the positive x-axis, etc) without having to manually convert between pixel and data spaces like when using matplotlib's `FancyArrow` and `FancyArrowPatch`.

- **Seamless box-arrow integration**  
  Attach arrows to any edge or corner of a `LogicBox` using `sideA`, `sideB`, and pixel offsets.

- **Bezier curves and elbow routing**  
  Use preset or custom Bezier styles for natural curves, or route segmented arrows around obstacles and between misaligned elements.

- **Built-in image feature detection**  
  Use `VectorDetector` to automatically locate keypoints in diagrams and images, then label and link them programmatically.

- **Pixel-perfect arrow widths**  
  All arrow geometry is computed in pixel space, ensuring consistent widths regardless of axis scale or skew.

- **LaTeX + publication-ready styles**  
  Full support for LaTeX typesetting, true-to-theme dark mode, and high-DPI figure export without extra configuration.

- **Modular and extensible**  
  Each component (logic tree, arrows, feature detection) is usable independently or together - no lock-in or boilerplate. 


---

## Coordinate Scaling Notice

The `ArrowETC` class requires the final aspect ratio of your `matplotlib.axes.Axes` object to compute arrow geometry correctly. Always call `set_xlim()` and `set_ylim()` before drawing arrows. Otherwise, the rendered arrows may appear skewed.

---

## Examples

### Decision Tree for Analytical Filtering

A logic tree representing filtering steps in a dummy non-targeted analysis dataset.

<div align="center">
    <img src="https://raw.githubusercontent.com/carret1268/LogicTreeETC/main/resources/logictree_examples/DecisionTree_NTA-Example.png" width="500"/>
</div>

Code: `examples/decision_tree-NTA-Example.py`

Features:

- Custom box styling and LaTeX text  
- Straight and bifurcating arrows with labeled branches  
- Optional text rotation  

---

### Annotated Nephron Diagram with Auto-Detected Features

This example combines a background image, automatic feature detection, and curved arrows.

<div align="center">
    <img src="https://raw.githubusercontent.com/carret1268/LogicTreeETC/main/resources/logictree_examples/anatomical_diagram-nephron-Example.png" width="500"/>
</div>

Code: `examples/anatomical_diagram-nephron-Example.py`

Features:

- Automatic vertex detection via `VectorDetector`

<div align="center">
    <img src="https://raw.githubusercontent.com/carret1268/LogicTreeETC/main/resources/vector_detector_results_from_nephron_example/nephron-verts_auto_detected-Example.png" width="400"/>
</div>

- User-labeled vertices for later access

<div align="center">
    <img src="https://raw.githubusercontent.com/carret1268/LogicTreeETC/main/resources/vector_detector_results_from_nephron_example/nephron-labeled_verts_auto_detected-Example.png" width="400"/>
</div>

- Curved arrows with fine-tuned styling and proportional arrowheads

---

### Normalized Blackbody Spectrum

Demonstrates the normalized blackbody spectrum at 5 temperatures, highlighting Wien's law and the ultraviolet catastrophe.

<div align="center">
    <img src="https://raw.githubusercontent.com/carret1268/LogicTreeETC/main/resources/logictree_examples/normalized_blackbody_spectrum-Example.png" width="600"/>
</div>

Features:

- Compatible with scientific plots  
- Arrow rendering is robust to skewed aspect ratios  
- Optional heads at both ends of arrows  
- Preset styles like `colormode="dark"` save time  

---

### Pedagogy: Showing the Product Rule

Illustrates term-by-term application of the product rule in differentiation.

<div align="center">
    <img src="https://raw.githubusercontent.com/carret1268/LogicTreeETC/main/resources/logictree_examples/information_flow-Calculus-Example.png" width="600"/>
</div>

Code: `examples/information_flow-calculus-Example.py`

Features:

- Explicit font color control  
- Multi-segment arrows with arbitrary angles  

---

### Custom Arrows with ArrowETC

See `examples/example_arrows.py` for all examples below.

#### Basic arrow with head

<div align="center">
  <img src="https://raw.githubusercontent.com/carret1268/LogicTreeETC/main/resources/arrow_examples/basic_arrow_with_head.png" width="400"/>
</div>

#### Multi-segment arrow with head

<div align="center">
  <img src="https://raw.githubusercontent.com/carret1268/LogicTreeETC/main/resources/arrow_examples/multi_segment_arrow_with_head.png" width="400"/>
</div>

#### Obtuse angle arrow

<div align="center">
  <img src="https://raw.githubusercontent.com/carret1268/LogicTreeETC/main/resources/arrow_examples/obtuse_arrow_with_head.png" width="400"/>
</div>

#### Acute angle arrow

<div align="center">
  <img src="https://raw.githubusercontent.com/carret1268/LogicTreeETC/main/resources/arrow_examples/acute_arrow_with_head.png" width="400"/>
</div>

#### Complex multi-segmented arrow

<div align="center">
  <img src="https://raw.githubusercontent.com/carret1268/LogicTreeETC/main/resources/arrow_examples/many_segments_with_head.png" width="400"/>
</div>

#### Basic Bezier arrow

<div align="center">
  <img src="https://raw.githubusercontent.com/carret1268/LogicTreeETC/main/resources/arrow_examples/basic_bezier_with_head.png" width="400"/>
</div>

#### Complex Bezier arrow

<div align="center">
  <img src="https://raw.githubusercontent.com/carret1268/LogicTreeETC/main/resources/arrow_examples/crazier_bezier_with_head.png" width=400/>
</div>

---

## Font Installation (Times New Roman)

If Times New Roman is missing from your system, install it manually:

```bash
# Linux
mkdir -p ~/.local/share/fonts
cp logictree/fonts/Times\ New\ Roman.ttf ~/.local/share/fonts/
fc-cache -fv
```

On Windows or macOS, open:

```
logictree/fonts/Times New Roman.ttf
```

Then click **Install**.

To verify installation:

```bash
fc-list | grep -i times
```

If matplotlib cannot find the font:

```bash
rm -rf ~/.cache/matplotlib
```

---

## LaTeX Rendering (Optional)

Enable LaTeX rendering with `use_tex_rendering=True` when calling `LogicTree.add_box()`.

### Windows

- Install [MiKTeX](https://miktex.org/download)

### macOS

- Install [MacTeX](https://tug.org/mactex/)

### Linux

```bash
sudo apt install texlive-latex-base texlive-fonts-recommended texlive-fonts-extra texlive-latex-extra dvipng
```

---

## Development

This project uses the following tools for code quality and security:

- **Ruff**: for fast linting and auto-formatting.
- **mypy**: for static type checking.
- **Bandit**: for security scanning.
- **GitHub Actions CI**: runs tests, type checks, linting, and security checks on every push.
- **Dependabot**: automatically checks for dependency updates.

See the [.github/workflows/](https://github.com/carret1268/LogicTreeETC/tree/main/.github/workflows) directory for CI configurations.

---

## License

This project is licensed under CC0 (public domain). See the `LICENSE` file for details.
