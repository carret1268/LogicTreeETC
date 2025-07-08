# LogicTreeETC

[![Documentation Status](https://readthedocs.org/projects/logictreeetc/badge/?version=latest)](https://logictreeetc.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/carret1268/logictreeetc/branch/main/graph/badge.svg)](https://codecov.io/gh/carret1268/logictreeetc)

**Create flexible, publication-quality logic tree diagrams and multi-segment arrows with full vertex control in Python.**

📖 **[Read the Docs: Full Documentation](https://logictreeetc.readthedocs.io/en/latest/)**

---

## Installation

You can install the latest version from PyPI with:

```bash
pip install logictreeetc
```

Or upgrade existing installation with:

```bash
pip install --upgrade logictreeetc
```

---

## Why use LogicTreeETC?

Matplotlib's built-in `FancyArrow` and `FancyArrowPatch` only support single straight or curved arrows without exposing vertex information. That means:
- You can’t create arrows with multiple segments or right-angle bends.
- You have no access to arrow vertices for debugging or integration.
- You’re stuck with fixed styling options.

**LogicTreeETC fixes this by:**

✅ Letting you build multi-segment arrows with arbitrary paths.  
✅ Storing all vertex positions in class attributes so you can reuse or debug them.  
✅ Giving you control over the width, head, color, and path of every arrow.  
✅ Integrating tightly with logic boxes for decision trees or flowcharts.

---

## Note

The `ArrowETC` class **assumes an equal aspect ratio** (i.e., 1:1 x/y scaling). If you plot with a non-square aspect ratio, the arrow will appear distorted since vertex coordinates assume equal scaling.

**To avoid this:**

1. Always set `ax.set_aspect('equal')` on your own matplotlib axes.
2. Or manually adjust vertex coordinates to compensate for uneven scaling.

---

## Examples

### Example 1: Decision Tree for Non-Targeted Analysis (NTA)

Creates a logic tree showing how samples progress through replicate, CV, and MDL checks, including:
- Reading counts from a CSV
- Adding boxes for decisions
- Annotating thresholds with LaTeX
- Connecting boxes with arrows and bifurcations

<p align="center">
  <img src="examples/DecisionTree_NTA-Example.png" alt="Decision Tree for NTA" width="600"/>
</p>

See [examples/decisionTreeNTAExample.py](examples/decisionTreeNTAExample.py) for full code.

---

### Example 2: Suggested Study Order for Data Structures & Algorithms (DSA)

Shows a recommended sequence for learning key data structures, from arrays to graphs, with arrows indicating the progression.

<p align="center">
  <img src="examples/DecisionTree_DSA-Example.png" alt="DSA Study Order" width="600"/>
</p>

See [examples/decisionTreeDSAExample.py](examples/decisionTreeDSAExample.py) for full code.

---

### Example 3: Standalone Arrows with ArrowETC

You can use `ArrowETC` by itself to build complex, multi-segment arrows or straight rectangular connectors. Arrows don’t have to include arrowheads—they can simply define a series of segments with consistent width:

```python
from logictree.ArrowETC import ArrowETC

arrow = ArrowETC([(0, 0), (-10, 0), (-10, -10)], 2, True)
arrow.save_arrow(name="./single_joint_arrow.png")
```


## Check and Install Fonts

This project uses the **Times New Roman** font by default.

To check if the font is already installed, call the `check_for_font("Times New Roman")` function in `./examples/decisionTreeExample.py`.  
- If it prints a file path, the font is installed.  
- If it prints “Times New Roman not found,” you’ll need to install it manually.

The font file is included with the project at:  
```
"logictree/fonts/Times New Roman.ttf"
```

---

### 🪟 Windows
- Double-click `"Times New Roman.ttf"` to open the font preview window.
- Click **Install** to add the font to your system.

---

### 🍏 macOS
- Double-click `"Times New Roman.ttf"` to open it in **Font Book**.
- Click **Install Font** to install it system-wide.

**Optional verification:**  
You can check with `fc-list` if you have the `fontconfig` tools installed:
```bash
fc-list | grep -i times
```
If you don’t have `fc-list`, you can install it with Homebrew:
```bash
brew install fontconfig
```

---

### 🐧 Linux (Debian/Ubuntu/WSL)
- Copy the font file to your local fonts directory and update the font cache:
  ```bash
  mkdir -p ~/.local/share/fonts
  cp logictree/fonts/Leelawadee.ttf ~/.local/share/fonts/
  fc-cache -fv
  ```
- Confirm installation:
  ```bash
  fc-list | grep -i times
  ```

---

**Note:**  
After installing the font, you may need to restart applications or your graphical environment for the font to be recognized.

If you still see an error like `findfont: Font family 'Times New Roman' not found.`, you might need to refresh your matplotlib cache. Try running
```bash
rm -rf ~/.cache/matplotlib
```

## Optional: LaTeX Support for Matplotlib

This package **does not require LaTeX** to function. However, if you enable LaTeX text rendering in `matplotlib` (e.g., by setting `plt.rc('text', usetex=True)` or by calling `LogicTreeETC.add_box()` method with `use_tex_rendering=True`), you must have a LaTeX installation available on your system.

Without LaTeX installed, trying to use tex rendering will cause errors like:
```
RuntimeError: Failed to process string with tex because latex could not be found
```

---

### 🪟 Windows
- Download and install [MiKTeX](https://miktex.org/download) (recommended for Windows).  
- During installation, choose the option to install missing packages on-the-fly if prompted.  
- After installation, restart your terminal or IDE to make sure the `latex` command is in your system PATH.

---

### 🍏 macOS
- Install MacTeX, the standard TeX distribution for macOS, from the [MacTeX website](https://tug.org/mactex/).  
- The download is large (~4GB), but it provides everything you need for LaTeX rendering.  
- After installing, you may need to restart your terminal or IDE for changes to take effect.

---

### 🐧 Linux (Debian/Ubuntu/WSL)
Install a minimal LaTeX environment with:
```bash
sudo apt update
sudo apt install texlive-latex-base texlive-fonts-recommended texlive-fonts-extra texlive-latex-extra dvipng
```

---

**Note:**  
If you don’t plan to use LaTeX rendering in your plots, you can safely ignore these installation steps — LaTeX is not required to use the core functionality of this package.

## License

This project is licensed under a CC0 License. See LICENSE file for details.
