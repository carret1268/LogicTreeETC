# LogicTreeETC

A framework for building custom logic trees in Python, built on top of matplotlib.

## 💡 Motivation

This package was created because existing matplotlib arrow tools, such as `FancyArrow` and `FancyArrowPatch`, make it difficult to programmatically build logic tree diagrams or flowcharts. Specifically:

- There is no straightforward way to access or calculate the exact vertex positions of arrows after transformations.
- Aligning arrows with nodes requires manual calculations without bounding box or layout support.
- Matplotlib lacks tools for dynamically arranging boxes and drawing connecting arrows with precision.

LogicTreeETC fills this gap by providing an intuitive API for positioning labeled boxes and drawing arrows between them, enabling easy creation of clear and accurate logic tree diagrams.

![Logic Tree Example 0](./examples/logic_tree-sample_occurence03.png)

## 📜 Check and Install Fonts

This project uses the **Leelawadee** font by default.

To check if the font is already installed, call the `check_for_font("Leelawadee")` function in `./examples/decisionTreeExample.py`.  
- If it prints a file path, the font is installed.  
- If it prints “Leelawadee not found,” you’ll need to install it manually.

The font file is included with the project at:  
```
logictree/fonts/Leelawadee.ttf
```

---

### 🪟 Windows
- Double-click `Leelawadee.ttf` to open the font preview window.
- Click **Install** to add the font to your system.

---

### 🍏 macOS
- Double-click `Leelawadee.ttf` to open it in **Font Book**.
- Click **Install Font** to install it system-wide.

**Optional verification:**  
You can check with `fc-list` if you have the `fontconfig` tools installed:
```bash
fc-list | grep -i Leelawadee
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
  fc-list | grep -i Leelawadee
  ```

---

**🔔 Note:**  
After installing the font, you may need to restart applications or your graphical environment for the font to be recognized.

If you still see an error like `findfont: Font family 'Leelawadee' not found.`, you might need to refresh your matplotlib cache. Try running
```bash
rm -rf ~/.cache/matplotlib
```

## ⚠️ Optional: LaTeX Support for Matplotlib

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

**🔔 Note:**  
If you don’t plan to use LaTeX rendering in your plots, you can safely ignore these installation steps — LaTeX is not required to use the core functionality of this package.
