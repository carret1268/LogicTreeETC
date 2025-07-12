"""
LogicTree - A Flow Diagram Engine for Visual Logic Structures

This module defines the `LogicTree` class, which provides a high-level interface for creating
logic tree diagrams using `LogicBox` and `ArrowETC` objects. It supports fast creation of labeled
boxes, styled arrow connections (including Bezier curves and segmented paths), and LaTeX-enhanced
text rendering. Diagrams are rendered with `matplotlib`, and the resulting figures can be saved
as high-resolution images.

Features
--------
- Add boxes with custom text, alignment, rotation, and visual styling.
- Connect boxes using straight, segmented, or curved arrows with optional arrowheads.
- Full support for LaTeX rendering inside boxes and titles.
- Export to PNG with tight layout control and adjustable aspect ratio.

Examples
--------
>>> from logictree.LogicTreeETC import LogicTree
>>> logic_tree = LogicTree(xlims=(0, 100), ylims=(0, 100), title="My Logic Tree")

# Add boxes
>>> logic_tree.add_box(20, 80, "Start", "Start", "black", "white", ha="center")
>>> logic_tree.add_box(20, 50, "Decision", "Decision", "black", "white", ha="center")
>>> logic_tree.add_box(10, 20, "Option A", "OptionA", "black", "green", ha="center")
>>> logic_tree.add_box(30, 20, "Option B", "OptionB", "black", "red", ha="center")

# Add arrows
>>> logic_tree.add_connection(
...     logic_tree.boxes["Start"], logic_tree.boxes["Decision"], arrow_head=True, shaft_width=25
... )
>>> logic_tree.add_connection_biSplit(
...     logic_tree.boxes["Decision"],
...     logic_tree.boxes["OptionA"],
...     logic_tree.boxes["OptionB"],
...     arrow_head=True, shaft_width=30
... )

# Finalize and save
>>> logic_tree.make_title(pos="center")
>>> logic_tree.save_as_png("logic_tree_example.png", dpi=300)

Notes
-----
- If LaTeX rendering is enabled, you must have packages such as `bm`, `amsmath`, `soul`, and `relsize` installed.
  On most Linux systems, these can be installed using:

  sudo apt install -y texlive-latex-base texlive-latex-recommended texlive-fonts-recommended \\
      texlive-latex-extra texlive-humanities dvipng cm-super
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

from math import atan2, degrees
from matplotlib.patches import BoxStyle
import matplotlib.pyplot as plt
from numpy import hypot

from .arrow_etc import ArrowETC
from .logicbox import LogicBox


class LogicTree:
    """
    LogicTree - High-Level Diagram Layout Engine for Visualizing Logic Structures

    The `LogicTree` class provides a complete system for building logic diagrams by placing labeled boxes
    (`LogicBox` objects) and connecting them with highly customizable arrows (`ArrowETC` objects).
    It abstracts away most of the layout, rendering, and styling logic, while remaining fully
    compatible with the `matplotlib` workflow.

    Key features include:
    - Flexible arrow routing (straight, segmented, or curved paths)
    - Optional arrowhead placement and geometry control
    - Box labeling with alignment, LaTeX rendering, and font customization
    - Tight integration with `matplotlib` for exporting high-resolution plots

    Parameters
    ----------
    fig_size : tuple of float, optional
        Size of the output matplotlib figure in inches (width, height). Default is (9, 9).
    xlims : tuple of float, optional
        Min and max limits of the x-axis. Used to constrain box layout. Default is (0, 100).
    ylims : tuple of float, optional
        Min and max limits of the y-axis. Default is (0, 100).
    colormode : {'dark', 'light'}, optional
        Sets default font and figure background colors. Default is 'light'.
    title : str, optional
        Title of the logic diagram. Can also be set later using `make_title()`.
    font_dict : dict, optional
        Dictionary of font properties to use for box text. If None, a default style is used.
    font_dict_title : dict, optional
        Dictionary of font properties to use for the figure title. If None, a default style is used.
    text_color : str, optional
        Override for all box text color (applies to `font_dict` if provided).
    title_color : str, optional
        Override for the title color (applies to `font_dict_title` if provided).

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The main matplotlib figure object.
    ax : matplotlib.axes.Axes
        The axes object where boxes and arrows are drawn.
    boxes : dict[str, LogicBox]
        Dictionary of box name → LogicBox instances. Populated via `add_box()`.
    arrows : list[ArrowETC]
        List of all ArrowETC arrows added to the figure.
    title : str or None
        Title string used for rendering the figure heading.
    xlims, ylims : tuple of float
        Axis limits, controlling visual boundaries.
    font_dict : dict
        Default font properties for box text.
    title_font_dict : dict
        Font properties used when rendering the diagram title.
    latex_ul_depth : str
        Thickness setting used when underlining LaTeX-rendered box text.
    latex_ul_width : str
        Depth setting used when underlining LaTeX-rendered box text.
    """

    def __init__(
        self,
        fig_size: Tuple[float, float] = (9, 9),
        xlims: Tuple[float, float] = (0, 100),
        ylims: Tuple[float, float] = (0, 100),
        colormode: Literal["dark", "light"] = "light",
        title: Optional[str] = None,
        font_dict: Optional[Dict[str, Any]] = None,
        font_dict_title: Optional[Dict[str, Any]] = None,
        text_color: Optional[str] = None,
        title_color: Optional[str] = None,
    ) -> None:
        self.boxes: Dict[str, LogicBox] = {}
        self.arrows: List[ArrowETC] = []
        self.title = title
        self.xlims = xlims
        self.ylims = ylims

        # font dictionary for title
        if font_dict_title is None:
            font_dict_title = {
                "fontname": "Times New Roman",
                "fontsize": 34,
                "color": "white" if colormode == "dark" else "black",
            }
        if title_color is not None:
            font_dict_title["color"] = title_color
        self.title_font_dict = font_dict_title

        # default font dictionary for boxes
        if font_dict is None:
            font_dict = {
                "fontname": "Times New Roman",
                "fontsize": 15,
                "color": "white" if colormode == "dark" else "black",
            }
        if text_color is not None:
            font_dict["color"] = text_color
        self.font_dict = font_dict

        # underlining options for LaTeX rendering
        self.latex_ul_depth = "1pt"
        self.latex_ul_width = "1pt"

        # set style presets
        font_defaults = {
            "font.family": "Times New Roman",
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.titlecolor": "white" if colormode == "dark" else "black",
            "axes.labelsize": 20,
            "axes.labelcolor": "white" if colormode == "dark" else "black",
            "axes.facecolor": "black" if colormode == "dark" else "white",
            "figure.facecolor": "black" if colormode == "dark" else "white",
            "xtick.color": "white" if colormode == "dark" else "black",
            "ytick.color": "white" if colormode == "dark" else "black",
            "legend.fontsize": 14,
            "legend.title_fontsize": 16,
        }

        plt.rcParams.update(font_defaults)

        # generate figure and axes
        fig, ax = plt.subplots(figsize=fig_size, frameon=True)
        ax.set_xlim(xlims[0], xlims[1])
        ax.set_ylim(ylims[0], ylims[1])
        ax.axis("off")

        if colormode == "dark":
            for spine in ax.spines.values():
                spine.set_color("white")

        fig.canvas.draw_idle()

        self.fig = fig
        self.ax = ax

        self._colormode = colormode

    def _get_pathsForBi_left_then_right(
        self,
        Ax2: float,
        Ay2: float,
        left_box: LogicBox,
        right_box: LogicBox,
        tip_offset: float,
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Construct bifurcated arrow paths from a common stem point to left and right child boxes.

        This helper method is used internally by `add_connection_biSplit()` to generate
        two separate arrow paths branching from a shared vertical stem, typically used for
        decision or fork structures.

        The method computes elbow-style paths (three points) for each child box:
        vertical -> horizontal -> vertical.

        Parameters
        ----------
        Ax2, Ay2 : float
            The (x, y) coordinates where the vertical stem ends and the bifurcation begins.
        left_box, right_box : LogicBox
            The target boxes for the left and right branches. `left_box` must be horizontally
            to the left of `right_box`.
        tip_offset : float
            Vertical offset to apply to the final point of each branch. Helps prevent overlap
            with box edges.

        Returns
        -------
        tuple of list of (float, float)
            Two paths (left and right), each a list of three (x, y) points defining an arrow route.

        Raises
        ------
        ValueError
            If any required coordinates in `left_box` or `right_box` are not initialized.
        """
        if (
            left_box.yTop is None
            or left_box.yBottom is None
            or left_box.xCenter is None
            or left_box.yCenter is None
        ):
            raise ValueError(
                "left_box LogicBox layout not initialized before accessing coordinates."
            )
        if (
            right_box.yTop is None
            or right_box.yBottom is None
            or right_box.xCenter is None
            or right_box.yCenter is None
        ):
            raise ValueError(
                "right_box LogicBox layout not initialized before accessing coordinates."
            )

        # create the leftward arrow
        Lx1 = Ax2
        Ly1 = Ay2
        Lx2 = left_box.xCenter
        Ly2 = Ly1
        Lx3 = Lx2
        Ly3 = (
            left_box.yTop + tip_offset
            if Ay2 > left_box.yCenter
            else left_box.yBottom - tip_offset
        )

        # create the rightward arrow
        Rx1 = Ax2
        Ry1 = Ay2
        Rx2 = right_box.xCenter
        Ry2 = Ry1
        Rx3 = Rx2
        Ry3 = (
            right_box.yTop + tip_offset
            if Ay2 > right_box.yCenter
            else right_box.yBottom - tip_offset
        )

        # set paths
        path_left = [(Lx1, Ly1), (Lx2, Ly2), (Lx3, Ly3)]
        path_right = [(Rx1, Ry1), (Rx2, Ry2), (Rx3, Ry3)]

        return path_left, path_right

    def add_box(
        self,
        xpos: float,
        ypos: float,
        text: str,
        box_name: str,
        bbox_fc: str,
        bbox_ec: str,
        font_dict: Optional[Dict[str, Any]] = None,
        text_color: Optional[str] = None,
        fs: Optional[int] = None,
        font_weight: Optional[float] = None,
        lw: float = 1.6,
        bbox_style: BoxStyle = BoxStyle("Round", pad=0.6),
        va: Literal["top", "center", "bottom"] = "center",
        ha: Literal["left", "center", "right"] = "right",
        use_tex_rendering: bool = False,
        ul: bool = False,
        ul_depth_width: Optional[Tuple[float, float]] = None,
        angle: float = 0.0,
    ) -> LogicBox:
        """
        Add a styled LogicBox to the diagram at the specified position.

        This method places a new labeled box on the logic tree, using customizable
        styling for fonts, alignment, and appearance. Optionally, LaTeX rendering and
        underlining can be enabled for rich formatting.

        Parameters
        ----------
        xpos, ypos : float
            Coordinates for box placement in data space.
        text : str
            Text to display inside the box. Can include LaTeX if `use_tex_rendering=True`.
        box_name : str
            Unique identifier for the box. Used to reference the box in connections.
        bbox_fc, bbox_ec : str
            Face color and edge color of the box. Accepts color names or RGBA values.
        font_dict : dict, optional
            Dictionary of font properties (e.g., fontname, fontsize, weight).
        text_color : str, optional
            Override for the text color (applied to `font_dict`).
        fs : int, optional
            Override for font size.
        font_weight : float or str, optional
            Font weight (e.g., 'bold', 'normal').
        lw : float, optional
            Line width of the box outline. Default is 1.6.
        bbox_style : matplotlib.patches.BoxStyle, optional
            Shape and padding style for the box. Default is BoxStyle("Round", pad=0.6).
        va : {'top', 'center', 'bottom'}, optional
            Vertical alignment of text within the box. Default is 'center'.
        ha : {'left', 'center', 'right'}, optional
            Horizontal alignment of text. Default is 'right'.
        use_tex_rendering : bool, optional
            If True, enables LaTeX rendering for the box text. Requires proper LaTeX installation.
        ul : bool, optional
            If True, applies underlining to the box text using LaTeX.
        ul_depth_width : tuple of (float, float), optional
            Tuple specifying LaTeX underline depth and thickness.
        angle : float, optional
            Angle (in degrees) to rotate the box around its center. Default is 0.

        Returns
        -------
        LogicBox
            The constructed and registered LogicBox object.

        Raises
        ------
        ValueError
            If `box_name` is already used or if the rendered text lacks a valid bounding box.
        """
        if box_name in self.boxes:
            raise ValueError(
                f"Box name '{box_name}' already exists. Please use a unique name."
            )

        # option to use latex rendering (minimal font options with latex, so not default)
        if use_tex_rendering:
            # our latex preamble for importing latex packages and making a command
            # \bigsymbol{} for enlarging latex math symbols
            latex_preamble = (
                r"\usepackage{bm}"
                r"\usepackage{amsmath}"
                r"\usepackage{soul}"
                r"\setul{2pt}{1pt}"
                r"\usepackage{relsize}"
                r"\newcommand{\bigsymbol}[1]{\mathlarger{\mathlarger{\mathlarger{#1}}}}"
            )

            # update rcParams to use latex
            plt.rcParams.update(
                {
                    "text.usetex": True,
                    "font.family": "cm",
                    "text.latex.preamble": latex_preamble,
                }
            )
        else:
            plt.rcParams.update({"text.usetex": False})

        # set fontidct of not provided
        if font_dict is None:
            font_dict = self.font_dict.copy()
        # if specific text color is specified, change it in font_dict
        if text_color is not None:
            font_dict["color"] = text_color
        # if specific fontsize is specified, change it in font_dict
        if fs is not None:
            font_dict["fontsize"] = fs
        # if weight is specified, change it in font_dict
        if font_weight is not None:
            font_dict["weight"] = font_weight

        # create a logicBox object which stores all of this information
        myBox = LogicBox(
            xpos=xpos,
            ypos=ypos,
            text=text,
            box_name=box_name,
            bbox_fc=bbox_fc,
            bbox_ec=bbox_ec,
            bbox_style=bbox_style,
            font_dict=font_dict,
            va=va,
            ha=ha,
            lw=lw,
            angle=angle,
        )

        # add latex commands to text for underlining
        if use_tex_rendering and (ul or ul_depth_width is not None):
            text_str = r"\ul{" + myBox.text + r"}"
            # if underlining parameters are set, add the command to change them
            if ul_depth_width is not None:
                text_str = (
                    r"\setul{"
                    + f"{ul_depth_width[0]}"
                    + r"}{"
                    + f"{ul_depth_width[1]}"
                    + r"}"
                    + text_str
                )
        else:
            text_str = myBox.text
        # make the text
        txt = self.ax.text(
            x=myBox.x,
            y=myBox.y,
            s=text_str,
            fontdict=myBox.font_dict,
            bbox=myBox.style,
            va=myBox.va,
            ha=myBox.ha,
            rotation=myBox.angle,
        )

        # Ensure the figure is rendered so bbox extents are valid
        self.fig.canvas.draw()

        # Get the full bounding box of the text box (includes padding and styling)
        bbox_patch = txt.get_bbox_patch()
        if bbox_patch is None:
            raise ValueError("Text object has no bounding box patch.")

        # Convert the patch bbox from display to data coordinates
        bbox_data = self.ax.transData.inverted().transform_bbox(
            bbox_patch.get_window_extent(renderer=self.fig.canvas.get_renderer())  # type: ignore
        )

        # Set box dimensions and positions
        myBox.xLeft, myBox.xRight = bbox_data.x0, bbox_data.x1
        myBox.yBottom, myBox.yTop = bbox_data.y0, bbox_data.y1
        myBox.width = myBox.xRight - myBox.xLeft
        myBox.height = myBox.yTop - myBox.yBottom
        myBox.xCenter = (myBox.xLeft + myBox.xRight) / 2
        myBox.yCenter = (myBox.yBottom + myBox.yTop) / 2

        # store box in our LogicTree object's box dictionary to grab dimensions when needed
        self.boxes[myBox.name] = myBox

        return myBox

    def add_arrow(self, arrow: ArrowETC, fill_arrow: bool = True) -> None:
        """
        Add a preconstructed ArrowETC object to the LogicTree figure.

        This method allows manual control over arrow creation by letting you construct
        an `ArrowETC` instance externally and then add it to the tree. Useful for advanced
        customizations or layout debugging.

        The arrow is appended to `self.arrows` and drawn onto the existing matplotlib Axes.

        Parameters
        ----------
        arrow : ArrowETC
            A fully constructed ArrowETC object with geometry and styling already defined.
        fill_arrow : bool, optional
            Whether to fill the arrow body (`ArrowETC.fc`). If False, only the outline is drawn.
            Default is True.

        Raises
        ------
        ValueError
            If the arrow has fewer than two path points.
        """
        if not arrow.path or len(arrow.path) < 2:
            raise ValueError("ArrowETC must have a path with at least two points.")

        self.arrows.append(arrow)
        self.ax = arrow.draw_to_ax(self.ax, fill_arrow=fill_arrow)

    def add_arrow_between(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        shaft_width: float = 20,
        arrow_head: bool = True,
        arrow_head_at_tail: bool = False,
        arrow_head_width_multiplier: float = 2,
        arrow_head_length_multiplier: float = 1.5,
        tip_offset: float = 0.0,
        butt_offset: float = 0.0,
        fc: str = "black",
        ec: str = "black",
        zorder: float = 1.0,
        lw: float = 1.0,
        ls: str = "-",
        fill_arrow: bool = True,
    ) -> None:
        """
        Draw a single arrow between two points using ArrowETC.

        This low-level method is ideal for freeform annotations, callouts, or diagram embellishments.
        Arrow geometry is computed between two (x, y) coordinates with optional tip and butt offsets.

        Parameters
        ----------
        start : tuple of float
            (x, y) coordinates for the arrow base (tail).
        end : tuple of float
            (x, y) coordinates for the arrow tip (head).
        shaft_width : float, optional
            Width of the arrow shaft in pixels. Default is 20.
        arrow_head : bool, optional
            Whether to draw an arrowhead at the tip. Default is True.
        arrow_head_at_tail : bool, optional
            Whether to draw a second arrowhead pointing backward from the tail. Default is False.
        arrow_head_width_multiplier : float, optional
            Width multiplier for arrowhead relative to shaft. Default is 2.
        arrow_head_length_multiplier : float, optional
            Length multiplier for arrowhead relative to shaft width. Default is 1.5.
        tip_offset : float, optional
            Distance to shorten the arrow tip, to avoid overlapping a target. Default is 0.0.
        butt_offset : float, optional
            Distance to push the arrow base forward, away from its start point. Default is 0.0.
        fc : str, optional
            Fill color of the arrow. Default is "black".
        ec : str, optional
            Edge (stroke) color of the arrow. Default is "black".
        zorder : float, optional
            Drawing order for the arrow. Higher values appear on top. Default is 1.0.
        lw : float, optional
            Line width for the arrow outline. Default is 1.0.
        ls : str, optional
            Line style for the outline (e.g., "-", "--"). Default is "-".
        fill_arrow : bool, optional
            Whether to fill the interior of the arrow. Default is True.

        Raises
        ------
        ValueError
            If the start and end points are identical (zero-length arrow).
        """
        if start == end:
            raise ValueError("Arrow start and end points must differ.")

        # Vector from start to end
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = hypot(dx, dy)

        # Unit direction vector
        ux, uy = dx / length, dy / length

        # Apply offsets
        new_start = (start[0] + ux * butt_offset, start[1] + uy * butt_offset)
        new_end = (end[0] - ux * tip_offset, end[1] - uy * tip_offset)

        path = [new_start, new_end]
        arrow = ArrowETC(
            ax=self.ax,
            path=path,
            shaft_width=shaft_width,
            arrow_head=arrow_head,
            arrow_head_at_tail=arrow_head_at_tail,
            arrow_head_width_multiplier=arrow_head_width_multiplier,
            arrow_head_length_multiplier=arrow_head_length_multiplier,
            fc=fc,
            ec=ec,
            zorder=zorder,
            lw=lw,
            ls=ls,
        )
        self.add_arrow(arrow, fill_arrow=fill_arrow)

    def add_connection_biSplit(
        self,
        boxA: LogicBox,
        boxB: LogicBox,
        boxC: LogicBox,
        arrow_head: bool = True,
        shaft_width: float = 20,
        fill_connection: bool = True,
        fc_A: Optional[str] = None,
        ec_A: Optional[str] = None,
        fc_B: Optional[str] = None,
        ec_B: Optional[str] = None,
        fc_C: Optional[str] = None,
        ec_C: Optional[str] = None,
        lw: float = 0.5,
        butt_offset: float = 0,
        tip_offset: float = 0,
        textLeft: Optional[str] = None,
        textRight: Optional[str] = None,
        textLeftOffset: Literal["above", "below"] = "above",
        textRightOffset: Literal["above", "below"] = "above",
        text_kwargs: Optional[dict] = None,
    ) -> None:
        """
        Draw a bifurcating arrow connection from a parent box (`boxA`) to two child boxes (`boxB`, `boxC`).

        This method creates a split connection: a single vertical shaft from `boxA` that branches
        into two lateral arrows, one for each child. Labels can optionally be placed along each branch.

        Left/right ordering is automatically inferred based on the horizontal positions of `boxB` and `boxC`.
        Vertical direction (upward vs. downward) is inferred from the y-position of `boxA` relative to the children.

        Parameters
        ----------
        boxA : LogicBox
            The parent box initiating the bifurcation.
        boxB : LogicBox
            One child box. Its left/right role is automatically determined.
        boxC : LogicBox
            The other child box.
        arrow_head : bool, optional
            Whether to draw arrowheads on both branches. Default is True.
        shaft_width : float, optional
            Width of the arrow shafts in pixels. Default is 20.
        fill_connection : bool, optional
            Whether to fill the arrows with color (`fc_*`). Default is True.
        fc_A, ec_A : str, optional
            Face and edge colors for the vertical stem from `boxA`. If None, defaults to `boxA`'s colors.
            Use "ec" or "fc" to inherit from the complementary color.
        fc_B, ec_B : str, optional
            Colors for the branch leading to `boxB`. Same logic as above.
        fc_C, ec_C : str, optional
            Colors for the branch leading to `boxC`.
        lw : float, optional
            Line width for arrow outlines. Default is 0.5.
        butt_offset : float, optional
            Distance to nudge the stem away from `boxA`. Default is 0.
        tip_offset : float, optional
            Distance to nudge the arrow tips away from the target boxes. Default is 0.
        textLeft, textRight : str, optional
            Optional labels placed along the left and right arrow branches.
        textLeftOffset, textRightOffset : {'above', 'below'}, optional
            Whether to place the corresponding label above or below its arrow shaft. Default is 'above'.
        text_kwargs : dict, optional
            Additional text styling options for labels, e.g.:
                - fontsize : int
                - fontname : str
                - color : str
                - fontstyle : str

        Raises
        ------
        ValueError
            If any of the boxes are missing layout attributes.
            If `boxA` is not clearly above or below both `boxB` and `boxC`.

        Notes
        -----
        This method is useful for visualizing binary decisions or logical forks in flow diagrams.
        Each arrow path consists of three segments: vertical stem → horizontal split → vertical drop.
        """
        if (
            boxA.xLeft is None
            or boxA.xCenter is None
            or boxA.xRight is None
            or boxA.yTop is None
            or boxA.yCenter is None
            or boxA.yBottom is None
        ):
            raise ValueError(
                "boxA LogicBox layout is not initialized before accessing coordinates."
            )
        if (
            boxB.xLeft is None
            or boxB.xCenter is None
            or boxB.xRight is None
            or boxB.yTop is None
            or boxB.yCenter is None
            or boxB.yBottom is None
        ):
            raise ValueError(
                "boxB LogicBox layout is not initialized before accessing coordinates."
            )
        if (
            boxC.xLeft is None
            or boxC.xCenter is None
            or boxC.xRight is None
            or boxC.yTop is None
            or boxC.yCenter is None
            or boxC.yBottom is None
        ):
            raise ValueError(
                "boxC LogicBox layout is not initialized before accessing coordinates."
            )

        # Resolve text styling
        if text_kwargs is None:
            text_kwargs = {}
        fontname = text_kwargs.get("fontname", "sans-serif")
        fontsize = text_kwargs.get("fontsize", 12)
        fontcolor = text_kwargs.get("color", "white")
        fontstyle = text_kwargs.get("fontstyle", "normal")

        def annotate_segment(
            text: Optional[str],
            path: list[tuple[float, float]],
            offset: Literal["above", "below"],
        ) -> None:
            """
            Place text at the midpoint of a given arrow segment, offset vertically above or below.

            Parameters
            ----------
            text : str, optional
                The text to render. If None or empty, nothing is drawn.
            path : list of (float, float)
                The path representing the arrow segment.
            offset : {'above', 'below'}
                Whether the label is placed above or below the arrow shaft.
            """
            if not text:
                return
            (x1, y1), (x2, _) = path[0], path[-1]
            xm = (x1 + x2) / 2 + (
                0.2 * shaft_width / 2 if x1 < x2 else -0.2 * shaft_width / 2
            )
            ym = y1 + shaft_width * 0.2 if offset == "above" else y1 - shaft_width * 0.2
            va = "bottom" if offset == "above" else "top"
            self.ax.text(
                xm,
                ym,
                text,
                ha="center",
                va=va,
                fontsize=fontsize,
                fontname=fontname,
                color=fontcolor,
                fontstyle=fontstyle,
            )

        def resolve_colors(
            box: LogicBox, fc: Optional[str], ec: Optional[str]
        ) -> tuple[Optional[str], str]:
            """
            Resolve fill and edge color settings using box defaults and shorthand keywords.

            Parameters
            ----------
            box : LogicBox
                The box used to provide default or fallback colors.
            fc : str, optional
                The face color. Can be None, "ec", or a valid color string.
            ec : str, optional
                The edge color. Can be None, "fc", or a valid color string.

            Returns
            -------
            tuple of (str, str)
                The resolved face color and edge color.
            """
            if fill_connection:
                fc = (
                    box.edge_color
                    if fc == "ec"
                    else (box.face_color if fc is None else fc)
                )
            ec = (
                box.face_color if ec == "fc" else (box.edge_color if ec is None else ec)
            )

            return fc, ec

        fc_A, ec_A = resolve_colors(boxA, fc_A, ec_A)
        fc_B, ec_B = resolve_colors(boxB, fc_B, ec_B)
        fc_C, ec_C = resolve_colors(boxC, fc_C, ec_C)

        # Determine vertical direction of arrows
        if boxA.yCenter > boxB.yCenter and boxA.yCenter > boxC.yCenter:
            Ax1, Ay1 = boxA.xCenter, boxA.yBottom - butt_offset
            Ay2 = (Ay1 + max(boxB.yTop, boxC.yTop)) / 2
        elif boxA.yCenter < boxB.yCenter and boxA.yCenter < boxC.yCenter:
            Ax1, Ay1 = boxA.xCenter, boxA.yTop + butt_offset
            Ay2 = (Ay1 + min(boxB.yBottom, boxC.yBottom)) / 2
        else:
            raise ValueError("boxA must be clearly above or below both boxB and boxC.")

        Ax2 = Ax1
        path_vertical = [(Ax1, Ay1), (Ax2, Ay2)]
        arrow = ArrowETC(
            ax=self.ax,
            path=path_vertical,
            arrow_head=False,
            shaft_width=shaft_width,
            ec=cast(str, ec_A),
            fc=cast(str, fc_A),
            lw=lw,
        )
        self.add_arrow(arrow)

        # Determine left/right order
        left_box, right_box = (
            (boxB, boxC) if boxB.xCenter < boxC.xCenter else (boxC, boxB)
        )
        path_left, path_right = self._get_pathsForBi_left_then_right(
            Ax2, Ay2, left_box=left_box, right_box=right_box, tip_offset=tip_offset
        )

        def draw_branch(
            path: list[tuple[float, float]],
            ec: str,
            fc: str,
            lw: float,
            label: Optional[str],
            label_offset: Literal["above", "below"],
        ) -> None:
            """
            Draw a single arrow branch with optional fill and text annotation.

            Parameters
            ----------
            path : list of (float, float)
                The arrow path from the split point to the destination box.
            ec : str
                Edge color of the arrow.
            fc : str
                Fill color of the arrow.
            label : str, optional
                Optional text to annotate the arrow shaft.
            label_offset : {'above', 'below'}
                Vertical position of the text relative to the arrow shaft.
            """
            arrow = ArrowETC(
                ax=self.ax,
                path=path,
                arrow_head=arrow_head,
                shaft_width=shaft_width,
                ec=ec,
                fc=fc,
                lw=lw,
                close_tail=False,
                zorder=1000,
            )
            self.add_arrow(arrow)
            annotate_segment(label, path, label_offset)

        # Draw left
        if left_box is boxB:
            draw_branch(
                path_left,
                cast(str, ec_B),
                cast(str, fc_B),
                lw,
                textLeft,
                textLeftOffset,
            )
            draw_branch(
                path_right,
                cast(str, ec_C),
                cast(str, fc_C),
                lw,
                textRight,
                textRightOffset,
            )
        else:
            draw_branch(
                path_left,
                cast(str, ec_C),
                cast(str, fc_C),
                lw,
                textLeft,
                textLeftOffset,
            )
            draw_branch(
                path_right,
                cast(str, ec_B),
                cast(str, fc_B),
                lw,
                textRight,
                textRightOffset,
            )

    def _get_side_coords(
        self, box: LogicBox, side: str, offset: float = 0.0
    ) -> tuple[float, float]:
        """
        Get a coordinate on a specific edge or corner of a LogicBox, with optional outward offset.

        This utility method returns a coordinate useful for routing arrows. If `offset` is non-zero,
        the point is shifted outward in the direction implied by `side`.

        Parameters
        ----------
        box : LogicBox
            The box from which to extract a connection point.
        side : str
            The location on the box to target. Must be one of:
            {'left', 'right', 'top', 'bottom', 'center',
            'topLeft', 'topRight', 'bottomLeft', 'bottomRight'}.
        offset : float, optional
            Distance to shift the point outward (away from the box edge). Default is 0.

        Returns
        -------
        tuple of float
            The (x, y) coordinates of the specified point, offset if requested.

        Raises
        ------
        ValueError
            If any box coordinates are missing (i.e., not initialized).
            If `side` is not a recognized keyword.
        """

        if (
            box.xLeft is None
            or box.xCenter is None
            or box.xRight is None
            or box.yTop is None
            or box.yCenter is None
            or box.yBottom is None
        ):
            raise ValueError(
                "box LogicBox layout is not initialized before accessing coordinates."
            )

        match side:
            case "left":
                return box.xLeft - offset, box.yCenter
            case "right":
                return box.xRight + offset, box.yCenter
            case "top":
                return box.xCenter, box.yTop + offset
            case "bottom":
                return box.xCenter, box.yBottom - offset
            case "center":
                return box.xCenter, box.yCenter
            case "topLeft":
                return box.xLeft - offset, box.yTop + offset
            case "topRight":
                return box.xRight + offset, box.yTop + offset
            case "bottomLeft":
                return box.xLeft - offset, box.yBottom - offset
            case "bottomRight":
                return box.xRight + offset, box.yBottom - offset
            case _:
                raise ValueError(f"Invalid side: '{side}'")

    def add_connection(
        self,
        boxA: LogicBox,
        boxB: Union[LogicBox, Tuple[float, float]],
        segmented: bool = False,
        arrow_head: bool = True,
        arrow_head_at_tail: bool = False,
        arrow_head_width_multiplier: float = 2,
        arrow_head_length_multiplier: float = 1.5,
        shaft_width: float = 20,
        fill_connection: bool = True,
        butt_offset: float = 0,
        tip_offset: float = 0,
        fc: Optional[str] = None,
        ec: Optional[str] = None,
        lw: float = 0.7,
        sideA: Optional[
            Literal[
                "left",
                "topLeft",
                "top",
                "topRight",
                "right",
                "bottomRight",
                "bottom",
                "bottomLeft",
                "center",
            ]
        ] = None,
        sideB: Optional[
            Literal[
                "left",
                "topLeft",
                "top",
                "topRight",
                "right",
                "bottomRight",
                "bottom",
                "bottomLeft",
                "center",
            ]
        ] = None,
    ) -> None:
        """
        Draw a straight or elbow-style arrow between two boxes or from a box to a point.

        This method provides full control over routing, arrowhead style, and shaft offsets.
        It supports both direct and segmented paths and allows entry/exit points on specific
        box edges or corners.

        Parameters
        ----------
        boxA : LogicBox
            The source box where the arrow begins.
        boxB : LogicBox or tuple of float
            The destination - either another LogicBox or a fixed (x, y) coordinate.
        segmented : bool, optional
            If True, draws an elbow-style arrow with horizontal/vertical joints. If False,
            draws a straight line. Default is False.
        arrow_head : bool, optional
            Whether to include an arrowhead pointing at the destination. Default is True.
        arrow_head_at_tail : bool, optional
            Whether to include a second arrowhead pointing backward from the source. Default is False.
        arrow_head_width_multiplier : float, optional
            Width multiplier for the arrowhead. Default is 2.
        arrow_head_length_multiplier : float, optional
            Length multiplier for the arrowhead. Default is 1.5.
        shaft_width : float, optional
            Thickness of the arrow shaft in pixels. Default is 20.
        fill_connection : bool, optional
            Whether to fill the arrow polygon with color. Default is True.
        butt_offset : float, optional
            Offset the arrow's base outward from the source box. Default is 0.
        tip_offset : float, optional
            Offset the arrow's tip away from the target. Helps avoid overlap. Default is 0.
        fc : str, optional
            Fill color for the arrow body. If None, uses the target box's face color.
        ec : str, optional
            Edge (outline) color. If None, uses the target box's edge color.
        lw : float, optional
            Line width for the arrow outline. Default is 0.7.
        sideA : str, optional
            Edge or corner of `boxA` to start from. If None, inferred based on angle.
        sideB : str, optional
            Edge or corner of `boxB` to point to. Ignored if `boxB` is a coordinate.

        Raises
        ------
        ValueError
            If box coordinates are missing or if the boxes have identical centers.
        """
        if (
            boxA.xLeft is None
            or boxA.xCenter is None
            or boxA.xRight is None
            or boxA.yTop is None
            or boxA.yCenter is None
            or boxA.yBottom is None
        ):
            raise ValueError(
                "boxA LogicBox layout is not initialized before accessing coordinates."
            )

        if isinstance(boxB, LogicBox):
            if (
                boxB.xLeft is None
                or boxB.xCenter is None
                or boxB.xRight is None
                or boxB.yTop is None
                or boxB.yCenter is None
                or boxB.yBottom is None
            ):
                raise ValueError(
                    "boxB LogicBox layout is not initialized before accessing coordinates."
                )
            if fill_connection:
                if fc is None or fc == "fc":
                    fc = boxB.face_color
                elif fc == "ec":
                    fc = boxB.edge_color
            if ec is None or ec == "ec":
                ec = boxB.edge_color
            elif ec == "fc":
                ec = boxB.face_color

            if boxA.xCenter == boxB.xCenter and boxA.yCenter == boxB.yCenter:
                raise ValueError("Boxes cannot have the same position.")

            dx = boxB.xCenter - boxA.xCenter
            dy = boxB.yCenter - boxA.yCenter
        else:
            # boxB is a coordinate point
            xB, yB = boxB
            dx = xB - boxA.xCenter
            dy = yB - boxA.yCenter

        theta = degrees(atan2(dy, dx))

        def auto_side(theta: float, for_A: bool) -> str:
            if -45 <= theta <= 45:
                return "right" if for_A else "left"
            elif 45 < theta <= 135:
                return "top" if for_A else "bottom"
            elif theta > 135 or theta < -135:
                return "left" if for_A else "right"
            else:
                return "bottom" if for_A else "top"

        resolved_sideA = sideA or auto_side(theta, for_A=True)
        resolved_sideB = sideB or auto_side(theta, for_A=False)

        start = self._get_side_coords(boxA, resolved_sideA)

        if isinstance(boxB, LogicBox):
            end = self._get_side_coords(boxB, resolved_sideB)
        else:
            end = boxB  # (x, y) tuple

        if butt_offset:
            match resolved_sideA:
                case "left":
                    start = (start[0] - butt_offset, start[1])
                case "right":
                    start = (start[0] + butt_offset, start[1])
                case "top":
                    start = (start[0], start[1] + butt_offset)
                case "bottom":
                    start = (start[0], start[1] - butt_offset)
                case "topLeft":
                    start = (start[0] - butt_offset, start[1] + butt_offset)
                case "topRight":
                    start = (start[0] + butt_offset, start[1] + butt_offset)
                case "bottomLeft":
                    start = (start[0] - butt_offset, start[1] - butt_offset)
                case "bottomRight":
                    start = (start[0] + butt_offset, start[1] - butt_offset)

        if tip_offset:
            match resolved_sideB:
                case "left":
                    end = (end[0] - tip_offset, end[1])
                case "right":
                    end = (end[0] + tip_offset, end[1])
                case "top":
                    end = (end[0], end[1] + tip_offset)
                case "bottom":
                    end = (end[0], end[1] - tip_offset)
                case "topLeft":
                    end = (end[0] - tip_offset, end[1] + tip_offset)
                case "topRight":
                    end = (end[0] + tip_offset, end[1] + tip_offset)
                case "bottomLeft":
                    end = (end[0] - tip_offset, end[1] - tip_offset)
                case "bottomRight":
                    end = (end[0] + tip_offset, end[1] - tip_offset)

        if segmented:
            if isinstance(boxB, LogicBox):
                if (
                    boxB.xLeft is None
                    or boxB.xCenter is None
                    or boxB.xRight is None
                    or boxB.yTop is None
                    or boxB.yCenter is None
                    or boxB.yBottom is None
                ):
                    raise ValueError(
                        "boxB LogicBox layout is not initialized before accessing coordinates."
                    )
                yA = boxA.yCenter
                yB = boxB.yCenter
                xA = boxA.xCenter
                xB = boxB.xCenter
            else:
                yA = boxA.yCenter
                yB = end[1]
                xA = boxA.xCenter
                xB = end[0]

            if yA == yB or xA == xB:
                path = [start, end]
            elif yA < yB:
                midY = (boxA.yTop + yB) / 2
                path = [start, (start[0], midY), (end[0], midY), end]
            else:
                midY = (boxA.yBottom + yB) / 2
                path = [start, (start[0], midY), (end[0], midY), end]
        else:
            path = [start, end]

        arrow = ArrowETC(
            ax=self.ax,
            path=path,
            arrow_head=arrow_head,
            arrow_head_at_tail=arrow_head_at_tail,
            arrow_head_width_multiplier=arrow_head_width_multiplier,
            arrow_head_length_multiplier=arrow_head_length_multiplier,
            shaft_width=shaft_width,
            ec=cast(str, ec),
            fc=cast(str, fc),
            lw=lw,
        )
        self.add_arrow(arrow)

    def add_bezier_connection(
        self,
        boxA: LogicBox,
        boxB: Union[LogicBox, Tuple[float, float]],
        style: Literal["smooth", "elbow", "s-curve"] = "smooth",
        control_points: Optional[list[tuple[float, float]]] = None,
        arrow_head: bool = True,
        arrow_head_width_multiplier: float = 2,
        arrow_head_length_multiplier: float = 1.5,
        shaft_width: float = 20,
        fill_connection: bool = True,
        fc: Optional[str] = None,
        ec: Optional[str] = None,
        lw: float = 0.7,
        sideA: Optional[
            Literal[
                "left",
                "topLeft",
                "top",
                "topRight",
                "right",
                "bottomRight",
                "bottom",
                "bottomLeft",
                "center",
            ]
        ] = None,
        sideB: Optional[
            Literal[
                "left",
                "topLeft",
                "top",
                "topRight",
                "right",
                "bottomRight",
                "bottom",
                "bottomLeft",
                "center",
            ]
        ] = None,
        butt_offset: float = 0,
        tip_offset: float = 0,
        n_bezier: int = 600,
    ) -> None:
        """
        Draw a curved arrow between two boxes or from a box to a coordinate using a smooth Bezier curve.

        This method builds a flowing, organic connection path ideal for reducing visual clutter in dense
        diagrams. Curvature is determined automatically using a spline fit through inferred or specified
        side coordinates.

        Parameters
        ----------
        boxA : LogicBox
            The source box where the arrow begins.
        boxB : LogicBox or tuple of float
            The destination - either another LogicBox or a fixed (x, y) coordinate.
        arrow_head : bool, optional
            Whether to include an arrowhead at the tip. Default is True.
        arrow_head_at_tail : bool, optional
            Whether to draw an arrowhead pointing backward from the tail. Default is False.
        arrow_head_width_multiplier : float, optional
            Width multiplier for the arrowhead. Default is 2.
        arrow_head_length_multiplier : float, optional
            Length multiplier for the arrowhead. Default is 1.5.
        shaft_width : float, optional
            Width of the arrow shaft in pixels. Default is 20.
        fill_connection : bool, optional
            Whether to fill the arrow polygon. Default is True.
        butt_offset : float, optional
            Distance to offset the base of the arrow away from the source box. Default is 0.
        tip_offset : float, optional
            Distance to offset the arrow tip away from the destination. Useful to avoid overlap. Default is 0.
        fc : str, optional
            Fill color of the arrow body. If None, inherits from destination box (if applicable).
        ec : str, optional
            Edge color of the arrow outline. If None, inherits from destination box.
        lw : float, optional
            Line width for the arrow outline. Default is 0.7.
        bezier_n : int, optional
            Number of points used to sample the Bezier curve. Higher values yield smoother curves. Default is 400.
        sideA : str, optional
            Edge or corner of `boxA` to connect from. Default is inferred from angle.
        sideB : str, optional
            Edge or corner of `boxB` to connect to. Ignored if `boxB` is a coordinate.

        Raises
        ------
        ValueError
            If box coordinates are uninitialized or the curve cannot be constructed.
        """

        if (
            boxA.xLeft is None
            or boxA.xCenter is None
            or boxA.xRight is None
            or boxA.yTop is None
            or boxA.yCenter is None
            or boxA.yBottom is None
        ):
            raise ValueError(
                "boxA LogicBox layout is not initialized before accessing coordinates."
            )

        if isinstance(boxB, LogicBox):
            if (
                boxB.xLeft is None
                or boxB.xCenter is None
                or boxB.xRight is None
                or boxB.yTop is None
                or boxB.yCenter is None
                or boxB.yBottom is None
            ):
                raise ValueError(
                    "boxB LogicBox layout is not initialized before accessing coordinates."
                )

            if fill_connection:
                if fc is None or fc == "fc":
                    fc = boxB.face_color
                elif fc == "ec":
                    fc = boxB.edge_color
            if ec is None or ec == "ec":
                ec = boxB.edge_color
            elif ec == "fc":
                ec = boxB.face_color

            if boxA.xCenter == boxB.xCenter and boxA.yCenter == boxB.yCenter:
                raise ValueError("Boxes cannot have the same position.")

            dx = boxB.xCenter - boxA.xCenter
            dy = boxB.yCenter - boxA.yCenter
        else:
            # boxB is a coordinate
            xB, yB = boxB
            dx = xB - boxA.xCenter
            dy = yB - boxA.yCenter

        theta = degrees(atan2(dy, dx))

        def auto_side(theta: float, for_A: bool) -> str:
            if -45 <= theta <= 45:
                return "right" if for_A else "left"
            elif 45 < theta <= 135:
                return "top" if for_A else "bottom"
            elif theta > 135 or theta < -135:
                return "left" if for_A else "right"
            else:
                return "bottom" if for_A else "top"

        resolved_sideA = sideA or auto_side(theta, for_A=True)
        resolved_sideB = sideB or auto_side(theta, for_A=False)

        start = self._get_side_coords(boxA, resolved_sideA)

        if isinstance(boxB, LogicBox):
            end = self._get_side_coords(boxB, resolved_sideB)
        else:
            end = boxB  # raw coordinate

        # Apply butt offset
        if butt_offset:
            match resolved_sideA:
                case "left":
                    start = (start[0] - butt_offset, start[1])
                case "right":
                    start = (start[0] + butt_offset, start[1])
                case "top":
                    start = (start[0], start[1] + butt_offset)
                case "bottom":
                    start = (start[0], start[1] - butt_offset)
                case "topLeft":
                    start = (start[0] - butt_offset, start[1] + butt_offset)
                case "topRight":
                    start = (start[0] + butt_offset, start[1] + butt_offset)
                case "bottomLeft":
                    start = (start[0] - butt_offset, start[1] - butt_offset)
                case "bottomRight":
                    start = (start[0] + butt_offset, start[1] - butt_offset)

        # Apply tip offset
        if tip_offset:
            match resolved_sideB:
                case "left":
                    end = (end[0] - tip_offset, end[1])
                case "right":
                    end = (end[0] + tip_offset, end[1])
                case "top":
                    end = (end[0], end[1] + tip_offset)
                case "bottom":
                    end = (end[0], end[1] - tip_offset)
                case "topLeft":
                    end = (end[0] - tip_offset, end[1] + tip_offset)
                case "topRight":
                    end = (end[0] + tip_offset, end[1] + tip_offset)
                case "bottomLeft":
                    end = (end[0] - tip_offset, end[1] - tip_offset)
                case "bottomRight":
                    end = (end[0] + tip_offset, end[1] - tip_offset)

        if control_points is not None:
            path = [start] + control_points + [end]
        else:
            match style:
                case "smooth":
                    cx = (start[0] + end[0]) / 2
                    cy = (start[1] + end[1]) / 2
                    normal = (-dy, dx)
                    mag = (dx**2 + dy**2) ** 0.5 or 1e-6
                    offset = 0.2 * mag
                    ctrl = (
                        cx + normal[0] / mag * offset,
                        cy + normal[1] / mag * offset,
                    )
                    path = [start, ctrl, end]
                case "elbow":
                    ctrl1 = (end[0], start[1])
                    ctrl2 = (end[0], end[1])
                    path = [start, ctrl1, ctrl2]
                case "s-curve":
                    d = 0.3 * (dx**2 + dy**2) ** 0.5
                    ctrl1 = (
                        (2 * start[0] + end[0]) / 3,
                        (2 * start[1] + end[1]) / 3 - d,
                    )
                    ctrl2 = (
                        (start[0] + 2 * end[0]) / 3,
                        (start[1] + 2 * end[1]) / 3 + d,
                    )
                    path = [start, ctrl1, ctrl2, end]
                case _:
                    raise ValueError(f"Unknown style '{style}'")

        arrow = ArrowETC(
            ax=self.ax,
            path=path,
            arrow_head=arrow_head,
            shaft_width=shaft_width,
            arrow_head_width_multiplier=arrow_head_width_multiplier,
            arrow_head_length_multiplier=arrow_head_length_multiplier,
            bezier=True,
            bezier_n=n_bezier,
            fc=cast(str, fc),
            ec=cast(str, ec),
            lw=lw,
        )
        self.add_arrow(arrow)

    def make_title(
        self,
        pos: Literal["left", "center", "right"] = "left",
        consider_box_x: bool = True,
        new_title: Optional[str] = None,
    ) -> None:
        """
        Place a title above the LogicTree figure with optional alignment and dynamic layout positioning.

        This method adds a title to the top of the logic tree using the tree's bounding boxes
        (if `consider_box_x=True`) or the full axis limits otherwise. Title alignment can be left,
        center, or right. You may also provide a new title string inline.

        Parameters
        ----------
        pos : {'left', 'center', 'right'}, optional
            Horizontal alignment of the title. Default is 'left'.
        consider_box_x : bool, optional
            If True (default), aligns the title based on LogicBox horizontal positions;
            otherwise uses `xlims` from the Axes.
        new_title : str, optional
            If given, replaces the current `self.title` string with this value before placing it.

        Raises
        ------
        ValueError
            If `pos` is invalid.
        ValueError
            If `self.title` is None when rendering the title.
        ValueError
            If any LogicBox lacks xLeft or xRight coordinates when `consider_box_x=True`.
        """
        if new_title is not None:
            self.title = new_title

        # if we are to ignore consider_box_x, use xlims to find the horizontal placement of title
        if not consider_box_x:
            if pos == "left":
                ha = "left"
                x = self.xlims[0]
            elif pos == "center":
                ha = "center"
                x = (self.xlims[1] + self.xlims[0]) / 2
            elif pos == "right":
                ha = "right"
                x = self.xlims[1]
            else:
                raise ValueError("pos must be one of ['left', 'center', 'right']")

        # if we are to consider_box_x
        else:
            xFarLeft = float("inf")
            xFarRight = float("-inf")
            for box in self.boxes:
                x_left = self.boxes[box].xLeft
                x_right = self.boxes[box].xRight

                if x_left is None or x_right is None:
                    raise ValueError(
                        f"LogicBox '{box}' layout not initialized: xLeft or xRight is None."
                    )

                if x_left < xFarLeft:
                    xFarLeft = x_left
                if x_right > xFarRight:
                    xFarRight = x_right
            if pos == "left":
                ha = "left"
                x = xFarLeft
            elif pos == "right":
                ha = "right"
                x = xFarRight
            elif pos == "center":
                ha = "center"
                x = (xFarRight + xFarLeft) / 2
            else:
                raise ValueError("pos must be one of ['left', 'center', 'right']")

        # finally make the title
        if self.title is None:
            raise ValueError("LogicTree.title is None. Please provide a title.")

        self.ax.text(
            x=x,
            y=self.ylims[1],
            s=self.title,
            va="top",
            ha=ha,
            fontdict=self.title_font_dict,
        )

    def save_as_png(
        self,
        file_name: str,
        dpi: int = 800,
        bbox_inches: Optional[Literal["tight"]] = "tight",
        content_padding: float = 0.0,
        aspect: Literal["auto", "equal"] = "equal",
    ) -> None:
        """
        Export the LogicTree diagram as a high-resolution PNG image.

        Saves the current figure with optional DPI, padding, and aspect ratio control.
        Useful for publication or presentation-quality outputs.

        Parameters
        ----------
        file_name : str
            Full path and name of the output PNG file.
        dpi : int, optional
            Resolution of the output image in dots per inch. Default is 800.
        bbox_inches : {'tight'} or None, optional
            Whether to automatically crop whitespace around the figure. Default is "tight".
        content_padding : float, optional
            Padding (in inches) around the figure content. Helps avoid clipped labels or boxes.
        aspect : {'auto', 'equal'}, optional
            Axes aspect ratio mode. Default is 'equal'.
        """
        self.ax.set_aspect(aspect)
        self.fig.savefig(
            file_name, dpi=dpi, bbox_inches=bbox_inches, pad_inches=content_padding
        )


__all__ = ["LogicTree"]
