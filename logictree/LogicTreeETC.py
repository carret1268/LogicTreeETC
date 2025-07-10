"""
This module defines the LogicTree class, which helps create logic tree diagrams
using LogicBox and ArrowETC objects. LogicTree manages adding labeled boxes,
connecting them with multi-segmented arrows, and rendering the final figure
with matplotlib. LaTeX rendering is supported for advanced text formatting.

Examples
--------
Here's a minimal example of how to build a logic tree diagram:

>>> from logictree.LogicTreeETC import LogicTree
>>> logic_tree = LogicTree(xlims=(0, 100), ylims=(0, 100), title="My Logic Tree")

# Add some boxes

>>> logic_tree.add_box(xpos=20, ypos=80, text="Start", box_name="Start", bbox_fc="black", bbox_ec="white", ha="center")
>>> logic_tree.add_box(xpos=20, ypos=50, text="Decision", box_name="Decision", bbox_fc="black", bbox_ec="white", ha="center")
>>> logic_tree.add_box(xpos=10, ypos=20, text="Option A", box_name="OptionA", bbox_fc="black", bbox_ec="green", ha="center")
>>> logic_tree.add_box(xpos=30, ypos=20, text="Option B", box_name="OptionB", bbox_fc="black", bbox_ec="red", ha="center")

# Connect boxes

>>> logic_tree.add_connection(boxA=logic_tree.boxes["Start"], boxB=logic_tree.boxes["Decision"], arrow_head=True, arrow_width=2)
>>> logic_tree.add_connection_biSplit(boxA=logic_tree.boxes["Decision"], boxB=logic_tree.boxes["OptionA"], boxC=logic_tree.boxes["OptionB"], arrow_head=True, arrow_width=2)

# Add a title and save

>>> logic_tree.make_title(pos="center")
>>> logic_tree.save_as_png("logic_tree_example.png", dpi=300)

Notes
-----
- If LaTeX rendering is enabled, packages such as bm, amsmath, soul, and relsize must be installed.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from math import atan2, degrees
from matplotlib.patches import BoxStyle
import matplotlib.pyplot as plt
from numpy import hypot

from arrowetc import ArrowETC
from .LogicBoxETC import LogicBox


class LogicTree:
    """
    Build logic tree diagrams by placing LogicBox objects and connecting them with ArrowETC arrows.

    LogicTree allows you to:
    - Add labeled boxes using `add_box()`
    - Connect boxes with straight or segmented arrows using `add_connection()` or `add_connection_biSplit()`
    - Style your logic tree with fonts, colors, figure titles, and LaTeX-rendered text.

    Parameters
    ----------
    fig_size : tuple of float, optional
        Size of the matplotlib figure (width, height). Default is (9, 9).
    xlims : tuple of float, optional
        Min and max x-axis limits. Default is (0, 100).
    ylims : tuple of float, optional
        Min and max y-axis limits. Default is (0, 100).
    fig_fc : str, optional
        Background color of the figure. Default is 'black'.
    title : str, optional
        Title to display on the figure. Can be updated later with `make_title()`.
    font_dict : dict, optional
        Font settings for general text in boxes. If None, a default font dict is used.
    font_dict_title : dict, optional
        Font settings for the figure title. If None, a default font dict is used.
    text_color : str, optional
        Override for font color in boxes.
    title_color : str, optional
        Override for font color of the figure title.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib figure instance.
    ax : matplotlib.axes.Axes
        The main matplotlib axes for drawing.
    boxes : dict
        Dictionary storing LogicBox objects keyed by their `box_name`.
    arrows : list[ArrowETC]
        List storing all ArrowETC objects.
    title : str
        The figure's title.
    xlims, ylims : tuple of float
        Axis limits used for positioning and layout.
    font_dict : dict
        Default font settings for text in boxes.
    title_font : dict
        Font settings for the figure title.
    latex_ul_depth, latex_ul_width : str
        Settings for LaTeX underlining (depth and width).
    """

    def __init__(
        self,
        fig_size: Tuple[float, float] = (9, 9),
        xlims: Tuple[float, float] = (0, 100),
        ylims: Tuple[float, float] = (0, 100),
        fig_fc: str = "black",
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

        # Font dictionary for title
        if font_dict_title is None:
            font_dict_title = dict(
                fontname="Times New Roman", fontsize=34, color="white"
            )
        if title_color is not None:
            font_dict_title["color"] = title_color
        self.title_font = font_dict_title

        # Default font dictionary for boxes
        if font_dict is None:
            font_dict = {
                "fontname": "Times New Roman",
                "fontsize": 15,
                "color": "white",
            }
        if text_color is not None:
            font_dict["color"] = text_color
        self.font_dict = font_dict

        # Underlining options for LaTeX rendering
        self.latex_ul_depth = "1pt"
        self.latex_ul_width = "1pt"

        # Generate figure and axes
        fig, ax = plt.subplots(figsize=fig_size, frameon=True, facecolor=fig_fc)
        ax.set_xlim(xlims[0], xlims[1])
        ax.set_ylim(ylims[0], ylims[1])
        ax.axis("off")
        fig.canvas.draw_idle()

        self.fig = fig
        self.ax = ax

    def _get_pathsForBi_left_then_right(
        self,
        Ax2: float,
        Ay2: float,
        left_box: LogicBox,
        right_box: LogicBox,
        tip_offset: float,
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Generate the paths for a bifurcating connection with left and right branches.

        Used internally by `add_connection_biSplit()` to compute the two three-segment paths
        from a common parent point to two child boxes.

        Parameters
        ----------
        Ax2, Ay2 : float
            Starting point of the split (usually end of vertical line from boxA).
        left_box, right_box : LogicBox
            Boxes to connect left and right paths to. left_box must be left of right_box.
        tip_offset : float
            Vertical offset for the arrow tips.

        Returns
        -------
        tuple of list of tuple
            Paths for the left and right connections, each a list of (x, y) points.

        Raises
        ------
        ValueError
            If `yTop`, `yBottom`, xCenter or `yCenter` are None because the layout has
            not yet been initialized. This is for either left_box or right_box.
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
        Add a LogicBox to the LogicTree with specified text and styling.

        Parameters
        ----------
        xpos, ypos : float
            Coordinates for box placement.
        text : str
            Text displayed inside the box. Supports LaTeX if `use_tex_rendering=True`.
        box_name : str
            Unique identifier for the LogicBox; used to reference the box in connections.
        bbox_fc, bbox_ec : str
            Face and edge colors of the box. RGBA tuples allowed for transparency.
        font_dict : dict, optional
            Font properties. Defaults to LogicTree's font_dict.
        text_color : str, optional
            Override for the text color.
        fs : int, optional
            Override for font size.
        font_weight : str, optional
            Font weight (e.g., 'normal', 'bold').
        lw : float, optional
            Line width of the box's edge. Default is 1.6.
        bbox_style : BoxStyle, optional
            Matplotlib BoxStyle object for box shape and padding. Default is 'Round'.
        va : str, optional
            Vertical alignment: 'top', 'center', or 'bottom'. Default is 'center'.
        ha : str, optional
            Horizontal alignment: 'left', 'center', or 'right'. Default is 'right'.
        use_tex_rendering : bool, optional
            Enable LaTeX text rendering.
        ul : bool, optional
            Underline text if LaTeX rendering is enabled.
        ul_depth_width : tuple of (float, float), optional
            Underline depth and width for LaTeX.
        angle : float, optional
            Angle in degrees to rotate your box. Rotations are about the center of the box.

        Returns
        -------
        LogicBox
            The new LogicBox object.

        Raises
        ------
        ValueError
            If `box_name` is already used.
        ValueError
            If the rendered text object has no bounding box patch.
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
                    + f"{ul_depth_width[0]}pt"
                    + r"}{"
                    + f"{ul_depth_width[1]}pt"
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
        Add a pre-constructed ArrowETC object to the LogicTree canvas.

        This method allows advanced users to configure complex arrows externally
        and then attach them to the logic tree. The arrow is stored for later access
        and drawn using the existing matplotlib axes.

        Parameters
        ----------
        arrow : ArrowETC
            A ready-to-render ArrowETC object.
        fill_arrow : bool, optional
            If True, the arrow will be filled with its ArrowETC.fc attribute. If False,
            there will be no fill. Default is True.

        Raises
        ------
        ValueError
            If the ArrowETC object is missing a valid path.
        """
        if not arrow.path or len(arrow.path) < 2:
            raise ValueError("ArrowETC must have a path with at least two points.")

        self.arrows.append(arrow)
        self.ax = arrow.draw_to_ax(self.ax, fill_arrow=fill_arrow)

    def add_arrow_between(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        arrow_width: float = 0.5,
        arrow_head: bool = True,
        tip_offset: float = 0.0,
        butt_offset: float = 0.0,
        facecolor: str = "black",
        edgecolor: str = "black",
        zorder: float = 1.0,
        linewidth: float = 1.0,
        linestyle: str = "-",
        fill_arrow: bool = True,
    ) -> None:
        """
        Add a quick arrow between two points using default ArrowETC settings.

        This wrapper creates a new ArrowETC object from two coordinates and adds it
        to the logic tree. Useful for free-floating annotations or diagram embellishment.

        Parameters
        ----------
        start : tuple of float
            (x, y) coordinates for the base of the arrow.
        end : tuple of float
            (x, y) coordinates for the tip of the arrow.
        arrow_width : float, optional
            Width of the arrow shaft. Default is 0.5.
        arrow_head : bool, optional
            Whether to draw an arrowhead. Default is True.
        tip_offset : float, optional
            Distance to offset the arrow tip (e.g., to avoid overlap). Default is 0.0.
        butt_offset : float, optional
            Distance to offset the arrow base. Default is 0.0.
        facecolor : str, optional
            Fill color of the arrow. Default is "black".
        edgecolor : str, optional
            Outline color of the arrow. Default is "black".
        zorder : float, optional
            Drawing layer priority. Default is 1.0.
        linewidth : float, optional
            Outline thickness. Default is 1.0.
        linestyle : str, optional
            Line style (e.g., "-", "--"). Default is "-".
        fill_arrow : bool, optional
            If True, the arrow will be filled with its ArrowETC.fc attribute. If False,
            there will be no fill. Default is True.

        Raises
        ------
        ValueError
            If start and end points are the same.
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
            path=path,
            arrow_width=arrow_width,
            arrow_head=arrow_head,
            fc=facecolor,
            ec=edgecolor,
            zorder=zorder,
            lw=linewidth,
            ls=linestyle,
        )
        self.add_arrow(arrow, fill_arrow=fill_arrow)

    def add_connection_biSplit(
        self,
        boxA: LogicBox,
        boxB: LogicBox,
        boxC: LogicBox,
        arrow_head: bool = True,
        arrow_width: float = 0.5,
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
        Create a bifurcating arrow connection from a parent LogicBox (`boxA`) to two child boxes (`boxB` and `boxC`),
        using a stem that splits into two branching segments. Labels can optionally be placed along the left and right
        arrow branches with customizable position and styling.

        This method automatically infers the orientation of the bifurcation (downward or upward) based on the vertical
        positions of the input boxes. It also determines the left/right ordering based on horizontal positions. Arrow
        styling (head, width, fill, color) and label appearance are all configurable.

        Parameters
        ----------
        boxA : LogicBox
            The parent box from which the bifurcation begins. Must be clearly vertically above or below both child boxes.
        boxB : LogicBox
            One of the two child boxes. The method will automatically determine whether this is the left or right branch
            based on xCenter.
        boxC : LogicBox
            The other child box. Must be on the same vertical side of `boxA` as `boxB`.
        arrow_head : bool, optional
            If True (default), draws arrowheads at the ends of the left and right branches.
        arrow_width : float, optional
            Width of the arrows in data units. Default is 0.5.
        fill_connection : bool, optional
            Whether to fill the arrows with face color (True by default). If False, only outlines are drawn.
        fc_A : str, optional
            Face color of the vertical stem arrow from boxA. If None, defaults to `boxA.face_color`.
            If "ec", uses `boxA.edge_color`.
        ec_A : str, optional
            Edge color of the vertical stem arrow from boxA. If None, defaults to `boxA.edge_color`.
            If "fc", uses `boxA.face_color`.
        fc_B : str, optional
            Face color of the arrow branch toward boxB. If None, defaults to `boxB.face_color`.
            If "ec", uses `boxB.edge_color`.
        ec_B : str, optional
            Edge color of the arrow branch toward boxB. If None, defaults to `boxB.edge_color`.
            If "fc", uses `boxB.face_color`.
        fc_C : str, optional
            Face color of the arrow branch toward boxC. If None, defaults to `boxC.face_color`.
            If "ec", uses `boxC.edge_color`.
        ec_C : str, optional
            Edge color of the arrow branch toward boxC. If None, defaults to `boxC.edge_color`.
            If "fc", uses `boxC.face_color`.
        lw : float, optional
            Line width of the arrow edges. Default is 0.5.
        butt_offset : float, optional
            Distance to offset the base of the vertical stem away from the parent boxA. Prevents visual overlap.
            Default is 0.
        tip_offset : float, optional
            Distance to offset the tips of the arrows from boxB and boxC. Prevents overlap with box edges.
            Default is 0.
        textLeft : str, optional
            Optional text label to display above or below the arrow leading to the left box. Centered along the shaft.
        textRight : str, optional
            Optional text label to display above or below the arrow leading to the right box. Centered along the shaft.
        textLeftOffset : {'above', 'below'}, optional
            Whether to place the `textLeft` label above or below the arrow shaft. Default is 'above'.
        textRightOffset : {'above', 'below'}, optional
            Whether to place the `textRight` label above or below the arrow shaft. Default is 'above'.
        text_kwargs : dict, optional
            Dictionary of matplotlib-compatible text styling options.

            Keys may include:

                - 'fontsize' (int): font size (default: 12)
                - 'fontname' (str): font family (default: 'sans-serif')
                - 'color' (str): font color (default: 'white')

        Raises
        ------
        ValueError
            If any required coordinates (`xLeft`, `xCenter`, `xRight`, `yTop`, `yCenter`, `yBottom`) of any input box
            are uninitialized (i.e., None).
        ValueError
            If `boxA` is not clearly vertically above or below both `boxB` and `boxC`.

        Notes
        -----
        This function is intended for use with properly initialized `LogicBox` instances, such as those added via
        LogicTree's `add_box()` method. It is useful for visualizing binary decision splits in flow diagrams or logic trees.
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
            xm = (x1 + x2) / 2 + (arrow_width / 2 if x1 < x2 else -arrow_width / 2)
            ym = (
                y1 + arrow_width * 0.95
                if offset == "above"
                else y1 - arrow_width * 0.95
            )
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
            path=path_vertical,
            arrow_head=False,
            arrow_width=arrow_width,
            ec=ec_A,
            fc=fc_A,
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
            fc: Optional[str],
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
                path=path,
                arrow_head=arrow_head,
                arrow_width=arrow_width,
                ec=ec,
                fc=fc,
                lw=lw,
                close_butt=False,
                zorder=1000,
            )
            self.add_arrow(arrow)
            annotate_segment(label, path, label_offset)

        # Draw left
        if left_box is boxB:
            draw_branch(path_left, ec_B, fc_B, lw, textLeft, textLeftOffset)
            draw_branch(path_right, ec_C, fc_C, lw, textRight, textRightOffset)
        else:
            draw_branch(path_left, ec_C, fc_C, lw, textLeft, textLeftOffset)
            draw_branch(path_right, ec_B, fc_B, lw, textRight, textRightOffset)

    def _get_side_coords(
        self, box: LogicBox, side: str, offset: float = 0.0
    ) -> tuple[float, float]:
        """
        Return coordinates on a box edge or corner, optionally nudged outward.

        Parameters
        ----------
        box : LogicBox
            The box to extract a coordinate from.
        side : str
            One of 'left', 'right', 'top', 'bottom', 'center', or a corner like 'topLeft'.
        offset : float, optional
            Distance to offset the point outward in the direction of connection.
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
        arrow_width: float = 0.5,
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
        Draw a straight or segmented arrow connection between two LogicBoxes.

        The arrow can be automatically routed or user-directed by specifying entry and
        exit sides (edges or corners). Optional offsets ensure that arrows avoid overlap
        with box borders.

        Parameters
        ----------
        boxA : LogicBox
            The source LogicBox from which the arrow originates.
        boxB : LogicBox | Tuple[float, float]
            The target LogicBox, or the exact coordinates (x, y) to which the arrow points. If you want
            padding between the point (or LogicBox edge), update the `tip_offset` parameter.
        segmented : bool, optional
            If True, non-aligned boxes will be connected with segmented (elbow) arrows.
            If False, a direct connection is used. Default is False.
        arrow_head : bool, optional
            Whether to draw an arrowhead pointing at `boxB`. Default is True.
        arrow_width : float, optional
            Width of the arrow shaft in data units. Default is 0.5.
        fill_connection : bool, optional
            Whether to fill the arrow body with color. Default is True.
        butt_offset : float, optional
            Distance to offset the starting point of the arrow (away from `boxA`) in the
            direction of the connection. Default is 0.
        tip_offset : float, optional
            Distance to offset the tip of the arrow (away from `boxB`) to avoid overlap.
            Default is 0.
        fc : str, optional
            Fill color of the arrow body. If None, uses `boxB`'s face color. If 'ec', uses `boxB`'s edge color.
        ec : str, optional
            Edge color (outline) of the arrow. If None, uses `boxB`'s edge color. If 'fc', uses face color.
        lw : float, optional
            Line width of the arrow outline. Default is 0.7.
        sideA : {'left', 'topLeft', 'right', 'topRight', 'top', 'bottomRight', 'bottom', 'bottomLeft' 'center'}, optional
            The side or corner of `boxA` where the arrow starts. Options include:
            'left', 'right', 'top', 'bottom', 'center', 'topLeft', 'topRight', 'bottomLeft', 'bottomRight'.
            If not provided, it is inferred automatically based on box positions.
        sideB : {'left', 'topLeft', 'right', 'topRight', 'top', 'bottomRight', 'bottom', 'bottomLeft' 'center'}, optional
            The side or corner of `boxB` where the arrow ends. Same options as `sideA`.

        Raises
        ------
        ValueError
            If boxA and boxB have the same center position.
        ValueError
            If required box coordinates are not initialized.
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

        if segmented and isinstance(boxB, LogicBox):
            # need another type check to appease mypy
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
            if boxA.yCenter == boxB.yCenter:
                path = [start, end]
            elif boxA.yCenter < boxB.yCenter:
                if boxA.xCenter == boxB.xCenter:
                    path = [start, end]
                else:
                    midY = (boxA.yTop + boxB.yBottom) / 2
                    path = [start, (start[0], midY), (end[0], midY), end]
            else:
                if boxA.xCenter == boxB.xCenter:
                    path = [start, end]
                else:
                    midY = (boxA.yBottom + boxB.yTop) / 2
                    path = [start, (start[0], midY), (end[0], midY), end]
        else:
            path = [start, end]

        arrow = ArrowETC(
            path=path,
            arrow_head=arrow_head,
            arrow_width=arrow_width,
            ec=ec,
            fc=fc,
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
        arrow_width: float = 0.5,
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
        Draw a curved Bezier arrow connection between two LogicBoxes or from a LogicBox to a fixed point.

        The arrow path may be automatically shaped using preset styles or manually customized with control points.
        You can specify exact exit and entry sides (edges or corners) of the boxes, and apply optional offsets to
        avoid overlap with box borders.

        Parameters
        ----------
        boxA : LogicBox
            The source LogicBox from which the arrow originates.
        boxB : LogicBox | Tuple[float, float]
            The target LogicBox, or the exact coordinates (x, y) to which the arrow points. If you want
            padding between the point (or LogicBox edge), update the `tip_offset` parameter.
        style : {'smooth', 'elbow', 's-curve'}, optional
            If `control_points` is not provided, determines the default Bezier shape:
            - 'smooth': a gently arced curve perpendicular to the connection line
            - 'elbow': a right-angle step shape
            - 's-curve': a symmetric double bend for greater visual separation
        control_points : list of (float, float), optional
            Explicit control points for the Bezier curve. Overrides `style` if provided.
        arrow_head : bool, optional
            Whether to draw an arrowhead pointing at `boxB`. Default is True.
        arrow_width : float, optional
            Width of the arrow shaft in data units. Default is 0.5.
        fill_connection : bool, optional
            Whether to fill the arrow body with color. Default is True.
        butt_offset : float, optional
            Distance to offset the starting point of the arrow (away from `boxA`) in the
            direction of the connection. Default is 0.
        tip_offset : float, optional
            Distance to offset the tip of the arrow (away from `boxB`) to avoid overlap.
            Default is 0.
        fc : str, optional
            Fill color of the arrow body. If None, uses `boxB`'s face color. If 'ec', uses `boxB`'s edge color.
        ec : str, optional
            Edge color (outline) of the arrow. If None, uses `boxB`'s edge color. If 'fc', uses face color.
        lw : float, optional
            Line width of the arrow outline. Default is 0.7.
        sideA : {'left', 'topLeft', 'right', 'topRight', 'top', 'bottomRight', 'bottom', 'bottomLeft', 'center'}, optional
            The side or corner of `boxA` where the arrow starts. If not provided, inferred automatically.
        sideB : {'left', 'topLeft', 'right', 'topRight', 'top', 'bottomRight', 'bottom', 'bottomLeft', 'center'}, optional
            The side or corner of `boxB` where the arrow ends. Ignored if `boxB` is a coordinate.
        n_bezier : int, optional
            Number of interpolation points used to render the Bezier curve. Increase if your arrowhead looks distorted.
            Default is 600.

        Raises
        ------
        ValueError
            If boxA and boxB have the same center position.
        ValueError
            If required box coordinates are not initialized.
        ValueError
            If style is unknown and control_points is not provided.
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
            path=path,
            arrow_head=arrow_head,
            arrow_width=arrow_width,
            bezier=True,
            bezier_n=n_bezier,
            fc=fc,
            ec=ec,
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
        Place a title on the LogicTree figure.

        Parameters
        ----------
        pos : str, optional
            Horizontal alignment of the title; one of ['left', 'center', 'right']. Default is 'left'.
        consider_box_x : bool, optional
            If True, aligns the title based on box positions; otherwise aligns using xlims. Default is True.
        new_title : str, optional
            If provided, updates the LogicTree's title before placing it.

        Raises
        ------
        ValueError
            If `pos` is not one of ['left', 'center', 'right'].
        ValueError
            If `self.title` is None when attempting to create the title.
        ValueError
            If any LogicBox in the layout is missing `xLeft` or `xRight` coordinates (if `consider_box_x=True`).

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
            fontdict=self.title_font,
        )

    def save_as_png(
        self, file_name: str, dpi: int = 800, content_padding: float = 0.0
    ) -> None:
        """
        Save the LogicTree diagram as a PNG file.

        Parameters
        ----------
        file_name : str
            Path and name of the output PNG file.
        dpi : int, optional
            Resolution of the output image. Default is 800.
        content_padding : float, optional
            The padding in inches to place around the content. This can be helpful
            to prevent your boxes from touching the edge of the figures.
        """
        self.ax.set_aspect("equal")
        self.fig.savefig(
            file_name, dpi=dpi, bbox_inches="tight", pad_inches=content_padding
        )


__all__ = ["LogicTree"]
