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
- ArrowETC connectors are limited to straight lines with right-angle bends (90-degree only).
- If LaTeX rendering is enabled, packages such as bm, amsmath, soul, and relsize must be installed.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple

from matplotlib.patches import BoxStyle
import matplotlib.pyplot as plt

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
        fig_fc: str = 'black',
        title: Optional[str] = None,
        font_dict: Optional[Dict[str, Any]] = None,
        font_dict_title: Optional[Dict[str, Any]] = None,
        text_color: Optional[str] = None,
        title_color: Optional[str] = None,
    ) -> None:
        self.boxes = {}  
        self.title = title
        self.xlims = xlims
        self.ylims = ylims
        
        # Font dictionary for title
        if font_dict_title is None:
            font_dict_title = dict(fontname='Times New Roman', fontsize=34, color='white')
        if title_color is not None:
            font_dict_title['color'] = title_color
        self.title_font = font_dict_title
        
        # Default font dictionary for boxes
        if font_dict is None:
            font_dict = {
                'fontname': 'Times New Roman',
                'fontsize': 15,
                'color': 'white'
            }
        if text_color is not None:
            font_dict['color'] = text_color
        self.font_dict = font_dict
        
        # Underlining options for LaTeX rendering
        self.latex_ul_depth = '1pt'
        self.latex_ul_width = '1pt'
        
        # Generate figure and axes
        fig, ax = plt.subplots(figsize=fig_size, frameon=True, facecolor=fig_fc)
        ax.set_xlim(xlims[0], xlims[1])
        ax.set_ylim(ylims[0], ylims[1])
        ax.axis('off')
        fig.canvas.draw_idle()
        
        self.fig = fig
        self.ax = ax
        
    def _get_pathsForBi_left_then_right(
        self, 
        Ax2: float, 
        Ay2: float, 
        left_box: LogicBox, 
        right_box: LogicBox, 
        tip_offset: float
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
        """
        # create the leftward arrow
        Lx1 = Ax2
        Ly1 = Ay2
        Lx2 = left_box.x_center
        Ly2 = Ly1
        Lx3 = Lx2
        Ly3 = left_box.yTop + tip_offset if Ay2 > left_box.y_center else left_box.yBottom - tip_offset
        
        # create the rightward arrow
        Rx1 = Ax2
        Ry1 = Ay2
        Rx2 = right_box.x_center
        Ry2 = Ry1
        Rx3 = Rx2
        Ry3 = right_box.yTop + tip_offset if Ay2 > right_box.y_center else right_box.yBottom - tip_offset

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
        bbox_style: BoxStyle = BoxStyle('Round', pad=0.6), 
        va: Literal['top', 'center', 'bottom'] = 'center', 
        ha: Literal['left', 'center', 'right'] = 'right', 
        use_tex_rendering: bool = False, 
        ul: bool = False, 
        ul_depth_width: Optional[Tuple[float, float]] = None,
        angle: float = 0.0
    ) -> None:
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

        Raises
        ------
        ValueError
            If `box_name` is already used.
        """
        if box_name in self.boxes:
            raise ValueError(f"Box name '{box_name}' already exists. Please use a unique name.")

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
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "cm",
                'text.latex.preamble': latex_preamble
            })
        else:
            plt.rcParams.update({"text.usetex": False})
            
        # set fontidct of not provided
        if font_dict is None:
            font_dict = self.font_dict.copy()
        # if specific text color is specified, change it in font_dict
        if text_color is not None:
            font_dict['color'] = text_color
        # if specific fontsize is specified, change it in font_dict
        if fs is not None:
            font_dict['fontsize'] = fs
        # if weight is specified, change it in font_dict
        if font_weight is not None:
            font_dict['weight'] = font_weight
            
        # create a logicBox object which stores all of this information
        myBox = LogicBox(
            xpos=xpos, ypos=ypos, text=text, box_name=box_name, 
            bbox_fc=bbox_fc, bbox_ec=bbox_ec, bbox_style=bbox_style, 
            font_dict=font_dict, va=va, ha=ha, lw=lw, angle=angle
        )
        
        # add latex commands to text for underlining 
        if use_tex_rendering and (ul or ul_depth_width is not None):
            text_str = r'\ul{' + myBox.text + r'}'
            # if underlining parameters are set, add the command to change them
            if ul_depth_width is not None:
                text_str = f'\\setul{{{ul_depth_width[0]}}}{{{ul_depth_width[1]}}}' + text_str
        else:
            text_str = myBox.text
        # make the text
        txt = self.ax.text(
            x=myBox.x, y=myBox.y, s=text_str, fontdict=myBox.font_dict,
            bbox=myBox.style, va=myBox.va, ha=myBox.ha, rotation=myBox.angle
        )
        
        # get our box's dims and edge positions to store in myBox object
        bbox = plt.gca().transData.inverted().transform_bbox(
            txt.get_window_extent(renderer=self.fig.canvas.get_renderer())
        ) # coords of text
        wpad = txt.get_bbox_patch().get_extents().width # pad size for width
        hpad = txt.get_bbox_patch().get_extents().height # pad size for height
        myBox.xLeft, myBox.xRight = bbox.x0 - wpad, bbox.x1 + wpad
        myBox.yBottom, myBox.yTop = bbox.y0 - hpad, bbox.y1 + wpad
        myBox.width = myBox.xRight - myBox.xLeft
        myBox.height = myBox.yTop - myBox.yBottom
        myBox.x_center = myBox.xRight - myBox.width/2
        myBox.y_center = myBox.yTop - myBox.height/2
        
        # store box in our LogicTree object's box dictionary to grab dimensions when needed
        self.boxes[myBox.name] = myBox
        
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
        tip_offset: float = 0
    ) -> None:
        """
        Create a bifurcating connection from boxA to both boxB and boxC.

        Parameters
        ----------
        boxA, boxB, boxC : LogicBox
            Parent and child boxes for the connection. boxA must be above or below both boxB and boxC.
        arrow_head : bool, optional
            If True, draws arrowheads at boxB and boxC.
        arrow_width : float, optional
            Width of the arrows in data coordinates.
        fill_connection : bool, optional
            Whether to fill the arrows with color.
        fc_A, ec_A, fc_B, ec_B, fc_C, ec_C : str, optional
            Fill and edge colors for the three parts of the connection.
        lw : float, optional
            Line width of the arrows.
        butt_offset, tip_offset : float, optional
            Offsets for avoiding overlap at the base or tips of the arrows.

        Raises
        ------
        ValueError
            If boxA is not clearly above or below both boxB and boxC.
        """
        # do stylizing stuff
        if fill_connection:
            # option for face color to equal edgecolor
            if fc_A == 'ec':
                fc_A = boxA.edge_color
            # if no option specified, face color of arrow is same as face color of box
            elif fc_A is None:
                fc_A = boxA.face_color
            # option for face color to equal edgecolor
            if fc_B == 'ec':
                fc_B = boxB.edge_color
            # if no option specified, face color of arrow is same as face color of box
            elif fc_B is None:
                fc_B = boxB.face_color
            # option for face color to equal edgecolor
            if fc_C == 'ec':
                fc_C = boxC.edge_color
            # if no option specified, face color of arrow is same as face color of box
            elif fc_C is None:
                fc_C = boxC.face_color
        
        if ec_A =='fc':
            ec_A = boxA.face_color
        elif ec_A is None:
            ec_A = boxA.edge_color
        if ec_B =='fc':
            ec_B = boxB.face_color
        elif ec_B is None:
            ec_B = boxB.edge_color
        if ec_C =='fc':
            ec_C = boxC.face_color
        elif ec_C is None:
            ec_C = boxC.edge_color
        
        # first take the case of boxA being above boxes B and C
        if (boxA.y_center > boxB.y_center) and (boxA.y_center > boxC.y_center):
            # create the downward line from BoxA to center
            Ax1 = boxA.x_center
            Ay1 = boxA.yBottom - butt_offset
            Ax2 = Ax1
            # take it down to the midpoint of boxA and the highest of boxes B and C
            if boxB.yTop >= boxC.yTop:
                Ay2 = (Ay1 + boxB.yTop)/2
            else:
                Ay2 = (Ay1 + boxC.yTop)/2
            # set path for downward segment
            path = [(Ax1, Ay1), (Ax2, Ay2)]
            arrow = ArrowETC(path=path, arrow_head=False, arrow_width=arrow_width)

            # get vertices
            x = arrow.x_vertices[:-1]
            y = arrow.y_vertices[:-1]
            self.ax.plot(x, y, color=ec_A, lw=0.01)
            # fill arrow if desired
            if fill_connection:
                self.ax.fill(x, y, color=fc_A)
                
            # take the case that boxB is to the left of boxC 
            if boxB.x_center < boxC.x_center:
                # get paths
                path_left, path_right = self._get_pathsForBi_left_then_right(Ax2, Ay2, left_box=boxB, \
                                                                             right_box=boxC, tip_offset=tip_offset)
                # make left arrow
                arrow = ArrowETC(path=path_left, arrow_head=arrow_head, arrow_width=arrow_width)
                # get vertices
                x = arrow.x_vertices[:-1]
                y = arrow.y_vertices[:-1]
                self.ax.plot(x, y, color=ec_B, lw=lw)
                # fill arrow if desired
                if fill_connection:
                    self.ax.fill(x, y, color=fc_B)
                
                # make right arrow
                arrow = ArrowETC(path=path_right, arrow_head=arrow_head, arrow_width=arrow_width)
                # get vertices
                x = arrow.x_vertices[:-1]
                y = arrow.y_vertices[:-1]
                self.ax.plot(x, y, color=ec_C, lw=lw)
                # fill arrow if desired
                if fill_connection:
                    self.ax.fill(x, y, color=fc_C)
                    
            # take the case that boxB is to the right of boxC 
            elif boxB.x_center > boxC.x_center:
                # get paths
                path_left, path_right = self._get_pathsForBi_left_then_right(Ax2, Ay2, left_box=boxC, right_box=boxB, tip_offset=tip_offset)
                # make left arrow
                arrow = ArrowETC(path=path_left, arrow_head=arrow_head, arrow_width=arrow_width)
                # get vertices
                x = arrow.x_vertices[:-1]
                y = arrow.y_vertices[:-1]
                self.ax.plot(x, y, color=ec_C, lw=lw)
                # fill arrow if desired
                if fill_connection:
                    self.ax.fill(x, y, color=fc_C)
                
                # make right arrow
                arrow = ArrowETC(path=path_right, arrow_head=arrow_head, arrow_width=arrow_width)
                # get vertices
                x = arrow.x_vertices[:-1]
                y = arrow.y_vertices[:-1]
                self.ax.plot(x, y, color=ec_B, lw=lw)
                # fill arrow if desired
                if fill_connection:
                    self.ax.fill(x, y, color=fc_B)
                
        # now take the case of boxA being below boxes B and C
        elif (boxA.y_center < boxB.y_center) and (boxA.y_center < boxC.y_center):
            # create the upward line from BoxA to center
            Ax1 = boxA.x_center
            Ay1 = boxA.yTop + butt_offset
            Ax2 = Ax1
            # take it down to the midpoint of boxA and the highest of boxes B and C
            if boxB.yBottom <= boxC.yBottom:
                Ay2 = (Ay1 + boxB.yBottom)/2
            else:
                Ay2 = (Ay1 + boxC.yBottom)/2
            # set path for downward segment
            path = [(Ax1, Ay1), (Ax2, Ay2)]
            arrow = ArrowETC(path=path, arrow_head=arrow_head, arrow_width=arrow_width)

            # get vertices
            x = arrow.x_vertices[:-1]
            y = arrow.y_vertices[:-1]
            self.ax.plot(x, y, color=ec_A, lw=0.01)
            # fill arrow if desired
            if fill_connection:
                self.ax.fill(x, y, color=fc_A)

            # take the case that boxB is to the left of boxC 
            if boxB.x_center < boxC.x_center:
                # get paths
                path_left, path_right = self._get_pathsForBi_left_then_right(Ax2, Ay2, left_box=boxB, \
                                                                             right_box=boxC, tip_offset=tip_offset)
                # make left arrow
                arrow = ArrowETC(path=path_left, arrow_head=arrow_head, arrow_width=arrow_width)
                # get vertices
                x = arrow.x_vertices[:-1]
                y = arrow.y_vertices[:-1]
                self.ax.plot(x, y, color=ec_B, lw=lw)
                # fill arrow if desired
                if fill_connection:
                    self.ax.fill(x, y, color=fc_B)
                
                # make right arrow
                arrow = ArrowETC(path=path_right, arrow_head=arrow_head, arrow_width=arrow_width)
                # get vertices
                x = arrow.x_vertices[:-1]
                y = arrow.y_vertices[:-1]
                self.ax.plot(x, y, color=ec_C, lw=lw)
                # fill arrow if desired
                if fill_connection:
                    self.ax.fill(x, y, color=fc_C)
                    
            # take the case that boxB is to the right of boxC 
            elif boxB.x_center > boxC.x_center:
                # get paths
                path_left, path_right = self._get_pathsForBi_left_then_right(Ax2, Ay2, left_box=boxC, right_box=boxB, tip_offset=tip_offset)
                # make left arrow
                arrow = ArrowETC(path=path_left, arrow_head=arrow_head, arrow_width=arrow_width)
                # get vertices
                x = arrow.x_vertices[:-1]
                y = arrow.y_vertices[:-1]
                self.ax.plot(x, y, color=ec_C, lw=lw)
                # fill arrow if desired
                if fill_connection:
                    self.ax.fill(x, y, color=fc_C)
                
                # make right arrow
                arrow = ArrowETC(path=path_right, arrow_head=arrow_head, arrow_width=arrow_width)
                # get vertices
                x = arrow.x_vertices[:-1]
                y = arrow.y_vertices[:-1]
                self.ax.plot(x, y, color=ec_B, lw=lw)
                # fill arrow if desired
                if fill_connection:
                    self.ax.fill(x, y, color=fc_B)
            
    def add_connection(
        self, 
        boxA: LogicBox, 
        boxB: LogicBox, 
        arrow_head: bool = True, 
        arrow_width: float = 0.5, 
        fill_connection: bool= True,
        butt_offset: float = 0, 
        tip_offset: float = 0, 
        fc: Optional[str] = None, 
        ec: Optional[str] = None, 
        lw: float = 0.7
    ) -> None:
        """
        Create a straight or segmented connection (arrow) from boxA to boxB.

        Parameters
        ----------
        boxA, boxB : LogicBox
            Source and target boxes for the connection.
        arrow_head : bool, optional
            If True, draws an arrowhead at boxB.
        arrow_width : float, optional
            Width of the arrow in data coordinates. Default is 0.5.
        fill_connection : bool, optional
            Whether to fill the arrow with color.
        butt_offset : float, optional
            Offset of the arrow's butt to avoid overlapping with boxA.
        tip_offset : float, optional
            Offset of the arrow's tip to avoid overlapping with boxB.
        fc : str, optional
            Fill color; if None, uses boxB's face color. If 'ec', uses boxB's edge color.
        ec : str, optional
            Edge color; if None, uses boxB's edge color. If 'fc', uses boxB's face color.
        lw : float, optional
            Line width of the arrow edges.

        Raises
        ------
        ValueError
            If boxes are not aligned in the same row or column and cannot be connected directly.
        """
        # handle colors
        if fill_connection:
            # if no fc is chosen, take the fc of connection to be fc of boxB
            if fc is None or fc == 'fc':
                fc = boxB.face_color
            elif fc == 'ec':
                fc = boxB.edge_color
        # if no ec is chosen, take ec of connection to be ec of boxB
        if ec is None or ec == 'ec':
            ec = boxB.edge_color
        elif ec == 'fc':
            ec = boxB.face_color
            
        # first case, boxA and boxB are on the same row
        if boxA.y_center == boxB.y_center:
            # boxA is to the left of boxB
            if boxA.x_center < boxB.x_center:
                Ax, Ay = boxA.xRight + butt_offset, boxA.y_center
                Bx, By = boxB.xLeft - tip_offset, boxB.y_center
            # boxA is to the right of boxB
            elif boxA.x_center > boxB.x_center:
                Ax, Ay = boxA.xLeft - butt_offset, boxA.y_center
                Bx, By = boxB.xRight + tip_offset, boxB.y_center
            path = [(Ax, Ay), (Bx, By)]
        # second case, boxA is below boxB
        elif boxA.y_center < boxB.y_center:
            # same column
            if boxA.x_center == boxB.x_center:
                Ax, Ay = boxA.x_center, boxA.yTop + butt_offset
                Bx, By = boxB.x_center, boxB.yBottom - tip_offset
                path = [(Ax, Ay), (Bx, By)]
            # boxes are offset in the x-axis
            else:
                Ax, Ay =  boxA.x_center, boxA.yTop + butt_offset
                Bx = boxB.x_center
                By = (boxB.yBottom + boxA.yTop)/2
                Cx, Cy = Bx, boxB.yBottom - tip_offset
                path = [(Ax, Ay), (Bx, By), (Cx, Cy)]
        # third case, boxA is above boxB
        elif boxA.y_center > boxB.y_center:
            # same column
            if boxA.x_center == boxB.x_center:
                Ax, Ay = boxA.x_center, boxA.yBottom - butt_offset
                Bx, By = boxB.x_center, boxB.yTop + tip_offset
                path = [(Ax, Ay), (Bx, By)]
            # boxes are offset in the x-axis
            else:
                Ax, Ay =  boxA.x_center, boxA.yBottom - butt_offset
                Bx = boxA.x_center
                By = (boxB.yTop + boxA.yBottom)/2
                Cx, Cy = boxB.x_center, By
                Dx, Dy = Cx, boxB.yTop + tip_offset
                path = [(Ax, Ay), (Bx, By), (Cx, Cy), (Dx, Dy)]
        else:
            raise ValueError("Boxes must be aligned horizontally or vertically to create a connection.")
                
        # create arrow object and 
        arrow = ArrowETC(path=path, arrow_head=arrow_head, arrow_width=arrow_width)
        x = arrow.x_vertices
        y = arrow.y_vertices
        self.ax.plot(x, y, color=ec, lw=lw)
        # fill arrow if desired
        if fill_connection:
            self.ax.fill(x, y, color=fc)
            
    def make_title(
        self, 
        pos: Literal['left', 'center', 'right'] = 'left', 
        consider_box_x: bool = True, 
        new_title: Optional[str] = None
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
            If `pos` is not one of the accepted options.
        """
        if new_title is not None:
            self.title = new_title
        
        # if we are to ignore consider_box_x, use xlims to find the horizontal placement of title
        if not consider_box_x:
            if pos == 'left':
                ha = 'left'
                x = self.xlims[0]
            elif pos == 'center':
                ha = 'center'
                x = (self.xlims[1] + self.xlims[0])/2
            elif pos == 'right':
                ha = 'right'
                x = self.xlims[1]
            else:
                raise ValueError("pos must be one of ['left', 'center', 'right']")
        
        # if we are to consider_box_x
        else:
            xFarLeft = float('inf')
            xFarRight = float('-inf')
            for box in self.boxes:
                if self.boxes[box].xLeft < xFarLeft:
                    xFarLeft = self.boxes[box].xLeft
                if self.boxes[box].xRight > xFarRight:
                    xFarRight = self.boxes[box].xRight
            if pos == 'left':
                ha = 'left'
                x = xFarLeft
            elif pos == 'right':
                ha = 'right'
                x = xFarRight
            elif pos == 'center':
                ha = 'center'
                x = (xFarRight + xFarLeft)/2
            else:
                raise ValueError("pos must be one of ['left', 'center', 'right']")
        
        # finally make the title
        self.ax.text(x=x, y=self.ylims[1], s=self.title, va='top', ha=ha, fontdict=self.title_font)
                
    def save_as_png(self, file_name: str, dpi: int = 800, content_padding: float = 0.0) -> None:
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
        self.ax.set_aspect('equal')
        # self.fig.subplots_adjust(right=28)
        self.fig.savefig(file_name, dpi=dpi, bbox_inches='tight', pad_inches=content_padding)

__all__ = ["LogicTree"]
