"""
This module defines the LogicBox class for creating labeled, stylable boxes
in logic tree diagrams. Each LogicBox supports custom text, styling, and
positioning options using matplotlib's BoxStyle.
"""

from typing import Any, Dict, Literal, Optional

from matplotlib.patches import BoxStyle


class LogicBox:
    """
    A box object for use in logic tree diagrams, containing text and visual styling.

    LogicBox provides a labeled, stylable rectangle positioned by its alignment
    parameters. Boxes can be referenced by name, styled with custom fonts and
    colors, and later connected by arrows in logic diagrams.

    Parameters
    ----------
    xpos : float or int
        X-coordinate of the box, interpreted as the left, center, or right
        depending on the `ha` alignment parameter.
    ypos : float or int
        Y-coordinate of the box, interpreted as the top, center, or bottom
        depending on the `va` alignment parameter.
    text : str
        The text displayed within the box.
    box_name : str
        A unique identifier for the box; useful for referencing the LogicBox object.
    bbox_fc : str
        Face color of the box.
    bbox_ec : str
        Edge color of the box.
    font_dict : dict
        Dictionary specifying text styling. See
        https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text
    bbox_style : BoxStyle, optional
        Matplotlib BoxStyle object specifying box shape and padding. Default is
        BoxStyle("Square", pad=0.5).
    lw : float or int, optional
        Line width of the box's edge. Default is 1.6.
    va : str, optional
        Vertical alignment, one of ["top", "center", "bottom"]. Default is "center".
    ha : str, optional
        Horizontal alignment, one of ["left", "center", "right"]. Default is "left".
    angle : float, optional
        Angle in degrees to rotate your box. Rotations are about the center of the box.

    Raises
    ------
    ValueError
        If `va` is not one of "top", "center", "bottom", or if `ha` is not one of
        "left", "center", "right".

    Attributes
    ----------
    x, y : float
        Coordinates of the box position.
    text : str
        Displayed text in the box.
    name : str
        Identifier for the box.
    face_color : str
        Face color of the box.
    edge_color : str
        Edge color of the box.
    style : dict
        Dictionary of box styling parameters.
    font_dict : dict
        Dictionary of text styling parameters.
    va, ha : str
        Vertical and horizontal alignments.
    lw : float
        Line width of the box edge.
    xLeft, xRight, yBottom, yTop : float or None
        Coordinates of the box edges, set during layout.
    width, height : float or None
        Dimensions of the box, set during layout.
    xCenter, yCenter : float or None
        Coordinates of the box center, set during layout.
    """

    def __init__(
        self,
        xpos: float,
        ypos: float,
        text: str,
        box_name: str,
        bbox_fc: str,
        bbox_ec: str,
        font_dict: Dict[str, Any],
        bbox_style: BoxStyle = BoxStyle("Square", pad=0.5),
        lw: float = 1.6,
        va: Literal["top", "center", "bottom"] = "center",
        ha: Literal["left", "center", "right"] = "left",
        angle: float = 0.0,
    ) -> None:
        # data validation for literals (va and ha parameters)
        if va not in ("top", "center", "bottom"):
            raise ValueError(f"Invalid va: {va}. Must be 'top', 'center', or 'bottom'.")
        if ha not in ("left", "center", "right"):
            raise ValueError(f"Invalid ha: {ha}. Must be 'left', 'center', or 'right'.")

        # create a bbox style object for styling text box
        my_style = self._my_bbox_style(
            facecolor=bbox_fc, edgecolor=bbox_ec, linewidth=lw, boxstyle=bbox_style
        )

        self.x = xpos
        self.y = ypos
        self.text = text
        self.name = box_name
        self.face_color = bbox_fc
        self.edge_color = bbox_ec
        self.style = my_style
        self.font_dict = font_dict
        self.va = va
        self.ha = ha
        self.lw = lw
        self.angle = angle
        self.xLeft: Optional[float] = None
        self.xRight: Optional[float] = None
        self.yBottom: Optional[float] = None
        self.yTop: Optional[float] = None
        self.width: Optional[float] = None
        self.height: Optional[float] = None
        self.xCenter: Optional[float] = None
        self.yCenter: Optional[float] = None

    def _my_bbox_style(
        self, facecolor: str, edgecolor: str, linewidth: float, boxstyle: BoxStyle
    ) -> Dict[str, Any]:
        """
        Create a dictionary of styling parameters for the box's visual appearance.

        Parameters
        ----------
        facecolor : str
            Face color of the box.
        edgecolor : str
            Edge color of the box.
        linewidth : float or int
            Line width of the box's edge.
        boxstyle : BoxStyle
            Matplotlib BoxStyle object specifying box shape and padding.

        Returns
        -------
        dict
            Dictionary containing styling properties for use with matplotlib text boxes.
        """
        my_style = {
            "boxstyle": boxstyle,
            "facecolor": facecolor,
            "edgecolor": edgecolor,
            "linewidth": linewidth,
        }

        return my_style

    def __repr__(self) -> str:
        """
        Return a developer-friendly string representation of the LogicBox.

        Returns
        -------
        str
            A concise description of the LogicBox including its name, text, and position.
        """
        return (
            f"<LogicBox(name={self.name!r}, text={self.text!r}, "
            f"x={self.x}, y={self.y})>"
        )


__all__ = ["LogicBox"]
