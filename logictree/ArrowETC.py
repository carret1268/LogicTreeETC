"""ArrowETC module for creating multi-segmented arrows with explicit vertex control.

This module defines the ArrowETC class, which allows building complex
arrows composed of multiple line segments aligned along the x and y axes.
It enables precise control over arrow geometry for logic tree diagrams,
flowcharts, and custom annotations.
"""

from typing import List, Optional, Tuple 

import matplotlib.pyplot as plt
import numpy as np

class ArrowETC:
    """
    An arrow object with detailed vertex control for multi-segmented arrows.

    ArrowETC provides arrows constructed from a series of connected line segments,
    storing coordinates of every vertex to make it easy to generate complex arrows
    with multiple joints. Unlike matplotlib's FancyArrow, it gives explicit access
    to arrow geometry for alignment and advanced layout tasks.

    Parameters
    ----------
    path : list of tuple of float or int
        List of points defining the path of the arrow. Each tuple is the
        center of an endpoint of a line segment. The first point is the
        "butt" (tail), and the last point is the arrow "head".
    arrow_width : float or int
        The width of the arrow shaft in data coordinates.
    arrow_head : bool, optional
        If True, an arrowhead will be added at the end of the arrow path,
        using the last point in `path` as the tip. If False, the arrow ends
        with a flat edge.

    Attributes
    ----------
    path : list of tuple
        Input path defining the arrow's geometry.
    x_path : list of float
        List of x-coordinates along the arrow path.
    y_path : list of float
        List of y-coordinates along the arrow path.
    n_path : int
        Number of points in the path.
    n_segments : int
        Number of line segments (n_path - 1).
    segment_lengths : list of float
        Lengths of each line segment.
    path_angles : list of float
        Angles (radians) each segment makes with the positive x-axis.
    vertices : ndarray of shape (N, 2)
        Array of vertices defining the arrow polygon.
    x_vertices : ndarray of float
        X-coordinates of the arrow polygon vertices.
    y_vertices : ndarray of float
        Y-coordinates of the arrow polygon vertices.
    """
    def __init__(
        self, 
        path: List[Tuple[float, float]], 
        arrow_width: float, 
        arrow_head: bool = False
    ) -> None:
        self.path = path
        self.x_path = [coord[0] for coord in path]
        self.y_path = [coord[1] for coord in path]
        self.n_path = len(path)
        self.n_segments = self.n_path - 1 # actual number of line segments
        self.n_segment_vertices = 2*(1 + self.n_segments) # vertex count w/o arrow head
        self.segment_lengths = self._get_segment_length()
        if arrow_head == True:
            self.n_vertices = self.n_segment_vertices + 3 # vertex count w/ arrow head
        else:
            self.n_vertices = self.n_segment_vertices
        # find the angles each segment makes with the (+) horizontal (CCW)
        self.path_angles = self._get_angles(path=path)
        # getting angles in reverse is essential for the way vertices are calculated
        self.reverse_path_angles = self._get_angles(path=path[::-1])
        self.arrow_width = arrow_width
        self.arrow_head = arrow_head
        # verts need to wrap back around to first vertex for plotting
        verts = self._get_vertices()
        self.vertices = np.vstack((verts, verts[0]))
        self.x_vertices = self.vertices[:, 0]
        self.y_vertices = self.vertices[:, 1]
    
    def _get_vertices(self) -> np.ndarray:
        """
        Compute the vertices outlining the multi-segment arrow polygon.

        Vertices are calculated by traversing the arrow path twice:
        once in forward order to generate one side of the arrow shaft,
        and once in reverse order to generate the other side, optionally
        inserting an arrowhead at the tip.

        Returns
        -------
        ndarray of shape (N, 2)
            Array of vertices as (x, y) coordinates in data space,
            ordered clockwise around the arrow polygon.
        """
        path = self.path
        vertices = []
        # iterate through the path normally first, get first half of vertices
        for i in range(self.n_path-1):
            # get the next two neighboring points starting at 'butt'
            A, B = path[i], path[i+1]
            Ax, Ay = A[0], A[1]
            Bx, By = B[0], B[1]
            theta_1 = self.path_angles[i] # angle of this line segment
            # at the end of this half of vertices, there wont be an angle for next segment
            theta_2 = self.path_angles[i+1] if i + 1 < self.n_segments else None
            
            # first vertex is special and needs to be calculated separately
            if i == 0:
                vert = self._get_first_vertex(Ax, Ay, theta_1)
                vertices.append(vert)
            # Get the vertex
            vert = self._vertex_from_angle(Bx, By, theta_1, theta_2)
            vertices.append(vert)
        
        # generate an arrow head if desired
        if self.arrow_head:
            B = vertices[-1]
            Bx, By = B[0], B[1]
            verts = self._get_arrow_head_vertices(Bx, By, path[-1][0], path[-1][1], theta_1)
            # replace last vertex with new one to make room for arrow head
            vertices[-1] = verts[0]
            # fill in the 3 vertices of arrow head
            for point in verts[1:]:
                vertices.append(point)
    
            
        # now iterate through path backwards to get the last half of vertices
        path = path[::-1]
        for i in range(self.n_path-1):
            # get the next two neighboring points starting at 'butt'
            A, B = path[i], path[i+1]
            Ax, Ay = A[0], A[1]
            Bx, By = B[0], B[1]
            theta_1 = self.reverse_path_angles[i] # angle of this line segment
            # at the end of this half of vertices, there wont be an angle for next segment
            theta_2 = self.reverse_path_angles[i+1] if i + 1 < self.n_segments else None
                
            # first vertex is special and needs to be calculated separately, If we have no arrow head
            if i == 0 and not self.arrow_head:
                vert = self._get_first_vertex(Ax, Ay, theta_1)
                vertices.append(vert)
            # Get the vertex
            vert = self._vertex_from_angle(Bx, By, theta_1, theta_2)
            vertices.append(vert)

        return np.array(vertices, dtype=float)

    def _get_arrow_head_vertices(
        self, 
        Bx: float, 
        By: float, 
        tipx: float, 
        tipy: float, 
        theta_1: float
    ) -> List[np.ndarray]:
        """
        Calculate vertices needed to draw the arrowhead.

        This method computes the vertices forming the arrowhead polygon
        based on the final arrow segment direction and specified dimensions.
        It also adjusts the last shaft vertex to make space for the arrowhead.

        Parameters
        ----------
        Bx, By : float
            Coordinates of the last shaft vertex before the arrowhead.
        tipx, tipy : float
            Coordinates of the arrow tip.
        theta_1 : float
            Angle of the final arrow segment in radians.

        Returns
        -------
        list of ndarray
            List of (x, y) points forming the arrowhead polygon.
        """
        width = self.arrow_width
        arrow_head_width = width*0.55
        arrow_head_angle = 50*np.pi/180 # angle vertex0 and vert2 make with tip (vert1)
        tip_to_tip_dist = np.tan(arrow_head_angle)*(arrow_head_width + width/2)
        # 4 cases
        verts = []
        if theta_1 == 0:
            # move last vertex back
            vertx = Bx - tip_to_tip_dist
            verty = By
            verts.append(np.array([vertx, verty], dtype=float))
            # get first head vertex
            vertx = vertx
            verty = verty + arrow_head_width
            verts.append(np.array([vertx, verty], dtype=float))
            # get tip of arrow
            vertx = tipx
            verty = tipy
            verts.append(np.array([vertx, verty], dtype=float))
            # get last vertex of arrow
            vertx = verts[0][0]
            verty = verts[0][1] - width - arrow_head_width
            verts.append(np.array([vertx, verty], dtype=float))
            # complete the cycle putting us back on the base arrow body
            vertx = vertx
            verty = verty + arrow_head_width
            verts.append(np.array([vertx, verty], dtype=float))
        elif theta_1 == np.pi/2:
            # move last vertex down
            vertx = Bx
            verty = By - tip_to_tip_dist
            verts.append(np.array([vertx, verty], dtype=float))
            # get first head vertex
            vertx = vertx - arrow_head_width
            verty = verty
            verts.append(np.array([vertx, verty], dtype=float))
            # get tip of arrow
            vertx = tipx
            verty = tipy
            verts.append(np.array([vertx, verty], dtype=float))
            # get last vertex of arrow
            vertx = verts[0][0] + width + arrow_head_width
            verty = verts[0][1]
            verts.append(np.array([vertx, verty], dtype=float))
            # complete the cycle putting us back on the base arrow body
            vertx = vertx - arrow_head_width
            verty = verty
            verts.append(np.array([vertx, verty], dtype=float))
        elif theta_1 == np.pi:
            # move last vertex back
            vertx = Bx + tip_to_tip_dist
            verty = By
            verts.append(np.array([vertx, verty], dtype=float))
            # get first head vertex
            vertx = vertx
            verty = verty - arrow_head_width
            verts.append(np.array([vertx, verty], dtype=float))
            # get tip of arrow
            vertx = tipx
            verty = tipy
            verts.append(np.array([vertx, verty], dtype=float))
            # get last vertex of arrow
            vertx = verts[0][0]
            verty = verts[0][1] + width + arrow_head_width
            verts.append(np.array([vertx, verty], dtype=float))
            # complete the cycle putting us back on the base arrow body
            vertx = vertx
            verty = verty - arrow_head_width
            verts.append(np.array([vertx, verty], dtype=float))
        elif theta_1 == 3*np.pi/2:
            # move last vertex back
            vertx = Bx
            verty = By + tip_to_tip_dist
            verts.append(np.array([vertx, verty], dtype=float))
            # get first head vertex
            vertx = vertx + arrow_head_width
            verty = verty
            verts.append(np.array([vertx, verty], dtype=float))
            # get tip of arrow
            vertx = tipx
            verty = tipy
            verts.append(np.array([vertx, verty], dtype=float))
            # get last vertex of arrow
            vertx = verts[0][0] - width - arrow_head_width
            verty = verts[0][1]
            verts.append(np.array([vertx, verty], dtype=float))
            # complete the cycle putting us back on the base arrow body
            vertx = vertx + arrow_head_width
            verty = verty
            verts.append(np.array([vertx, verty], dtype=float))

        return verts
    
    def _get_first_vertex(self, Ax: float, Ay: float, theta_1: float) -> np.ndarray:
        """
        Calculate the first vertex of the arrow shaft polygon.

        Used to determine the initial side vertex at the base of the arrow,
        based on the starting path point and direction of the first segment.

        Parameters
        ----------
        Ax, Ay : float
            Coordinates of the starting point of the arrow path.
        theta_1 : float
            Angle of the first arrow segment in radians.

        Returns
        -------
        ndarray of float
            Coordinates of the first vertex as [x, y].
        """
        width = self.arrow_width
        if theta_1 == 0:
            vertx = Ax
            verty = Ay + width/2
            vert = np.array([vertx, verty], dtype=float)
        elif theta_1 == np.pi/2:
            vertx = Ax - width/2
            verty = Ay
            vert = np.array([vertx, verty], dtype=float)
        elif theta_1 == np.pi:
            vertx = Ax
            verty = Ay - width/2
            vert = np.array([vertx, verty], dtype=float)
        elif theta_1 == 3*np.pi/2:
            vertx = Ax + width/2
            verty = Ay
            vert = np.array([vertx, verty], dtype=float) 

        return vert
            
    def _vertex_from_angle(self, Bx: float, By: float, theta_1: float, theta_2: Optional[float]) -> np.ndarray:
        """
        Calculate a polygon vertex at a path joint.

        Given two segment angles, computes the intersection point of the
        two sides of the arrow at the joint.

        Parameters
        ----------
        Bx, By : float
            Coordinates of the joint between two segments.
        theta_1 : float
            Angle of the incoming segment.
        theta_2 : float or None
            Angle of the outgoing segment, or None if it's the last segment.

        Returns
        -------
        ndarray of float
            Coordinates of the calculated vertex as [x, y].
        """
        width = self.arrow_width
        # first when there is a segment ahead
        if ((theta_1 == 0) and (theta_2 == np.pi/2)) or ((theta_1 == np.pi/2) and (theta_2 == 0)):
            vertx = Bx - width/2
            verty = By + width/2
            vert = np.array([vertx, verty], dtype=float)
        elif ((theta_1 == 0) and (theta_2 == 3*np.pi/2)) or ((theta_1 == 3*np.pi/2) and (theta_2 == 0)):
            vertx = Bx + width/2
            verty = By + width/2
            vert = np.array([vertx, verty], dtype=float)
        elif ((theta_1 == np.pi) and (theta_2 == np.pi/2)) or ((theta_1 == np.pi/2) and (theta_2 == np.pi)):
            vertx = Bx - width/2
            verty = By - width/2
            vert = np.array([vertx, verty], dtype=float)
        elif ((theta_1 == np.pi) and (theta_2 == 3*np.pi/2)) or ((theta_1 == 3*np.pi/2) and (theta_2 == np.pi)):
            vertx = Bx + width/2
            verty = By - width/2
            vert = np.array([vertx, verty], dtype=float)
        # now if there is no segment ahead
        elif (theta_1 == 0) and (theta_2 is None):
            vertx = Bx
            verty = By + width/2
            vert = np.array([vertx, verty], dtype=float)
        elif (theta_1 == np.pi/2) and (theta_2 is None):
            vertx = Bx - width/2
            verty = By
            vert = np.array([vertx, verty], dtype=float)
        elif (theta_1 == np.pi) and (theta_2 is None):
            vertx = Bx
            verty = By - width/2
            vert = np.array([vertx, verty], dtype=float)
        elif (theta_1 == 3*np.pi/2) and (theta_2 is None):
            vertx = Bx + width/2
            verty = By
            vert = np.array([vertx, verty], dtype=float)
        return vert
    
    def _get_angles(self, path: List[Tuple[float, float]]) -> List[float]:
        """
        Calculate angles each segment makes with the positive x-axis.

        Each segment is defined by consecutive points in the path. Only
        horizontal or vertical segments aligned with the x or y axes are allowed.

        Parameters
        ----------
        path : list of tuple of float
            List of points defining the arrow path.

        Returns
        -------
        list of float
            Angles in radians of each segment relative to the +x axis.

        Raises
        ------
        ValueError
            If any segment does not align exactly with the x or y axis.
        """
        angles = []
        for i in range(self.n_segments):
            # get the next two neighboring points starting at 'butt'
            p1, p2 = path[i], path[i+1]
            x1, y1 = p1[0], p1[1]
            x2, y2 = p2[0], p2[1]
            
            # for (+) horizontal line
            if (x2 > x1) and (y2 == y1):
                theta = 0
            # for (+) vertical line
            elif (x2 == x1) and (y2 > y1):
                theta = np.pi/2
            # for (-) horizontal line
            elif (x2 < x1) and (y2 == y1):
                theta = np.pi
            # for (-) vertical line
            elif (x2 == x1) and (y2 < y1):
                theta = 3*np.pi/2
            # Get the angles starting with Quadrant 1
            elif (x2 > x1) and (y2 > y1):
                theta = np.arctan((y2-y1)/(x2-x1))
            # Quadrant 2
            elif (x2 < x1) and (y2 > y1):
                # start with angle CCW from (+) vertical
                phi = np.arctan((x2-x1)/(y2-y1))
                theta = np.pi/2 + abs(phi)
            # Quadrant 3
            elif (x2 < x1) and (y2 < y1):
                # start with angle CCW from (-) horizontal
                phi = np.arctan((y2-y1)/(x2-x1))
                theta = np.pi + abs(phi)
            # Quadrant 4
            else:
                # start with angle CW from (+) horizontal
                phi = np.arctan((y2-y1)/(x2-x1))
                theta = 2*np.pi - abs(phi)
            # throw an error for non-Right angles
            if (theta != 0) and (theta != np.pi) and (theta != np.pi/2) and (theta != 3*np.pi/2):
                raise ValueError(
                    f"The arrow path must be limited to straight line segments along +/- x or y axes. "
                    f"The segment between point {p1} and point {p2} does not fall on an axis."
                )
            angles.append(theta)

        return angles
    
    def _get_segment_length(self) -> List[float]:
        """
        Compute the Euclidean length of each arrow segment.

        Returns
        -------
        list of float
            Distances between consecutive path points defining each segment.
        """
        distances = []
        for i in range(self.n_segments):
            p1, p2 = self.path[i], self.path[i+1]
            x1, y1 = p1[0], p1[1]
            x2, y2 = p2[0], p2[1]
            d = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            distances.append(d)

        return distances
    
    def show_arrow(self, ec: str = 'white', fc: str = 'cyan', lw: float = 0.6) -> None:
        """
        Display the arrow using matplotlib.

        Generates a plot of the arrow polygon with specified line and
        fill colors.

        Parameters
        ----------
        ec : str, optional
            Edge color of the arrow outline. Default is 'white'.
        fc : str, optional
            Fill color of the arrow body. Default is 'cyan'.
        lw : float, optional
            Line width of the arrow outline. Default is 0.6.
        """
        x = self.x_vertices
        y = self.y_vertices
        # generate figure and axis to put boxes in
        _, ax = plt.subplots(figsize=(8, 8), frameon=True, facecolor='black')
        ax.axis('off')
        ax.set_aspect('equal')
        # set axis bounds
        xdiff = (max(x) - min(x)) * 0.2
        ydiff = (max(y) - min(y)) * 0.2
        xmin = min(x) - xdiff
        xmax = max(x) + xdiff
        ymin = min(y) - ydiff
        ymax = max(y) + ydiff
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        # plot lines and vertices
        ax.plot(x, y, color=ec, lw=lw, zorder=100)
        ax.fill(x, y, color=fc)
                    
__all__ = ["ArrowETC"]
