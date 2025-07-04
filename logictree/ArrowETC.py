"""
ArrowETC module for creating multi-segmented arrows and path-based shapes with explicit vertex control.

This module defines the ArrowETC class, which allows building complex polygonal
shapes by connecting multiple line segments in sequence. While it can produce
classic arrows with optional arrowheads, it can also create segmented rectangles
and arbitrary paths useful for connectors, pipes, or other linear features
in logic tree diagrams, flowcharts, or technical illustrations.

ArrowETC stores extensive metadata for each arrow, including segment lengths,
angles, and a complete set of polygon vertices outlining the arrow body. These
attributes remain accessible after construction, enabling downstream tasks such as
collision detection, dynamic alignment, or generating custom labels tied to
specific arrow joints.

**WARNING**: ArrowETC assumes the arrows or segmented shapes will be plotted
in an environment with an **equal aspect ratio**. The saved or displayed
arrow polygon does not automatically account for distorted aspect ratios—if you
use an unequal aspect ratio (e.g., `ax.set_aspect('auto')`), your shapes may appear
skewed or "out of whack." It is the user's responsibility to either:
1) ensure plots using ArrowETC have an equal aspect ratio, or
2) manually transform the arrow vertices to compensate for an intended uneven aspect ratio.

Features
---------
- Explicit calculation of each vertex, including miter joints at corners.
- Supports straight and multi-bend paths with arbitrary angles.
- Optional flared arrowhead at the final path point.
- Suitable for creating segmented rectangles (shaft-only shapes) by disabling the arrowhead.
- Stores metadata such as:
  - `self.vertices`: polygon vertex coordinates,
  - `self.segment_lengths`: lengths of all segments,
  - `self.path_angles`: angles each segment makes with the x-axis.

Examples
---------
Basic arrow with head:

>>> from logictree.ArrowETC import ArrowETC
>>> arrow = ArrowETC(path=[(0, 0), (0, 5)], arrow_width=1.5, arrow_head=True)
>>> arrow.save_arrow(name='example_arrow.png')

Segmented rectangular path without arrowhead:

>>> rect_arrow = ArrowETC(path=[(0, 0), (5, 0), (5, -3)], arrow_width=1.0, arrow_head=False)
>>> rect_arrow.save_arrow(name='segmented_rect.png')

Plotting in a custom figure:

>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots()
>>> ax.set_aspect('equal')
>>> ax.fill(rect_arrow.x_vertices, rect_arrow.y_vertices, color='lightblue')
>>> ax.plot(rect_arrow.x_vertices, rect_arrow.y_vertices, color='black')
>>> plt.show()

Typical use cases
------------------
- Drawing arrows or right-angle connectors in logic diagrams.
- Building segmented pipes, buses, or flow lines with multiple corners.
- Aligning shapes precisely in custom diagrams using stored vertex metadata.

Dependencies
-------------
- Python packages: numpy, matplotlib.
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
            verts = self._get_arrow_head_vertices(path[-1][0], path[-1][1], theta_1)
            # replace last vertex with new one to make room for arrow head
            vertices[-1] = verts[0]
            # fill in the 3 vertices of arrow head
            vertices.extend(verts[1:])
            
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
        tipx: float,
        tipy: float,
        theta_1: float
    ) -> List[np.ndarray]:
        """
        Calculate five points forming the arrowhead with shaft sides extending
        straight to the arrowhead base line without kinks.
        Returns [A, left_base, tip, right_base, E].
        """
        shaft_width = self.arrow_width
        head_width = shaft_width * 2.0
        head_length = shaft_width * 1.5

        # Unit vectors
        dir_x, dir_y = np.cos(theta_1), np.sin(theta_1)
        perp_x, perp_y = -dir_y, dir_x

        # Tip point
        tip = np.array([tipx, tipy], dtype=float)

        # Base center: base of the arrowhead along shaft
        base_cx = tipx - head_length * dir_x
        base_cy = tipy - head_length * dir_y

        # Left and right points on the arrowhead base line
        left_base = np.array([
            base_cx + (head_width / 2) * perp_x,
            base_cy + (head_width / 2) * perp_y
        ])
        right_base = np.array([
            base_cx - (head_width / 2) * perp_x,
            base_cy - (head_width / 2) * perp_y
        ])

        # Shaft left line: parallel to shaft, offset by +shaft_width/2
        shaft_dx, shaft_dy = dir_x, dir_y
        shaft_left_point = np.array([
            base_cx + (shaft_width/2) * perp_x,
            base_cy + (shaft_width/2) * perp_y
        ])

        # Shaft right line: parallel to shaft, offset by -shaft_width/2
        shaft_right_point = np.array([
            base_cx - (shaft_width/2) * perp_x,
            base_cy - (shaft_width/2) * perp_y
        ])

        def line_intersection(p1, d1, p2, d2):
            """
            Computes intersection of lines p1 + t*d1 and p2 + s*d2.
            """
            A = np.array([d1, -d2]).T
            if np.linalg.matrix_rank(A) < 2:
                # Parallel lines: return base point directly to avoid NaN
                return p2
            t_s = np.linalg.solve(A, p2 - p1)
            return p1 + t_s[0]*d1

        # Compute A: where shaft left edge intersects base line
        A = line_intersection(
            shaft_left_point, np.array([shaft_dx, shaft_dy]),
            left_base, right_base - left_base
        )

        # Compute E: where shaft right edge intersects base line
        E = line_intersection(
            shaft_right_point, np.array([shaft_dx, shaft_dy]),
            left_base, right_base - left_base
        )

        return [A, left_base, tip, right_base, E]

    
    def _get_first_vertex(self, Ax: float, Ay: float, theta_1: float) -> np.ndarray:
        """
        Calculate the first side vertex at the butt of the arrow,
        offset perpendicular to the first segment angle.
        """
        w2 = self.arrow_width / 2
        offset_angle = theta_1 + np.pi/2  # left side offset
        dx = w2 * np.cos(offset_angle)
        dy = w2 * np.sin(offset_angle)

        return np.array([Ax + dx, Ay + dy])
            
    def _vertex_from_angle(self, Bx: float, By: float, theta_1: float, theta_2: Optional[float]) -> np.ndarray:
        """
        Calculate a polygon vertex at a joint between two arbitrary segments,
        using miter-join logic to produce sharp corners without kinks.

        Parameters
        ----------
        Bx, By : float
            Coordinates of the joint between segments.
        theta_1 : float
            Angle of incoming segment.
        theta_2 : float or None
            Angle of outgoing segment. None if it's the last segment.

        Returns
        -------
        ndarray of float
            Coordinates of the calculated vertex as [x, y].
        """
        w2 = self.arrow_width / 2
        point = np.array([Bx, By], dtype=float)

        dir1 = np.array([np.cos(theta_1), np.sin(theta_1)])
        perp1 = np.array([-dir1[1], dir1[0]])
        A = point + w2 * perp1
        dA = dir1

        if theta_2 is None:
            return A

        dir2 = np.array([np.cos(theta_2), np.sin(theta_2)])
        perp2 = np.array([-dir2[1], dir2[0]])
        B = point + w2 * perp2
        dB = dir2

        mat = np.column_stack((dA, -dB))
        if np.linalg.matrix_rank(mat) < 2:
            avg_normal = (perp1 + perp2) / 2
            avg_normal /= np.linalg.norm(avg_normal)
            return point + w2 * avg_normal

        t = np.linalg.solve(mat, B - A)[0]
        return A + t * dA
    
    def _get_angles(self, path: List[Tuple[float, float]]) -> List[float]:
        """
        Calculate angles each segment makes with the positive x-axis,
        allowing arbitrary directions.

        Parameters
        ----------
        path : list of (x, y)
            Arrow path points.

        Returns
        -------
        list of float
            Angles (radians) of each segment relative to +x axis.
        """
        angles = []
        for i in range(self.n_segments):
            p1, p2 = path[i], path[i + 1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            theta = np.arctan2(dy, dx) % (2 * np.pi)
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
    
    def save_arrow(self, name: str = './arrow.png', ec: str = 'white', fc: str = 'cyan', lw: float = 0.6) -> None:
        """
        Display the arrow using matplotlib.

        Generates a plot of the arrow polygon with specified line and
        fill colors.

        Parameters
        ----------
        name : str, optional
            Name / path of the resulting png. Default is './arrow.png'.
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
        ax.set_aspect('equal')
        
        plt.savefig(name)
                    
__all__ = ["ArrowETC"]
