"""
ArrowETC - Custom Arrow Rendering with Vertex-Level Control

This module defines the `ArrowETC` class, which generates precise, customizable arrows
for use in diagrams, flowcharts, technical illustrations, and scientific visualizations.
Unlike `matplotlib.patches.FancyArrow` or `FancyArrowPatch`, `ArrowETC` provides full
acess and control over the arrow's geometry, including vertices, segment lengths, and angles.

Arrows can be constructed from straight line segments or smoothed via Bezier interpolation.
An optional arrowhead can be appended to the tip and/or tail. All geometry is computed
in display (pixel) space and transformed back to data coordinates to ensure accurate rendering
under any axis scaling or aspect ratio.

Features
--------
- Explicit calculation of every polygon vertex, including miter joins at segment corners.
- Optional Bezier-style curves for smooth arrow shapes.
- Customizable arrowhead geometry and placement.
- Compatible with both data-space and pixel-space workflows.
- Full access to metadata: segment lengths, angles, and vertex coordinates.
- Optionally close the butt (tail) of the arrow polygon.

Examples
--------
>>> from logictree.ArrowETC import ArrowETC
>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots()
>>> arrow = ArrowETC(ax=ax, path=[(0, 0), (0, 5)], shaft_width=30, arrow_head=True)
>>> arrow.draw_to_ax(ax)
>>> plt.show()

>>> curved = ArrowETC(ax=ax, path=[(0, 0), (2, 4), (4, 0)], shaft_width=25, bezier=True)
>>> curved.save_arrow(name='curved_arrow.png')

"""

from typing import List, Tuple, Union

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import splprep, splev

FloatLike = Union[float, np.float64]


class ArrowETC:
    """
    A customizable arrow object with explicit control over geometry, styling, and vertex layout.

    The `ArrowETC` class generates straight or curved arrows based on a list of path points.
    It builds the arrow body polygon in display (pixel) space to ensure consistent appearance
    across different axis scales and aspect ratios, then transforms it back to data coordinates
    for plotting.

    Arrows can be constructed using straight-line segments or smoothed with a Bezier (B-spline)
    curve. Optional arrowheads can be added at the tip and/or tail, with configurable width
    and length multipliers. All geometry and metadata-including vertices, segment lengths, and angles-
    are accessible after construction.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib Axes instance used to determine coordinate transforms.
    path : list of tuple[float, float]
        Sequence of (x, y) points defining the arrow path. The first point is the tail,
        and the last point is the tip.
    shaft_width : float, optional
        Width of the arrow shaft in pixels. Default is 20.
    arrow_head : bool, optional
        If True (default), draws an arrowhead at the tip.
    arrow_head_at_tail : bool, optional
        If True, draws a second arrowhead at the tail (inverted direction). Default is False.
    arrow_head_width_multiplier : float, optional
        Multiplier controlling how much wider the arrowhead is compared to the shaft. Default is 2.
    arrow_head_length_multiplier : float, optional
        Multiplier controlling how much longer the arrowhead is compared to the shaft width. Default is 1.5.
    ec : str, optional
        Edge color of the arrow outline. Default is "grey".
    fc : str, optional
        Fill color of the arrow body. Default is "white".
    lw : float, optional
        Line width of the arrow edge. Default is 1.5.
    ls : str, optional
        Line style for the arrow outline (e.g., '-', '--'). Default is "-".
    zorder : float, optional
        Drawing order on the plot. Higher values appear on top. Default is 100.
    bezier : bool, optional
        If True, the path is interpolated as a smooth Bezier (B-spline) curve. Default is False.
    bezier_n : int, optional
        Number of points used to sample the Bezier curve. Higher values improve smoothness. Default is 400.
    close_tail : bool, optional
        If True, closes the polygon at the arrow's tail by connecting the final vertex to the first. Default is True.

    Attributes
    ----------
    path : list of tuple[float, float]
        The input list of (x, y) coordinates defining the arrow path.
    path_px : list of tuple[float, float]
        The input list of (x, y) coordinates defining the arrow path after being transformed to display (pixel) space.
    x_path : list of float
        X-coordinates of the path.
    y_path : list of float
        Y-coordinates of the path.
    n_path : int
        Number of path points.
    n_segments : int
        Number of segments connecting the path points (n_path - 1).
    segment_lengths : list of float or None
        Euclidean lengths of each straight segment. None if Bezier is used.
    path_angles : list of float
        Angle (in radians) each straight segment makes with the x-axis. Undefined for Bezier.
    vertices : np.ndarray of shape (N, 2)
        Final arrow polygon vertices in data coordinates.
    x_vertices : np.ndarray
        X-coordinates of polygon vertices.
    y_vertices : np.ndarray
        Y-coordinates of polygon vertices.
    """

    def __init__(
        self,
        ax: Axes,
        path: List[Tuple[FloatLike, FloatLike]],
        shaft_width: FloatLike = 20.0,
        arrow_head: bool = True,
        arrow_head_at_tail: bool = False,
        arrow_head_width_multiplier: float = 2,
        arrow_head_length_multiplier: float = 1.5,
        ec: str = "grey",
        fc: str = "white",
        lw: FloatLike = 1.5,
        ls: str = "-",
        zorder: FloatLike = 100,
        bezier: bool = False,
        bezier_n: int = 400,
        close_tail: bool = True,
    ) -> None:
        # data validation
        self.n_path = len(path)
        if self.n_path < 2:
            raise ValueError(
                f"The `path` parameter must have at least 2 points, not {self.n_path}"
            )
        if shaft_width <= 0:
            raise ValueError(
                f"The `shaft_width` parameter must be greater than 0, not {shaft_width}"
            )

        # set parameters
        self.path = path
        self.ax = ax
        self.path_px = ax.transData.transform(np.array(self.path))
        self.arrow_head_width_multiplier = arrow_head_width_multiplier
        self.arrow_head_length_multiplier = arrow_head_length_multiplier
        self.ec = ec
        self.fc = fc
        self.lw = lw
        self.ls = ls
        self.zorder = zorder
        self.bezier = bezier
        self.bezier_n = bezier_n
        self.x_path = [coord[0] for coord in path]
        self.y_path = [coord[1] for coord in path]
        self.close_tail = close_tail
        self.n_segments = self.n_path - 1  # actual number of line segments
        self.n_segment_vertices = 2 * (
            1 + self.n_segments
        )  # vertex count w/o arrow head
        self.segment_lengths = self._get_segment_length() if not self.bezier else None

        if arrow_head:
            self.n_vertices = self.n_segment_vertices + 3  # vertex count w/ arrow head
        else:
            self.n_vertices = self.n_segment_vertices

        # find the angles each segment makes with the (+) horizontal (CCW)
        self.path_angles = self._get_angles(path=path)

        # getting angles in reverse is essential for the way vertices are calculated
        self.reverse_path_angles = self._get_angles(path=path[::-1])
        self.shaft_width = shaft_width
        self.arrow_head = arrow_head
        self.arrow_head_at_tail = arrow_head_at_tail

        if self.bezier:
            self.curve_samples = self._get_bezier_samples()
            verts = self._build_vertices_from_bezier_display_path()
        else:
            verts = self._build_vertices_from_display_path()

        # optionally close the polygon at the butt end
        if self.close_tail:
            self.vertices = np.vstack((verts, verts[0]))
        else:
            self.vertices = np.asarray(verts)

        self.x_vertices = self.vertices[:, 0]
        self.y_vertices = self.vertices[:, 1]

    def _build_vertices_from_display_path(self) -> np.ndarray:
        """
        Construct the arrow polygon by sampling each path segment in display (pixel) space,
        computing the shaft edges and mitered corners, then transforming the result
        back into data coordinates.

        This method is used for arrows composed of straight-line segments (i.e., `bezier=False`)
        and ensures consistent visual thickness regardless of axis scaling or aspect ratio.

        For each segment:
        - The left and right shaft boundaries are computed using perpendicular vectors.
        - Miter joins are used to smoothly connect adjacent segments.
        - Optionally adds arrowheads at the tip and/or tail using display-space geometry.

        Returns
        -------
        np.ndarray of shape (N, 2)
            The list of polygon vertices in data coordinate space representing the full arrow body.

        Raises
        ------
        AttributeError
            If the `ax` attribute is not set, which is required to access transformation methods.
        """
        if not hasattr(self, "ax"):
            raise AttributeError(
                "ArrowETC must have `self.ax` set to use display-based geometry."
            )

        path = self.path
        trans = self.ax.transData
        inv_trans = trans.inverted()

        # transform path points from data to display space
        path_disp = np.vstack([trans.transform(p) for p in path])
        self.path_px = path_disp
        angles_disp = self._get_angles(path_disp)
        self.angles_px = angles_disp
        angles_disp_rev = self._get_angles(path_disp[::-1])

        # setup
        w2 = self.shaft_width / 2
        vertices_disp = []

        # forward traversal (left side)
        for i in range(len(path_disp) - 1):
            A_disp = path_disp[i]
            B_disp = path_disp[i + 1]
            theta_1 = angles_disp[i]
            theta_2 = angles_disp[i + 1] if i + 1 < len(angles_disp) else None

            if i == 0:
                dx = np.cos(theta_1 + np.pi / 2)
                dy = np.sin(theta_1 + np.pi / 2)
                normal = np.array([dx, dy])
                normal /= np.linalg.norm(normal)
                vertex = np.array(A_disp) + w2 * normal
                vertices_disp.append(vertex)

            # miter join vertex at joint B
            point = np.array(B_disp)
            d1 = np.array([np.cos(theta_1), np.sin(theta_1)])
            d2 = (
                np.array([np.cos(theta_2), np.sin(theta_2)])
                if theta_2 is not None
                else None
            )
            perp1 = np.array([-d1[1], d1[0]])
            A = point + w2 * perp1

            if theta_2 is None or d2 is None:
                vertex = A
            else:
                perp2 = np.array([-d2[1], d2[0]])
                B = point + w2 * perp2
                mat = np.column_stack((d1, -d2))
                if np.linalg.matrix_rank(mat) < 2:
                    avg_normal = (perp1 + perp2) / 2
                    avg_normal /= np.linalg.norm(avg_normal)
                    vertex = point + w2 * avg_normal
                else:
                    t = np.linalg.solve(mat, B - A)[0]
                    vertex = A + t * d1

            vertices_disp.append(vertex)

        # reverse traversal (right side)
        path_disp_rev = path_disp[::-1]
        for i in range(len(path_disp_rev) - 1):
            A_disp = path_disp_rev[i]
            B_disp = path_disp_rev[i + 1]
            theta_1 = angles_disp_rev[i]
            theta_2 = angles_disp_rev[i + 1] if i + 1 < len(angles_disp_rev) else None

            if i == 0 and not self.arrow_head:
                dx = np.cos(theta_1 + np.pi / 2)
                dy = np.sin(theta_1 + np.pi / 2)
                normal = np.array([dx, dy])
                normal /= np.linalg.norm(normal)
                vertex = np.array(A_disp) + w2 * normal
                vertices_disp.append(vertex)

            point = np.array(B_disp)
            d1 = np.array([np.cos(theta_1), np.sin(theta_1)])
            d2 = (
                np.array([np.cos(theta_2), np.sin(theta_2)])
                if theta_2 is not None
                else None
            )
            perp1 = np.array([-d1[1], d1[0]])
            A = point + w2 * perp1

            if theta_2 is None or d2 is None:
                vertex = A
            else:
                perp2 = np.array([-d2[1], d2[0]])
                B = point + w2 * perp2
                mat = np.column_stack((d1, -d2))
                if np.linalg.matrix_rank(mat) < 2:
                    avg_normal = (perp1 + perp2) / 2
                    avg_normal /= np.linalg.norm(avg_normal)
                    vertex = point + w2 * avg_normal
                else:
                    t = np.linalg.solve(mat, B - A)[0]
                    vertex = A + t * d1

            vertices_disp.append(vertex)

        if self.arrow_head or self.arrow_head_at_tail:
            vertices_disp = self._add_arrow_head_vertices_px(
                vertices_px=vertices_disp,
            )

        # transform all display-space vertices back to data-space
        self.vertices_px = vertices_disp
        vertices_data = np.array(
            [inv_trans.transform(p) for p in vertices_disp], dtype=np.float64
        )

        return vertices_data

    def _add_arrow_head_vertices_px(
        self,
        vertices_px: list[np.ndarray],
    ) -> list[np.ndarray]:
        """
        Append arrowhead geometry in display space to the provided shaft polygon.

        Arrowheads are constructed using a fan-shaped polygon defined by:
        - A tip point (the end of the arrow),
        - A wide base (arrowhead base width),
        - A narrow base (shaft width), to create a smooth transition.

        The method handles arrowheads at both the tip (`arrow_head=True`) and the tail
        (`arrow_head_at_tail=True`), and adapts logic depending on whether the arrow is
        straight or Bezier-smoothed.

        Parameters
        ----------
        vertices_px : list of np.ndarray
            Display-space coordinates of the arrow shaft polygon. This list is typically
            composed of "left-side" and "right-side" vertices.

        Returns
        -------
        list of np.ndarray
            Updated list of polygon vertices including arrowhead(s), in display coordinates.
        """
        if not (self.arrow_head or self.arrow_head_at_tail):
            return vertices_px

        def build_arrow_head(tip: np.ndarray, theta: FloatLike) -> List[np.ndarray]:
            # unit direction and perpendicular vectors
            dir_vec = np.array([np.cos(theta), np.sin(theta)])
            perp_vec = np.array([-dir_vec[1], dir_vec[0]])

            # base center point
            base_center = tip - head_length * dir_vec

            # base left and right points (wide)
            base_left = base_center + (head_width / 2) * perp_vec
            base_right = base_center - (head_width / 2) * perp_vec

            # shaft left and right points (narrow)
            shaft_left = base_center + (shaft_width / 2) * perp_vec
            shaft_right = base_center - (shaft_width / 2) * perp_vec

            return [
                np.array(shaft_left),
                np.array(base_left),
                np.array(tip),
                np.array(base_right),
                np.array(shaft_right),
            ]

        shaft_width = self.shaft_width
        head_width = shaft_width * self.arrow_head_width_multiplier
        head_length = shaft_width * self.arrow_head_length_multiplier

        # split our points where the arrow head ought to go
        left = vertices_px[: len(vertices_px) // 2]
        right = vertices_px[len(vertices_px) // 2 + 1 :]

        # if head at tail
        if self.arrow_head_at_tail:
            theta = self.angles_px[0] + np.pi
            tip = self.path_px[0]
            tail_head_verts = build_arrow_head(tip, theta)

        # now tip of arrow
        if self.arrow_head:
            theta = self.angles_px[-1]
            tip = self.path_px[-1]
            head_verts = build_arrow_head(tip, theta)

        # combine our vertices to get the final arrow (if tail has head, its tip is first vertex)
        if self.arrow_head_at_tail and self.arrow_head:
            verts_with_head = tail_head_verts[2:]
            verts_with_head.extend(left[1:])
            verts_with_head.extend(np.array(head_verts))
            verts_with_head.extend(right[:-1])
            verts_with_head.extend(tail_head_verts[:2])
        elif self.arrow_head_at_tail:
            verts_with_head = tail_head_verts[2:]
            verts_with_head.extend(left[1:])
            verts_with_head.append(vertices_px[len(vertices_px) // 2])
            verts_with_head.extend(right[:-1])
            verts_with_head.extend(tail_head_verts[:2])
        else:
            if self.bezier:
                # we need to find the points in left and right closest to our arrow head base
                left_base_p = head_verts[0]
                right_base_p = head_verts[-1]
                d2_left = np.sum((left - left_base_p) ** 2, axis=1)
                d2_right = np.sum((right - right_base_p) ** 2, axis=1)
                i_left = np.argmin(d2_left)
                i_right = np.argmin(d2_right)

                # now get the vertices
                verts_with_head = left[:i_left]
                verts_with_head.extend(np.array(head_verts))
                verts_with_head.extend(right[i_right + 1 :])

            else:
                verts_with_head = left
                verts_with_head.extend(np.array(head_verts))
                verts_with_head.extend(right)

        return verts_with_head

    def _get_bezier_samples(self) -> NDArray[np.float64]:
        """
        Sample a smooth B-spline curve through the provided path points.

        This method fits a B-spline (via `scipy.interpolate.splprep`) to the original path
        and returns a dense sequence of interpolated coordinates. The number of points is
        controlled by `self.bezier_n`.

        If the number of path segments is less than 3, the spline order is reduced accordingly.

        Returns
        -------
        np.ndarray of shape (N, 2)
            Array of [x, y] coordinates representing the sampled Bezier curve in data space.
        """
        x = np.array([p[0] for p in self.path])
        y = np.array([p[1] for p in self.path])

        # Use scipy splprep for B-spline parameterization
        k = min(3, self.n_segments)
        if self.n_segments < 2:
            k = 1  # fallback to linear spline for small paths
        tck, _u = splprep([x, y], s=0, k=k)
        unew = np.linspace(0, 1, self.bezier_n)
        out = splev(unew, tck)
        sampled_curve = np.column_stack(out)

        return sampled_curve

    def _build_vertices_from_bezier_display_path(self) -> np.ndarray:
        """
        Construct the arrow polygon for a Bezier-smoothed path using display-space geometry.

        This method samples the arrow path using a B-spline interpolation (via `_get_bezier_samples()`),
        computes the left and right shaft boundaries in display (pixel) space, and adds optional
        arrowheads at the tip and/or tail. Final polygon vertices are transformed back to data coordinates.

        The resulting arrow maintains visual consistency across plots with unequal axis scaling.

        Returns
        -------
        np.ndarray of shape (N, 2)
            Array of polygon vertices in data coordinate space defining the full Bezier arrow.

        Raises
        ------
        AttributeError
            If `self.ax` is not defined, which is required for coordinate transformations.
        """
        if not hasattr(self, "ax"):
            raise AttributeError(
                "ArrowETC must have `self.ax` set to use display-based geometry."
            )

        # sample the Bezier curve in data space
        samples_data = self._get_bezier_samples()

        # transform to display (pixel) space
        trans = self.ax.transData
        inv_trans = trans.inverted()
        path_disp = np.vstack([trans.transform(p) for p in samples_data])
        self.path_px = path_disp

        # compute angles along the curve
        angles_disp = self._get_angles(path_disp)
        self.angles_px = angles_disp
        angles_disp_rev = self._get_angles(path_disp[::-1])

        w2 = self.shaft_width / 2
        left_side = []
        right_side = []

        # build left-side vertices (butt → tip)
        for i in range(len(path_disp) - 1):
            A = path_disp[i]
            B = path_disp[i + 1]
            theta1 = angles_disp[i]
            theta2 = angles_disp[i + 1] if i + 1 < len(angles_disp) else None

            # butt cap at the very first segment
            if i == 0:
                normal = np.array(
                    [np.cos(theta1 + np.pi / 2), np.sin(theta1 + np.pi / 2)]
                )
                normal /= np.linalg.norm(normal)
                left_side.append(np.array(A) + w2 * normal)

            # miter-join at B
            dir1 = np.array([np.cos(theta1), np.sin(theta1)])
            perp1 = np.array([-dir1[1], dir1[0]])
            A_pt = np.array(B) + w2 * perp1

            if theta2 is None:
                vertex = A_pt
            else:
                dir2 = np.array([np.cos(theta2), np.sin(theta2)])
                perp2 = np.array([-dir2[1], dir2[0]])
                B_pt = np.array(B) + w2 * perp2
                mat = np.column_stack((dir1, -dir2))
                if np.linalg.matrix_rank(mat) < 2:
                    avg = (perp1 + perp2) / 2
                    avg /= np.linalg.norm(avg)
                    vertex = np.array(B) + w2 * avg
                else:
                    t = np.linalg.solve(mat, B_pt - A_pt)[0]
                    vertex = A_pt + t * dir1

            left_side.append(vertex)

        # build right-side vertices (tip → butt)
        rev = path_disp[::-1]
        for i in range(len(rev) - 1):
            A = rev[i]
            B = rev[i + 1]
            theta1 = angles_disp_rev[i]
            theta2 = angles_disp_rev[i + 1] if i + 1 < len(angles_disp_rev) else None

            # butt cap on tail end if no arrowhead
            if i == 0 and not self.arrow_head:
                normal = np.array(
                    [np.cos(theta1 + np.pi / 2), np.sin(theta1 + np.pi / 2)]
                )
                normal /= np.linalg.norm(normal)
                right_side.append(np.array(A) + w2 * normal)

            # miter-join at B
            dir1 = np.array([np.cos(theta1), np.sin(theta1)])
            perp1 = np.array([-dir1[1], dir1[0]])
            A_pt = np.array(B) + w2 * perp1

            if theta2 is None:
                vertex = A_pt
            else:
                dir2 = np.array([np.cos(theta2), np.sin(theta2)])
                perp2 = np.array([-dir2[1], dir2[0]])
                B_pt = np.array(B) + w2 * perp2
                mat = np.column_stack((dir1, -dir2))
                if np.linalg.matrix_rank(mat) < 2:
                    avg = (perp1 + perp2) / 2
                    avg /= np.linalg.norm(avg)
                    vertex = np.array(B) + w2 * avg
                else:
                    t = np.linalg.solve(mat, B_pt - A_pt)[0]
                    vertex = A_pt + t * dir1

            right_side.append(vertex)

        # stitch left + right back into a closed polygon (butt→tip→butt)
        vertices_disp = left_side + right_side

        # inject arrowheads if requested
        if self.arrow_head or self.arrow_head_at_tail:
            vertices_disp = self._add_arrow_head_vertices_px(vertices_px=vertices_disp)

        # transform final display-space vertices back to data coordinates
        self.vertices_px = vertices_disp
        verts_data = np.array(
            [inv_trans.transform(p) for p in vertices_disp], dtype=np.float64
        )

        return verts_data

    def _get_angles(
        self,
        path: Union[
            List[Tuple[FloatLike, FloatLike]],
            NDArray[np.float64],
            List[np.ndarray],
        ],
    ) -> List[FloatLike]:
        """
        Compute the angle (in radians) that each segment in the path makes with the positive x-axis.

        The angles are returned in the range [0, 2π) and represent the counter-clockwise
        orientation of each segment. Input can be provided as a list of (x, y) tuples, a
        list of NumPy arrays, or a 2D NumPy array.

        Parameters
        ----------
        path : list of tuple, list of np.ndarray, or np.ndarray of shape (N, 2)
            Sequence of path points from which to calculate segment angles.

        Returns
        -------
        list of float
            List of angles (in radians) between each consecutive pair of points.
        """
        angles: List[FloatLike] = []
        # if it's an ndarray we can vectorize, otherwise fall back to the loop
        if isinstance(path, np.ndarray):
            dx = np.diff(path[:, 0])
            dy = np.diff(path[:, 1])
            thetas = np.mod(np.arctan2(dy, dx), 2 * np.pi)
            return thetas.tolist()
        else:
            # covers both List[tuple] and List[np.ndarray]
            for i in range(len(path) - 1):
                p1, p2 = path[i], path[i + 1]
                dx = np.array(p2[0] - p1[0])
                dy = np.array(p2[1] - p1[1])
                angles.append(float(np.arctan2(dy, dx) % (2 * np.pi)))

            return angles

    def _get_segment_length(self) -> List[FloatLike]:
        """
        Compute the Euclidean distance between each consecutive pair of path points.

        This method is used only for straight-arrow paths (`bezier=False`).
        The result is a list of segment lengths in data units.

        Returns
        -------
        list of float
            Distances between adjacent path points, one for each segment in the arrow.
        """
        distances = []
        for i in range(self.n_segments):
            p1, p2 = self.path[i], self.path[i + 1]
            x1, y1 = p1[0], p1[1]
            x2, y2 = p2[0], p2[1]
            d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            distances.append(d)

        return distances

    def draw_to_ax(self, ax: Axes, fill_arrow: bool = True) -> Axes:
        """
        Render the arrow polygon onto a specified matplotlib Axes.

        This method draws both the filled interior (if `fill_arrow=True`) and
        the stroked edge using the styling parameters provided during initialization
        (`fc`, `ec`, `lw`, `ls`, and `zorder`).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The Axes object on which to render the arrow.
        fill_arrow : bool, optional
            Whether to fill the interior of the arrow using `self.fc`. If False, only the outline
            will be rendered. Default is True.

        Returns
        -------
        matplotlib.axes.Axes
            The same Axes object with the arrow drawn onto it.
        """
        # fill the shape (face only)
        if fill_arrow:
            ax.fill(
                self.x_vertices,
                self.y_vertices,
                color=self.fc,
                zorder=self.zorder,
                ec="none",
            )

        # draw the outline (stroke/edge only)
        ax.plot(
            self.x_vertices,
            self.y_vertices,
            color=self.ec,
            linewidth=self.lw,
            linestyle=self.ls,
            zorder=self.zorder,
        )

        return ax

    def save_arrow(
        self,
        name: str = "./arrow.png",
    ) -> None:
        """
        Save a standalone image of the arrow to a PNG file.

        This method creates a new figure and axes, draws the arrow using internal geometry,
        and exports the result to the specified file path. The plot is rendered without axes,
        ticks, or frame, and includes padding to fit the arrow bounds.

        Parameters
        ----------
        name : str, optional
            File path for the saved image. Must end in '.png'. Default is './arrow.png'.
        """
        # plot lines and vertices
        self.ax = self.draw_to_ax(self.ax)

        plt.savefig(name, pad_inches=-0.1, dpi=500)


__all__ = ["ArrowETC"]
