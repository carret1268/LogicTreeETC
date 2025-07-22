"""
VectorML: A lightweight feature point detection module for annotated vector graphics.

This module defines the `VectorDetector` class, which allows users to extract, map,
and label salient image features (e.g., corners or vertices) using OpenCV's Harris and
Shi-Tomasi algorithms. It supports visualization of raw and labeled points, and rescaling
coordinates to match `extent`-based data axes used in `matplotlib`.

Typical use case:
    - Load an image using `matplotlib.image.imread` or `cv2.imread`
    - Initialize a `VectorDetector` with the image and optional extent
    - Call `detect_features()` to find candidate points
    - Use `plot_detected_points()` to visualize the detected points
    - Retroactively label points by using `label_point()`
    - Use `plot_labeled_points()` for visualizing just your labeled points
    - Access labeled points by calling `get_point_by_label()`

Example
-------
```python
import matplotlib.image as mpimg
from logictree.VectorML import VectorDetector

# Load image and create detector
img = mpimg.imread("nephron_diagram.png")
detector = VectorDetector(img, extent=(0, 100, 0, 100))

# Detect features using Shi-Tomasi
detector.detect_features(method="shi-tomasi", max_points=50)

# Label a few key points
detector.label_point(0, "glomerulus_center")
detector.label_point(5, "loop_of_Henle_tip")

# Save annotated image
detector.plot_detected_points("outputs/all_detected.png")
detector.plot_labeled_points("outputs/labeled_points.png")
```
"""

from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import cv2
from matplotlib import colormaps
import matplotlib.cm as cm  # For ScalarMappable
import matplotlib.colors as mcolors
from matplotlib.pyplot import colorbar, close, subplots
import numpy as np

PathType = Union[Path, str]


class VectorDetector:
    """
    Detects feature points (vertices of interest) from a grayscale or color image using OpenCV.
    Automatically maps detected coordinates to a user-defined `extent` used in matplotlib `imshow()`.

    Attributes
    ----------
    original_image : np.ndarray
        The input image in RGB or grayscale.
    gray : np.ndarray
        Grayscale version of the input image, normalized to uint8.
    extent : Optional[tuple[float, float, float, float]]
        Image extent in the form (x0, x1, y0, y1) to map pixel coordinates to data space.
    points : list[tuple[float, float]]
        List of detected (x, y) coordinates, optionally mapped to extent.
    labels : dict[str, tuple[float, float]]
        Mapping from user-defined string labels to detected coordinates.
    """

    def __init__(
        self,
        image: np.ndarray,
        extent: Optional[Tuple[float, float, float, float]] = None,
    ):
        """
        Parameters
        ----------
        image : np.ndarray
            Input image as a NumPy array (grayscale or RGB/RGBA).
        extent : tuple[float, float, float, float], optional
            Data coordinates to map to, matching matplotlib's `imshow(..., extent=...)`.
        """
        self.original_image = image

        if image.ndim == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            self.gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.ndim == 2:
            self.gray = image.copy()
        else:
            raise ValueError(
                "Unsupported image shape: must be grayscale, RGB, or RGBA."
            )

        if self.gray.dtype != np.uint8:
            self.gray = cv2.normalize(
                src=self.gray,
                dst=self.gray.copy(),
                alpha=0.0,
                beta=255.0,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_8U,
            ).astype(np.uint8)
        else:
            self.gray = self.gray.astype(np.uint8)

        self.extent = extent
        self.points: list[tuple[float, float]] = []
        self.labels: dict[str, tuple[float, float]] = {}

    def detect_features(
        self,
        method: Literal["harris", "shi-tomasi"] = "harris",
        max_points: int = 100,
        quality: float = 0.01,
        min_distance: float = 10.0,
        mask_border: int = 0,
    ) -> None:
        """
        Detects feature points from the image using the specified method.

        Parameters
        ----------
        method : {'harris', 'shi-tomasi'}
            Corner detection algorithm to use.
        max_points : int
            Maximum number of feature points to return.
        quality : float
            Quality level threshold (used in Shi-Tomasi).
        min_distance : float
            Minimum spacing between detected features (used in Shi-Tomasi).
        mask_border : int
            If > 0, masks out a border of pixels around the image to reduce noisy edge detection.
        """
        mask: Optional[np.ndarray] = None
        if mask_border > 0:
            mask = np.ones_like(self.gray, dtype=np.uint8)
            mask[:mask_border, :] = 0
            mask[-mask_border:, :] = 0
            mask[:, :mask_border] = 0
            mask[:, -mask_border:] = 0

        if method == "harris":
            dst = cv2.cornerHarris(self.gray, blockSize=2, ksize=3, k=0.04)
            kernel = np.ones((3, 3), dtype=np.uint8)
            dst_dilated = cv2.dilate(dst.astype(np.float32), kernel)
            threshold = 0.01 * float(dst_dilated.max())
            corners_raw = np.argwhere(dst_dilated > threshold)
            corners: np.ndarray = corners_raw[:, ::-1].astype(np.float32)  # (x, y)

            self.points = []
            for pt in corners[:max_points]:
                x, y = pt
                self.points.append(self._rescale_to_extent(float(x), float(y)))

        elif method == "shi-tomasi":
            corners = cv2.goodFeaturesToTrack(
                self.gray,
                maxCorners=max_points,
                qualityLevel=quality,
                minDistance=min_distance,
                mask=mask,
            )
            self.points = []
            if corners is not None:
                for pt in corners:
                    x, y = map(float, pt.reshape(-1))
                    self.points.append(self._rescale_to_extent(x, y))
            else:
                self.points = []

        else:
            raise ValueError("Unsupported method: choose 'harris' or 'shi-tomasi'.")

    def _rescale_to_extent(self, x: float, y: float) -> Tuple[float, float]:
        if self.extent is None:
            return x, y

        x0, x1, y0, y1 = self.extent
        height, width = self.gray.shape

        x_scaled = x0 + (x / width) * (x1 - x0)
        y_scaled = y0 + (1 - y / height) * (y1 - y0)
        return x_scaled, y_scaled

    def label_point(self, index: int, label: str) -> None:
        """
        Assigns a label to a previously detected point by index.

        Parameters
        ----------
        index : int
            The index of the detected point in `self.points`.
        label : str
            A user-defined string label to assign to the selected point.

        Raises
        ------
        IndexError
            If the provided index is out of range for the list of detected points.
        """
        if 0 <= index < len(self.points):
            self.labels[label] = self.points[index]
        else:
            raise IndexError("Point index out of range.")

    def get_point_by_label(self, label: str) -> Optional[Tuple[float, float]]:
        """
        Retrieves the coordinates of a labeled point.

        Parameters
        ----------
        label : str
            The user-defined label assigned to a point.

        Returns
        -------
        tuple[float, float] or None
            The (x, y) coordinates of the labeled point, or None if the label is not found.
        """
        return self.labels.get(label)

    def list_labeled_points(self) -> dict[str, Tuple[float, float]]:
        """
        Returns a dictionary of all user-labeled points.

        Returns
        -------
        dict[str, tuple[float, float]]
            A copy of the internal label-to-point mapping.
        """
        return self.labels.copy()

    def plot_detected_points(self, save_path: PathType) -> None:
        """
        Visualizes all detected feature points with numeric labels and saves the plot.

        Parameters
        ----------
        save_path : str | Path
            File path where the output figure will be saved (e.g., 'outputs/all_detected.png').

        Notes
        -----
        Each detected point is colored and numbered in the colorbar legend.
        The image background will reflect the original image with optional extent mapping.
        """
        save_path = Path(save_path).resolve()
        fig, ax = subplots()

        if self.extent:
            ax.imshow(self.original_image, extent=self.extent)
        else:
            ax.imshow(self.original_image)

        n = len(self.points)
        if n == 0:
            print("No points to plot.")
            return

        cmap = colormaps.get_cmap("viridis").resampled(n)
        bounds = np.arange(n + 1)
        norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=n)

        for i, (x, y) in enumerate(self.points):
            ax.plot(x, y, "o", markersize=6, color=cmap(i))

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = colorbar(sm, ax=ax, boundaries=bounds, ticks=np.arange(n) + 0.5)
        cbar.ax.set_yticklabels([f"P{i}" for i in range(n)])
        cbar.ax.tick_params(which="minor", length=0)

        ax.set_title("Detected Vertices")
        ax.axis("off")
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        close(fig)

    def plot_labeled_points(
        self,
        save_path: PathType,
        legend_loc: Literal[
            "best",
            "upper right",
            "upper left",
            "lower left",
            "lower right",
            "right",
            "center left",
            "center right",
            "lower center",
            "upper center",
            "center",
        ] = "best",
    ) -> None:
        """
        Plots only the points that have been user-labeled and saves the image.

        Parameters
        ----------
        save_path : str | Path
            File path where the output figure will be saved (e.g., 'outputs/labeled_points.png').
        legend_loc : str, optional
            Location of the legend in the plot. Defaults to "best".

        Notes
        -----
        Each labeled point is assigned a color and displayed with its label in the legend.
        """
        if not self.labels:
            print("No labeled points to plot.")
            return

        save_path = Path(save_path).resolve()

        fig, ax = subplots()

        if self.extent:
            ax.imshow(self.original_image, extent=self.extent)
        else:
            ax.imshow(self.original_image)

        label_names = list(self.labels.keys())
        n = len(label_names)
        cmap = colormaps.get_cmap("tab10" if n <= 10 else "tab20").resampled(n)

        for i, label in enumerate(label_names):
            x, y = self.labels[label]
            ax.plot(x, y, "o", color=cmap(i), markersize=6, label=label)

        ax.legend(title="Labeled Points", loc=legend_loc, fontsize=8, title_fontsize=9)
        ax.set_title("User-Labeled Vertices")
        ax.axis("off")
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        close(fig)


__all__ = ["VectorDetector"]
