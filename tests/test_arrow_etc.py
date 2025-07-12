import numpy as np
import pytest

from matplotlib.pyplot import subplots, close

from logictree import ArrowETC


def test_basic_straight_arrow():
    """Straight arrow with two points and arrowhead should initialize correctly and produce expected geometry."""
    _, ax = subplots()
    ax.set_aspect("equal")
    path = [(0, 0), (0, 5)]
    arrow = ArrowETC(ax=ax, path=path, shaft_width=1.0, arrow_head=True)

    assert arrow.n_path == 2
    assert arrow.n_segments == 1
    assert arrow.segment_lengths[0] == pytest.approx(5.0)
    assert arrow.vertices.shape[1] == 2
    assert arrow.path_angles[0] == pytest.approx(np.pi / 2)

    close()


def test_multi_segmented_arrow():
    """Arrow with multiple segments and bends should compute correct attributes and vertices."""
    _, ax = subplots()
    ax.set_aspect("equal")
    path = [(0, 0), (0, 2), (4, -1)]
    arrow = ArrowETC(
        ax=ax,
        path=path,
        shaft_width=1.0,
        arrow_head=True,
        ec="black",
        fc="green",
        lw=2,
        ls="--",
        zorder=5,
    )

    assert arrow.n_path == 3
    assert arrow.segment_lengths[0] == pytest.approx(2)
    assert arrow.ec == "black"
    assert arrow.fc == "green"
    assert arrow.lw == 2
    assert arrow.ls == "--"
    assert arrow.zorder == 5

    expected_angle = np.arctan2(-3, 4) % (2 * np.pi)
    assert arrow.path_angles[1] == pytest.approx(expected_angle)

    close()


def test_headless_arrow():
    """Arrow without arrowhead should still produce correct geometry and attributes."""
    _, ax = subplots()
    ax.set_aspect("equal")
    path = [(0, 0), (0, 2), (4, -1)]
    arrow = ArrowETC(ax=ax, path=path, shaft_width=1.0, arrow_head=False)

    assert arrow.n_segments == 2

    expected_angle = np.arctan2(-3, 4) % (2 * np.pi)
    assert arrow.path_angles[1] == pytest.approx(expected_angle)

    close()


def test_bezier_arrow():
    """Arrow constructed with bezier=True should generate curve samples and vertices matching bezier_n."""
    _, ax = subplots()
    path = [(0, 0), (2, 4), (4, 0)]
    arrow = ArrowETC(
        ax=ax, path=path, shaft_width=0.5, arrow_head=True, bezier=True, bezier_n=100
    )

    assert arrow.curve_samples.shape[0] == 100
    assert arrow.vertices.shape[1] == 2

    close()


def test_bezier_arrow_no_head():
    """Arrow with bezier=True and arrow_head=False should generate correct curve vertices without head."""
    _, ax = subplots()
    path = [(0, 0), (2, 4), (4, 0)]
    arrow = ArrowETC(
        ax=ax, path=path, shaft_width=0.5, arrow_head=False, bezier=True, bezier_n=50
    )

    # Should have correct number of bezier samples
    assert hasattr(arrow, "curve_samples")
    assert arrow.curve_samples.shape[0] == 50

    # The last curve vertex should be near the last path point - shaft_width/2, since no arrowhead
    last_curve_point = arrow.curve_samples[-1]
    last_vertex = arrow.vertices[len(arrow.curve_samples) - 1]
    assert abs(np.linalg.norm(last_curve_point - last_vertex) - 0.5 / 2) < 0.3

    # Vertices should still wrap around to first point (closing the polygon)
    assert np.allclose(arrow.vertices[0], arrow.vertices[-1], atol=1e-8)

    close()


def test_invalid_inputs():
    """Invalid cases like too few points or negative width should raise ValueError."""
    _, ax = subplots()
    with pytest.raises(ValueError):
        ArrowETC(ax=ax, path=[(0, 0)], shaft_width=1.0)
    with pytest.raises(ValueError):
        ArrowETC(ax=ax, path=[(0, 0), (1, 1)], shaft_width=-1.0)

    close()


def test_save_arrow_creates_image(tmp_path):
    """save_arrow() should produce a PNG file."""
    _, ax = subplots()
    path = [(0, 0), (3, 3)]
    arrow = ArrowETC(ax=ax, path=path, shaft_width=0.5, arrow_head=True)
    out_file = tmp_path / "arrow.png"
    arrow.save_arrow(name=str(out_file))

    assert out_file.exists()
    assert out_file.stat().st_size > 0

    close()


def test_get_segment_length_calculates_distances():
    """_get_segment_length should compute expected segment lengths."""
    _, ax = subplots()
    path = [(0, 0), (3, 4)]
    arrow = ArrowETC(ax=ax, path=path, shaft_width=0.5)
    lengths = arrow._get_segment_length()

    assert lengths[0] == pytest.approx(5.0)

    close()


def test_get_angles_horizontal_and_vertical():
    """_get_angles should return correct angles for horizontal and vertical paths."""
    _, ax = subplots()
    path = [(0, 0), (2, 0), (2, 3)]
    arrow = ArrowETC(ax=ax, path=path, shaft_width=0.5)
    angles = arrow._get_angles(path)

    assert angles[0] == pytest.approx(0)  # right
    assert angles[1] == pytest.approx(np.pi / 2)  # up

    close()
