from pathlib import Path
import tempfile
import matplotlib.image as mpimg
import pytest

from logictree import VectorDetector


def test_vector_detector_pipeline_full_coverage():
    image_path = (
        Path(__file__).resolve().parent / "../examples/img/image_of_nephron.png"
    )
    img = mpimg.imread(image_path)

    # test with and without extent
    detector = VectorDetector(img, extent=(0, 100, 0, 100))
    detector_no_extent = VectorDetector(img)

    # Harris detection with mask
    detector.detect_features(method="harris", max_points=20, mask_border=5)
    assert len(detector.points) <= 20

    # Shi-Tomasi detection
    detector.detect_features(method="shi-tomasi", max_points=20)
    assert len(detector.points) <= 20

    # Label and retrieve
    if detector.points:
        detector.label_point(0, "first")
        assert detector.get_point_by_label("first") == detector.points[0]
        assert "first" in detector.list_labeled_points()

        with pytest.raises(IndexError):
            detector.label_point(999, "fail")

    # Rescale passthrough
    if detector.points:
        raw_x, raw_y = detector.points[0]
        passthrough = detector_no_extent._rescale_to_extent(raw_x, raw_y)
        assert isinstance(passthrough, tuple)

    # Invalid method check
    with pytest.raises(ValueError):
        detector.detect_features(method="bad-method")  # type: ignore

    # Plot outputs (including empty-plot paths)
    with tempfile.TemporaryDirectory() as tmpdir:
        p1 = Path(tmpdir) / "detected.png"
        p2 = Path(tmpdir) / "labeled.png"

        detector.plot_detected_points(str(p1))
        detector.plot_labeled_points(str(p2))
        assert p1.exists() and p1.stat().st_size > 0
        assert p2.exists() and p2.stat().st_size > 0

        # Empty-labeled/point cases
        empty = VectorDetector(img)
        empty.plot_labeled_points(str(p2))
        empty.plot_detected_points(str(p1))
