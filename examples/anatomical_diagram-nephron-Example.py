"""
Annotated Nephron Diagram with Auto-Detected Vertices and Biological Labels

This example demonstrates the combined use of the LogicTreeETC and ArrowETC packages to
construct a labeled nephron diagram from a "hand-drawn" image. It highlights the power of
VectorDetector for automatically identifying visual anchor points and applying precise,
annotated logic-based visual structures to scientific figures.

The workflow performs the following:
- Loads a stylized nephron image from disk
- Uses OpenCV-based feature detection (via VectorDetector) to identify high-interest vertices
- Automatically annotates selected points by label, enabling programmatic access later
- Plots all detected points for visual validation
- Draws region-specific bracket arrows around the proximal and distal convoluted tubules
- Creates labeled boxes (e.g., "Glomerulus", "Loop of Henle") and connects them to
  precise image coordinates using straight or Bezier arrows
- Adds arrows to indicate the direction of filtrate and fluid flow throughout the nephron

Key Features Illustrated:
- Integration of imshow() with a custom extent for accurate coordinate mapping
- Labeling auto-detected vertices for semantic reference
- Composing curved arrows using ArrowETC
- High-quality, publication-ready diagram composition using LogicTreeETC

Biological structures labeled include:
- Glomerulus
- Bowman's Capsule
- Proximal Convoluted Tubule (PCT)
- Loop of Henle
- Distal Convoluted Tubule (DCT)
- Collecting Duct

Generated output:
    resources/logictree_examples/anatomical_diagram-nephron-Example.png
"""

from pathlib import Path
import sys
import os

import matplotlib.image as mpimg

# Compute absolute path to the parent directory of examples/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from logictree import ArrowETC, LogicTree, VectorDetector  # noqa: E402


def main():
    tree = LogicTree(ylims=(10, 100))
    dir_path = Path(__file__).resolve().parent.parent / "resources"
    detector_output_path = (
        dir_path
        / "vector_detector_results_from_nephron_example/nephron-verts_auto_detected-Example.png"
    )
    detector_labeled_output_path = (
        dir_path
        / "vector_detector_results_from_nephron_example/nephron-labeled_verts_auto_detected-Example.png"
    )
    img_path = dir_path.parent / "examples/img/image_of_nephron.png"

    # use VectorDetector to automatically determine vertices
    extent = [0, 100, 0, 100]
    img = mpimg.imread(img_path)
    detector = VectorDetector(image=img, extent=extent)
    detector.detect_features(
        method="shi-tomasi", max_points=30, quality=0.2, min_distance=30
    )

    # this outputs a file "nephron-verts_auto_detected.png" which will be visually inspected
    detector.plot_detected_points(detector_output_path)

    # we determined some points of interest, P0, P1, P2, P6 and P7, so we will label them for easy access later
    detector.label_point(
        0, "proximal convoluted tubule start"
    )  # we are only interested in the x coordinate
    detector.label_point(
        1, "glomerulus"
    )  # we can access this later with detector.get_point_by_label("glomerulus")
    detector.label_point(2, "distal convoluted tubule end")
    detector.label_point(6, "proximal convoluted tubule end")
    detector.label_point(7, "distal convoluted tubule start")

    # we can now plot just the labeled points
    detector.plot_labeled_points(detector_labeled_output_path)

    # now plot the image on our tree.ax
    tree.ax.imshow(img, extent=extent)

    # add missing line to collecting duct for aesthetics
    arrow_path = [
        (90, 79),
        (90, 85.5),
        (93, 85.5),
        (93, 17.6),
    ]
    arrow = ArrowETC(
        ax=tree.ax,
        path=arrow_path,
        shaft_width=2,
        arrow_head=False,
        ec="black",
        fc="black",
        lw=0.15,
    )
    tree.add_arrow(arrow)

    arrow_path = [(90, 17.6), (90, 73.4)]
    arrow = ArrowETC(
        ax=tree.ax,
        path=arrow_path,
        shaft_width=2,
        arrow_head=False,
        ec="black",
        fc="black",
        lw=0.15,
    )
    tree.add_arrow(arrow)

    # draw brackets to indicate tubules
    pct_xi, _ = detector.get_point_by_label("proximal convoluted tubule start")
    pct_xf, _ = detector.get_point_by_label("proximal convoluted tubule end")
    pct_xf += 1  # shift end point a pinch
    arrow_path = [(pct_xi, 85), (pct_xi, 87), (pct_xf, 87), (pct_xf, 85)]
    arrow = ArrowETC(
        ax=tree.ax,
        path=arrow_path,
        shaft_width=1,
        arrow_head=False,
        ec="black",
        fc="black",
        lw=0.1,
    )
    tree.add_arrow(arrow)

    dct_xi, _ = detector.get_point_by_label("distal convoluted tubule start")
    dct_xf, _ = detector.get_point_by_label("distal convoluted tubule end")
    dct_xf -= 1
    arrow_path = [(dct_xi, 85), (dct_xi, 87), (dct_xf, 87), (dct_xf, 85)]
    arrow = ArrowETC(
        ax=tree.ax,
        path=arrow_path,
        shaft_width=1,
        arrow_head=False,
        ec="black",
        fc="black",
        lw=0.1,
    )
    tree.add_arrow(arrow)

    # add labels - first Glomerulus
    box_fc = "#dfdfdf"
    glom_box = tree.add_box(
        xpos=10,
        ypos=62,
        text="Glomerulus",
        box_name="glomerulus label",
        bbox_fc=box_fc,
        bbox_ec="black",
        text_color="black",
    )

    # we want to add connect the top of our annotation box to the vertex determined by VectorDetector
    glom_vert_x, glom_vert_y = detector.get_point_by_label("glomerulus")
    # after plotting once, I want to push the x-component back a hair
    glom_vert_x -= 0.3
    tree.add_bezier_connection(
        glom_box,
        (glom_vert_x, glom_vert_y),
        shaft_width=15,
        sideA="top",
        fc="#00b8c4",
        ec="black",
        tip_offset=1,
        butt_offset=1,
        n_bezier=1000,
        lw=1.2,
    )

    # proximal convoluted tubule
    pct_box = tree.add_box(
        xpos=15,
        ypos=96,
        text="Proximal Convoluted Tubule",
        box_name="pct label",
        bbox_fc=box_fc,
        bbox_ec="black",
        text_color="black",
        ha="center",
        va="bottom",
    )
    tree.add_bezier_connection(
        pct_box,
        (38.5, 87.5),
        shaft_width=15,
        sideA="right",
        fc="#00b8c4",
        ec="black",
        tip_offset=1,
        butt_offset=1,
        n_bezier=1000,
        lw=1.2,
    )

    # distal convoluted tubule
    dct_box = tree.add_box(
        xpos=74.25,
        ypos=96,
        text="Distal Convoluted Tubule",
        box_name="dct label",
        bbox_fc=box_fc,
        bbox_ec="black",
        text_color="black",
        ha="center",
        va="bottom",
    )
    tree.add_connection(
        dct_box,
        (74.25, 87.5),
        shaft_width=14,
        sideA="bottom",
        fc="#00b8c4",
        ec="black",
        tip_offset=0.4,
        butt_offset=0.7,
        lw=1.2,
        arrow_head_length_multiplier=1.2,
    )

    # bowman's capsule
    bc_box = tree.add_box(
        xpos=25,
        ypos=62,
        text="Bowman's Capsule",
        box_name="dc label",
        bbox_fc=box_fc,
        bbox_ec="black",
        text_color="black",
        ha="left",
        va="bottom",
    )
    tree.add_bezier_connection(
        bc_box,
        (16, 72),
        shaft_width=15,
        sideA="left",
        fc="#00b8c4",
        ec="black",
        tip_offset=0.4,
        butt_offset=0.7,
        n_bezier=1000,
        lw=1.2,
    )

    # loop of henle
    loh_box = tree.add_box(
        xpos=20,
        ypos=38,
        text="Loop of Henle",
        box_name="loh label",
        bbox_fc=box_fc,
        bbox_ec="black",
        text_color="black",
        ha="left",
        va="bottom",
    )
    tree.add_connection(
        loh_box,
        (52.2, 30),
        shaft_width=15,
        sideA="bottomRight",
        fc="#00b8c4",
        ec="black",
        tip_offset=0.4,
        butt_offset=0.1,
        lw=1.2,
    )

    # collecting duct
    cd_box = tree.add_box(
        xpos=76,
        ypos=30,
        text="Collecting Duct",
        box_name="cd label",
        bbox_fc=box_fc,
        bbox_ec="black",
        text_color="black",
        ha="center",
        va="bottom",
    )
    tree.add_bezier_connection(
        cd_box,
        (92, 43),
        shaft_width=15,
        sideA="top",
        fc="#00b8c4",
        ec="black",
        tip_offset=0.4,
        butt_offset=0.1,
        n_bezier=1000,
        lw=1.2,
    )

    # add a few arrows to indicate fluid flow
    path0 = [
        (34, 76.2),
        (38, 73.2),
        (43, 76.2),
    ]
    arrow0 = ArrowETC(
        ax=tree.ax,
        path=path0,
        shaft_width=5,
        arrow_head_width_multiplier=2.5,
        arrow_head_length_multiplier=7.5 / 4,
        fc="grey",
        ec="black",
        bezier=True,
        bezier_n=1000,
    )
    tree.add_arrow(arrow0)

    path1 = [
        (60.7, 54),
        (60.7, 68),
    ]
    arrow1 = ArrowETC(
        ax=tree.ax,
        path=path1,
        shaft_width=5,
        arrow_head_width_multiplier=2.5,
        arrow_head_length_multiplier=7.5 / 4,
        fc="grey",
        ec="black",
    )
    tree.add_arrow(arrow1)

    path2 = [
        (91.5, 68),
        (91.5, 54),
    ]
    arrow2 = ArrowETC(
        ax=tree.ax,
        path=path2,
        shaft_width=5,
        arrow_head_width_multiplier=3,
        arrow_head_length_multiplier=9 / 4,
        fc="grey",
        ec="black",
    )
    tree.add_arrow(arrow2)

    path3 = [
        (91.5, 34),
        (91.5, 22),
    ]
    arrow3 = ArrowETC(
        ax=tree.ax,
        path=path3,
        shaft_width=5,
        arrow_head_width_multiplier=3,
        arrow_head_length_multiplier=9 / 4,
        fc="grey",
        ec="black",
    )
    tree.add_arrow(arrow3)

    output_path = dir_path / "logictree_examples/anatomical_diagram-nephron-Example.png"
    tree.save_as_png(output_path, dpi=800, content_padding=0.5)


if __name__ == "__main__":
    main()
