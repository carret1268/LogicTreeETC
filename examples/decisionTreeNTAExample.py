"""
Example decision tree diagram from dummy non-targeted analysis (NTA) data.

This example shows how to create a decision (logic) tree using the LogicTree package.
It demonstrates:
- Reading decision thresholds and counts from a CSV file.
- Adding boxes for each decision node, with labels, colors, alignment, and rotation.
- Connecting boxes with multi-segment arrows to show the decision flow.
- Annotating thresholds with LaTeX-formatted text (using `use_tex_rendering=True`).
- Styling everything: border colors, background colors, font properties, and titles.

This script produces a figure illustrating how samples in a dataset progress
through a decision tree based on replicate, CV, and MDL thresholds, with kept/removed flags.

Usage:
------
Run this script directly. It will save a PNG named
'DecisionTree_DSA-Example.png' in the examples directory.

Dependencies:
-------------
- logic_tree_data.csv: CSV containing the counts and threshold parameters.
- Python packages: matplotlib, pandas, your installed logictree package.
"""

from pathlib import Path
import sys
import os

from matplotlib.patches import BoxStyle
import pandas as pd

# Compute absolute path to the parent directory of examples/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from logictree import LogicTree  # noqa: E402


def make_tree():
    # Load CSV with counts and thresholds
    df = pd.read_csv(f"{Path(__file__).resolve().parent}/logic_tree_data.csv")

    # Build text for first row (total/missing samples)
    n_total_sample_occurence = df["n_total_sample_occurence"].iloc[0]
    str_total_sample_occurence = (
        f"Total Sample Occurence (N = {n_total_sample_occurence:,})"
    )
    n_missing_occurence = df["n_missing_occurence"].iloc[0]
    str_missing_occurence = f"Missing (N = {n_missing_occurence:,})"

    # Build text for replicate threshold results
    n_over_replicate = df["n_over_replicate"].iloc[0]
    str_over_replicate = f"$\\geq$ Replicate Threshold (N = {n_over_replicate:,})"
    n_under_replicate = df["n_under_replicate"].iloc[0]
    str_under_replicate = f"$<$ Replicate Threshold (N = {n_under_replicate:,})"

    # Build text for CV threshold results
    n_under_CV = df["n_under_CV"].iloc[0]
    str_under_CV = f"$\\leq$ CV Threshold (N = {n_under_CV:,})"
    n_over_CV = df["n_over_CV"].iloc[0]
    str_over_CV = f"$>$ CV Threshold (N = {n_over_CV:,})"

    # Build text for MDL threshold results under CV and over CV branches
    n_under_CV_over_MDL = df["n_under_CV_over_MDL"].iloc[0]
    str_under_CV_over_MDL = f"$\\geq$ MDL (N = {n_under_CV_over_MDL:,})"
    n_under_CV_under_MDL = df["n_under_CV_under_MDL"].iloc[0]
    str_under_CV_under_MDL = f"$<$ MDL (N = {n_under_CV_under_MDL:,})"
    n_over_CV_over_MDL = df["n_over_CV_over_MDL"].iloc[0]
    str_over_CV_over_MDL = f"$\\geq$ MDL (N = {n_over_CV_over_MDL:,})"
    n_over_CV_under_MDL = df["n_over_CV_under_MDL"].iloc[0]
    str_over_CV_under_MDL = f"$<$ MDL (N = {n_over_CV_under_MDL:,})"

    # Threshold values for annotations
    replicate_threshold = df["replicate_threshold"].iloc[0]
    replicate_threshold_str = f"\\textbf{{Replicate Threshold = {replicate_threshold}}}"
    CV_threshold = df["CV_threshold"].iloc[0]
    CV_threshold_str = f"\\textbf{{CV Threshold = {CV_threshold}}}"
    MDL = r"$\bigsymbol{\mu}_{\text{MB}} \text{ + } \bigsymbol{3\sigma}_{\text{MB}}$"
    MDL_str = f"\\textbf{{MDL = {MDL}}}"

    # Box y-positions and arrow width
    y_row1, y_row2, y_row3, y_row4 = 110, 60, 10, -30
    arr_width = 3.8
    tip_offset = 0.9

    # Axis limits
    xlims = (-50, 135)
    ylims = (-50, 135)
    logic_tree = LogicTree(
        xlims=xlims, ylims=ylims, title="Logic Tree - Sample Occurence"
    )

    # Add first row boxes
    logic_tree.add_box(
        xpos=75,
        ypos=y_row1,
        text=str_total_sample_occurence,
        box_name="Total Sample Occurence",
        bbox_fc="black",
        bbox_ec="white",
    )
    logic_tree.add_box(
        xpos=99,
        ypos=y_row1,
        text=str_missing_occurence,
        ha="left",
        box_name="Missing",
        bbox_fc="dimgrey",
        bbox_ec="xkcd:light blue grey",
    )

    # Add second row boxes
    logic_tree.add_box(
        xpos=55,
        ypos=y_row2,
        text=str_over_replicate,
        ha="right",
        box_name="Over Replicate",
        bbox_fc="black",
        bbox_ec="xkcd:bright sky blue",
    )
    logic_tree.add_box(
        xpos=65,
        ypos=y_row2,
        text=str_under_replicate,
        ha="left",
        box_name="Under Replicate",
        bbox_fc="dimgrey",
        bbox_ec="xkcd:light blue grey",
    )

    # Add third row boxes
    logic_tree.add_box(
        xpos=20,
        ypos=y_row3,
        text=str_under_CV,
        ha="right",
        box_name="Under CV",
        bbox_fc="black",
        bbox_ec="xkcd:water blue",
    )
    logic_tree.add_box(
        xpos=71,
        ypos=y_row3,
        text=str_over_CV,
        ha="left",
        box_name="Over CV",
        bbox_fc="xkcd:cherry",
        bbox_ec="xkcd:rosa",
    )

    # Add fourth row boxes
    logic_tree.add_box(
        xpos=-15,
        ypos=y_row4,
        text=str_under_CV_over_MDL,
        ha="right",
        box_name="Under CV, Over MDL",
        bbox_fc="black",
        bbox_ec="xkcd:ocean",
    )
    logic_tree.add_box(
        xpos=-6,
        ypos=y_row4,
        text=str_under_CV_under_MDL,
        ha="left",
        box_name="Under CV, Under MDL",
        bbox_fc="dimgrey",
        bbox_ec="xkcd:light blue grey",
    )
    logic_tree.add_box(
        xpos=96,
        ypos=y_row4,
        text=str_over_CV_over_MDL,
        ha="right",
        box_name="Over CV, Over MDL",
        bbox_fc="xkcd:rust orange",
        bbox_ec="xkcd:light salmon",
    )
    logic_tree.add_box(
        xpos=105,
        ypos=y_row4,
        text=str_over_CV_under_MDL,
        ha="left",
        box_name="Over CV, Under MDL",
        bbox_fc="dimgrey",
        bbox_ec="xkcd:light blue grey",
    )

    # Add arrows and bifurcations connecting boxes
    logic_tree.add_connection(
        logic_tree.boxes["Total Sample Occurence"],
        logic_tree.boxes["Missing"],
        arrow_head=True,
        arrow_width=arr_width,
        fill_connection=True,
        tip_offset=0.8,
        lw=1.2,
    )
    logic_tree.add_connection_biSplit(
        logic_tree.boxes["Total Sample Occurence"],
        logic_tree.boxes["Over Replicate"],
        logic_tree.boxes["Under Replicate"],
        arrow_head=True,
        arrow_width=arr_width,
        fill_connection=True,
        fc_A="ec",
        ec_B="xkcd:off white",
        fc_B="ec",
        lw=1.3,
        tip_offset=tip_offset,
    )
    logic_tree.add_connection_biSplit(
        logic_tree.boxes["Over Replicate"],
        logic_tree.boxes["Under CV"],
        logic_tree.boxes["Over CV"],
        arrow_head=True,
        arrow_width=arr_width,
        fill_connection=True,
        fc_A="ec",
        ec_B="xkcd:off white",
        fc_B="ec",
        lw=1.3,
        tip_offset=tip_offset,
    )
    logic_tree.add_connection_biSplit(
        logic_tree.boxes["Under CV"],
        logic_tree.boxes["Under CV, Over MDL"],
        logic_tree.boxes["Under CV, Under MDL"],
        arrow_head=True,
        arrow_width=arr_width,
        fill_connection=True,
        fc_A="ec",
        ec_B="xkcd:off white",
        fc_B="ec",
        lw=1.3,
        tip_offset=tip_offset,
    )
    logic_tree.add_connection_biSplit(
        logic_tree.boxes["Over CV"],
        logic_tree.boxes["Over CV, Over MDL"],
        logic_tree.boxes["Over CV, Under MDL"],
        arrow_head=True,
        arrow_width=arr_width,
        fill_connection=True,
        lw=1.3,
        tip_offset=tip_offset,
    )

    # Add annotation boxes for thresholds
    annotation_font = {
        "fontsize": 16,
        "color": "white",
    }  # you could adjust 'fontname' here too!
    y_row1_5 = (y_row1 + y_row2) / 2
    y_row2_5 = (y_row2 + y_row3) / 2
    y_row3_5 = (y_row3 + y_row4) / 2

    logic_tree.add_box(
        xpos=-4,
        ypos=y_row1_5,
        text=replicate_threshold_str,
        box_name="Replicate Threshold",
        bbox_fc=(1, 1, 1, 0),
        bbox_ec=(1, 1, 1, 0),
        ha="right",
        va="center",
        bbox_style=BoxStyle("Square", pad=0.3),
        font_dict=annotation_font,
        lw=1,
        use_tex_rendering=True,
        ul=True,
    )
    logic_tree.add_box(
        xpos=-32,
        ypos=y_row2_5,
        text=CV_threshold_str,
        box_name="CV Threshold",
        bbox_fc=(1, 1, 1, 0),
        bbox_ec=(1, 1, 1, 0),
        ha="right",
        va="center",
        bbox_style=BoxStyle("Square", pad=0.3),
        font_dict=annotation_font,
        lw=1,
        use_tex_rendering=True,
        ul=True,
    )
    logic_tree.add_box(
        xpos=-44,
        ypos=y_row3_5,
        text=MDL_str,
        box_name="MDL",
        bbox_fc=(1, 1, 1, 0),
        bbox_ec=(1, 1, 1, 0),
        ha="right",
        va="center",
        bbox_style=BoxStyle("Square", pad=0.3),
        font_dict=annotation_font,
        lw=1,
        use_tex_rendering=True,
        ul=True,
        ul_depth_width=("8pt", "1pt"),
        angle=20,
    )

    # Add kept/removed/flag text annotations
    logic_tree.add_box(
        xpos=27,
        ypos=y_row1_5 + arr_width * 0.85,
        text=r"\textit{\textbf{Kept}}",
        box_name="Kept",
        bbox_fc=(1, 1, 1, 0),
        bbox_ec=(1, 1, 1, 0),
        ha="right",
        va="bottom",
        bbox_style=BoxStyle("Square", pad=0.1),
        font_dict=annotation_font,
        use_tex_rendering=True,
        fs=12,
    )
    logic_tree.add_box(
        xpos=65,
        ypos=y_row1_5 + arr_width * 0.85,
        text=r"\textit{\textbf{Removed}}",
        box_name="Removed",
        bbox_fc=(1, 1, 1, 0),
        bbox_ec=(1, 1, 1, 0),
        ha="center",
        va="bottom",
        bbox_style=BoxStyle("Square", pad=0.1),
        font_dict=annotation_font,
        use_tex_rendering=True,
        fs=12,
    )
    logic_tree.add_box(
        xpos=56,
        ypos=y_row2_5 + arr_width * 0.85,
        text=r"\textit{\textbf{CV Flag}}",
        box_name="CV Flag",
        bbox_fc=(1, 1, 1, 0),
        bbox_ec=(1, 1, 1, 0),
        ha="center",
        va="bottom",
        bbox_style=BoxStyle("Square", pad=0.1),
        font_dict=annotation_font,
        use_tex_rendering=True,
        fs=12,
    )
    logic_tree.add_box(
        xpos=-9,
        ypos=y_row3_5 + arr_width * 0.85,
        text=r"\textit{\textbf{MDL Flag}}",
        box_name="MDL Flag Left",
        bbox_fc=(1, 1, 1, 0),
        bbox_ec=(1, 1, 1, 0),
        ha="left",
        va="bottom",
        bbox_style=BoxStyle("Square", pad=0.1),
        font_dict=annotation_font,
        use_tex_rendering=True,
        fs=12,
    )
    logic_tree.add_box(
        xpos=105,
        ypos=y_row3_5 + arr_width * 0.85,
        text=r"\textit{\textbf{MDL Flag}}",
        box_name="MDL Flag Right",
        bbox_fc=(1, 1, 1, 0),
        bbox_ec=(1, 1, 1, 0),
        ha="left",
        va="bottom",
        bbox_style=BoxStyle("Square", pad=0.1),
        font_dict=annotation_font,
        use_tex_rendering=True,
        fs=12,
    )

    # Add title and save
    logic_tree.make_title(pos="left")
    logic_tree.save_as_png(
        file_name=Path(__file__).resolve().parent / "DecisionTree_NTA-Example.png",
        dpi=900,
        content_padding=0.25,
    )


if __name__ == "__main__":
    make_tree()
