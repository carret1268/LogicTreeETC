"""
Visual Decision Tree for Sample Filtering in a Non-Targeted Analysis (NTA) Workflow

This example illustrates how to construct a logic tree using the LogicTree package
to model how samples are retained or removed based on a sequence of analytical
threshold checks. The thresholds applied are:

- Replicate count threshold
- Coefficient of Variation (CV) threshold
- Method Detection Limit (MDL) threshold

The tree is constructed from a CSV file containing sample counts at each decision
point and dynamically labels each box with both logical conditions and sample counts.

Key Features:
- Loads parameters and counts from a CSV file
- Adds labeled boxes and arrows for each stage of decision logic
- Uses LaTeX-rendered threshold annotations
- Visually distinguishes kept, missing, and failed samples with colors
- Embeds intermediate annotations like "Replicate Threshold" and "CV Threshold"

Generated Output:
    resources/logictree_examples/DecisionTree_NTA-Example.png
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
    df = pd.read_csv(f"{Path(__file__).resolve().parent}/data/logic_tree_data.csv")

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
    arr_width = 15
    tip_offset = 0.3
    butt_offset = -0.4

    # Axis limits
    xlims = (-50, 135)
    ylims = (-50, 135)
    logic_tree = LogicTree(
        xlims=xlims,
        ylims=ylims,
        title="Logic Tree - Sample Occurence",
    )

    # predefine colors
    pass_fc = "white"
    pass_ec = "black"
    missing_fc = "#afafaf"
    missing_ec = "#3D3D3D"
    fail_fc = "#f58181"
    fail_ec = "#a40000"

    # Add first row boxes
    logic_tree.add_box(
        xpos=75,
        ypos=y_row1,
        text=str_total_sample_occurence,
        box_name="Total Sample Occurence",
        bbox_fc=pass_fc,
        bbox_ec=pass_ec,
    )
    logic_tree.add_box(
        xpos=99,
        ypos=y_row1,
        text=str_missing_occurence,
        ha="left",
        box_name="Missing",
        bbox_fc=missing_fc,
        bbox_ec=missing_ec,
    )

    # Add second row boxes
    logic_tree.add_box(
        xpos=55,
        ypos=y_row2,
        text=str_over_replicate,
        ha="right",
        box_name="Over Replicate",
        bbox_fc=pass_fc,
        bbox_ec=pass_ec,
    )
    logic_tree.add_box(
        xpos=65,
        ypos=y_row2,
        text=str_under_replicate,
        ha="left",
        box_name="Under Replicate",
        bbox_fc=missing_fc,
        bbox_ec=missing_ec,
    )

    # Add third row boxes
    logic_tree.add_box(
        xpos=20,
        ypos=y_row3,
        text=str_under_CV,
        ha="right",
        box_name="Under CV",
        bbox_fc=pass_fc,
        bbox_ec=pass_ec,
    )
    logic_tree.add_box(
        xpos=71,
        ypos=y_row3,
        text=str_over_CV,
        ha="left",
        box_name="Over CV",
        bbox_fc=fail_fc,
        bbox_ec=fail_ec,
    )

    # Add fourth row boxes
    logic_tree.add_box(
        xpos=-15,
        ypos=y_row4,
        text=str_under_CV_over_MDL,
        ha="right",
        box_name="Under CV, Over MDL",
        bbox_fc=pass_fc,
        bbox_ec=pass_ec,
    )
    logic_tree.add_box(
        xpos=-6,
        ypos=y_row4,
        text=str_under_CV_under_MDL,
        ha="left",
        box_name="Under CV, Under MDL",
        bbox_fc=missing_fc,
        bbox_ec=missing_ec,
    )
    logic_tree.add_box(
        xpos=96,
        ypos=y_row4,
        text=str_over_CV_over_MDL,
        ha="right",
        box_name="Over CV, Over MDL",
        bbox_fc=fail_fc,
        bbox_ec=fail_ec,
    )
    logic_tree.add_box(
        xpos=105,
        ypos=y_row4,
        text=str_over_CV_under_MDL,
        ha="left",
        box_name="Over CV, Under MDL",
        bbox_fc=missing_fc,
        bbox_ec=missing_ec,
    )

    # Add arrows and bifurcations connecting boxes
    arrow_text_style = {
        "fontname": "Times New Roman",
        "fontsize": 12,
        "color": "black",
        "fontstyle": "italic",
    }
    logic_tree.add_connection(
        logic_tree.boxes["Total Sample Occurence"],
        logic_tree.boxes["Missing"],
        arrow_head=True,
        shaft_width=arr_width,
        fill_connection=True,
        tip_offset=0.8,
        lw=1.2,
    )
    logic_tree.add_connection_biSplit(
        logic_tree.boxes["Total Sample Occurence"],
        logic_tree.boxes["Over Replicate"],
        logic_tree.boxes["Under Replicate"],
        arrow_head=True,
        shaft_width=arr_width,
        fill_connection=True,
        lw=1.3,
        tip_offset=tip_offset,
        textLeft="Kept",
        textRight="Removed",
        text_kwargs=arrow_text_style,
    )
    logic_tree.add_connection_biSplit(
        logic_tree.boxes["Over Replicate"],
        logic_tree.boxes["Under CV"],
        logic_tree.boxes["Over CV"],
        arrow_head=True,
        shaft_width=arr_width,
        fill_connection=True,
        lw=1.3,
        tip_offset=tip_offset,
        textRight="CV Flag",
        text_kwargs=arrow_text_style,
        butt_offset=butt_offset,
    )
    logic_tree.add_connection_biSplit(
        logic_tree.boxes["Under CV"],
        logic_tree.boxes["Under CV, Over MDL"],
        logic_tree.boxes["Under CV, Under MDL"],
        arrow_head=True,
        shaft_width=arr_width,
        fill_connection=True,
        lw=1.3,
        tip_offset=tip_offset,
        textRight="MDL Flag",
        text_kwargs=arrow_text_style,
        butt_offset=butt_offset,
    )
    logic_tree.add_connection_biSplit(
        logic_tree.boxes["Over CV"],
        logic_tree.boxes["Over CV, Over MDL"],
        logic_tree.boxes["Over CV, Under MDL"],
        arrow_head=True,
        shaft_width=arr_width,
        fill_connection=True,
        lw=1.3,
        tip_offset=tip_offset,
        textRight="MDL Flag",
        text_kwargs=arrow_text_style,
        butt_offset=butt_offset,
    )

    # Add annotation boxes for thresholds
    annotation_font = {
        "fontsize": 16,
        "color": "black",
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

    # Add title and save
    logic_tree.make_title(pos="left")
    logic_tree.save_as_png(
        file_name=Path(__file__).resolve().parent.parent
        / "resources/logictree_examples/DecisionTree_NTA-Example.png",
        dpi=900,
        content_padding=0.25,
    )


if __name__ == "__main__":
    make_tree()
