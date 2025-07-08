import sys
import os

from matplotlib.patches import BoxStyle
import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from logictree import LogicBox  # noqa: E402


def test_logic_box_initialization_defaults():
    font_dict = {"fontsize": 12, "color": "white"}
    box = LogicBox(
        xpos=10,
        ypos=20,
        text="Test",
        box_name="box1",
        bbox_fc="black",
        bbox_ec="white",
        font_dict=font_dict,
    )
    assert box.x == 10
    assert box.y == 20
    assert box.text == "Test"
    assert box.name == "box1"
    assert box.face_color == "black"
    assert box.edge_color == "white"
    assert box.font_dict == font_dict
    assert box.va == "center"
    assert box.ha == "left"
    assert box.lw == 1.6
    assert box.angle == 0.0
    assert box.style["boxstyle"].__class__.__name__ == "Square"
    assert box.xLeft is None
    assert box.xRight is None
    assert box.yBottom is None
    assert box.yTop is None
    assert box.width is None
    assert box.height is None
    assert box.xCenter is None
    assert box.yCenter is None


def test_logic_box_initialization_with_style():
    font_dict = {"fontsize": 14}
    style = BoxStyle("Round", pad=0.2)
    box = LogicBox(
        xpos=5,
        ypos=5,
        text="Styled",
        box_name="box2",
        bbox_fc="blue",
        bbox_ec="red",
        font_dict=font_dict,
        bbox_style=style,
        lw=2.5,
        va="top",
        ha="right",
        angle=15,
    )
    assert box.va == "top"
    assert box.ha == "right"
    assert box.lw == 2.5
    assert box.angle == 15
    assert box.style["boxstyle"].__class__.__name__ == "Round"


def test_logic_box_invalid_alignment_raises():
    with pytest.raises(ValueError, match="Invalid va: middle"):
        LogicBox(
            xpos=0,
            ypos=0,
            text="Bad VA",
            box_name="box3",
            bbox_fc="black",
            bbox_ec="black",
            font_dict={},
            va="middle",
        )

    with pytest.raises(ValueError, match="Invalid ha: mid"):
        LogicBox(
            xpos=0,
            ypos=0,
            text="Bad HA",
            box_name="box4",
            bbox_fc="black",
            bbox_ec="black",
            font_dict={},
            ha="mid",
        )


def test_logic_box_repr():
    box = LogicBox(
        xpos=100,
        ypos=200,
        text="Hello",
        box_name="title",
        bbox_fc="white",
        bbox_ec="black",
        font_dict={},
    )
    rep = repr(box)
    assert rep == "<LogicBox(name='title', text='Hello', x=100, y=200)>"
