from numpy import allclose
import matplotlib
import pytest
from unittest.mock import MagicMock

from logictree.LogicTreeETC import LogicTree
from logictree.LogicBoxETC import LogicBox

matplotlib.use("Agg")

def create_logic_box(tree, name, x, y, **kwargs):
    tree.add_box(
        xpos=x, ypos=y, text=name, box_name=name,
        bbox_fc="black", bbox_ec="white", ha="center", **kwargs
    )
    return tree.boxes[name]

def test_logic_tree_init():
    tree = LogicTree()
    assert tree.title is None
    assert tree.xlims == (0, 100)
    assert tree.ylims == (0, 100)
    assert tree.title_font["fontname"] == "Times New Roman"
    assert tree.font_dict["fontsize"] == 15

    tree = LogicTree(
        fig_size=(5, 5), xlims=(0, 10), ylims=(-10, 10),
        fig_fc="white", title="Test Tree",
        font_dict={"fontname": "Calibri", "fontsize": 20, "color": "black"},
        font_dict_title={"fontname": "Comic Sans", "fontsize": 24, "color": "magenta"},
        text_color=None, title_color=None,
    )
    assert tree.title == "Test Tree"
    assert tree.title_font["fontname"] == "Comic Sans"

    tree = LogicTree(
        font_dict={"fontname": "Calibri", "fontsize": 20, "color": "black"},
        font_dict_title={"fontname": "Comic Sans", "fontsize": 24, "color": "magenta"},
        text_color="green", title_color="cyan",
    )
    assert tree.title_font["color"] == "cyan"
    assert tree.font_dict["color"] == "green"

def test_get_pathsForBi_left_then_right():
    tree = LogicTree()
    tree.add_box(0, 0, "", "boxA", "black", "white")
    tree.add_box(0, 10, "", "boxB", "black", "white")

    with pytest.raises(ValueError):
        tree._get_pathsForBi_left_then_right(5, 10, tree.boxes["boxA"], LogicBox(0, 10, "fail", "fail", "white", "black", {}), 0)
    with pytest.raises(ValueError):
        tree._get_pathsForBi_left_then_right(5, 10, LogicBox(0, 10, "fail", "fail", "white", "black", {}), tree.boxes["boxB"], 0)

    expected = ([(5, 10), (0, 10), (0, 2.2)], [(5, 10), (0, 10), (0, 12.2)])
    actual = tree._get_pathsForBi_left_then_right(5, 10, tree.boxes["boxA"], tree.boxes["boxB"], 0)
    assert all(allclose(a, b) for a, b in zip(actual[0], expected[0]))
    assert all(allclose(a, b) for a, b in zip(actual[1], expected[1]))

def test_add_box():
    tree = LogicTree()
    tree.add_box(0, 2, "boxAText", "boxA", "black", "white")
    boxA = tree.boxes["boxA"]
    assert allclose([boxA.xRight], [2.2])
    assert boxA.text == "boxAText"

    tree.add_box(0, 2, "boxBText", "boxB", "black", "white", va="bottom", ha="left")
    boxB = tree.boxes["boxB"]
    assert allclose([boxB.xLeft], [-2.2])

    tree.add_box(0, 2, "boxCText", "boxC", "black", "white", use_tex_rendering=True, ul=True, ul_depth_width=(2, 3))
    assert tree.boxes["boxC"].name == "boxC"

    with pytest.raises(ValueError):
        tree.add_box(10, 20, "boxDText", "boxC", "green", "cyan")

def test_add_connection_biSplit():
    tree = LogicTree()
    # downward pointing tree
    tree.add_box(5, 5, "boxAText", "boxA", "black", "white")
    tree.add_box(0, 0, "boxBText", "boxB", "black", "white")
    tree.add_box(10, 0, "boxCText", "boxC", "black", "white")

    # upward pointing tree
    tree.add_box(0, 10, "boxUpB", "boxUpB", "black", "white")
    tree.add_box(10, 10, "boxUpC", "boxUpC", "black", "white")
    tree.add_box(5, 5, "boxUpA", "boxUpA", "black", "white")

    tree.ax = MagicMock()

    # Downward connection
    tree.add_connection_biSplit(
        tree.boxes["boxA"],
        tree.boxes["boxB"],
        tree.boxes["boxC"],
        fill_connection=True,
        fc_A="ec", ec_B="fc", ec_C="fc"
    )

    # Upward connection
    tree.add_connection_biSplit(
        tree.boxes["boxUpA"],
        tree.boxes["boxUpC"],
        tree.boxes["boxUpB"],
        arrow_head=False,
        fill_connection=False,
        fc_A="black", ec_B="white", ec_C="white"
    )

    # Confirm drawing calls were made
    assert tree.ax.plot.call_count >= 5
    assert tree.ax.fill.call_count >= 2

    for args, kwargs in tree.ax.fill.call_args_list:
        assert "color" in kwargs
        assert kwargs["color"] in {"black", "white"}

    # raise errors for uninitialized boxes
    with pytest.raises(ValueError):
        tree.add_connection_biSplit(LogicBox(0, 10, "fail", "fail", "white", "black", {}), tree.boxes["boxB"], tree.boxes["boxC"])
    with pytest.raises(ValueError):
        tree.add_connection_biSplit(tree.boxes["boxA"], LogicBox(0, 10, "fail", "fail", "white", "black", {}), tree.boxes["boxC"])
    with pytest.raises(ValueError):
        tree.add_connection_biSplit(tree.boxes["boxA"], tree.boxes["boxB"], LogicBox(0, 10, "fail", "fail", "white", "black", {}))

def test_add_connection():
    tree = LogicTree()
    tree.add_box(0, 0, "boxAText", "boxA", "black", "white")
    tree.add_box(0, 5, "boxBText", "boxB", "black", "white")
    tree.add_box(5, 0, "boxCRight", "boxCR", "black", "white")
    tree.add_box(5, 10, "boxD", "boxD", "black", "white")

    tree.ax = MagicMock()
    tree.add_connection(tree.boxes["boxA"], tree.boxes["boxB"], arrow_head=True, fill_connection=True, fc="ec", ec="fc", lw=1.0)
    assert tree.ax.plot.called
    assert tree.ax.fill.called

    tree.add_connection(tree.boxes["boxCR"], tree.boxes["boxA"])
    assert tree.ax.plot.call_count >= 2

    tree.add_connection(tree.boxes["boxA"], tree.boxes["boxD"])
    assert tree.ax.plot.call_count >= 3

    bad_box = LogicBox(0, 0, "fail", "fail", "white", "black", {})
    with pytest.raises(ValueError, match="boxA LogicBox layout is not initialized"):
        tree.add_connection(bad_box, tree.boxes["boxB"])

    with pytest.raises(ValueError, match="boxB LogicBox layout is not initialized"):
        tree.add_connection(tree.boxes["boxA"], bad_box)

    aligned_box = LogicBox(0, 0, "fail", "fail", "white", "black", {})
    aligned_box.xCenter = aligned_box.yCenter = 0
    aligned_box.xLeft = aligned_box.xRight = 0
    aligned_box.yTop = aligned_box.yBottom = 0

    with pytest.raises(ValueError, match="Boxes must be aligned"):
        tree.add_connection(aligned_box, aligned_box)

def test_save_as_png(tmp_path):
    tree = LogicTree(title="Arrow Test")
    a = create_logic_box(tree, "A", 10, 20)
    b = create_logic_box(tree, "B", 10, 10)
    tree.add_connection(a, b)
    output = tmp_path / "out.png"
    tree.save_as_png(str(output))

    assert output.exists()

def test_make_title(tmp_path):
    tree = LogicTree(title="BiSplit Test")
    a = create_logic_box(tree, "A", 20, 30)
    b = create_logic_box(tree, "B", 10, 10)
    c = create_logic_box(tree, "C", 30, 10)
    tree.add_connection_biSplit(a, b, c)
    tree.make_title(pos="center", new_title="New Title")
    tree.make_title(pos="center", new_title="New Title", consider_box_x=False)
    output = tmp_path / "tree.png"
    tree.save_as_png(str(output))
    assert output.exists()

    tree = LogicTree()
    with pytest.raises(ValueError, match="LogicTree.title is None"):
        tree.make_title()

    with pytest.raises(ValueError, match="pos must be one of"):
        tree.make_title(pos="invalid", new_title="New Title 2")
