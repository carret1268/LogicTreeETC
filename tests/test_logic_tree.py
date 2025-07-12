from numpy import allclose, array
import matplotlib
import pytest

from logictree.logictree import ArrowETC, LogicBox, LogicTree

matplotlib.use("Agg")


def create_logic_box(tree, name, x, y, **kwargs):
    tree.add_box(
        xpos=x,
        ypos=y,
        text=name,
        box_name=name,
        bbox_fc="black",
        bbox_ec="white",
        ha="center",
        **kwargs,
    )
    return tree.boxes[name]


def test_logic_tree_init():
    tree = LogicTree()
    assert tree.title is None
    assert tree.xlims == (0, 100)
    assert tree.ylims == (0, 100)
    assert tree.title_font_dict["fontname"] == "Times New Roman"
    assert tree.font_dict["fontsize"] == 15

    tree = LogicTree(
        fig_size=(5, 5),
        xlims=(0, 10),
        ylims=(-10, 10),
        title="Test Tree",
        font_dict={"fontname": "Calibri", "fontsize": 20, "color": "black"},
        font_dict_title={"fontname": "Comic Sans", "fontsize": 24, "color": "magenta"},
        text_color=None,
        title_color=None,
    )
    assert tree.title == "Test Tree"
    assert tree.title_font_dict["fontname"] == "Comic Sans"

    tree = LogicTree(
        font_dict={"fontname": "Calibri", "fontsize": 20, "color": "black"},
        font_dict_title={"fontname": "Comic Sans", "fontsize": 24, "color": "magenta"},
        text_color="green",
        title_color="cyan",
    )
    assert tree.title_font_dict["color"] == "cyan"
    assert tree.font_dict["color"] == "green"


def test_get_pathsForBi_left_then_right():
    tree = LogicTree()
    tree.add_box(0, 0, "", "boxA", "black", "white")
    tree.add_box(0, 10, "", "boxB", "black", "white")

    with pytest.raises(ValueError):
        tree._get_pathsForBi_left_then_right(
            5,
            10,
            tree.boxes["boxA"],
            LogicBox(0, 10, "fail", "fail", "white", "black", {}),
            0,
        )
    with pytest.raises(ValueError):
        tree._get_pathsForBi_left_then_right(
            5,
            10,
            LogicBox(0, 10, "fail", "fail", "white", "black", {}),
            tree.boxes["boxB"],
            0,
        )

    expected = (
        [
            (5, 10),
            (-16.057347670250895, 10),
            (-16.057347670250895, -14.054834054834053),
        ],
        [
            (5, 10),
            (-16.057347670250895, 10),
            (-16.057347670250895, -14.054834054834053),
        ],
    )
    actual = tree._get_pathsForBi_left_then_right(
        5, 10, tree.boxes["boxA"], tree.boxes["boxB"], 0
    )
    assert all(allclose(a, b) for a, b in zip(actual[0], expected[0]))
    assert all(allclose(a, b) for a, b in zip(actual[1], expected[1]))


def test_add_box():
    tree = LogicTree(colormode="dark")

    tree.add_box(0, 2, "boxAText", "boxA", "black", "white")
    boxA = tree.boxes["boxA"]
    assert allclose([boxA.xRight], [1.8], 0.01)
    assert boxA.text == "boxAText"
    assert boxA.name == "boxA"
    assert boxA.face_color == "black"
    assert boxA.edge_color == "white"
    assert boxA.va == "center"
    assert boxA.ha == "right"
    assert boxA.angle == 0
    assert boxA.font_dict == {
        "fontname": "Times New Roman",
        "fontsize": 15,
        "color": "white",
    }

    tree.add_box(
        0,
        2,
        "boxBText",
        "boxB",
        "white",
        "black",
        va="bottom",
        ha="left",
        font_dict=dict(fontname="sans-serif", fontsize=22, color="purple"),
        angle=10,
    )
    boxB = tree.boxes["boxB"]
    assert allclose([boxB.xLeft], [-2.657], 0.01)
    assert boxB.text == "boxBText"
    assert boxB.name == "boxB"
    assert boxB.face_color == "white"
    assert boxB.edge_color == "black"
    assert boxB.va == "bottom"
    assert boxB.ha == "left"
    assert boxB.angle == 10
    assert boxB.font_dict == dict(fontname="sans-serif", fontsize=22, color="purple")

    tree.add_box(
        0,
        2,
        "boxCText",
        "boxC",
        "black",
        "white",
        use_tex_rendering=True,
        ul=True,
    )
    assert tree.boxes["boxC"].name == "boxC"

    with pytest.raises(ValueError):
        tree.add_box(10, 20, "boxDText", "boxC", "green", "cyan")


def test_add_arrow():
    tree = LogicTree()

    arrow0 = ArrowETC(ax=tree.ax, path=[(0, 0), (2, 2)], shaft_width=0.1)
    tree.add_arrow(arrow0)

    assert len(tree.boxes) == 0
    assert len(tree.arrows) == 1
    assert len(tree.arrows[0].vertices) == 8

    arrow1 = ArrowETC(ax=tree.ax, path=[(2, 4), (10, 10), (5, 5)], shaft_width=0.2)
    tree.add_arrow(arrow1, fill_arrow=False)

    assert len(tree.boxes) == 0
    assert len(tree.arrows) == 2
    assert len(tree.arrows[1].vertices) == 10


def test_add_arrow_between():
    tree = LogicTree()

    tree.add_arrow_between((0, 0), (1, 1))

    assert len(tree.boxes) == 0
    assert len(tree.arrows) == 1
    assert len(tree.arrows[0].vertices) == 8

    tree.add_arrow_between(
        (2, 0), (1, 1), fill_arrow=False, arrow_head=False, arrow_head_at_tail=True
    )

    assert len(tree.boxes) == 0
    assert len(tree.arrows) == 2
    assert len(tree.arrows[1].vertices) == 8

    tree.add_arrow_between(
        (2, 0), (1, 1), fill_arrow=False, arrow_head_at_tail=True, arrow_head=True
    )

    assert len(tree.boxes) == 0
    assert len(tree.arrows) == 3
    assert len(tree.arrows[2].vertices) == 11


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

    # Downward connection
    tree.add_connection_biSplit(
        tree.boxes["boxA"],
        tree.boxes["boxB"],
        tree.boxes["boxC"],
        fill_connection=True,
        fc_A="ec",
        ec_B="fc",
        ec_C="fc",
    )

    # Upward connection
    tree.add_connection_biSplit(
        tree.boxes["boxUpA"],
        tree.boxes["boxUpC"],
        tree.boxes["boxUpB"],
        arrow_head=False,
        fill_connection=False,
        fc_A="black",
        ec_B="white",
        fc_B="green",
        ec_C="white",
        fc_C="yellow",
        textLeft="Left Text",
        textRight="Right Text",
        text_kwargs={"fontsize": 20, "fontname": "Times New Roman", "color": "red"},
    )

    # Ensure arrows and boxes are correct
    assert len(tree.boxes) == 6
    assert len(tree.arrows) == 6
    assert len(tree.arrows[0].vertices) == 5
    assert len(tree.arrows[3].vertices) == 5
    assert len(tree.arrows[1].vertices) == 9
    assert len(tree.arrows[2].vertices) == 9
    assert len(tree.arrows[4].vertices) == 6
    assert len(tree.arrows[5].vertices) == 6

    # check stylings
    assert tree.arrows[0].fc == "white"
    assert tree.arrows[0].ec == "white"
    assert tree.arrows[1].fc == "black"
    assert tree.arrows[1].ec == "black"
    assert tree.arrows[2].fc == "black"
    assert tree.arrows[2].ec == "black"
    assert tree.arrows[3].fc == "black"
    assert tree.arrows[3].ec == "white"
    assert tree.arrows[4].fc == "yellow"
    assert tree.arrows[4].ec == "white"
    assert tree.arrows[5].fc == "green"
    assert tree.arrows[5].ec == "white"

    # raise errors for uninitialized boxes
    with pytest.raises(ValueError):
        tree.add_connection_biSplit(
            LogicBox(0, 10, "fail", "fail", "white", "black", {}),
            tree.boxes["boxB"],
            tree.boxes["boxC"],
        )
    with pytest.raises(ValueError):
        tree.add_connection_biSplit(
            tree.boxes["boxA"],
            LogicBox(0, 10, "fail", "fail", "white", "black", {}),
            tree.boxes["boxC"],
        )
    with pytest.raises(ValueError):
        tree.add_connection_biSplit(
            tree.boxes["boxA"],
            tree.boxes["boxB"],
            LogicBox(0, 10, "fail", "fail", "white", "black", {}),
        )


def test_add_connection():
    tree = LogicTree()

    # Create boxes for multiple directional connections
    tree.add_box(0, 0, "boxAText", "boxA", "black", "white")
    tree.add_box(0, 5, "boxBText", "boxB", "black", "white")
    tree.add_box(5, 0, "boxCRight", "boxC", "black", "white")
    tree.add_box(5, 10, "boxD", "boxD", "black", "white")

    # Simple top-down arrow, with explicit fill/edge color switch
    tree.add_connection(
        tree.boxes["boxA"],
        tree.boxes["boxB"],
        arrow_head=True,
        fill_connection=True,
        fc="ec",
        ec="fc",
        lw=1.0,
    )

    arr = tree.arrows[0]
    assert len(tree.boxes) == 4
    assert len(tree.arrows) == 1
    assert len(arr.vertices) == 8  # tail is closed
    assert arr.ec == "black"
    assert arr.fc == "white"
    assert arr.lw == 1.0

    # Add a right-to-left connection
    tree.add_connection(
        tree.boxes["boxC"],
        tree.boxes["boxA"],
        arrow_head=False,
        sideA="right",
        sideB="left",
        shaft_width=10,
        fill_connection=False,
        tip_offset=4,
        butt_offset=2,
    )
    arr = tree.arrows[1]
    assert len(tree.boxes) == 4
    assert len(tree.arrows) == 2
    assert len(arr.vertices) == 5
    assert arr.ec == "white"
    assert arr.fc == None  # only when fill_connect=False
    assert arr.lw == 0.7

    # Add a bottom-to-top connection
    tree.add_connection(
        tree.boxes["boxA"],
        tree.boxes["boxD"],
        arrow_head=False,
        sideA="bottomRight",
        sideB="bottomLeft",
    )

    arr = tree.arrows[2]
    assert len(tree.boxes) == 4
    assert len(tree.arrows) == 3
    assert len(arr.vertices) == 5
    assert arr.ec == "white"
    assert arr.fc == "black"

    # Connect to a raw coordinate (not a LogicBox)
    tree.add_connection(
        tree.boxes["boxC"],
        (20, 20),
        arrow_head=False,
        sideA="bottom",
        sideB="top",
        ec="yellow",
        fc="orange",
    )

    arr = tree.arrows[3]
    assert len(tree.boxes) == 4
    assert len(tree.arrows) == 4
    assert len(arr.vertices) == 5
    assert arr.ec == "yellow"
    assert arr.fc == "orange"

    # Connection with side offsets and directional logic
    tree.add_connection(
        tree.boxes["boxC"],
        tree.boxes["boxB"],
        sideA="topLeft",
        sideB="topRight",
        butt_offset=0.5,
        tip_offset=1.0,
        arrow_head=True,
        arrow_head_at_tail=True,
    )

    arr = tree.arrows[4]
    assert len(tree.boxes) == 4
    assert len(tree.arrows) == 5
    assert len(arr.vertices) == 11

    # Invalid LogicBox without initialized layout
    bad_box = LogicBox(20, 40, "fail", "fail", "white", "black", {})
    with pytest.raises(ValueError, match="boxA LogicBox layout is not initialized"):
        tree.add_connection(bad_box, tree.boxes["boxB"])

    with pytest.raises(ValueError, match="boxB LogicBox layout is not initialized"):
        tree.add_connection(tree.boxes["boxA"], bad_box)


def test_add_bezier_connection():
    tree = LogicTree()
    tree.ax.set_aspect("equal")

    # Setup logic boxes in different positions
    tree.add_box(0, 0, "A", "boxA", "black", "white")
    tree.add_box(10, 10, "B", "boxB", "white", "black")
    tree.add_box(15, 5, "C", "boxC", "black", "white")
    tree.add_box(-10, -10, "D", "boxD", "white", "black")

    # SMOOTH default connection
    tree.add_bezier_connection(
        boxA=tree.boxes["boxA"],
        boxB=tree.boxes["boxB"],
        style="smooth",
        arrow_head=True,
        fill_connection=True,
        fc="fc",
        ec="ec",
        lw=1.0,
        shaft_width=1,
    )

    arr = tree.arrows[0]
    assert len(tree.boxes) == 4
    assert len(tree.arrows) == 1
    assert len(arr.vertices) > 500  # bezier path
    assert arr.ec == "black"
    assert arr.fc == "white"

    # elbow style connection
    tree.add_bezier_connection(
        tree.boxes["boxA"],
        tree.boxes["boxC"],
        style="elbow",
        arrow_head=False,
        lw=0.5,
        sideA="top",
        sideB="topRight",
        arrow_head_length_multiplier=9 / 4,
        arrow_head_width_multiplier=3,
        fill_connection=False,
        fc="green",
        ec="purple",
    )

    arr = tree.arrows[1]
    assert len(tree.boxes) == 4
    assert len(tree.arrows) == 2
    assert len(arr.vertices) > 500  # bezier path
    assert arr.fc == "green"
    assert arr.ec == "purple"
    assert arr.lw == 0.5

    # s-curve style connection with butt/tip offsets
    tree.add_bezier_connection(
        tree.boxes["boxC"],
        tree.boxes["boxD"],
        style="s-curve",
        butt_offset=1.0,
        tip_offset=1.5,
        shaft_width=0.8,
        sideA="right",
        sideB="bottomRight",
    )

    arr = tree.arrows[2]
    assert len(tree.boxes) == 4
    assert len(tree.arrows) == 3
    assert len(arr.vertices) > 500  # bezier path
    assert arr.fc == "white"
    assert arr.ec == "black"
    assert arr.lw == 0.7

    # Manually specified control points
    custom_cp = [(5, 12), (7, 4)]
    tree.add_bezier_connection(
        tree.boxes["boxA"],
        tree.boxes["boxC"],
        control_points=custom_cp,
        fc="black",
        ec="white",
        sideA="bottom",
        sideB="bottomLeft",
    )

    arr = tree.arrows[3]
    assert len(tree.boxes) == 4
    assert len(tree.arrows) == 4
    assert len(arr.vertices) > 500  # bezier path

    # Connecting to a coordinate with smooth default
    tree.add_bezier_connection(
        tree.boxes["boxB"],
        (20, -10),
        style="smooth",
        arrow_head=True,
        tip_offset=1.0,
        sideA="left",
        sideB="topLeft",
    )

    arr = tree.arrows[4]
    assert len(tree.boxes) == 4
    assert len(tree.arrows) == 5
    assert len(arr.vertices) > 500  # bezier path

    # Invalid box layout should raise ValueError
    broken_box = LogicBox(0, 0, "fail", "fail", "white", "black", {})
    with pytest.raises(ValueError, match="boxA LogicBox layout is not initialized"):
        tree.add_bezier_connection(broken_box, tree.boxes["boxB"])

    with pytest.raises(ValueError, match="boxB LogicBox layout is not initialized"):
        tree.add_bezier_connection(tree.boxes["boxA"], broken_box)


# smoke test for match case blocks determining tip and butt offset
@pytest.mark.parametrize(
    "side",
    [
        "left",
        "topLeft",
        "top",
        "topRight",
        "right",
        "bottomRight",
        "bottom",
        "bottomLeft",
    ],
)
def test_add_connection_sides_cover_match_cases(side):
    tree = LogicTree()

    # Add two dummy boxes
    tree.add_box(0, 0, "A", "boxA", "black", "white")
    tree.add_box(10, 10, "B", "boxB", "white", "black")
    tree.add_box(-10, 10, "C", "boxC", "white", "black")

    # This should run the match-case logic for both sideA and sideB
    tree.add_connection(
        tree.boxes["boxA"],
        tree.boxes["boxB"],
        sideA=side,
        sideB=side,
        tip_offset=0.1,
        butt_offset=0.1,
    )
    tree.add_bezier_connection(
        tree.boxes["boxB"],
        tree.boxes["boxC"],
        sideA=side,
        sideB=side,
        tip_offset=0.2,
        butt_offset=0.1,
    )

    matplotlib.pyplot.close()


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
