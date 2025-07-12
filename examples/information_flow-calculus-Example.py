"""
Visual Derivation Tree for Differentiating a Composite Function using the Product Rule

This example demonstrates the use of the LogicTreeETC and ArrowETC packages to visually
break down the symbolic differentiation of a composite function using both the product
rule and chain rule. The function being differentiated is:

    f(x) = e^{x^2} * sin(x)

The visualization highlights:

- The original function and the symbolic derivative form
- The product rule decomposition into u(x) and v(x)
- The application of the chain rule to differentiate u(x)
- The individual derivatives u'(x) and v'(x)
- The final combined expression for f'(x)
- A tree of logical relationships with annotated arrows showing data flow
- The ready-to-go dark theme when colormode="dark"

Boxes are color-coded and arranged to emphasize branching logic, making the derivation
visually intuitive for teaching or presentation purposes.

Generated output:
    resources/DecisionTreeCalculus-Example.png

Usage:
    Run this file as a script to generate the image:
        python examples/logictree_examples/information_flow-Calculus-Example.png
"""

from pathlib import Path
import sys
import os

# Compute absolute path to the parent directory of examples/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from logictree import ArrowETC, LogicTree  # noqa: E402


def main():
    tree = LogicTree(xlims=(-2, 100), ylims=(5, 97), colormode="dark")
    font_dict = dict(fontsize=32, color="white")

    problem_str = r"$f(x) = e^{x^2}\cdot\sin(x)$"
    func_box = tree.add_box(
        xpos=5,
        ypos=88,
        text=problem_str,
        bbox_fc=(0, 0, 0, 0),
        bbox_ec=(0, 0, 0, 0),
        box_name="function",
        use_tex_rendering=True,
        font_dict=font_dict,
        ha="left",
    )

    question_str = r"$\frac{d}{dx}f(x) = $ ?"
    q_box = tree.add_box(
        xpos=95,
        ypos=88,
        text=question_str,
        bbox_fc=(0, 0, 0, 0),
        bbox_ec=(0, 0, 0, 0),
        box_name="Problem Statement",
        use_tex_rendering=True,
        font_dict=font_dict,
        ha="right",
    )

    tree.add_connection(
        boxA=func_box, boxB=q_box, shaft_width=20, fc="#b0b0b0", ec="white", lw=1.8
    )

    product_rule_str = r"$f'(x) = u'(x)\cdot v(x) + u(x)\cdot v'(x)$"
    tree.add_box(
        xpos=50,
        ypos=74,
        text=product_rule_str,
        bbox_fc="#B0B0B0",
        bbox_ec="#ffff00",
        box_name="Product Rule",
        use_tex_rendering=True,
        font_dict=dict(color="black", fontsize=24),
        ha="center",
        lw=2,
    )

    u_str = r"$u(x) = e^{x^2}$"
    tree.add_box(
        xpos=24,
        ypos=58,
        text=u_str,
        bbox_fc=(0, 0, 0, 0),
        bbox_ec=(0, 0, 0, 0),
        box_name="u(x)",
        use_tex_rendering=True,
        font_dict=dict(color="#f480ff", fontsize=32),
        ha="center",
        lw=2,
    )

    v_str = r"$v(x) = sin(x)$"
    tree.add_box(
        xpos=76,
        ypos=58,
        text=v_str,
        bbox_fc=(0, 0, 0, 0),
        bbox_ec=(0, 0, 0, 0),
        box_name="v(x)",
        use_tex_rendering=True,
        font_dict=dict(fontsize=32, color="cyan"),
        ha="center",
        lw=2,
    )

    u_prime_str = r"$u'(x) = 2xe^{x^2}$"
    tree.add_box(
        xpos=41,
        ypos=50,
        text=u_prime_str,
        bbox_fc=(0, 0, 0, 0),
        bbox_ec=(0, 0, 0, 0),
        box_name="u'(x)",
        use_tex_rendering=True,
        font_dict=dict(color="#9b38ff", fontsize=32),
        ha="right",
        lw=2,
    )

    v_prime_str = r"$v'(x) = cos(x)$"
    tree.add_box(
        xpos=76,
        ypos=50,
        text=v_prime_str,
        bbox_fc=(0, 0, 0, 0),
        bbox_ec=(0, 0, 0, 0),
        box_name="v'(x)",
        use_tex_rendering=True,
        font_dict=dict(color="#279eff", fontsize=32),
        ha="center",
        lw=2,
    )

    answer_str = r"$f'(x) = 2xe^{x^2}\cdot\sin(x) + e^{x^2}\cdot\cos(x)$"
    tree.add_box(
        xpos=50,
        ypos=25,
        text=answer_str,
        bbox_fc="#e0ffe0",
        bbox_ec="#00981f",
        box_name="answer",
        use_tex_rendering=True,
        font_dict=dict(color="#000000", fontsize=30),
        ha="center",
        lw=5,
    )

    # add arrows from u, u', v and v'
    arrow_fc = "#777"
    u_path = [(9, 57), (1, 57), (1, 10), (68.5, 10), (68.5, 21.6)]
    u_arrow = ArrowETC(
        ax=tree.ax, path=u_path, shaft_width=13, fc=arrow_fc, ec="#000000", lw=2
    )
    tree.add_arrow(u_arrow)

    u_prime_path = [(10, 45), (10, 40), (32, 40), (32, 28)]
    u_prime_arrow = ArrowETC(
        ax=tree.ax, path=u_prime_path, shaft_width=13, fc=arrow_fc, ec="black", lw=2
    )
    tree.add_arrow(u_prime_arrow)

    v_path = [
        (56, 58.1),
        (50, 58.1),
        (50, 28),
    ]
    v_arrow = ArrowETC(
        ax=tree.ax, path=v_path, shaft_width=13, fc=arrow_fc, ec="#000000", lw=2
    )
    tree.add_arrow(v_arrow)

    v_prime_path = [
        (61, 45),
        (61, 40),
        (82, 40),
        (82, 28),
    ]
    v_prime_arrow = ArrowETC(
        ax=tree.ax, path=v_prime_path, shaft_width=13, fc=arrow_fc, ec="#000000", lw=2
    )
    tree.add_arrow(v_prime_arrow)

    output_path = (
        Path(__file__).resolve().parent.parent
        / "resources/logictree_examples/information_flow-Calculus-Example.png"
    )
    tree.save_as_png(
        file_name=output_path,
        dpi=900,
        content_padding=0.25,
    )


if __name__ == "__main__":
    main()
