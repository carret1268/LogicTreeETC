"""
Example logic tree diagram: Suggested Study Order for Data Structures & Algorithms.

This example demonstrates how to create a logic tree using the LogicTree package
to visualize a recommended progression for studying key data structures. Boxes
represent topics like arrays, stacks, queues, and linked lists, while arrows
indicate the suggested learning sequence.

It shows:
- Creating boxes for each data structure/topic with customized labels and styles.
- Connecting boxes with directional arrows to indicate the recommended study order.
- Organizing topics into multiple rows to illustrate the progression clearly.
- Adding a title and saving the figure as a high-resolution PNG.

This script produces a figure illustrating a suggested path for learning
fundamental data structures, starting from arrays and ending with graphs.

Usage:
------
Run this script directly. It will save a PNG named
'DecisionTree_DSA-Example.png' in the examples directory.

Dependencies:
-------------
- Python packages: matplotlib, your installed logictree package.
"""
from pathlib import Path
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from logictree.LogicTreeETC import LogicTree

def make_ds_algo_study_order_tree():
    y_row1, y_row2, y_row3 = 85, 20, -40
    arr_width = 2.8
    tip_offset = 0.8
    xlims = (-30, 140)
    ylims = (-60, 110)

    logic_tree = LogicTree(xlims=xlims, ylims=ylims, title='Suggested Study Order: DSA')

    # Top row sequence: Arrays -> Stacks -> Queues -> Singly Linked Lists
    logic_tree.add_box(xpos=0, ypos=y_row1, text="Arrays", box_name="Arrays", bbox_fc='black', bbox_ec='xkcd:lime green')
    logic_tree.add_box(xpos=30, ypos=y_row1, text="Stacks", box_name="Stacks", bbox_fc='black', bbox_ec='xkcd:teal')
    logic_tree.add_box(xpos=60, ypos=y_row1, text="Queues", box_name="Queues", bbox_fc='black', bbox_ec='xkcd:sky blue')
    logic_tree.add_box(xpos=90, ypos=y_row1, text="Singly Linked Lists", box_name="SinglyLinked", bbox_fc='black', bbox_ec='xkcd:purple', ha='left')

    # Second row: Circular Linked Lists -> Doubly Linked Lists
    logic_tree.add_box(xpos=-5, ypos=y_row2, text="Circular Linked Lists", box_name="CircularLinked", bbox_fc='black', bbox_ec='xkcd:lavender', ha='left')
    logic_tree.add_box(xpos=120, ypos=y_row2, text="Doubly Linked Lists", box_name="DoublyLinked", bbox_fc='black', bbox_ec='xkcd:violet')

    # Third row: Trees and Graphs
    logic_tree.add_box(xpos=122.15, ypos=y_row3, text="Trees (BST, AVL, etc.)", box_name="Trees", bbox_fc='black', bbox_ec='xkcd:goldenrod')
    logic_tree.add_box(xpos=20, ypos=y_row3, text="Graph Algorithms", box_name="Graphs", bbox_fc='black', bbox_ec='xkcd:red orange')

    # Connect boxes in suggested order:
    logic_tree.add_connection(boxA=logic_tree.boxes['Arrays'], boxB=logic_tree.boxes['Stacks'],
                              arrow_head=True, arrow_width=arr_width, tip_offset=tip_offset, fc=(1,1,1,0.3), ec="white")
    logic_tree.add_connection(boxA=logic_tree.boxes['Stacks'], boxB=logic_tree.boxes['Queues'],
                              arrow_head=True, arrow_width=arr_width, tip_offset=tip_offset, fc=(1,1,1,0.3), ec="white")
    logic_tree.add_connection(boxA=logic_tree.boxes['Queues'], boxB=logic_tree.boxes['SinglyLinked'],
                              arrow_head=True, arrow_width=arr_width, tip_offset=tip_offset, fc=(1,1,1,0.3), ec="white")
    logic_tree.add_connection(boxA=logic_tree.boxes['SinglyLinked'], boxB=logic_tree.boxes['CircularLinked'],
                              arrow_head=True, arrow_width=arr_width, tip_offset=tip_offset, fc=(1,1,1,0.3), ec="white")
    logic_tree.add_connection(boxA=logic_tree.boxes['CircularLinked'], boxB=logic_tree.boxes['DoublyLinked'],
                              arrow_head=True, arrow_width=arr_width, tip_offset=tip_offset, fc=(1,1,1,0.3), ec="white")
    logic_tree.add_connection(boxA=logic_tree.boxes['DoublyLinked'], boxB=logic_tree.boxes['Trees'],
                              arrow_head=True, arrow_width=arr_width, tip_offset=tip_offset, fc=(1,1,1,0.3), ec="white")
    logic_tree.add_connection(boxA=logic_tree.boxes['Trees'], boxB=logic_tree.boxes['Graphs'],
                              arrow_head=True, arrow_width=arr_width, tip_offset=tip_offset, fc=(1,1,1,0.3), ec="white")

    logic_tree.make_title(pos='center')
    logic_tree.save_as_png(file_name=Path(__file__).resolve().parent / 'DecisionTree_DSA-Example.png', dpi=600, content_padding=0.1)

if __name__ == '__main__':
    make_ds_algo_study_order_tree()
