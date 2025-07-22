"""
LogicTreeETC: Flexible logic trees and multi-segment arrows with vertex-level control.

Modules:
- LogicTree: Create and connect labeled decision boxes.
- LogicBoxETC: Render styled, rotatable text boxes.

See the documentation for details and examples.
"""

__version__ = "0.1.2"
__author__ = "E. Tyler Carr"
__email__ = "carret1268@gmail.com"
__license__ = "CC0 1.0 Universal"

from .logictree import LogicTree
from .logicbox import LogicBox
from .vector_detector import VectorDetector
from .arrow_etc import ArrowETC

__all__ = ["LogicTree", "LogicBox", "VectorDetector", "ArrowETC"]

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
