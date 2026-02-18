from chython import depict_settings

from .node import *
from .tree import *

depict_settings(aam=False)

__all__ = ["Node", "Tree"]
