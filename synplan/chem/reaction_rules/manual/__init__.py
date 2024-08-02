from .decompositions import rules as d_rules
from .transformations import rules as t_rules

hardcoded_rules = t_rules + d_rules

__all__ = ["hardcoded_rules"]
