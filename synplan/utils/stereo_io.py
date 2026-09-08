"""Checked MDL output; the paired Chython backend prepares stereo geometry."""

from chython.files.RDFrw import ERDFWrite as RDFWrite
from chython.files.RDFrw import RDFRead as _RDFRead
from chython.files.SDFrw import ESDFWrite as SDFWrite
from chython.files.SDFrw import SDFRead as _SDFRead

__all__ = ["RDFRead", "RDFWrite", "SDFRead", "SDFWrite"]


class _CheckedRead:
    def __init__(self, file, **kwargs):
        kwargs.setdefault("calc_cis_trans", True)
        kwargs.setdefault("strict_stereo", not kwargs.get("ignore_stereo", False))
        # The concrete MDL reader follows this cooperative mixin in the MRO.
        super().__init__(file, **kwargs)  # ty: ignore[too-many-positional-arguments]


class RDFRead(_CheckedRead, _RDFRead):
    """Preserve E/Z and enable native strict MDL parsing when available."""


class SDFRead(_CheckedRead, _SDFRead):
    """Preserve E/Z and enable native strict MDL parsing when available."""
