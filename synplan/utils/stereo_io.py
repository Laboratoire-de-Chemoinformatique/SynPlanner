"""Checked MDL output; the paired Chython backend prepares stereo geometry."""

from inspect import signature
from io import StringIO

from chython.files.RDFrw import ERDFWrite
from chython.files.RDFrw import RDFRead as _RDFRead
from chython.files.SDFrw import ESDFWrite
from chython.files.SDFrw import SDFRead as _SDFRead

try:
    from chython.files.mdl.write import prepare_stereo_molecule
except ImportError:
    prepare_stereo_molecule = None


class _CheckedMDL:
    def _write_molecule(self, molecule, write3d=None):
        if prepare_stereo_molecule is None:
            # Released 1.105 has no coordinate-safety contract. Verify the
            # serialized structure before accepting its output. This fallback
            # can refuse E/Z depictions that require the paired backend fix.
            prepared = molecule.copy()
            if write3d is None:
                prepared.clean2d(engine="smilesdrawer")
            stream = StringIO()
            ESDFWrite(stream).write(prepared)
            restored = next(
                iter(
                    SDFRead(
                        StringIO(stream.getvalue()), ignore=False, calc_cis_trans=True
                    )
                )
            )
            if str(restored) != str(molecule):
                raise ValueError(
                    "MDL stereo cannot be preserved with this Chython version; use CXSMILES or the updated backend"
                )
            molecule = prepared
        return super()._write_molecule(molecule, write3d)


class RDFWrite(_CheckedMDL, ERDFWrite):
    """V3000 reaction output that refuses stereo loss."""


class SDFWrite(_CheckedMDL, ESDFWrite):
    """V3000 molecule output that refuses stereo loss."""


class _CheckedRead:
    def __init__(self, file, **kwargs):
        kwargs.setdefault("calc_cis_trans", True)
        if "strict_stereo" in signature(super().__init__).parameters:
            kwargs.setdefault("strict_stereo", not kwargs.get("ignore_stereo", False))
        super().__init__(file, **kwargs)


class RDFRead(_CheckedRead, _RDFRead):
    """Preserve E/Z and enable native strict MDL parsing when available."""


class SDFRead(_CheckedRead, _SDFRead):
    """Preserve E/Z and enable native strict MDL parsing when available."""
