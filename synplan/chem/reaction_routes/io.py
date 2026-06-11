"""Back-compat shim; import from `synplan.chem.reaction.routes.io` instead."""

from synplan._compat import deprecated_module

deprecated_module(__name__, "synplan.chem.reaction.routes.io")

from synplan.chem.reaction.routes.io import *
from synplan.chem.reaction.routes.io import (
    make_dict as make_dict,
    make_json as make_json,
    read_routes_csv as read_routes_csv,
    read_routes_json as read_routes_json,
    write_routes_csv as write_routes_csv,
    write_routes_json as write_routes_json,
)
