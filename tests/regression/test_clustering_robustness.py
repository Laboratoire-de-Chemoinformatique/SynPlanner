"""Route-CGR clustering robustness invariants.

When a route's CGR cannot be composed (stereo, unbalanced atom maps,
multi-product issues), ``compose_all_route_cgrs`` in the dict-branch silently
stores ``{route_id: None}`` in the result. The downstream
``compose_all_sb_cgrs`` iterates that dict and calls ``compose_sb_cgr(None)``,
which crashes on ``None.connected_components`` with an AttributeError that
has nothing to do with the root cause.

This is brittle: any caller that combines load-from-file with clustering
hits a confusing crash on the first stereo-bearing route. The invariant
asserted here is that ``compose_all_sb_cgrs`` is *robust to None values* in
the input dict — failed-CGR routes must either be filtered out or surface
a clear, route-id-bearing error.

The test uses no real CGRs; it constructs a minimal input dict with one None
value and a real CGR (from a conftest fixture) and runs the function.
Surviving without an unattributed ``AttributeError`` is the invariant.
"""

from __future__ import annotations

import pytest

from synplan.chem.reaction_routes.route_cgr import compose_all_sb_cgrs


def test_compose_all_sb_cgrs_handles_none_entries(simple_cgr):
    """A None value in the route_cgrs_dict must not crash the whole batch.

    Acceptable behaviours:
      * ``None`` is filtered out and the result simply omits that key, OR
      * a ValueError / typed exception is raised that identifies the route_id.

    Unacceptable: raw ``AttributeError: 'NoneType' object has no attribute
    'connected_components'`` — the user has no way to map the error back
    to which route failed.
    """
    mixed = {
        "r_good": simple_cgr,
        "r_bad_stereo": None,  # what dict-branch of compose_all_route_cgrs produces
    }
    try:
        result = compose_all_sb_cgrs(mixed)
    except AttributeError as e:
        # Specifically the silent-None crash is what we're catching.
        if "NoneType" in str(e) or "connected_components" in str(e):
            pytest.fail(
                "compose_all_sb_cgrs crashed with an unattributed "
                f"AttributeError on a None-valued input entry: {e}. The "
                "function must either filter Nones or raise a typed error "
                "that names the offending route_id."
            )
        raise
    except (ValueError, TypeError, KeyError) as e:
        # Typed error that mentions the route_id is acceptable.
        assert "r_bad_stereo" in str(e) or "route" in str(e).lower(), (
            f"compose_all_sb_cgrs raised {type(e).__name__} but the message "
            f"does not identify which route_id was problematic: {e}"
        )
        return
    # Success path: None was filtered out.
    assert "r_good" in result, (
        "compose_all_sb_cgrs dropped the valid CGR entry while filtering "
        "the None; only the bad entry should be filtered."
    )
    # Either omit r_bad_stereo entirely or include it with a sentinel.
    if "r_bad_stereo" in result:
        assert result["r_bad_stereo"] is None or result["r_bad_stereo"] is False, (
            "compose_all_sb_cgrs returned a non-None, non-False value for a "
            "None input — the function must not fabricate a CGR for a failed "
            "route."
        )
