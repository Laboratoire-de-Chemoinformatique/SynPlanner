"""Invariants for the raw RDF reader.

MDL's RDF format permits per-record framing in two equivalent ways:

* ``$RFMT`` / ``$MFMT`` between records (the form many tools emit),
* bare ``$RXN`` between records (the form used by, e.g., CHORISO and many
  ORCA/RDKit RDF outputs, equivalent to a concatenation of raw V2000 RXN
  records under a single RDF header).

``iter_rdf_text_blocks`` enters its "body" state on the first ``$RFMT``,
``$MFMT``, or ``$RXN`` line, but once in the body it only splits on
``$RFMT``/``$MFMT``. Bare-``$RXN`` files therefore yield exactly one block
no matter how many reactions they contain. ``count_rdf_records`` (counter
based on ``$RFMT``/``$MFMT`` only) returns 0 for the same input.

These tests assert the equivalence: both framings must produce the same
number of records.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synplan.utils.files import (
    RawReactionReader,
    count_rdf_records,
    iter_rdf_text_blocks,
)


def _minimal_rxn_block(diff: int) -> str:
    """A minimal MDL V2000 $RXN block representing a 1-atom 'reaction'.

    ``diff`` makes each call produce a slightly distinct molecule so the
    blocks aren't byte-identical, which protects against any deduplication that
    might otherwise mask multi-record handling.
    """
    return (
        "$RXN\n"
        "\n"
        "  ChemDraw\n"
        "\n"
        f"  1  1\n"
        "$MOL\n"
        "\n"
        "  Mrv2014\n"
        "\n"
        "  1  0  0  0  0  0            999 V2000\n"
        f"    {diff:.4f}    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "M  END\n"
        "$MOL\n"
        "\n"
        "  Mrv2014\n"
        "\n"
        "  1  0  0  0  0  0            999 V2000\n"
        f"    {diff + 1:.4f}    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "M  END\n"
    )


@pytest.fixture
def rdf_two_records_rfmt_framed(tmp_path: Path) -> Path:
    """Two reactions framed with $RFMT (the well-supported form)."""
    p = tmp_path / "two_rfmt.rdf"
    body = (
        "$RDFILE 1\n"
        "$DATM 01/01/26 00:00\n"
        "$RFMT\n" + _minimal_rxn_block(1.0) + "$RFMT\n" + _minimal_rxn_block(2.0)
    )
    p.write_text(body, encoding="utf-8")
    return p


@pytest.fixture
def rdf_two_records_rxn_framed(tmp_path: Path) -> Path:
    """Two reactions framed with bare $RXN (the CHORISO-style form)."""
    p = tmp_path / "two_rxn.rdf"
    body = (
        "$RDFILE 1\n"
        "$DATM 01/01/26 00:00\n" + _minimal_rxn_block(1.0) + _minimal_rxn_block(2.0)
    )
    p.write_text(body, encoding="utf-8")
    return p


def test_iter_rdf_text_blocks_splits_bare_rxn(rdf_two_records_rxn_framed: Path):
    """Two ``$RXN`` records must yield two blocks, just like ``$RFMT``."""
    blocks = list(iter_rdf_text_blocks(rdf_two_records_rxn_framed, 1))
    assert len(blocks) == 2, (
        f"iter_rdf_text_blocks yielded {len(blocks)} block(s) for a file "
        "with two bare-$RXN records. The reader only splits on "
        "$RFMT/$MFMT; CHORISO-style RDF files load only the first reaction."
    )


def test_count_rdf_records_counts_bare_rxn(rdf_two_records_rxn_framed: Path):
    """``count_rdf_records`` must report 2 for two bare-``$RXN`` records."""
    n = count_rdf_records(rdf_two_records_rxn_framed)
    assert n == 2, (
        f"count_rdf_records returned {n} for a file with two bare-$RXN "
        "records (expected 2). Reactor inputs whose total exceeds 0 should "
        "report a non-zero count regardless of which legal RDF framing the "
        "producer used."
    )


def test_raw_reader_iterates_all_rxn_records(
    rdf_two_records_rfmt_framed: Path,
    rdf_two_records_rxn_framed: Path,
):
    """``RawReactionReader`` returns the same record count from both framings."""
    with RawReactionReader(rdf_two_records_rfmt_framed) as r:
        rfmt_records = list(r)
    with RawReactionReader(rdf_two_records_rxn_framed) as r:
        rxn_records = list(r)
    assert len(rfmt_records) == 2, (
        f"$RFMT-framed RDF: got {len(rfmt_records)} records, expected 2 "
        "(fixture is broken)."
    )
    assert len(rxn_records) == len(rfmt_records), (
        f"$RXN-framed RDF yielded {len(rxn_records)} records vs "
        f"{len(rfmt_records)} for the equivalent $RFMT-framed file. Both "
        "are legal MDL framings; the reader must support both."
    )
