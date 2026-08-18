"""Raw building-block cost estimates for exported synthesis routes.

Cost calculation can be applied directly when building-block preparation made
no structural changes beyond standardization. If building blocks were
deprotected during preparation, calculate route cost only after restoring the
real protected inputs with ``expand_deprotected_building_blocks`` from
``synplan/chem/reaction/routes/postprocess/deprotected_building_blocks.py``;
otherwise the route leaves do not represent the catalogue building blocks that
must be priced.

The bb catalogue used to develop this module was aggregated and
prepared by the Protolaw bb_preparer project:
https://github.com/Protolaw/bb_preparer. The estimator itself is independent
of that project and accepts any compatible prepared price table.

The catalogue prices used here are deliberately treated as raw price-per-gram
values.  Vendor currencies are not normalized by the building-block dataset,
so the resulting numbers must not be labelled as a particular currency.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, Literal

from chython import smiles as smiles_parser
from chython.containers import MoleculeContainer

from synplan.chem.reaction.routes.tree_ops import node_children
from synplan.chem.utils import safe_canonicalization
from synplan.utils.files import ChemicalRecord, iter_chemical_records, open_text

_COST_UNITS = "raw_catalogue_units"
BuildingBlockCostStatus = Literal["priced", "missing", "unpriced"]


class RouteCostError(ValueError):
    """Raised when a route or building-block price catalogue is invalid."""


@dataclass(frozen=True, slots=True)
class BuildingBlockCost:
    """Cost contribution from one distinct terminal building block.

    ``equivalents`` is the number of occurrences of the molecule among the
    route leaves.  ``cost_per_gram`` is its contribution per gram of final
    target, rather than the catalogue price-per-gram value (which is stored in
    ``price_per_gram``).
    """

    smiles: str
    equivalents: int
    molecular_weight: float
    vendor: str | None
    price_column: str | None
    price_per_gram: float | None
    cost_per_mol: float | None
    cost_per_gram: float | None
    status: BuildingBlockCostStatus


@dataclass(frozen=True, slots=True)
class RouteCostEstimate:
    """Raw minimum catalogue cost for one synthesis-route tree.

    Principal totals are ``None`` whenever at least one leaf is missing from
    the catalogue or has no positive price.  In that case the corresponding
    ``priced_cost_*`` values retain the subtotal for the priced leaves.
    """

    target_smiles: str
    target_molecular_weight: float
    cost_per_mol: float | None
    cost_per_gram: float | None
    priced_cost_per_mol: float
    priced_cost_per_gram: float
    complete: bool
    building_blocks: tuple[BuildingBlockCost, ...]
    missing_smiles: tuple[str, ...]
    unpriced_smiles: tuple[str, ...]
    price_columns: tuple[str, ...]
    cost_units: str = field(default=_COST_UNITS, init=False)


@dataclass(slots=True)
class _LeafGroup:
    molecule: MoleculeContainer
    molecular_weight: float
    equivalents: int
    aliases: set[str]


@dataclass(frozen=True, slots=True)
class _PriceMatch:
    price_per_gram: float
    price_column: str
    column_index: int
    line_number: int

    @property
    def vendor(self) -> str:
        return self.price_column[:-4]

    @property
    def rank(self) -> tuple[float, int, int]:
        return (self.price_per_gram, self.column_index, self.line_number)


@dataclass(frozen=True, slots=True)
class _PreparedRoute:
    target_smiles: str
    target_molecular_weight: float
    groups: dict[str, _LeafGroup]


def _route_error(path: str, message: str) -> RouteCostError:
    return RouteCostError(f"invalid route node at {path}: {message}")


def _node_smiles(node: Mapping[str, Any], path: str) -> str:
    value = node.get("smiles")
    if not isinstance(value, str) or not value.strip():
        raise _route_error(path, "'smiles' must be a non-empty string")
    return value.strip()


def _collect_leaf_smiles(route: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Validate an alternating route tree and return target and leaf SMILES."""
    if not isinstance(route, Mapping):
        raise RouteCostError("route must be a mapping containing a molecule root")

    leaves: list[str] = []
    active_nodes: set[int] = set()

    def visit(node: Any, expected_type: str, path: str) -> None:
        if not isinstance(node, Mapping):
            raise _route_error(path, "node must be a mapping")
        node_id = id(node)
        if node_id in active_nodes:
            raise _route_error(path, "route tree contains a cycle")

        node_type = node.get("type")
        if node_type != expected_type:
            raise _route_error(
                path,
                f"expected type {expected_type!r}, found {node_type!r}",
            )
        _node_smiles(node, path)
        try:
            children = node_children(node, path=path)
        except ValueError as error:
            raise RouteCostError(str(error)) from error

        active_nodes.add(node_id)
        try:
            if expected_type == "mol":
                if not children:
                    leaves.append(_node_smiles(node, path))
                    return
                if len(children) != 1:
                    raise _route_error(
                        path,
                        "a non-terminal molecule must have exactly one reaction child",
                    )
                visit(children[0], "reaction", f"{path}.children[0]")
                return

            if not children:
                raise _route_error(
                    path, "a reaction must have at least one molecule child"
                )
            for index, child in enumerate(children):
                visit(child, "mol", f"{path}.children[{index}]")
        finally:
            active_nodes.remove(node_id)

    if route.get("type") != "mol":
        raise _route_error("$", "the route root must have type 'mol'")
    target_smiles = _node_smiles(route, "$")
    visit(route, "mol", "$")
    return target_smiles, leaves


def _canonical_route_molecule(
    value: str, *, role: str
) -> tuple[str, MoleculeContainer, float]:
    try:
        molecule = smiles_parser(value)
    except Exception as error:
        raise RouteCostError(f"invalid {role} SMILES {value!r}: {error}") from error
    if not isinstance(molecule, MoleculeContainer):
        raise RouteCostError(f"invalid {role} SMILES {value!r}: not a molecule")
    try:
        canonical = safe_canonicalization(molecule, clean_stereo=False)
        canonical_smiles = str(canonical)
        molecular_weight = float(canonical.molecular_mass)
    except Exception as error:
        raise RouteCostError(
            f"cannot canonicalize {role} SMILES {value!r}: {error}"
        ) from error
    if not canonical_smiles or not math.isfinite(molecular_weight):
        raise RouteCostError(
            f"cannot calculate a finite molecular weight for {role} SMILES {value!r}"
        )
    if molecular_weight <= 0.0:
        raise RouteCostError(
            f"molecular weight must be positive for {role} SMILES {value!r}"
        )
    return canonical_smiles, canonical, molecular_weight


def _group_leaves(
    leaf_smiles: list[str],
    parsed: dict[str, tuple[str, MoleculeContainer, float]] | None = None,
) -> dict[str, _LeafGroup]:
    groups: dict[str, _LeafGroup] = {}
    if parsed is None:
        parsed = {}
    for value in leaf_smiles:
        result = parsed.get(value)
        if result is None:
            result = _canonical_route_molecule(value, role="leaf")
            parsed[value] = result
        canonical_smiles, molecule, molecular_weight = result
        group = groups.get(canonical_smiles)
        if group is None:
            groups[canonical_smiles] = _LeafGroup(
                molecule=molecule,
                molecular_weight=molecular_weight,
                equivalents=1,
                aliases={value, canonical_smiles},
            )
        else:
            group.equivalents += 1
            group.aliases.add(value)
    return groups


def _prepare_route(
    route: Mapping[str, Any],
    parsed_leaves: dict[str, tuple[str, MoleculeContainer, float]],
) -> _PreparedRoute:
    target_value, leaf_values = _collect_leaf_smiles(route)
    target_smiles, _target_molecule, target_weight = _canonical_route_molecule(
        target_value, role="target"
    )
    return _PreparedRoute(
        target_smiles=target_smiles,
        target_molecular_weight=target_weight,
        groups=_group_leaves(leaf_values, parsed_leaves),
    )


def _combine_leaf_groups(
    prepared_routes: Iterator[_PreparedRoute],
) -> dict[str, _LeafGroup]:
    combined: dict[str, _LeafGroup] = {}
    for prepared in prepared_routes:
        for canonical_smiles, group in prepared.groups.items():
            existing = combined.get(canonical_smiles)
            if existing is None:
                combined[canonical_smiles] = _LeafGroup(
                    molecule=group.molecule,
                    molecular_weight=group.molecular_weight,
                    equivalents=group.equivalents,
                    aliases=set(group.aliases),
                )
            else:
                existing.aliases.update(group.aliases)
    return combined


def _is_price_column(name: str) -> bool:
    normalized = name.strip().casefold()
    return len(normalized) > len("_ppg") and normalized.endswith("_ppg")


def _empty_catalogue_price_columns(path: Path) -> tuple[str, ...]:
    """Recover header fields after a valid, header-only record iteration."""
    try:
        with open_text(path) as handle:
            header = next(csv.reader(handle, delimiter="\t"))
    except StopIteration as error:
        raise RouteCostError(f"{path}: expected a header row") from error
    except (OSError, UnicodeError, csv.Error) as error:
        raise RouteCostError(
            f"cannot read building-block catalogue {path}: {error}"
        ) from error
    return tuple(name.strip() for name in header if _is_price_column(name))


def _catalogue_records(
    path: Path,
) -> tuple[Iterator[ChemicalRecord], tuple[str, ...]]:
    records = iter_chemical_records(
        path,
        input_format="tsv",
        chemistry_columns={"smiles": "smiles"},
    )
    try:
        first = next(records)
    except StopIteration:
        price_columns = _empty_catalogue_price_columns(path)
        return iter(()), price_columns
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        raise RouteCostError(
            f"invalid building-block catalogue {path}: {error}"
        ) from error

    price_columns = tuple(
        name for name in first.metadata_names if _is_price_column(name)
    )
    return chain((first,), records), price_columns


def _record_prices(
    record: ChemicalRecord,
    price_columns: tuple[str, ...],
    path: Path,
) -> tuple[tuple[float, int, str], ...]:
    if record.format_error:
        raise RouteCostError(
            f"invalid building-block catalogue {path}:{record.line_number}: "
            f"{record.format_error}"
        )
    metadata = dict(zip(record.metadata_names, record.metadata))
    positive: list[tuple[float, int, str]] = []
    for index, column in enumerate(price_columns):
        raw_value = metadata[column].strip()
        if not raw_value:
            continue
        try:
            price = float(raw_value)
        except ValueError as error:
            raise RouteCostError(
                f"invalid price in {path}:{record.line_number}, column {column!r}: "
                f"{raw_value!r} is not numeric"
            ) from error
        if not math.isfinite(price):
            raise RouteCostError(
                f"invalid price in {path}:{record.line_number}, column {column!r}: "
                f"{raw_value!r} is not finite"
            )
        if price < 0.0:
            raise RouteCostError(
                f"invalid price in {path}:{record.line_number}, column {column!r}: "
                f"{raw_value!r} is negative"
            )
        if price > 0.0:
            positive.append((price, index, column))
    return tuple(positive)


def _catalogue_matches(
    path: Any, groups: Mapping[str, _LeafGroup]
) -> tuple[tuple[str, ...], set[str], dict[str, _PriceMatch]]:
    aliases: dict[str, str] = {}
    for canonical_smiles, group in groups.items():
        for alias in group.aliases:
            previous = aliases.setdefault(alias, canonical_smiles)
            if previous != canonical_smiles:
                raise RouteCostError(
                    f"route SMILES alias {alias!r} resolves to multiple leaf identities"
                )

    catalog_matcher = getattr(path, "best_prices", None)
    if catalog_matcher is not None:
        price_columns, matched, raw_best = catalog_matcher(aliases)
        if not price_columns:
            raise RouteCostError("building-block catalog contains no *_ppg prices")
        best = {
            canonical: _PriceMatch(
                price_per_gram=match[0],
                price_column=match[1],
                column_index=match[2],
                line_number=match[3],
            )
            for canonical, match in raw_best.items()
        }
        return price_columns, matched, best

    records, price_columns = _catalogue_records(path)
    if not price_columns:
        raise RouteCostError(
            f"invalid building-block catalogue {path}: expected at least one '*_ppg' "
            "column"
        )

    matched: set[str] = set()
    best: dict[str, _PriceMatch] = {}
    try:
        for record in records:
            positive_prices = _record_prices(record, price_columns, path)
            canonical_smiles = aliases.get(record.chemistry)
            if canonical_smiles is None:
                continue
            matched.add(canonical_smiles)
            for price, column_index, column in positive_prices:
                candidate = _PriceMatch(
                    price_per_gram=price,
                    price_column=column,
                    column_index=column_index,
                    line_number=record.line_number,
                )
                previous = best.get(canonical_smiles)
                if previous is None or candidate.rank < previous.rank:
                    best[canonical_smiles] = candidate
    except RouteCostError:
        raise
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        raise RouteCostError(
            f"invalid building-block catalogue {path}: {error}"
        ) from error
    return price_columns, matched, best


def _finite_cost(value: float, *, smiles: str) -> float:
    if not math.isfinite(value):
        raise RouteCostError(
            f"cost calculation is not finite for building block {smiles!r}"
        )
    return value


def _build_estimate(
    prepared: _PreparedRoute,
    price_columns: tuple[str, ...],
    matched: set[str],
    best_prices: Mapping[str, _PriceMatch],
) -> RouteCostEstimate:
    target_smiles = prepared.target_smiles
    target_weight = prepared.target_molecular_weight
    building_blocks: list[BuildingBlockCost] = []
    missing_smiles: list[str] = []
    unpriced_smiles: list[str] = []
    priced_costs: list[float] = []

    for canonical_smiles, group in prepared.groups.items():
        price_match = best_prices.get(canonical_smiles)
        if canonical_smiles not in matched:
            missing_smiles.append(canonical_smiles)
            building_blocks.append(
                BuildingBlockCost(
                    smiles=canonical_smiles,
                    equivalents=group.equivalents,
                    molecular_weight=group.molecular_weight,
                    vendor=None,
                    price_column=None,
                    price_per_gram=None,
                    cost_per_mol=None,
                    cost_per_gram=None,
                    status="missing",
                )
            )
            continue
        if price_match is None:
            unpriced_smiles.append(canonical_smiles)
            building_blocks.append(
                BuildingBlockCost(
                    smiles=canonical_smiles,
                    equivalents=group.equivalents,
                    molecular_weight=group.molecular_weight,
                    vendor=None,
                    price_column=None,
                    price_per_gram=None,
                    cost_per_mol=None,
                    cost_per_gram=None,
                    status="unpriced",
                )
            )
            continue

        cost_per_mol = _finite_cost(
            group.equivalents * group.molecular_weight * price_match.price_per_gram,
            smiles=canonical_smiles,
        )
        cost_per_gram = _finite_cost(
            cost_per_mol / target_weight, smiles=canonical_smiles
        )
        priced_costs.append(cost_per_mol)
        building_blocks.append(
            BuildingBlockCost(
                smiles=canonical_smiles,
                equivalents=group.equivalents,
                molecular_weight=group.molecular_weight,
                vendor=price_match.vendor,
                price_column=price_match.price_column,
                price_per_gram=price_match.price_per_gram,
                cost_per_mol=cost_per_mol,
                cost_per_gram=cost_per_gram,
                status="priced",
            )
        )

    priced_cost_per_mol = _finite_cost(math.fsum(priced_costs), smiles=target_smiles)
    priced_cost_per_gram = _finite_cost(
        priced_cost_per_mol / target_weight, smiles=target_smiles
    )
    complete = not missing_smiles and not unpriced_smiles
    return RouteCostEstimate(
        target_smiles=target_smiles,
        target_molecular_weight=target_weight,
        cost_per_mol=priced_cost_per_mol if complete else None,
        cost_per_gram=priced_cost_per_gram if complete else None,
        priced_cost_per_mol=priced_cost_per_mol,
        priced_cost_per_gram=priced_cost_per_gram,
        complete=complete,
        building_blocks=tuple(building_blocks),
        missing_smiles=tuple(missing_smiles),
        unpriced_smiles=tuple(unpriced_smiles),
        price_columns=price_columns,
    )


def estimate_route_costs(
    routes: Mapping[Any, Mapping[str, Any]],
    building_blocks_file: Any,
) -> dict[Any, RouteCostEstimate]:
    """Estimate a pool of routes with one shared catalogue scan.

    Route preparation and result order follow the input mapping's iteration
    order. The union of all terminal building-block aliases is collected before
    the catalogue is streamed, so a route pool costs approximately the same to
    look up as a single route. An empty pool returns immediately without
    opening the catalogue.

    :param routes: Mapping of stable route identifiers to nested route trees.
    :param building_blocks_file: Headered building-block price TSV.
    :return: Estimates keyed and ordered like the input mapping.
    :raises RouteCostError: If any route or the catalogue is malformed.
    """
    if not isinstance(routes, Mapping):
        raise RouteCostError("routes must be a mapping of identifiers to route trees")
    if not routes:
        return {}

    parsed_leaves: dict[str, tuple[str, MoleculeContainer, float]] = {}
    prepared = {
        route_id: _prepare_route(route, parsed_leaves)
        for route_id, route in routes.items()
    }
    combined_groups = _combine_leaf_groups(iter(prepared.values()))
    catalogue = (
        building_blocks_file
        if getattr(building_blocks_file, "best_prices", None) is not None
        else Path(building_blocks_file)
    )
    price_columns, matched, best_prices = _catalogue_matches(catalogue, combined_groups)
    return {
        route_id: _build_estimate(prepared_route, price_columns, matched, best_prices)
        for route_id, prepared_route in prepared.items()
    }


def estimate_route_cost(
    route: Mapping[str, Any], building_blocks_file: Any
) -> RouteCostEstimate:
    """Estimate the minimum raw building-block cost of one route.

    The final target is normalized to one mole.  Since route JSON has no
    stoichiometric coefficients, every terminal-molecule occurrence contributes
    one molar equivalent and all reactions are assumed to have 100% yield.  Zero
    and blank catalogue prices mean unavailable; positive prices are compared
    across every ``*_ppg`` column and matching duplicate row.

    Matching is intentionally limited to exact route and stereo-preserving
    canonical SMILES aliases.  The million-row catalogue is streamed without
    parsing or canonicalizing every catalogue molecule.

    :param route: One nested molecule/reaction route, for example
        ``routes_json[route_id]``.
    :param building_blocks_file: Headered TSV with exactly one ``SMILES`` column
        and at least one ``*_ppg`` price-per-gram column.
    :return: A complete total or an explicitly incomplete priced subtotal.
    :raises RouteCostError: If the route or catalogue is malformed.
    """
    return estimate_route_costs({0: route}, building_blocks_file)[0]


__all__ = [
    "BuildingBlockCost",
    "BuildingBlockCostStatus",
    "RouteCostError",
    "RouteCostEstimate",
    "estimate_route_cost",
    "estimate_route_costs",
]
