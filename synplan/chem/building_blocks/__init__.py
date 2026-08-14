"""Ordinary building-block preparation and planner stock APIs."""

from importlib import import_module

_MODULES = {
    "BuildingBlockPreparationConfig": "config",
    "BuildingBlockStockLoadConfig": "config",
    "DeprotectionOutput": "config",
    "DeprotectionPolicy": "config",
    "deprotect_molecule": "deprotection",
    "remove_protective_groups": "deprotection",
    "MoleculeIdentity": "identity",
    "MoleculeIdentityError": "identity",
    "inchi_to_inchi_key": "identity",
    "molecule_identity": "identity",
    "molecule_to_inchi": "identity",
    "molecule_to_inchi_key": "identity",
    "validate_standard_inchi_key": "identity",
    "BuildingBlockStock": "stock",
    "StockIdentityFormat": "stock",
    "coerce_building_block_stock": "stock",
    "detect_building_blocks_format": "stock",
}

__all__ = list(_MODULES)


def __getattr__(name: str):
    """Load focused submodules only when a package-level API is requested."""

    try:
        module_name = _MODULES[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    module = (
        import_module(module_name)
        if module_name.startswith("synplan.")
        else import_module(f".{module_name}", __name__)
    )
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
