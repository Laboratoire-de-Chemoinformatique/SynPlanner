__all__ = ["MappingConfig"]


def __getattr__(name):
    if name == "MappingConfig":
        from synplan.chem.reaction.curation.mapping import MappingConfig

        return MappingConfig
    raise AttributeError(name)
