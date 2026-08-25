"""Base machinery every configuration model inherits.

Individual config models live beside their domain — see rule 2 of
``docs/development/package_layout.rst``. The names that moved out are still
importable from here, resolved lazily so that this module stays a leaf.
"""

from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class BaseConfigModel(BaseModel):
    """Base class for all SynPlanner configuration models.

    Replaces ConfigABC. Provides backward-compatible from_dict(), from_yaml(),
    to_dict(), to_yaml() methods as thin wrappers over Pydantic's native API.
    """

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]):
        """Create instance from dictionary. Backward-compatible wrapper."""
        return cls.model_validate(config_dict)

    @classmethod
    def from_yaml(cls, file_path: str):
        """Create instance from YAML file. Backward-compatible wrapper."""
        with open(file_path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f) or {})

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary. Backward-compatible wrapper."""
        d = self.model_dump()
        return {k: str(v) if isinstance(v, Path) else v for k, v in d.items()}

    def to_yaml(self, file_path: str) -> None:
        """Serialize to YAML file. Backward-compatible wrapper."""
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Support unpickling from old dataclass-based configs.

        Old dataclass configs stored field values directly in __dict__.
        Pydantic models need proper __init__-based construction, so we
        re-validate the state dict through model_validate.
        """
        # Pydantic v2 pickle format uses __dict__ + __pydantic_fields_set__
        if "__dict__" in state and "__pydantic_fields_set__" in state:
            # Normal Pydantic pickle — delegate to default
            super().__setstate__(state)  # type: ignore[misc]
        else:
            # Legacy dataclass pickle: state is the raw field dict.
            # Strip fields unknown to the current model to avoid
            # extra="forbid" validation errors from removed fields.
            known = set(self.__class__.model_fields)
            cleaned = {k: v for k, v in state.items() if k in known}
            obj = self.__class__.model_validate(cleaned)
            super().__setstate__(  # type: ignore[misc]
                {
                    "__dict__": obj.__dict__,
                    "__pydantic_fields_set__": obj.__pydantic_fields_set__,
                }
            )


class NestedConfigContainer(BaseConfigModel):
    """Base for configs that hold optional ``XxxConfig | None = None`` nested fields.

    Subclasses declare a ``_NESTED_CONFIG_TYPES`` ``ClassVar`` mapping the
    field name to the concrete nested config class — for example::

        class MyConfig(NestedConfigContainer):
            substep_config: SubstepConfig | None = None

            _NESTED_CONFIG_TYPES: ClassVar[dict[str, type]] = {
                "substep_config": SubstepConfig,
            }

    The container then provides:

    * **YAML-friendly coercion.** Both ``substep_config:`` (parsed as
      ``None``) and ``substep_config: {}`` (empty mapping) produce a
      default-instantiated ``SubstepConfig``. Explicit dicts are validated
      against the declared type. To *disable* a nested config, omit the key
      from the input entirely (the field default of ``None`` is preserved).

      Pre-1.5.0 ``substep_config:`` silently left the field as ``None`` and
      the substep was skipped — see the 1.5.0 ⚠️ Backwards-incompatible
      block in the CHANGELOG.

    * ``to_dict()`` that emits each non-``None`` nested config (recursively)
      and skips top-level fields not listed in ``_NESTED_CONFIG_TYPES``.
      Subclasses with additional scalar fields override ``to_dict`` if they
      need them in the output.
    """

    _NESTED_CONFIG_TYPES: ClassVar[dict[str, type]] = {}

    @model_validator(mode="before")
    @classmethod
    def _coerce_nested_configs(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for field_name, cfg_cls in cls._NESTED_CONFIG_TYPES.items():
                if field_name not in data:
                    continue
                value = data[field_name]
                if value is None or isinstance(value, dict):
                    data[field_name] = cfg_cls(**(value or {}))
        return data

    def to_dict(self) -> dict[str, Any]:
        """Dict of nested configs, ``None`` entries excluded."""
        result: dict[str, Any] = {}
        for field_name in self._NESTED_CONFIG_TYPES:
            config = getattr(self, field_name)
            if config is not None:
                result[field_name] = config.to_dict()
        return result


# Moved to their domains; resolved lazily so `utils` keeps importing nothing.
_MOVED = {
    "ReactorConfig": "synplan.chem.reaction.config",
    "RuleExtractionConfig": "synplan.chem.reaction.rules.config",
    "GraphEmbedderConfig": "synplan.ml.config",
    "PolicyNetworkConfig": "synplan.ml.config",
    "LinearPolicyNetworkConfig": "synplan.ml.config",
    "MHNRankingPolicyNetworkConfig": "synplan.ml.config",
    "ValueNetworkConfig": "synplan.ml.config",
    "TuningConfig": "synplan.ml.config",
    "TreeConfig": "synplan.mcts.config",
    "RolloutEvaluationConfig": "synplan.mcts.config",
    "ValueNetworkEvaluationConfig": "synplan.mcts.config",
    "RDKitEvaluationConfig": "synplan.mcts.config",
    "PolicyEvaluationConfig": "synplan.mcts.config",
    "RandomEvaluationConfig": "synplan.mcts.config",
    "CombinedPolicyConfig": "synplan.mcts.config",
}

__all__ = ["BaseConfigModel", "NestedConfigContainer", *_MOVED]


def __getattr__(name: str):
    try:
        module = _MOVED[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    from importlib import import_module

    from synplan._compat import deprecated_module

    deprecated_module(f"{__name__}.{name}", module)
    return getattr(import_module(module), name)


def __dir__():
    return sorted(__all__)
