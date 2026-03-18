"""Shared result types for reaction processing pipelines.

Used by standardization, filtering, and other batch-processing functions
that need a common result format.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from chython.containers import ReactionContainer


@dataclass
class ErrorEntry:
    """A single processing error."""

    original: str
    stage: str
    error_type: str
    message: str
    line_number: int | None = None


@dataclass
class FilteredEntry:
    """A reaction intentionally excluded (not an error)."""

    original: str
    reason: str
    line_number: int | None = None


@dataclass
class ProcessResult:
    """Result of processing a batch of reactions."""

    reactions: list[ReactionContainer]
    filtered: list[FilteredEntry] = field(default_factory=list)
    errors: list[ErrorEntry] = field(default_factory=list)


@dataclass
class PipelineSummary:
    """Aggregate statistics from a pipeline run."""

    total_input: int = 0
    succeeded: int = 0
    filtered: int = 0
    errored: int = 0
    duplicates: int = 0
    elapsed_seconds: float = 0.0
    error_file: str | None = None
    error_breakdown: dict[str, int] = field(default_factory=dict)
    filter_breakdown: dict[str, int] = field(default_factory=dict)

    @property
    def total_output(self) -> int:
        return self.succeeded

    @property
    def total_excluded(self) -> int:
        return self.filtered + self.errored + self.duplicates

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path | None = None, **kwargs) -> str:
        """Serialize to JSON string. Optionally write to file."""
        data = self.to_dict()
        s = json.dumps(data, indent=2, default=str, **kwargs)
        if path is not None:
            Path(path).write_text(s)
        return s
