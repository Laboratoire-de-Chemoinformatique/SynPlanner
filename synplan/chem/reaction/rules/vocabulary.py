"""Bind newly extracted rule assets to the checkpoints trained with them."""

import json
from hashlib import sha256
from pathlib import Path


def manifest_digest(path):
    path = Path(path)
    manifest = path.with_suffix(".manifest.json")
    if not manifest.exists():
        return None
    metadata = json.loads(manifest.read_text())
    if metadata.get("schema") != "synplan-rules/2":
        raise ValueError(f"unsupported rule manifest schema in {manifest}")
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != metadata["rules_sha256"]:
        raise ValueError(
            "rule file changed since extraction; regenerate its vocabulary and policy assets"
        )
    return digest


class RuleLibrary(tuple):
    """An immutable ordered rule sequence with optional extraction provenance."""

    def __new__(cls, rules, vocabulary_digest=None):
        result = super().__new__(cls, rules)
        result.vocabulary_digest = vocabulary_digest
        return result


def bind_training_vocabulary(network, dataset):
    rules = getattr(dataset, "reaction_rules_path", None)
    if rules is None:
        policy_data = getattr(dataset, "policy_data_path", None)
        if policy_data and str(policy_data).endswith("_policy_data.tsv"):
            rules = str(policy_data)[: -len("_policy_data.tsv")] + ".tsv"
    if rules and Path(rules).exists():
        digest = manifest_digest(rules)
        if digest:
            network.rule_vocabulary_digest = digest
            network.hparams["rule_vocabulary_digest"] = digest
    return network
