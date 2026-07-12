"""Tests for the MHN-style dynamic ranking policy."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from chython import smarts
from torch_geometric.data import Batch, Data

from synplan.chem.reaction.rules.representation import (
    RULE_GRAPH_EDGE_FEATURE_DIM,
    RULE_GRAPH_NODE_FEATURE_DIM,
    RuleFingerprintConfig,
    RuleRepresentationConfig,
    reaction_rules_path_from_policy_data,
    rule_fingerprint_digest,
    rule_representation_digest,
)
from synplan.chem.utils import reaction_query_to_reaction
from synplan.ml.featurization.cache import (
    MAX_RULE_FINGERPRINT_CACHE_SIZE,
    cache_set,
)
from synplan.ml.featurization.fingerprints import (
    _side_fingerprint,
    rule_fingerprints_from_smarts,
)
from synplan.ml.featurization.rules import query_cgr_graphs_from_smarts
from synplan.ml.networks.checkpoint import policy_network_class_from_checkpoint
from synplan.ml.networks.embedding.molecule import build_graph_embedder
from synplan.ml.networks.policy.linear import RankingPolicyNetwork
from synplan.ml.networks.policy.mhnreact import MHNReact
from synplan.ml.training.trainers import build_mhn_ranking_network
from synplan.utils.config import MHNRankingPolicyNetworkConfig, PolicyNetworkConfig

RULE_A = "[c:1]-[N:2]>>[c:1]-[N+:2](-[O-:3])=[O:4]"
RULE_B = "[C:1]-[O:2]>>[C:1].[O:2]"
RULE_D2 = "[C;D2:1]-[O:2]>>[C:1].[O:2]"
RULE_D3 = "[C;D3:1]-[O:2]>>[C:1].[O:2]"
RULE_D124 = "[C;D1,D2,D4:1]>>[C:1]"
RULE_D134 = "[C;D1,D3,D4:1]>>[C:1]"
RULE_H1 = "[O;h1:1]>>[O:1]"
RULE_H0 = "[O;h0:1]>>[O:1]"
RULE_H124 = "[O;h1,h2,h4:1]>>[O:1]"
RULE_H134 = "[O;h1,h3,h4:1]>>[O:1]"
RULE_R5 = "[C;r5:1]-[O:2]>>[C:1].[O:2]"
RULE_R6 = "[C;r6:1]-[O:2]>>[C:1].[O:2]"
RULE_R568 = "[C;r5,r6,r8:1]>>[C:1]"
RULE_R578 = "[C;r5,r7,r8:1]>>[C:1]"
RULE_FORMED = "[C:1].[O:2]>>[C:1]-[O:2]"
RULE_DOUBLE_BROKEN = "[C:1]=[O:2]>>[C:1].[O:2]"
RULE_AROMATIC_SINGLE_BROKEN = "[c:1]-[c:2]>>[c:1].[c:2]"
RULE_AROMATIC_BROKEN = "[c:1]:[c:2]>>[c:1].[c:2]"
RULE_ANY_BROKEN = "[C:1]~[O:2]>>[C:1].[O:2]"
RULE_CHANGED = "[C:1]-[O:2]>>[C:1]=[O:2]"
RULE_REVERSE_CHANGED = "[C:1]=[O:2]>>[C:1]-[O:2]"
RULE_REMAP = "[C:10]-[O:20]>>[C:10].[O:20]"


def _fp_config(
    *,
    fp_size: int = 16,
    fp_type: str = "query_cgr",
    schema_version: str = "1",
) -> RuleFingerprintConfig:
    return RuleFingerprintConfig(
        fp_size=fp_size, fp_type=fp_type, schema_version=schema_version
    )


def _mhnreact_reference_rdk_fragment(fragment_smarts: str, fp_size: int) -> torch.Tensor:
    from rdkit import Chem, DataStructs
    from rdkit.Chem.rdmolops import FastFindRings

    mol = Chem.MolFromSmarts(str(fragment_smarts), mergeHs=False)
    if mol is None:
        raise ValueError(fragment_smarts)
    Chem.SanitizeMol(mol, catchErrors=True)
    FastFindRings(mol)
    mol.UpdatePropertyCache(strict=False)
    bit_vector = Chem.RDKFingerprint(mol, fpSize=fp_size, maxPath=6)
    array = np.zeros((fp_size,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bit_vector, array)
    return torch.from_numpy(array)


def _mhnreact_reference_rdk_side(side_smarts: str, fp_size: int) -> torch.Tensor:
    fragments = [fragment for fragment in str(side_smarts).split(".") if fragment]
    if not fragments:
        return torch.zeros(fp_size, dtype=torch.float)
    return torch.stack(
        [_mhnreact_reference_rdk_fragment(fragment, fp_size) for fragment in fragments]
    ).amax(dim=0)


def _mhnreact_reference_template_fp(rule_smarts: str, fp_size: int) -> torch.Tensor:
    parts = str(rule_smarts).split(">")
    return _mhnreact_reference_rdk_side(
        parts[0], fp_size
    ) - 0.5 * _mhnreact_reference_rdk_side(parts[-1], fp_size)


def _load_rules_like_mhnreact_converter(rules_tsv: Path) -> list[str]:
    rules: list[str] = []
    with rules_tsv.open(encoding="utf-8") as f:
        header = f.readline()
        if "rule_smarts" not in header:
            raise ValueError(f"unexpected header in {rules_tsv}: {header!r}")
        for line in f:
            if line.strip():
                rules.append(line.split("\t", 1)[0].strip())
    return rules


def _graph_batch() -> Batch:
    graph = Data(
        x=torch.tensor(
            [
                [6, 2, 14, 4, 2, 0, 0, 1, 1, 0, 0],
                [8, 2, 16, 2, 2, 0, 0, 1, 1, 0, 0],
            ],
            dtype=torch.uint8,
        ),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float,
        ),
        y_rules=torch.tensor([0], dtype=torch.long),
    )
    return Batch.from_data_list([graph])


def _fp_network(rule_fingerprints: torch.Tensor) -> MHNReact:
    config = MHNRankingPolicyNetworkConfig(
        vector_dim=8,
        batch_size=1,
        dropout=0.0,
        num_conv_layers=1,
        learning_rate=0.001,
        association_dim=4,
        rule_fp_size=rule_fingerprints.shape[1],
    )
    network = MHNReact(config=config, n_rules=rule_fingerprints.shape[0])
    network.set_training_rule_fingerprints(rule_fingerprints)
    return network


def _graph_rule_network(rule_graphs: list[Data]) -> MHNReact:
    config = MHNRankingPolicyNetworkConfig(
        vector_dim=8,
        batch_size=1,
        dropout=0.0,
        num_conv_layers=1,
        learning_rate=0.001,
        embedder_type="gps",
        heads=4,
        association_dim=4,
        rule_embedding_type="query_cgr_graph",
        rule_embedder={"embedder_type": "gps"},
        rule_graph_batch_size=1,
    )
    network = MHNReact(config=config, n_rules=len(rule_graphs))
    network.set_training_rule_graphs(rule_graphs)
    return network


def _graph_signature(graph: Data):
    return (
        graph.x.tolist(),
        graph.edge_index.tolist(),
        graph.edge_attr.tolist(),
    )


def test_rule_fingerprints_are_deterministic_and_permutation_equivariant():
    fingerprints_1 = rule_fingerprints_from_smarts((RULE_A, RULE_B), _fp_config())
    fingerprints_2 = rule_fingerprints_from_smarts((RULE_A, RULE_B), _fp_config())
    reversed_fingerprints = rule_fingerprints_from_smarts(
        (RULE_B, RULE_A), _fp_config()
    )

    assert torch.equal(fingerprints_1, fingerprints_2)
    assert tuple(fingerprints_1.shape) == (2, 16)
    assert torch.equal(reversed_fingerprints, fingerprints_1.flip(0))


def test_rule_fingerprint_error_identifies_rule():
    with pytest.raises(ValueError, match=r"index 0"):
        rule_fingerprints_from_smarts(("invalid",), _fp_config())


def test_legacy_rule_fingerprints_drop_query_labels():
    legacy_rule_fingerprints = rule_fingerprints_from_smarts(
        (RULE_D2, RULE_D3), _fp_config(fp_size=2048, fp_type="legacy")
    )

    assert torch.equal(legacy_rule_fingerprints[0], legacy_rule_fingerprints[1])


def test_mhnreact_rdkit_rule_fingerprints_match_native_rdk_reference():
    config = _fp_config(fp_size=2048, fp_type="mhnreact_rdkit")
    fingerprints = rule_fingerprints_from_smarts((RULE_A, RULE_B), config)
    expected = torch.stack(
        [
            _mhnreact_reference_template_fp(RULE_A, config.fp_size),
            _mhnreact_reference_template_fp(RULE_B, config.fp_size),
        ]
    )

    assert tuple(fingerprints.shape) == (2, 2048)
    assert torch.equal(fingerprints, expected)
    assert fingerprints.abs().sum() > 0


def test_mhnreact_rdkit_rule_fingerprints_are_deterministic_and_directional():
    config = _fp_config(fp_size=2048, fp_type="mhnreact_rdkit")
    fingerprints_1 = rule_fingerprints_from_smarts((RULE_B, RULE_FORMED), config)
    fingerprints_2 = rule_fingerprints_from_smarts((RULE_B, RULE_FORMED), config)

    assert torch.equal(fingerprints_1, fingerprints_2)
    assert not torch.equal(fingerprints_1[0], fingerprints_1[1])


def test_mhnreact_rdkit_fingerprint_error_identifies_unparseable_fragment():
    with pytest.raises(ValueError, match=r"index 0") as exc_info:
        rule_fingerprints_from_smarts(
            ("[C:1]>>[C",), _fp_config(fp_size=2048, fp_type="mhnreact_rdkit")
        )

    message = str(exc_info.value)
    assert "SMARTS: [C:1]>>[C" in message
    assert "reactant SMARTS fragment" in message


def test_mhnreact_converter_style_rule_order_is_preserved(tmp_path):
    rules_path = tmp_path / "reaction_rules.tsv"
    rules_path.write_text(
        f"rule_smarts\tpopularity\treaction_indices\n{RULE_A}\t1\t0\n{RULE_B}\t1\t1\n",
        encoding="utf-8",
    )

    rules = _load_rules_like_mhnreact_converter(rules_path)
    fingerprints = rule_fingerprints_from_smarts(
        rules, _fp_config(fp_size=2048, fp_type="mhnreact_rdkit")
    )

    assert rules == [RULE_A, RULE_B]
    assert tuple(fingerprints.shape) == (2, 2048)
    assert torch.equal(fingerprints[0], _mhnreact_reference_template_fp(RULE_A, 2048))
    assert torch.equal(fingerprints[1], _mhnreact_reference_template_fp(RULE_B, 2048))


@pytest.mark.parametrize(
    ("left_rule", "right_rule"),
    [
        (RULE_D2, RULE_D3),
        (RULE_H1, RULE_H0),
        (RULE_R5, RULE_R6),
    ],
)
def test_query_cgr_rule_fingerprints_keep_query_labels(left_rule, right_rule):
    query_cgr_rule_fingerprints = rule_fingerprints_from_smarts(
        (left_rule, right_rule), _fp_config(fp_size=2048, fp_type="query_cgr")
    )

    assert not torch.equal(
        query_cgr_rule_fingerprints[0], query_cgr_rule_fingerprints[1]
    )


@pytest.mark.parametrize(
    ("left_rule", "right_rule"),
    [
        (RULE_D2, RULE_D3),
        (RULE_H1, RULE_H0),
        (RULE_R5, RULE_R6),
    ],
)
def test_query_cgr_rule_graphs_keep_query_labels(left_rule, right_rule):
    left_graph, right_graph = query_cgr_graphs_from_smarts((left_rule, right_rule))

    assert tuple(left_graph.x.shape[1:]) == (RULE_GRAPH_NODE_FEATURE_DIM,)
    assert tuple(left_graph.edge_attr.shape[1:]) == (RULE_GRAPH_EDGE_FEATURE_DIM,)
    assert not torch.equal(left_graph.x, right_graph.x)


@pytest.mark.parametrize(
    ("left_rule", "right_rule"),
    [
        (RULE_D124, RULE_D134),
        (RULE_H124, RULE_H134),
        (RULE_R568, RULE_R578),
    ],
)
def test_query_cgr_rule_graphs_keep_set_valued_query_labels(left_rule, right_rule):
    left_graph, right_graph = query_cgr_graphs_from_smarts((left_rule, right_rule))

    assert tuple(left_graph.x.shape[1:]) == (RULE_GRAPH_NODE_FEATURE_DIM,)
    assert not torch.equal(left_graph.x, right_graph.x)


def test_query_cgr_rule_graphs_keep_dynamic_bond_labels():
    broken, formed, changed = query_cgr_graphs_from_smarts(
        (RULE_B, RULE_FORMED, RULE_CHANGED)
    )

    assert not torch.equal(broken.edge_attr, formed.edge_attr)
    assert not torch.equal(broken.edge_attr, changed.edge_attr)
    assert not torch.equal(formed.edge_attr, changed.edge_attr)


def test_query_cgr_rule_graph_embedder_keeps_bond_semantics():
    rule_pairs = [
        (RULE_B, RULE_DOUBLE_BROKEN),
        (RULE_AROMATIC_SINGLE_BROKEN, RULE_AROMATIC_BROKEN),
        (RULE_B, RULE_ANY_BROKEN),
        (RULE_CHANGED, RULE_REVERSE_CHANGED),
    ]
    for left_rule, right_rule in rule_pairs:
        torch.manual_seed(0)
        rule_graphs = query_cgr_graphs_from_smarts((left_rule, right_rule))
        network = _graph_rule_network(rule_graphs)
        network.eval()

        rule_associations = network.encode_rule_graphs(rule_graphs)

        assert tuple(rule_associations.shape) == (2, 4)
        assert not torch.allclose(rule_associations[0], rule_associations[1])


def test_query_cgr_rule_graphs_are_stable_under_atom_remapping():
    original, remapped = query_cgr_graphs_from_smarts((RULE_B, RULE_REMAP))

    assert _graph_signature(original) == _graph_signature(remapped)


def test_rule_fingerprint_digest_includes_fingerprint_config():
    legacy_digest = rule_fingerprint_digest((RULE_A,), _fp_config(fp_type="legacy"))
    mhnreact_digest = rule_fingerprint_digest(
        (RULE_A,), _fp_config(fp_type="mhnreact_rdkit")
    )
    query_cgr_digest = rule_fingerprint_digest(
        (RULE_A,), _fp_config(fp_type="query_cgr")
    )
    schema_digest = rule_fingerprint_digest(
        (RULE_A,), _fp_config(fp_type="query_cgr", schema_version="2")
    )

    assert legacy_digest != mhnreact_digest
    assert mhnreact_digest != query_cgr_digest
    assert query_cgr_digest != schema_digest


def test_rule_representation_digest_includes_encoder_contract():
    fingerprint_digest = rule_representation_digest(
        (RULE_A,),
        RuleRepresentationConfig(
            embedding_type="fingerprint", fingerprint_config=_fp_config()
        ),
    )
    graph_digest = rule_representation_digest(
        (RULE_A,), RuleRepresentationConfig(embedding_type="query_cgr_graph")
    )
    graph_schema_digest = rule_representation_digest(
        (RULE_A,),
        RuleRepresentationConfig(
            embedding_type="query_cgr_graph", graph_schema_version="2"
        ),
    )
    graph_batch_digest = rule_representation_digest(
        (RULE_A,),
        RuleRepresentationConfig(
            embedding_type="query_cgr_graph", graph_batch_size=2048
        ),
    )

    assert fingerprint_digest != graph_digest
    assert graph_digest != graph_schema_digest
    assert graph_digest != graph_batch_digest
    with pytest.raises(ValueError, match="embedder_type='gps'"):
        RuleRepresentationConfig(
            embedding_type="query_cgr_graph", graph_embedder_type="gcn"
        )


def test_rule_fingerprintcache_set_is_bounded_lru():
    cache = OrderedDict()

    for index in range(MAX_RULE_FINGERPRINT_CACHE_SIZE + 2):
        cache_set(cache, str(index), torch.tensor([float(index)]))

    assert len(cache) == MAX_RULE_FINGERPRINT_CACHE_SIZE
    assert list(cache) == [
        str(index) for index in range(2, MAX_RULE_FINGERPRINT_CACHE_SIZE + 2)
    ]

    cache_set(cache, "2", torch.tensor([2.0]))
    cache_set(cache, "new", torch.tensor([99.0]))

    assert "3" not in cache
    assert list(cache)[-1] == "new"


def test_reaction_rules_path_is_inferred_from_extracted_policy_mapping(tmp_path):
    rules_path = tmp_path / "reaction_rules.tsv"
    rules_path.write_text("rule_smarts\tpopularity\treaction_indices\n")
    policy_data_path = tmp_path / "reaction_rules_policy_data.tsv"
    policy_data_path.write_text("product_smiles\trule_id\n")

    assert reaction_rules_path_from_policy_data(policy_data_path) == rules_path


def test_reaction_rules_path_rejects_non_extracted_mapping_name(tmp_path):
    policy_data_path = tmp_path / "policy.tsv"
    policy_data_path.write_text("product_smiles\trule_id\n")

    with pytest.raises(ValueError, match=r"\*_policy_data.tsv"):
        reaction_rules_path_from_policy_data(policy_data_path)


def test_side_fingerprint_max_pools_fragments():
    reaction = reaction_query_to_reaction(smarts(RULE_B))
    pooled = _side_fingerprint(reaction.products, _fp_config())
    individual = [
        torch.as_tensor(
            molecule.morgan_fingerprint(
                min_radius=1,
                max_radius=4,
                length=16,
                number_active_bits=2,
            ),
            dtype=torch.float,
        )
        for molecule in reaction.products
    ]

    assert torch.equal(pooled, torch.stack(individual).amax(dim=0))


def test_mhn_config_validation():
    with pytest.raises(ValueError, match="requires policy_type='ranking'"):
        PolicyNetworkConfig(architecture="mhn_ranking", policy_type="filtering")
    with pytest.raises(ValueError, match="positive power of two"):
        MHNRankingPolicyNetworkConfig(rule_fp_size=1000)
    with pytest.raises(ValueError):
        MHNRankingPolicyNetworkConfig(rule_fp_min_radius=0)
    with pytest.raises(ValueError):
        MHNRankingPolicyNetworkConfig(rule_fp_type="unknown")
    with pytest.raises(ValueError):
        MHNRankingPolicyNetworkConfig(rule_embedding_type="unknown")
    with pytest.raises(ValueError):
        MHNRankingPolicyNetworkConfig(rule_embedder={"embedder_type": "unknown"})
    with pytest.raises(ValueError, match="embedder_type='gps'"):
        MHNRankingPolicyNetworkConfig(
            rule_embedding_type="query_cgr_graph",
            rule_embedder={"embedder_type": "gcn"},
        )
    with pytest.raises(ValueError, match="divisible"):
        PolicyNetworkConfig(
            embedder_type="gcn_concat", vector_dim=10, num_conv_layers=3
        )
    with pytest.raises(ValueError):
        MHNRankingPolicyNetworkConfig(rule_graph_batch_size=0)
    with pytest.raises(ValueError):
        MHNRankingPolicyNetworkConfig(rule_embedder={"vector_dim": 0})
    with pytest.raises(ValueError):
        MHNRankingPolicyNetworkConfig(rule_embedder={"num_conv_layers": 0})
    with pytest.raises(ValueError):
        MHNRankingPolicyNetworkConfig(rule_embedder={"heads": 0})
    with pytest.raises(ValueError):
        MHNRankingPolicyNetworkConfig(rule_embedder={"attn_type": "unknown"})
    with pytest.raises(ValueError):
        MHNRankingPolicyNetworkConfig(rule_embedder={"dropout": -0.1})
    with pytest.raises(ValueError):
        MHNRankingPolicyNetworkConfig(rule_embedder={"dropout": 1.1})
    with pytest.raises(ValueError):
        MHNRankingPolicyNetworkConfig(rule_embedder={"attn_dropout": -0.1})
    with pytest.raises(ValueError):
        MHNRankingPolicyNetworkConfig(rule_embedder={"attn_dropout": 1.1})
    with pytest.raises(ValueError):
        RuleFingerprintConfig(min_radius=0)

    config = MHNRankingPolicyNetworkConfig()
    assert config.architecture == "mhn_ranking"
    assert config.rule_embedding_type == "fingerprint"
    assert config.rule_embedder.embedder_type == "gps"
    assert config.rule_graph_batch_size == 1024
    assert config.rule_graph_schema_version == "1"
    assert config.rule_embedder.vector_dim is None
    assert config.rule_embedder.num_conv_layers is None
    assert config.rule_embedder.heads is None
    assert config.rule_embedder.attn_type is None
    assert config.rule_embedder.dropout is None
    assert config.rule_embedder.attn_dropout is None
    assert config.rule_fp_type == "query_cgr"
    assert config.rule_fp_schema_version == "1"

    mhnreact_config = MHNRankingPolicyNetworkConfig(rule_fp_type="mhnreact_rdkit")
    assert mhnreact_config.rule_fp_type == "mhnreact_rdkit"
    assert (
        mhnreact_config.rule_representation_config().fingerprint_config.fp_type
        == "mhnreact_rdkit"
    )

    graph_config = MHNRankingPolicyNetworkConfig(
        rule_embedding_type="query_cgr_graph",
        rule_embedder={"embedder_type": "gps"},
    )
    assert graph_config.rule_embedding_type == "query_cgr_graph"


@pytest.mark.parametrize(
    "config_kwargs",
    [
        {"embedder_type": "gcn", "rule_embedding_type": "fingerprint"},
        {
            "embedder_type": "gcn",
            "rule_embedding_type": "query_cgr_graph",
            "rule_embedder": {"embedder_type": "gps"},
        },
        {"embedder_type": "gps", "rule_embedding_type": "fingerprint"},
        {
            "embedder_type": "gps",
            "rule_embedding_type": "query_cgr_graph",
            "rule_embedder": {"embedder_type": "gps"},
        },
    ],
)
def test_mhn_config_scenarios_product_embedder_and_rule_encoder(config_kwargs):
    config = MHNRankingPolicyNetworkConfig(**config_kwargs)

    assert config.embedder_type == config_kwargs["embedder_type"]
    assert config.rule_embedding_type == config_kwargs["rule_embedding_type"]


def test_mhn_rule_side_dropout_overrides_product_gps_dropout():
    rule_graphs = query_cgr_graphs_from_smarts((RULE_A, RULE_B))
    config = MHNRankingPolicyNetworkConfig(
        vector_dim=8,
        batch_size=1,
        dropout=0.3,
        num_conv_layers=1,
        learning_rate=0.001,
        embedder_type="gps",
        heads=4,
        attn_type="multihead",
        attn_dropout=0.5,
        association_dim=4,
        rule_embedding_type="query_cgr_graph",
        rule_embedder={
            "embedder_type": "gps",
            "vector_dim": 6,
            "num_conv_layers": 2,
            "heads": 2,
            "attn_type": "performer",
            "dropout": 0.1,
            "attn_dropout": 0.2,
        },
    )
    network = MHNReact(config=config, n_rules=len(rule_graphs))

    assert network.molecule_embedding[1].p == pytest.approx(0.3)
    assert network.rule_embedding.projection[1].p == pytest.approx(0.1)
    assert network.rule_embedder is not None
    assert network.rule_embedder.node_expansion.out_features == 6
    assert len(network.rule_embedder.convs) == 2
    assert network.rule_embedder.convs[0].heads == 2
    assert network.rule_embedder.convs[0].attn_type == "performer"
    assert network.rule_embedder.convs[0].dropout == pytest.approx(0.1)
    assert network.rule_embedder.convs[0].attn.dropout.p == pytest.approx(0.2)
    assert network.rule_embedding.projection[0].in_features == 6
    assert network.hparams["config"]["rule_embedder"]["dropout"] == pytest.approx(0.1)
    assert network.hparams["config"]["rule_embedder"]["num_conv_layers"] == 2
    assert network.hparams["config"]["rule_embedder"]["heads"] == 2
    assert network.hparams["config"]["rule_embedder"]["attn_type"] == "performer"
    assert network.hparams["config"]["rule_embedder"]["attn_dropout"] == pytest.approx(
        0.2
    )


def test_graph_embedder_builder_validates_contract():
    with pytest.raises(ValueError, match="embedder_type"):
        build_graph_embedder("unknown", 8)
    with pytest.raises(ValueError, match="divisible"):
        build_graph_embedder("gcn_concat", 10, num_conv_layers=3)


def test_mhn_logits_probabilities_and_gradient_flow():
    fingerprints = rule_fingerprints_from_smarts((RULE_A, RULE_B), _fp_config())
    network = _fp_network(fingerprints)
    batch = _graph_batch()

    logits = network.get_logits(batch)
    probs = network(batch)
    assert tuple(logits.shape) == (1, 2)
    assert tuple(probs.shape) == (1, 2)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1))

    # Gradient flows through molecule embedding and rule embedding
    loss = logits.sum()
    loss.backward()
    assert network.molecule_embedding[0].weight.grad is not None
    assert network.rule_embedding.projection[0].weight.grad is not None


def test_mhn_query_cgr_graph_logits_and_gradient_flow():
    rule_graphs = query_cgr_graphs_from_smarts((RULE_A, RULE_B))
    network = _graph_rule_network(rule_graphs)
    batch = _graph_batch()

    logits = network.get_logits(batch)
    probs = network(batch)
    assert tuple(logits.shape) == (1, 2)
    assert tuple(probs.shape) == (1, 2)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1))

    loss = logits.sum()
    loss.backward()
    assert network.embedder.node_expansion.weight.grad is not None
    assert network.rule_embedder is not None
    assert network.rule_embedder.node_expansion.weight.grad is not None
    assert network.rule_embedding.projection[0].weight.grad is not None


def test_mhn_accepts_dynamic_rule_count_without_persisting_rules():
    fingerprints = rule_fingerprints_from_smarts((RULE_A, RULE_B), _fp_config())
    network = _fp_network(fingerprints)
    dynamic_rule_fingerprints = rule_fingerprints_from_smarts((RULE_B,), _fp_config())

    logits = network.get_logits(
        _graph_batch(), rule_fingerprints=dynamic_rule_fingerprints
    )

    assert tuple(logits.shape) == (1, 1)
    assert "_training_rule_fingerprints" not in network.state_dict()


def test_mhn_accepts_dynamic_query_cgr_graph_rule_count():
    network = _graph_rule_network(query_cgr_graphs_from_smarts((RULE_A, RULE_B)))
    dynamic_rule_graphs = query_cgr_graphs_from_smarts((RULE_B,))

    logits = network.get_logits(_graph_batch(), rule_graphs=dynamic_rule_graphs)

    assert tuple(logits.shape) == (1, 1)
    assert "_training_rule_graphs" not in network.state_dict()


def test_mhn_training_rule_artifacts_are_encoder_aware():
    fingerprints = rule_fingerprints_from_smarts((RULE_A, RULE_B), _fp_config())
    rule_graphs = query_cgr_graphs_from_smarts((RULE_A, RULE_B))

    fingerprint_network = _fp_network(fingerprints)
    with pytest.raises(ValueError, match="Rule graphs require"):
        fingerprint_network.set_training_rule_graphs(rule_graphs)

    graph_network = _graph_rule_network(rule_graphs)
    with pytest.raises(ValueError, match="Rule fingerprints require"):
        graph_network.set_training_rule_fingerprints(fingerprints)


def test_mhn_prepares_training_rules_from_policy_mapping(tmp_path):
    rules_path = tmp_path / "reaction_rules.tsv"
    rules_path.write_text(
        f"rule_smarts\tpopularity\treaction_indices\n{RULE_A}\t1\t0\n{RULE_B}\t1\t1\n",
        encoding="utf-8",
    )
    policy_data_path = tmp_path / "reaction_rules_policy_data.tsv"
    policy_data_path.write_text(
        "product_smiles\trule_id\nCC\t0\n",
        encoding="utf-8",
    )

    config = MHNRankingPolicyNetworkConfig(
        vector_dim=8,
        batch_size=1,
        dropout=0.0,
        num_conv_layers=1,
        learning_rate=0.001,
        association_dim=4,
        rule_fp_size=16,
        rule_fp_type="query_cgr",
        rule_fp_schema_version="1",
    )
    network = build_mhn_ranking_network(
        config=config,
        dataset=SimpleNamespace(
            policy_data_path=str(policy_data_path),
            _data=SimpleNamespace(y_rules=torch.tensor([0])),
        ),
    )

    assert network.n_rules == 2
    assert tuple(network._training_rule_fingerprints.shape) == (2, 16)
    assert network.hparams["n_rules"] == 2
    assert (
        network.hparams["rule_representation_digest"]
        == network.rule_representation_digest
    )
    assert network.rule_representation_digest is not None
    assert network.rule_representation_config.embedding_type == "fingerprint"
    assert network.rule_representation_config.fingerprint_config.fp_type == "query_cgr"
    assert network.hparams["config"]["rule_embedding_type"] == "fingerprint"
    assert "policy_data_path" not in network.hparams


def test_mhn_prepares_training_rule_graphs_from_policy_mapping(tmp_path):
    rules_path = tmp_path / "reaction_rules.tsv"
    rules_path.write_text(
        f"rule_smarts\tpopularity\treaction_indices\n{RULE_A}\t1\t0\n{RULE_B}\t1\t1\n",
        encoding="utf-8",
    )
    policy_data_path = tmp_path / "reaction_rules_policy_data.tsv"
    policy_data_path.write_text(
        "product_smiles\trule_id\nCC\t0\n",
        encoding="utf-8",
    )

    config = MHNRankingPolicyNetworkConfig(
        vector_dim=8,
        batch_size=1,
        dropout=0.0,
        num_conv_layers=1,
        learning_rate=0.001,
        embedder_type="gps",
        heads=4,
        association_dim=4,
        rule_embedding_type="query_cgr_graph",
        rule_embedder={"embedder_type": "gps"},
        rule_graph_batch_size=1,
    )
    network = build_mhn_ranking_network(
        config=config,
        dataset=SimpleNamespace(
            policy_data_path=str(policy_data_path),
            _data=SimpleNamespace(y_rules=torch.tensor([0])),
        ),
    )

    assert network.n_rules == 2
    assert len(network._training_rule_graphs) == 2
    assert network._training_rule_fingerprints.numel() == 0
    assert network.rule_representation_config.embedding_type == "query_cgr_graph"
    assert network.hparams["config"]["rule_embedding_type"] == "query_cgr_graph"
    assert (
        network.hparams["rule_representation_digest"]
        == network.rule_representation_digest
    )
    assert network.rule_representation_digest is not None


@pytest.mark.parametrize(
    ("hyperparameters", "expected_class"),
    [
        ({}, RankingPolicyNetwork),
        (
            {"config": {"architecture": "linear", "policy_type": "ranking"}},
            RankingPolicyNetwork,
        ),
        (
            {"config": {"architecture": "mhn_ranking", "policy_type": "ranking"}},
            MHNReact,
        ),
    ],
)
def test_checkpoint_class_dispatch_defaults_to_linear(
    tmp_path, hyperparameters, expected_class
):
    checkpoint = tmp_path / "policy.ckpt"
    torch.save({"hyper_parameters": hyperparameters}, checkpoint)

    assert policy_network_class_from_checkpoint(checkpoint) is expected_class
