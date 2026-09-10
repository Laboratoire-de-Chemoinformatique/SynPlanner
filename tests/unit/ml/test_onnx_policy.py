"""ONNX ranking export preserves the existing policy boundary."""

import subprocess
import sys

import numpy as np
import pytest
import torch
from chython import smiles

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

from scripts.export_policy_onnx import export_policy
from synplan.chem.precursor import Precursor
from synplan.chem.reaction.rules.vocabulary import RuleLibrary
from synplan.mcts.policy.template_based import LinearPolicy
from synplan.ml.config import LinearPolicyNetworkConfig
from synplan.ml.featurization.molecules import mol_to_numpy, mol_to_pyg
from synplan.ml.networks.policy.linear import RankingPolicyNetwork
from synplan.utils.loading import load_policy_function


@pytest.mark.parametrize("embedder", ["gcn", "gps"])
def test_onnx_ranking_roundtrip(tmp_path, monkeypatch, embedder):
    torch.manual_seed(42)
    config = LinearPolicyNetworkConfig(
        embedder_type=embedder,
        vector_dim=16,
        num_conv_layers=1,
        heads=2,
        attn_type="multihead",
    )
    network = RankingPolicyNetwork(config, n_rules=7).eval()
    network.hparams["rule_vocabulary_digest"] = "matching-rules"
    checkpoint, output = tmp_path / "policy.ckpt", tmp_path / "policy.onnx"
    torch.save(
        {"hyper_parameters": network.hparams, "state_dict": network.state_dict()},
        checkpoint,
    )
    export_policy(checkpoint, output)
    policy = load_policy_function(weights_path=output, top_rules=3)
    reference = LinearPolicy(network, top_rules=3)
    rules = RuleLibrary(range(7), vocabulary_digest="matching-rules")
    for smi in ["CC", "CCO", "CC(=O)Oc1ccccc1C(=O)O", "CC(=O)[O-].[Na+]"]:
        precursor = Precursor(smiles(smi))
        np.testing.assert_allclose(
            policy.get_probs(precursor),
            reference.get_probs(precursor).numpy(),
            atol=1e-6,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            policy.get_logits(precursor),
            reference.get_logits(precursor).numpy(),
            atol=1e-5,
            rtol=1e-4,
        )
        expected = list(reference.predict_reaction_rules_light(precursor, 7))
        actual = list(policy.predict_reaction_rules(precursor, rules))
        assert [r[2] for r in actual] == [r[1] for r in expected]

    with pytest.raises(ValueError, match="matching trained policy"):
        list(
            policy.predict_reaction_rules(
                precursor, RuleLibrary(range(7), "wrong-rules")
            )
        )
    with pytest.raises(Exception, match="output dimensionality"):
        list(policy.predict_reaction_rules_light(precursor, 8))
    with pytest.raises(ValueError, match="ranking only"):
        load_policy_function(weights_path=output, policy_type="filtering")
    assert policy.get_probs(Precursor(smiles("[Na+]"))) is None

    # A repeated proposal must use the inherited cache without another ORT call.
    with monkeypatch.context() as patch:
        patch.setattr(policy.session, "run", lambda *args: pytest.fail("cache miss"))
        assert list(policy.predict_reaction_rules(precursor, rules)) == actual
    policy.rule_prob_threshold = 1.0
    assert list(policy.predict_reaction_rules(precursor, rules)) == []

    # A fresh process must also run this policy and import the CLI without extras.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torch_geometric', 'chytorch',
                                      'pytorch_lightning', 'IPython', 'pandas', 'streamlit'}:
            raise AssertionError(f'Base planning imported {fullname}')
sys.meta_path.insert(0, CoreOnly())
from synplan.interfaces.cli import synplan
from synplan.chem.precursor import Precursor
from synplan.utils.loading import load_policy_function
from chython import smiles
policy = load_policy_function(weights_path=sys.argv[1], top_rules=3)
assert len(list(policy.predict_reaction_rules_light(Precursor(smiles('CCO')), 7))) == 3
""",
            str(output),
        ],
        check=True,
    )


def test_molecule_features_preserve_atom_and_edge_schema():
    molecule = smiles("CCO")
    arrays = mol_to_numpy(molecule, canonicalize=False)
    np.testing.assert_array_equal(
        arrays["x"],
        [
            [6, 2, 14, 2, 2, 3, 0, 1, 1, 0, 0],
            [6, 2, 14, 2, 2, 2, 0, 2, 2, 0, 0],
            [8, 2, 16, 4, 2, 1, 0, 1, 1, 0, 0],
        ],
    )
    np.testing.assert_array_equal(arrays["edge_index"], [[0, 1, 1, 2], [1, 0, 2, 1]])
    np.testing.assert_array_equal(arrays["edge_attr"], [[1, 0, 0, 0]] * 4)
    graph = mol_to_pyg(molecule, canonicalize=False)
    for name, values in arrays.items():
        np.testing.assert_array_equal(graph[name].numpy(), values)
    assert mol_to_numpy(smiles("[Na+].[Cl-]")) is None
