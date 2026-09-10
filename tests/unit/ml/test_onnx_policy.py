"""ONNX model export preserves the existing policy boundary."""

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
from synplan.mcts.config import ValueNetworkEvaluationConfig
from synplan.mcts.policy.template_based import LinearPolicy
from synplan.ml.config import LinearPolicyNetworkConfig
from synplan.ml.featurization.molecules import mol_to_numpy, mol_to_pyg
from synplan.ml.networks.policy.linear import (
    FilteringPolicyNetwork,
    RankingPolicyNetwork,
)
from synplan.ml.networks.value import ValueNetwork
from synplan.utils.loading import load_evaluation_function, load_policy_function


@pytest.mark.parametrize("embedder", ["gcn", "gps"])
@pytest.mark.parametrize("policy_type", ["ranking", "filtering"])
def test_onnx_policy_roundtrip(tmp_path, monkeypatch, embedder, policy_type):
    torch.manual_seed(42)
    config = LinearPolicyNetworkConfig(
        embedder_type=embedder,
        policy_type=policy_type,
        vector_dim=16,
        num_conv_layers=1,
        heads=2,
        attn_type="multihead",
    )
    network_class = (
        RankingPolicyNetwork if policy_type == "ranking" else FilteringPolicyNetwork
    )
    network = network_class(config, n_rules=7).eval()
    network.hparams["rule_vocabulary_digest"] = "matching-rules"
    checkpoint, output = tmp_path / "policy.ckpt", tmp_path / "policy.onnx"
    torch.save(
        {"hyper_parameters": network.hparams, "state_dict": network.state_dict()},
        checkpoint,
    )
    export_policy(checkpoint, output)
    policy = load_policy_function(
        weights_path=output, top_rules=3, policy_type=policy_type
    )
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
        np.testing.assert_allclose(
            [r[0] for r in actual], [r[0] for r in expected], atol=1e-6
        )

    with pytest.raises(ValueError, match="matching trained policy"):
        list(
            policy.predict_reaction_rules(
                precursor, RuleLibrary(range(7), "wrong-rules")
            )
        )
    with pytest.raises(Exception, match="output dimensionality"):
        list(policy.predict_reaction_rules_light(precursor, 8))
    with pytest.raises(ValueError, match="configured policy_type"):
        load_policy_function(
            weights_path=output,
            policy_type="filtering" if policy_type == "ranking" else "ranking",
        )
    assert policy.get_probs(Precursor(smiles("[Na+]"))) is None

    # A repeated proposal must use the inherited cache without another ORT call.
    with monkeypatch.context() as patch:
        patch.setattr(policy.session, "run", lambda *args: pytest.fail("cache miss"))
        assert list(policy.predict_reaction_rules(precursor, rules)) == actual
    policy.rule_prob_threshold = 1.0
    assert list(policy.predict_reaction_rules(precursor, rules)) == []

    if policy_type == "filtering":
        np.testing.assert_allclose(
            policy.get_filtering_probs_only(precursor),
            reference.get_filtering_probs_only(precursor).numpy(),
            atol=1e-6,
        )
        for coef in (0.0, 1.0):
            policy = load_policy_function(
                weights_path=output,
                policy_type=policy_type,
                priority_rules_fraction=coef,
            )
            reference.priority_rules_fraction = coef
            np.testing.assert_allclose(
                policy.get_probs(precursor),
                reference.get_probs(precursor).numpy(),
                atol=1e-6,
            )

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
                                      'pytorch_lightning', 'IPython', 'streamlit'}:
            raise AssertionError(f'Base planning imported {fullname}')
sys.meta_path.insert(0, CoreOnly())
from synplan.interfaces.cli import synplan
from synplan.chem.precursor import Precursor
from synplan.utils.loading import load_policy_function
from synplan.utils.frames import ChemFrame
from chython import smiles
frame = ChemFrame([{'molecule': smiles('CCO')}], depict_columns=['molecule'])
assert '<svg' in frame._repr_html_()
policy = load_policy_function(weights_path=sys.argv[1], top_rules=3, policy_type=sys.argv[2])
assert len(list(policy.predict_reaction_rules_light(Precursor(smiles('CCO')), 7))) == 3
""",
            str(output),
            policy_type,
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


def test_onnx_value_roundtrip(tmp_path):
    torch.manual_seed(42)
    network = ValueNetwork(vector_dim=16, batch_size=1, num_conv_layers=1).eval()
    checkpoint, output = tmp_path / "value.ckpt", tmp_path / "value.onnx"
    torch.save(
        {"hyper_parameters": network.hparams, "state_dict": network.state_dict()},
        checkpoint,
    )
    export_policy(checkpoint, output, value=True)
    actual = load_evaluation_function(
        ValueNetworkEvaluationConfig(weights_path=str(output))
    )
    reference = load_evaluation_function(
        ValueNetworkEvaluationConfig(weights_path=str(checkpoint))
    )
    for mols in (["CCO"], ["CC(=O)Oc1ccccc1C(=O)O", "CC"], ["CCCCCCCC", "c1ccccc1O"]):
        precursors = [Precursor(smiles(smi)) for smi in mols]
        assert actual.predict_value(precursors) == pytest.approx(
            reference.predict_value(precursors), abs=1e-6
        )
    assert actual.predict_value([Precursor(smiles("[Na+]"))]) == -1e6
    with pytest.raises(ValueError, match="policy export"):
        load_policy_function(weights_path=output)
