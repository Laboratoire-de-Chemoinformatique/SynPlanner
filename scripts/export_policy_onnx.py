"""Export a policy checkpoint to ONNX; pass --value for a value network."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import onnx
import torch
from chython import smiles

from synplan.ml.featurization.molecules import mol_to_pyg
from synplan.ml.networks.checkpoint import (
    load_policy_network_from_checkpoint,
    load_value_network_from_checkpoint,
)
from synplan.ml.networks.policy.linear import LinearPolicyNetwork
from synplan.ml.networks.value import ValueNetwork


class SingleGraphPool(torch.nn.Module):
    """Supply GPS pooling's graph count without extracting it from tensor values."""

    def __init__(self, pool):
        super().__init__()
        self.pool = pool

    def forward(self, x, index=None):
        return self.pool(x, index=index, dim_size=1)


class GraphExport(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x, edge_index, edge_attr):
        graph = SimpleNamespace(
            x=x, edge_index=edge_index, edge_attr=edge_attr, batch=None
        )
        if isinstance(self.network, ValueNetwork):
            return self.network(graph)
        embedding = self.network.embedder(graph)
        logits = self.network.y_predictor(embedding)
        if self.network.policy_type == "filtering":
            priority = torch.sigmoid(self.network.priority_predictor(embedding))
            return torch.sigmoid(logits), logits, priority
        return torch.softmax(logits, dim=-1), logits


def export_policy(checkpoint, output, *, value=False):
    """Export one molecule per call, with dynamic atom and bond counts."""
    output = Path(output)
    if output.suffix != ".onnx":
        raise ValueError("Output path must end in .onnx")
    network = (
        load_value_network_from_checkpoint(checkpoint)
        if value
        else load_policy_network_from_checkpoint(checkpoint)
    )
    if not isinstance(network, (LinearPolicyNetwork, ValueNetwork)):
        raise ValueError("ONNX export supports linear policies and value networks only")
    config = network.hparams.get("config", network.hparams)
    if config.get("embedder_type", "gcn") == "gps":
        network.embedder.pool = SingleGraphPool(network.embedder.pool)
    outputs = ["value"] if value else ["probabilities", "logits"]
    if not value and network.policy_type == "filtering":
        outputs.append("priority")
    graph = mol_to_pyg(smiles("CC(=O)Oc1ccccc1C(=O)O"))
    atoms = torch.export.Dim("atoms", min=2)
    edges = torch.export.Dim("edges", min=2)
    with torch.no_grad():
        torch.onnx.export(
            GraphExport(network).eval(),
            (graph.x, graph.edge_index, graph.edge_attr),
            str(output),
            input_names=["x", "edge_index", "edge_attr"],
            output_names=outputs,
            dynamic_shapes=({0: atoms}, {1: edges}, {0: edges}),
            opset_version=18,
            dynamo=True,
            external_data=False,
        )
    model = onnx.load(output)
    metadata = (
        {"synplan.value": "value-v1"}
        if value
        else {"synplan.policy": f"{network.policy_type}-v1"}
    )
    digest = getattr(network, "rule_vocabulary_digest", None)
    if digest is not None:
        metadata["synplan.rule_vocabulary_digest"] = digest
    onnx.helper.set_model_props(model, metadata)
    onnx.save(model, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--value", action="store_true", help="Export a value network")
    args = parser.parse_args()
    export_policy(args.checkpoint, args.output, value=args.value)
