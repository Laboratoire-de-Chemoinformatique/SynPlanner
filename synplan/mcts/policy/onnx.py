"""ONNX inference with the existing template-policy selection behavior."""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np

from synplan.mcts.policy.template_based import TemplateBasedPolicy
from synplan.ml.featurization.molecules import mol_to_numpy


class OnnxPolicy(TemplateBasedPolicy):
    """Run an exported ranking or filtering model on CPU without Torch."""

    def __init__(
        self,
        path,
        *,
        top_rules=50,
        rule_prob_threshold=0.0,
        priority_rules_fraction=0.5,
    ):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        metadata = self.session.get_modelmeta().custom_metadata_map
        policy_type = {"ranking-v1": "ranking", "filtering-v1": "filtering"}.get(
            metadata.get("synplan.policy")
        )
        if policy_type is None:
            raise ValueError(
                "Expected a SynPlanner ONNX ranking or filtering policy export"
            )
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.policy_net = SimpleNamespace(
            n_rules=self.session.get_outputs()[0].shape[-1],
            policy_type=policy_type,
            rule_vocabulary_digest=metadata.get("synplan.rule_vocabulary_digest"),
        )
        self.top_rules = top_rules
        self.rule_prob_threshold = rule_prob_threshold
        self.priority_rules_fraction = priority_rules_fraction
        self._proposal_cache = OrderedDict()

    def _predict(self, precursor, outputs):
        arrays = mol_to_numpy(precursor.policy_molecule, canonicalize=False)
        if arrays is None:
            return None
        inputs = {name: arrays[name] for name in self.input_names}
        values = self.session.run(outputs, inputs)
        return [value[0].astype(np.float64) for value in values]

    def _select_rules(self, probs):
        k = min(self.top_rules, probs.size)
        ids = np.argpartition(-probs, k - 1)[:k]
        ids = ids[np.argsort(-probs[ids])]
        selected = probs[ids]
        if self.policy_net.policy_type == "filtering":
            selected = np.exp(selected - selected.max())
            selected /= selected.sum()
        return selected, ids

    def get_probs(self, precursor):
        filtering = self.policy_net.policy_type == "filtering"
        outputs = ["probabilities", "priority"] if filtering else ["probabilities"]
        values = self._predict(precursor, outputs)
        if values is None:
            return None
        if filtering:
            coef = self.priority_rules_fraction
            return (1 - coef) * values[0] + coef * values[1]
        return values[0]

    def get_logits(self, precursor):
        values = self._predict(precursor, ["logits"])
        return values[0] if values is not None else None

    def get_filtering_probs_only(self, precursor):
        if self.policy_net.policy_type != "filtering":
            raise ValueError("This method is only for filtering policy networks")
        values = self._predict(precursor, ["probabilities"])
        return values[0] if values is not None else None
