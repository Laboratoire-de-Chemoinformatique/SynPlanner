"""ONNX ranking inference with the existing template-policy selection behavior."""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np

from synplan.mcts.policy.template_based import TemplateBasedPolicy
from synplan.ml.featurization.molecules import mol_to_numpy


class OnnxPolicy(TemplateBasedPolicy):
    """Run an exported ranking model on CPU without Torch."""

    def __init__(self, path, *, top_rules=50, rule_prob_threshold=0.0):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        metadata = self.session.get_modelmeta().custom_metadata_map
        if metadata.get("synplan.policy") != "ranking-v1":
            raise ValueError("Expected a SynPlanner ONNX ranking policy export")
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.policy_net = SimpleNamespace(
            n_rules=self.session.get_outputs()[0].shape[-1],
            policy_type="ranking",
            rule_vocabulary_digest=metadata.get("synplan.rule_vocabulary_digest"),
        )
        self.top_rules = top_rules
        self.rule_prob_threshold = rule_prob_threshold
        self.priority_rules_fraction = 0.5
        self._proposal_cache = OrderedDict()

    def _predict(self, precursor, output):
        arrays = mol_to_numpy(precursor.policy_molecule, canonicalize=False)
        if arrays is None:
            return None
        inputs = {name: arrays[name] for name in self.input_names}
        values = self.session.run([output], inputs)[0]
        return values[0].astype(np.float64)

    def _select_rules(self, probs):
        k = min(self.top_rules, probs.size)
        ids = np.argpartition(-probs, k - 1)[:k]
        ids = ids[np.argsort(-probs[ids])]
        return probs[ids], ids

    def get_probs(self, precursor):
        return self._predict(precursor, "probabilities")

    def get_logits(self, precursor):
        return self._predict(precursor, "logits")
