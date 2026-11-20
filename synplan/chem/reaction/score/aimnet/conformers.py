"""Load LoQI once and reuse phase-independent seeds during dual-phase assessment."""

import random
from functools import lru_cache

import numpy as np


class LoQIGenerator:
    def __init__(self, config, *, reuse_seeds=False):
        self.config = config
        self.model = None
        self.generate = (
            lru_cache(maxsize=16)(self._sample) if reuse_seeds else self._sample
        )

    def load(self, checkpoint):
        import loqi

        if self.model is None:
            self.model = loqi.load_model(
                checkpoint,
                config="loqi_flow.yaml",
                device=self.config.device,
                progress=False,
            )
        return self.model

    def _sample(self, smiles, seed):
        import loqi
        import torch

        # LoQI seeds Python, NumPy and Torch internally. Restore caller RNGs.
        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(
                devices=[] if self.config.device == "cpu" else None
            ):
                return loqi.generate_conformers(
                    smiles,
                    self.config.n_conformers,
                    model=self.model,
                    seed=seed,
                    steps=self.config.loqi_steps,
                    batch_atoms=self.config.batch_atoms,
                    add_hs=True,
                )[0]
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
