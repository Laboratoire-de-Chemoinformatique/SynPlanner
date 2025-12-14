"""
SAScore Benchmark Scripts for SynPlanner.

This package contains benchmark scripts for evaluating combined policy
(filtering + ranking) on SAScore datasets.

Quick Usage
-----------

Run benchmark from command line::

    uv run sascore-benchmark
    uv run sascore-plot

Load resources in your own code::

    from scripts.sascore_bench import load_policy_from_config, load_resources_from_config
    
    # Load just the policy
    policy = load_policy_from_config()
    
    # Load all resources (policy, rules, building blocks, config)
    resources = load_resources_from_config()
    tree = Tree(
        target=my_mol,
        config=resources["tree_config"],
        reaction_rules=resources["reaction_rules"],
        building_blocks=resources["building_blocks"],
        expansion_function=resources["policy_function"],
    )
"""

from scripts.sascore_bench.run_benchmark import (
    load_config,
    load_policy_from_config,
    load_resources_from_config,
    DEFAULT_CONFIG_PATH,
)

__all__ = [
    "load_config",
    "load_policy_from_config",
    "load_resources_from_config",
    "DEFAULT_CONFIG_PATH",
]
