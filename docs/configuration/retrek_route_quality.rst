.. _retrek_route_quality_config:

===========================
ReTReK route quality scorer
===========================

:class:`~synplan.chem.reaction.routes.quality.retrek.RetrekRouteScorer` ranks
detached :class:`~synplan.chem.reaction.routes.route.Route` objects after a
search. It does not configure or modify MCTS.

The example configuration is ``configs/retrek_route_quality.yaml``:

.. code-block:: yaml

    enabled_scores: ["cd", "as", "rd"]
    weights:
      cd: 5.0
      as: 0.5
      rd: 2.0
      st: 2.0

Load and use it from Python:

.. code-block:: python

    from synplan.chem.reaction.routes.quality.retrek import (
        RetrekRouteScorer,
        RetrekRouteScoringConfig,
    )

    config = RetrekRouteScoringConfig.from_yaml(
        "configs/retrek_route_quality.yaml"
    )
    scorer = RetrekRouteScorer(config)
    ranked_routes = scorer.rank(tree.routes())

``enabled_scores`` accepts ``cd``, ``as``, ``rd`` and ``st``. Each enabled
score must have a finite, non-negative weight, and at least one enabled weight
must be positive. The defaults enable CDScore, ASScore and RDScore.

.. warning::

    STScore is disabled by default because its current formula is known to be
    incorrect. It must not be enabled for published results until corrected.
