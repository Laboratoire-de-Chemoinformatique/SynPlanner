synplan.chem.reaction\_routes compatibility package
====================================================

``synplan.chem.reaction_routes`` is kept as a compatibility namespace for
existing user imports. New code should import route post-processing helpers from
``synplan.routes``.

Compatibility mapping
---------------------

.. list-table::
   :header-rows: 1

   * - Old import path
     - New import path
   * - ``synplan.chem.reaction_routes.analysis``
     - ``synplan.routes.analysis``
   * - ``synplan.chem.reaction_routes.clustering``
     - ``synplan.routes.clustering``
   * - ``synplan.chem.reaction_routes.depiction``
     - ``synplan.routes.depiction``
   * - ``synplan.chem.reaction_routes.hash_route``
     - ``synplan.routes.route_cgr.hash``
   * - ``synplan.chem.reaction_routes.io``
     - ``synplan.routes.io``
   * - ``synplan.chem.reaction_routes.leaving_groups``
     - ``synplan.routes.clustering.leaving_groups``
   * - ``synplan.chem.reaction_routes.notebook_plots``
     - ``synplan.routes.notebook_plots``
   * - ``synplan.chem.reaction_routes.route_cgr``
     - ``synplan.routes.route_cgr``
   * - ``synplan.chem.reaction_routes.route_cgr_container``
     - ``synplan.routes.route_cgr.container``
   * - ``synplan.chem.reaction_routes.route_cgr_depiction``
     - ``synplan.routes.route_cgr.depiction``
   * - ``synplan.chem.reaction_routes.route_cgr_state``
     - ``synplan.routes.route_cgr.state``

Module contents
---------------

.. automodule:: synplan.chem.reaction_routes
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
