synplan.route\_quality compatibility package
=============================================

``synplan.route_quality`` is kept as a compatibility namespace for existing
user imports. New code should import route-quality helpers from
``synplan.routes.quality``.

Compatibility mapping
---------------------

.. list-table::
   :header-rows: 1

   * - Old import path
     - New import path
   * - ``synplan.route_quality``
     - ``synplan.routes.quality``
   * - ``synplan.route_quality.scorer``
     - ``synplan.routes.quality.scorer``
   * - ``synplan.route_quality.protection``
     - ``synplan.routes.quality.protection``
   * - ``synplan.route_quality.protection.config``
     - ``synplan.routes.quality.protection.config``
   * - ``synplan.route_quality.protection.functional_groups``
     - ``synplan.routes.quality.protection.functional_groups``
   * - ``synplan.route_quality.protection.reaction_classifier``
     - ``synplan.routes.quality.protection.reaction_classifier``
   * - ``synplan.route_quality.protection.scanner``
     - ``synplan.routes.quality.protection.scanner``
   * - ``synplan.route_quality.protection.scorer``
     - ``synplan.routes.quality.protection.scorer``

Module contents
---------------

.. automodule:: synplan.route_quality
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
