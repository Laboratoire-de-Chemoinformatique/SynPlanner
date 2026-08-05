synplan.chem package
====================

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   synplan.chem.data
   synplan.chem.reaction.routes
   synplan.chem.reaction.rules

Submodules
----------

synplan.chem.molecule package
-----------------------------

For downstream applications and documentation, prefer package-level imports
such as ``from synplan.chem.molecule import Precursor, mol_from_smiles``. Within
SynPlanner, import from the focused ``molecule.precursor``,
``molecule.standardization``, and ``molecule.io`` modules so ownership and
dependency boundaries remain explicit. The historical ``synplan.chem.precursor``
and ``synplan.chem.utils`` imports remain supported for v1.6 compatibility.

.. automodule:: synplan.chem.molecule
   :members:
   :undoc-members:
   :show-inheritance:

Molecule submodules
~~~~~~~~~~~~~~~~~~~

.. automodule:: synplan.chem.molecule.standardization
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synplan.chem.molecule.io
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synplan.chem.molecule.precursor
   :members:
   :undoc-members:
   :show-inheritance:

synplan.chem.precursor compatibility module
-------------------------------------------

.. automodule:: synplan.chem.precursor
   :members:
   :undoc-members:
   :show-inheritance:

synplan.chem.reaction package
-----------------------------

.. automodule:: synplan.chem.reaction
   :members:
   :undoc-members:
   :show-inheritance:

synplan.chem.rdkit\_compat module
----------------------------------

.. automodule:: synplan.chem.rdkit_compat
   :members:
   :undoc-members:
   :show-inheritance:

synplan.chem.utils module
-------------------------

.. automodule:: synplan.chem.utils
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: synplan.chem
   :members:
   :undoc-members:
   :show-inheritance:
