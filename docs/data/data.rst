.. _data:

================================
Data
================================

``SynPlanner`` operates reaction and molecule data stored in different formats.

.. table::
    :widths: 30 50

    ================================== =================================================================================
    Data type                          Description
    ================================== =================================================================================
    Reactions                          Reactions can be loaded and stored as the list of reaction smiles in the file (.smi) or RDF File (.rdf)
    Molecules                          Molecules can be loaded and stored as the list of molecule smiles in the file (.smi) or SDF File (.sdf)
    Reaction rules                     Reaction rules can be loaded and stored as the pickled list of CGRtools ReactionContainer objects (.pickle)
    Retrosynthetic models              Retrosynthetic models (neural networks) can be loaded and stored as serialized PyTorch models (.ckpt)
    Retrosynthetic routes              Retrosynthetic routes can be visualized and stored as HTML files (.html) and can be stored as JSON files (.json)
    ================================== =================================================================================

.. note::
    Reaction and molecule file formats are parsed and recognized automatically by ``SynPlanner`` from file extensions.
    Be sure to store the data with the correct extension.

.. toctree::
    :hidden:
    :titlesonly:

    download