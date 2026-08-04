Use prebuilt Docker images
--------------------------

Prebuilt images are published to GitHub Container Registry (GHCR) for the Linux/AMD64 platform.

Registry and tags
~~~~~~~~~~~~~~~~~

- Registry path: ``ghcr.io/laboratoire-de-chemoinformatique/synplanner``
- Tag pattern: ``<version>-cli-amd64`` or ``<version>-gui-amd64``

Set your version once (replace with an existing tag if needed):

.. parsed-literal::

   VERSION=\ |release|

Pull images
~~~~~~~~~~~

.. code-block:: bash

   # CLI image
   docker pull ghcr.io/laboratoire-de-chemoinformatique/synplanner:${VERSION}-cli-amd64

   # GUI image
   docker pull ghcr.io/laboratoire-de-chemoinformatique/synplanner:${VERSION}-gui-amd64

Run the CLI
~~~~~~~~~~~

Show help:

.. code-block:: bash

   docker run --rm --platform linux/amd64 \
     ghcr.io/laboratoire-de-chemoinformatique/synplanner:${VERSION}-cli-amd64 --help

Quick planning example (mount config and data). Paths below are the layout
``synplan download_preset --preset synplanner-gps --save_to synplan_data``
writes; ``targets.smi`` is your own file of target SMILES:

.. code-block:: bash

   docker run --rm --platform linux/amd64 \
     -v "$(pwd)":/app -w /app \
     ghcr.io/laboratoire-de-chemoinformatique/synplanner:${VERSION}-cli-amd64 \
     planning \
       --config configs/planning_standard.yaml \
       --targets targets.smi \
       --reaction_rules synplan_data/policy/supervised_gps/v1/reaction_rules.tsv \
       --building_blocks synplan_data/building_blocks/emolecules-salt-ln/building_blocks.tsv \
       --policy_network synplan_data/policy/supervised_gps/v1/v1/ranking_policy.ckpt \
       --results_dir planning_results

Run the GUI
~~~~~~~~~~~

Expose port ``8501`` on the host and open your browser at ``http://localhost:8501``:

.. code-block:: bash

   docker run --rm --platform linux/amd64 -p 8501:8501 \
     ghcr.io/laboratoire-de-chemoinformatique/synplanner:${VERSION}-gui-amd64

Notes
~~~~~

- The repository path is lowercased for Docker compatibility.
- Images are built for ``linux/amd64``; running on arm64 hosts may require emulation (e.g. Docker Desktop with Rosetta/qemu).
- For data download instructions, see :doc:`data_download`.

