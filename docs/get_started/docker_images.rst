Use prebuilt Docker images
--------------------------

Prebuilt images are published to GitHub Container Registry (GHCR) for the Linux/AMD64 platform.

Registry and tags
~~~~~~~~~~~~~~~~~

- Registry path: ``ghcr.io/laboratoire-de-chemoinformatique/synplanner``
- Tag pattern: ``<version>-cli-amd64`` or ``<version>-gui-amd64``

Set your version once (replace with an existing tag if needed):

.. code-block:: bash

   VERSION=1.2.1

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

Quick planning example (mount config and data):

.. code-block:: bash

   docker run --rm --platform linux/amd64 \
     -v "$(pwd)":/app -w /app \
     ghcr.io/laboratoire-de-chemoinformatique/synplanner:${VERSION}-cli-amd64 \
     planning \
       --config configs/planning.yaml \
       --targets tutorials/synplan_data/benchmarks/sascore/targets_with_sascore_1.5_2.5.smi \
       --reaction_rules tutorials/synplan_data/uspto/uspto_reaction_rules.pickle \
       --building_blocks tutorials/synplan_data/building_blocks/building_blocks_em_sa_ln.smi \
       --policy_network tutorials/synplan_data/uspto/weights/ranking_policy_network.ckpt \
       --results_dir tutorials/planning_results

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


