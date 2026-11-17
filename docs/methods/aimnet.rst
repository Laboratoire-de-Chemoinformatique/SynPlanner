AIMNet2 reaction thermochemistry
================================

``AIMNetReactionAssessor`` annotates completed ``Route`` objects after planning.
It filters unsupported species, balances equations with SynPlanner's adapted
SynRBL strategy, enumerates missing stereochemistry, generates LoQI conformers,
and calculates gas-phase reaction energies, enthalpies, and Gibbs energies.
It returns route copies in input order; it does not rank routes or change search.

See :doc:`../user_guide/21_AIMNet_Thermochemistry` for an executable example
using all 435 pre-generated apatinib routes (1,192 steps) in saved order,
independent JSON/HTML exports, and persistent cache reuse. The bundled snapshot preserves
planning scores and Chemformer annotations; its adjacent provenance file records
the original collection hash and route selection. No planning is rerun.

Installation
------------

This workflow needs Python 3.11 or newer. Base SynPlanner retains Python 3.10
support. AIMNet2 and ASE are optional and are included in ``SynPlanner[cpu]``
and ``SynPlanner[all]``; ``SynPlanner[aimnet]`` installs just this optional stack.
LoQI is not yet distributed on PyPI and is installed separately at a pinned
revision:

.. code-block:: bash

   # Install CPU PyTorch first; pip extras do not select a PyTorch wheel index.
   python -m pip install 'torch>=2.10' --index-url https://download.pytorch.org/whl/cpu
   python -m pip install 'SynPlanner[cpu]'
   python -m pip install 'loqi @ git+https://github.com/isayevlab/LoQI@04f3caef4521b3fbcb236f8b01f4bc80c83cb918'

For a source checkout, replace ``SynPlanner[cpu]`` with ``-e '.[cpu]'``.
The repository's ``uv sync --extra cpu`` also selects CPU PyTorch. Install LoQI
after syncing, or use ``uv sync --inexact`` to retain the separate installation.
A future LoQI PyPI release can be added to the extras without changing the API.

Model weights resolve through upstream checksum-verified registries on first
use. Set ``AIMNET_CACHE_DIR`` and ``LOQI_CACHE_DIR`` to control download locations.
For offline use, provide ``aimnet_checkpoint`` and ``loqi_checkpoint`` paths to
local copies of the released gas-phase AIMNet2 wB97M-D3 and LoQI Flow v1.1.0
checkpoints. LoQI's bundled solvent-trained optimizer is not used: all optimization
and thermochemistry use the gas-phase AIMNet2 calculator.

Python interface
----------------

.. code-block:: python

   from synplan.chem.reaction.routes.io import read_routes_json, write_routes_json
   from synplan.chem.reaction.score.aimnet import AIMNetConfig, AIMNetReactionAssessor
   from synplan.utils.visualisation import thermochemistry_report_html

   routes = read_routes_json("routes.json", as_routes=True)
   config = AIMNetConfig(
       device="cpu", temperature=298.15, pressure=101325.0,
       n_conformers=10, seed=42,
       symmetry_numbers={"O": 2, "[H][H]": 2},  # Optional per-species overrides.
   )
   assessor = AIMNetReactionAssessor(config, cache_path="aimnet_cache.json")
   assessed = assessor.assess(routes, silent=False)
   write_routes_json(assessed, "routes_aimnet.json")
   thermochemistry_report_html(assessed, "routes_aimnet.html",
                              manifest={"statistics": assessor.last_stats})
   print(assessor.last_stats)

``assess_reactions(reactions, silent=True)`` accepts an iterable of Chython
``ReactionContainer`` objects and returns aligned JSON-compatible dictionaries.
``assess(routes)`` places those dictionaries on ``Step.thermochemistry`` and
coverage/provenance on ``Route.assessment["aimnet"]``. The input chemistry,
product identities, route order, forward annotations, and planning scores remain
intact. Chemformer and AIMNet assessment can run in either order. Saved Route
JSON is the persistence interface for this post-processing workflow.

Each annotation contains original/balanced equations, balancing confidence,
inferred species and multiplicities, per-variant equations and results, and a
stereospecific molecular-results dictionary. Reaction ``delta_e``, ``delta_h``,
and ``delta_g`` are in kcal/mol, with products minus reactants as the sign
convention. Molecular energies are explicitly labelled in eV, coordinates in
angstrom, masses in amu, and vibrational frequencies in cm^-1.

Chemistry and numerical contracts
---------------------------------

* Reactants and all products contribute, with repeated species counted repeatedly.
  Disconnected components are separate gas-phase species. Reagent-field catalysts
  and solvents are recorded but do not contribute unless balancing places them
  on a participating side. Unsupported participating elements or radicals cause
  a skip; metal-containing species are not silently removed from an equation.
* Supported elements are H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, and I.
  Charges are passed explicitly; isotope labels and masses are preserved.
* Already-balanced equations need no inference. Otherwise balancing requires
  confidence at least 0.8, refuses unsupported redox completion, and neither
  invents redox agents nor drops competing products. Element/isotope counts,
  net charge and applicability are rechecked afterwards. Confidence is a
  balancing heuristic, not an experimental probability.
* Unspecified tetrahedral and ordinary E/Z stereo is exhaustively enumerated,
  preserving specified labels. There is no cap or sampling. Equivalent species
  permutations are deduplicated without dropping stoichiometric multiplicities.
  Both sides are enumerated independently: these are thermodynamic alternatives,
  not predictions of selectivity, inversion, retention, or a coherent stereo path
  across an entire route. Unsupported stereo, including axial allenes and
  three-neighbor sulfoxide centers, is reported explicitly.
* LoQI generates ten conformers per unique stereoisomer using 25 flow steps and
  species-specific seeds. ASE BFGS optimizes them to 0.01 eV/angstrom, with at most
  200 steps each. Coordinates independently verify connectivity and R/S and E/Z
  labels before and after optimization. Nonfinite/collapsed structures,
  unconverged optimizations and wrong stereoisomers are rejected. Geminal
  hydrogen checks exempt the legitimate H-H bond in H2. A monatomic species
  has no conformational freedom and uses a single atom at the origin.
* Candidates are examined in ascending electronic energy. The first valid
  minimum supplies the molecular result. An analytical Hessian is mass-weighted
  and projected to remove rigid translations/rotations. Linear and monatomic
  species use the correct mode counts. Imaginary/nonpositive vibrational modes
  are rejected, never discarded or replaced by their absolute value.
* Thermal corrections use single-conformer ideal-gas rigid-rotor harmonic-
  oscillator (RRHO) statistics, at 298.15 K and 101325 Pa by default. Rotational
  symmetry defaults to 1; provide isomeric-SMILES overrides when appropriate.
  Each molecular result records whether symmetry was assumed. No solvent,
  low-frequency quasi-harmonic correction, conformer-ensemble entropy, stereo
  mixing entropy, or activation barrier is implied by these values.

A result has status ``ok``, ``partial``, ``failed``, or ``skipped``. Failure
records distinguish ``filtered_out`` policy exclusions from ``errored``
processing failures through stable categories and reason codes. Failed variants
retain null energies and stage/reason diagnostics rather than numerical zero.
A single rejected species makes its reaction variant unavailable. Other variants
and routes continue. Installation, device, checkpoint and unexpected programming
errors remain exceptions.

Caching and reproducibility
---------------------------

Successful molecular results are checkpointed atomically after each calculation.
Isolated protons use exact electronic energy 0 eV and isotope-aware monatomic
thermal contributions. Previous model-derived proton cache entries are
invalidated selectively; other compatible molecular entries remain reusable.

Cache identity includes stereochemistry, isotope/charge identity, model byte
hashes, software versions, sampling/optimization settings, and thermochemical
assumptions. Failures are reused only within the current assessment and retried
on the next call. Interrupted runs retain completed molecular calculations.
Use one writer per cache file; concurrent writers are not supported.

``cache_path=None`` uses an in-memory cache. ``last_stats`` records reaction and
variant coverage, unique species, cache hits, molecular calculations/failures,
and elapsed/calculation seconds. Reopening a compatible file cache requires
model identity resolution but performs no new generation/optimization for cached
species. CPU is the default; ``device="cuda"`` or ``"cuda:N"`` selects an explicitly
installed compatible GPU runtime. Seeds are reproducible within a runtime;
numerical equality across different devices/builds is not guaranteed. LoQI may
emit its own sampling progress even when the assessor's progress is disabled.

Exhaustive enumeration can grow exponentially. Generation and thermochemistry
are reused per molecular stereoisomer and reaction combinations are generated
lazily, but the complete returned annotation and exported report retain every
variant and can therefore be large.

Independent reports and complete runs
-------------------------------------

``thermochemistry_report_html(routes, html_path, manifest=None)`` renders every
route in input order, followed by a step table. The five values are reaction
ΔE, ΔH, ΔG (kcal/mol), balancing confidence, and calculated/total stereochemical
variants. Multiple variants show labeled min-max ranges over successful values;
expand the row to inspect every variant without averaging. Partially calculated
steps are Errored and retain their successful ranges and explicit coverage.

Processed, Filtered out, and Errored are separate display states. Intentional
filters include unsupported elements/radicals/stereo, unsupported redox
completion, and confidence refusals. Invalid input, unsolved balancing,
conservation failures, and failed generation/optimization/frequencies are errors.
Older unknown failure categories are treated as errors. Errored rows have a
copyable/downloadable compact log with stage, reason, species, variant IDs and
grouped conformer failures. Full coordinates/frequencies remain in Route JSON.

The Manifest button at the top right opens configuration, calculator/model
identity, source provenance, timing, cache statistics and reference conventions.
Conditions remain visible. Original search scores are shown; forward-validation
results are excluded. Both thermochemistry and forward HTML APIs default to
``show_policy_log_likelihood=False``. The forward scorer CLI offers
``--show-policy-log-likelihood`` as an explicit opt-in.

From the repository root, the complete apatinib run can be reproduced with:

.. code-block:: bash

   python -m scripts.assess_thermochemistry \
     --input tutorials/data/apatinib_routes.json \
     --output tutorial_results/aimnet_apatinib \
     --cache tutorial_results/aimnet_cache.json \
     --aimnet-model /path/aimnet2_wb97m_d3_0.pt \
     --loqi-model /path/loqi_flow_v1.1.0.ckpt \
     --workers 6 --threads 4 --verify-cache

The default is one CPU worker. Multiple isolated CPU workers preserve per-species
seeds; the parent is the only writer to the molecular cache. Each worker result
and each completed reaction is checkpointed atomically. Identical runs resume
completed reactions, including recorded failures; use a new output directory to
retry failed reactions while retaining the molecular cache. Changed input,
calculation code, models or settings require a new reaction-checkpoint directory.
Unexpected errors propagate and retain a traceback plus a partial report whose
unprocessed steps are explicitly Errored — run aborted. Cached verification
covers every previously successful reaction and forbids new molecular calculations.

CPU and GPU reaction benchmarks
-------------------------------

The benchmark uses six reactions from the saved apatinib collection: the final
substitution with H+ and Cl-, an already-balanced hydration, two amidations
with different molecular sizes, and reactions with two and four stereochemical
variants. It does not extrapolate these measurements to the full collection.

From the repository root, with local weights and a separate CUDA-enabled Python
environment containing the same scientific package versions:

.. code-block:: bash

   python -m scripts.benchmark_thermochemistry \
     --input tutorials/data/apatinib_routes.json \
     --output tutorial_results/aimnet_benchmark \
     --aimnet-model /path/aimnet2_wb97m_d3_0.pt \
     --loqi-model /path/loqi_flow_v1.1.0.ckpt \
     --gpu-python /path/cuda-environment/bin/python \
     --cpu-cores 1 4 8 --gpu-host-cores 4 --repeats 3

Each configuration uses one process pinned to distinct physical cores on one
NUMA node. The GPU configuration uses one CUDA device and four host CPU cores.
Configurations run sequentially; their order and reaction order rotate between
repetitions. Ten conformers, 25 sampling steps, seed 42, and the standard
optimization and thermochemical settings remain unchanged.

Every reaction and repetition starts with an empty molecular-results cache.
Species may be reused within that reaction, but never across benchmark trials.
Model loading and a water warm-up are measured separately; model weights and
operating-system file caches remain warm. CUDA is synchronized at calculation
timing boundaries. Any failed GPU preflight aborts the benchmark before CPU calculations begin.
The failure remains visible in the report and attempt diagnostics.

The output separates molecular calculation time, molecular-cache writing time,
and complete reaction thermo-data writing time. Both write measurements include
JSON serialization, buffered file close and atomic replacement, without forced
``fsync``; they are not measurements of durable disk latency. Remaining
preparation, balancing, enumeration and aggregation time is reported separately.
Total reaction time excludes startup, timing-record writes and report generation.

``benchmark.html``, ``benchmark.json`` and ``trials.csv`` contain individual
trials and per-reaction medians, means and ranges. Failed trials remain visible;
speedups require all repetitions to have complete successful variant coverage.
Raw records also retain conformer outcomes and CPU/GPU energy differences.
Configuration, identities and startup timings appear once in the Manifest.
These measurements reflect the machine's observed load, not exclusive access
to its hardware.

Completed trials are checkpointed. Repeating the same command resumes completed
trials; an interrupted trial starts again with a new empty cache. Changed source,
models, input or settings require a new output directory. Generate a partial
report at any time without rerunning calculations:

.. code-block:: bash

   python -m scripts.benchmark_thermochemistry \
     --report-only tutorial_results/aimnet_benchmark

Balancing audit scope
---------------------

The apatinib aryl-chloride substitution adds separate H+ and Cl- with confidence
0.88. Proton records, including isotope-labeled protons, do not authorize redox
completion. Hydrogen radicals, H2 and hydride retain their separate treatment.

Representative earlier USPTO quarantine cases remain refused by the existing
confidence/redox rules. Two concrete discrepancies remain in controlled examples:
a Suzuki equation can infer ``OB(O)Br``, and Boc protection can infer the unstable
carbonate monoester ``CC(C)(C)OC(=O)O``. Balanced atoms alone do not establish that
these are chemically appropriate by-products. This change does not introduce the
broader reagent-evidence policy or claim parity with the historical experimental
balancer; reported energies describe the recorded inferred equation.

Validation and references
-------------------------

Normal tests use injected deterministic calculators and never download weights.
To run the real-model CPU checks against local checkpoints:

.. code-block:: bash

   SYNPLAN_AIMNET_MODEL=/path/aimnet2_wb97m_d3_0.pt \
   SYNPLAN_LOQI_MODEL=/path/loqi_flow_v1.1.0.ckpt \
   OMP_NUM_THREADS=4 pytest tests/integration/test_aimnet_thermochemistry.py

The checks cover already-balanced and completed equations, two product
stereoisomers, direct AIMNet2/ASE parity, and reopening the cache with no new
molecular calculations.

* `AIMNet2 model and applicability <https://github.com/isayevlab/aimnetcentral/blob/v0.2.6/docs/models/aimnet2.md>`_
* `AIMNet2 optimization and thermochemistry <https://github.com/isayevlab/aimnetcentral/blob/v0.2.6/docs/tutorials/geometry_optimization.md>`_
* `Pinned LoQI runtime and model <https://github.com/isayevlab/LoQI/tree/04f3caef4521b3fbcb236f8b01f4bc80c83cb918>`_
* :doc:`rebalancing`
