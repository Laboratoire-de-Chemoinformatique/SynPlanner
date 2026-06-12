# Clustering Route Fixtures

These route fixtures were generated from SynPlanner MCTS trees and are used so
clustering tests do not have to solve retrosynthesis targets during CI.
Generation set Python, NumPy, and Torch seeds to `0` before running MCTS.

Targets and generated route counts:

- `routes_mol_simple.json` (634 routes): `CCNc1nc(Sc2ccc(C)cc2)cc(C(F)(F)F)n1`
- `routes_mol_medium.json` (28 routes): `c1cc(ccc1C2=NN(C(C=C2)=N)CCCC(O)=O)OCc3cc([N+]([O-])=O)ccc3`
- `routes_mol_complex.json` (8 routes): `c1cnccc1C(c2cncs2)(c3ccc4c(c(c(c(n4)Cl)Cc5ccc(cc5)Cl)Cl)c3)O`

Generation used the `synplanner-gps` preset. The simple and medium fixtures
used this tree configuration:

```python
TreeConfig(
    search_strategy="expansion_first",
    algorithm="UCT",
    enable_pruning=False,
    max_iterations=300,
    max_time=120,
    max_depth=6,
    min_mol_size=1,
    silent=True,
)
```

The complex fixture uses the same configuration except:

```python
max_iterations=500
max_time=180
```
