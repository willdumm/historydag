# HistoryDAG

This package provides an implementation for a history DAG object, which
compactly expresses a collection of internally labeled trees which share
a common set of leaf labels.

## Getting Started

HistoryDAG is on PyPI! Install with `pip install historydag`.

Alternatively, once you've cloned the repo, `pip install -e historydag` should be enough to
get set up.

There is sample data in `sample_data/`. For example:

```python
import historydag as hdag
import pickle

with open('sample_data/toy_trees.p', 'rb') as fh:
	ete_trees = pickle.load(fh)

dag = hdag.history_dag_from_etes(ete_trees, ['sequence'])
dag.count_trees()  # 1041

dag.add_all_allowed_edges()
dag.count_trees()  # 3431531

dag.hamming_parsimony_count()  # Shows counts of trees of different parsimony scores
dag.trim_optimal_weight()
# With default args, same as hamming_parsimony_count
dag.weight_count()  # Counter({75: 45983})

dag.convert_to_collapsed()
dag.weight_count()  # Counter({75: 1208})
dag.count_topologies()  # 1054 unique topologies, ignoring internal labels

# To count parsimony score and the number of unique nodes in each tree jointly:
node_count_funcs = hdag.utils.AddFuncDict(
    {
        "start_func": lambda n: 0,
        "edge_weight_func": lambda n1, n2: n1.label != n2.label,
        "accum_func": sum,
    },
    name="NodeCount",
)
dag.weight_count(**(node_count_funcs + hdag.utils.hamming_distance_countfuncs))
# Counter({(50, 75): 444, (51, 75): 328, (49, 75): 270, (52, 75): 94, (48, 75): 68, (53, 75): 4})

# To trim to only the trees with 48 unique node labels:
dag.trim_optimal_weight(**node_count_funcs, optimal_func=min)

# Sample a tree from the dag and make it an ete tree
t = dag.sample().to_ete()

# the history DAG also supports indexing and iterating:
t = dag[0].to_ete()
trees = [tree for tree in dag]

# Another method for fetching all trees in the dag is provided, but the order
# will not match index order:
scrambled_trees = list(dag.get_trees())


# Union is implemented as dag merging, including with sequences of dags
newdag = dag[0] | dag[1]
newdag = dag[0] | (dag[i] for i in range(3,5))
```

### Highlights
* History DAGs can be created with top-level functions like
    * `from_newick`
    * `from_ete`
    * `history_dag_from_newicks`
    * `history_dag_from_etes`
* Trees can be extracted from the history DAG with methods like
    * `HistoryDag.get_trees`
    * `HistoryDag.sample`
    * `HistoryDag.to_ete`
    * `HistoryDag.to_newick` and `HistoryDag.to_newicks`
* Simple history DAGs can be inspected with `HistoryDag.to_graphviz`
* The DAG can be trimmed according to arbitrary tree weight functions. Use
    `HistoryDag.trim_optimal_weight`.
* Disambiguation of sparse ambiguous labels can be done efficiently, but
    doesn't scale well. Use `HistoryDag.explode_nodes` followed by
    `HistoryDag.trim_optimal_weight`.
* Weights of trees in the DAG can be counted, according to arbitrary weight functions
    using `HistoryDag.weight_count`. The class `utils.AddFuncDict` is provided
    to manage these function arguments, and implements addition so that
    different weights can be counted jointly. These same functions can be used
    in trimming.

## Important Details

In order to create a history DAG from a collection of trees, each tree should
meet the following criteria:

* No unifurcations, including at the root node. Each node must have at least
    two children, unless it's a leaf node.
* The label attributes used to construct history DAG labels must be unique,
    because history DAG nodes which represent leaves must be labeled uniquely.


## Documentation

Docs are available at [https://matsengrp.github.io/historydag](https://matsengrp.github.io/historydag).

To build docs, after installing requirements from `requirements.txt`, do `make docs` to build
sphinx documentation locally. You'll find it at `docs/_build/html/index.html`.
