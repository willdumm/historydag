# HistoryDAG

This package provides an implementation for a history DAG object, which
compactly expresses a collection of internally labeled trees which share
a common set of leaf labels.

## Getting Started

Once you've cloned the repo, `pip install -e historydag` should be enough to
get setup.

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
```

## Important Details

In order to create a history DAG from a collection of trees, each tree should
meet the following criteria:

* No unifurcations, including at the root node. Each node must have at least
    two children, unless it's a leaf node.
* The label attributes used to construct history DAG labels must be unique,
    because history DAG nodes which represent leaves must be labeled uniquely.

In addition, all the trees in the collection must have identical sets of leaf labels.

## Documentation

After installing requirements from `requirements.txt`, do `make docs` to build
sphinx documentation locally. You'll find it at `docs/_build/html/index.html`.
