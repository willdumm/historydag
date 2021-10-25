import historydag.dag as hdag
from historydag import utils
import pickle
import random

with open('sample_data/toy_trees_100_collapsed.p', 'rb') as fh:
    collapsed = pickle.load(fh)
with open('sample_data/toy_trees_100_uncollapsed.p', 'rb') as fh:
    uncollapsed = pickle.load(fh)
trees = collapsed + uncollapsed

def test_collapse():
    uncollapsed_dag = hdag.history_dag_from_etes(trees)
    uncollapsed_dag.convert_to_collapsed()
    allcollapsedtrees = [utils.collapse_adjacent_sequences(tree) for tree in trees]
    collapsed_dag = hdag.history_dag_from_etes(allcollapsedtrees)
    assert(set(utils.deterministic_newick(tree.to_ete()) for tree in uncollapsed_dag.get_trees()) == set(utils.deterministic_newick(tree.to_ete()) for tree in collapsed_dag.get_trees()))
    
