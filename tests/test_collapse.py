import ete3
import historydag.dag as hdag
from historydag import utils
import pickle
import random
newickstring3 = (
        "(((4[&&NHX:name=4:sequence=K],(6[&&NHX:name=6:sequence=J],"
        "7[&&NHX:name=7:sequence=I])5[&&NHX:name=5:sequence=H])2[&&NHX:name=2:sequence=H],"
        "8[&&NHX:name=8:sequence=F],(11[&&NHX:name=11:sequence=E],"
        "10[&&NHX:name=10:sequence=D])9[&&NHX:name=9:sequence=C])"
        "3[&&NHX:name=3:sequence=H])1[&&NHX:name=1:sequence=A];"
)

etetree = ete3.TreeNode(newick=newickstring3, format=1)
etetree2 = utils.collapse_adjacent_sequences(etetree.copy())


with open('sample_data/toy_trees_100_collapsed.p', 'rb') as fh:
    collapsed = pickle.load(fh)
with open('sample_data/toy_trees_100_uncollapsed.p', 'rb') as fh:
    uncollapsed = pickle.load(fh)
trees = collapsed + uncollapsed

def test_fulltree():
    dag = hdag.history_dag_from_etes([etetree])
    dag.convert_to_collapsed()
    assert(set(utils.deterministic_newick(tree.to_ete()) for tree in dag.get_trees()) == set({utils.deterministic_newick(etetree2)}))

def test_twotrees():
    dag = hdag.history_dag_from_etes([etetree, etetree2])
    dag.convert_to_collapsed()
    assert(dag.count_trees() == 1)
    assert({utils.deterministic_newick(tree.to_ete()) for tree in dag.get_trees()} == {utils.deterministic_newick(tree) for tree in [etetree, etetree2]})


def test_collapse():
    uncollapsed_dag = hdag.history_dag_from_etes(trees)
    uncollapsed_dag.convert_to_collapsed()
    allcollapsedtrees = [utils.collapse_adjacent_sequences(tree) for tree in trees]
    collapsed_dag = hdag.history_dag_from_etes(allcollapsedtrees)
    assert(set(utils.deterministic_newick(tree.to_ete()) for tree in uncollapsed_dag.get_trees()) == set(utils.deterministic_newick(tree.to_ete()) for tree in collapsed_dag.get_trees()))
    
