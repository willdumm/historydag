import ete3
import historydag.dag as hdag
from historydag import utils
import pickle
import random
newickstring3 = (
        "((4[&&NHX:name=4:sequence=K],(6[&&NHX:name=6:sequence=J],"
        "7[&&NHX:name=7:sequence=I])5[&&NHX:name=5:sequence=H])2[&&NHX:name=2:sequence=H],"
        "8[&&NHX:name=8:sequence=F],(11[&&NHX:name=11:sequence=E],"
        "10[&&NHX:name=10:sequence=D])9[&&NHX:name=9:sequence=C], 12[&&NHX:name=9:sequence=Z])"
        "3[&&NHX:name=3:sequence=H];"
)

etetree = list(hdag.history_dag_from_etes([ete3.TreeNode(newick=newickstring3, format=1)], ['sequence']).get_trees())[0].to_ete(features=['sequence'])
etetree2 = utils.collapse_adjacent_sequences(etetree.copy())


with open('sample_data/toy_trees_100_collapsed.p', 'rb') as fh:
    collapsed = pickle.load(fh)
with open('sample_data/toy_trees_100_uncollapsed.p', 'rb') as fh:
    uncollapsed = pickle.load(fh)
trees = collapsed + uncollapsed
for tree in trees:
    if len(tree.children) == 1:
        newchild = tree.copy()
        for child in newchild.get_children():
            newchild.remove_child(child)
        tree.add_child(newchild)
        assert newchild.is_leaf()

def test_fulltree():
    dag = hdag.history_dag_from_etes([etetree], ['sequence'])
    dag.convert_to_collapsed()
    assert(set(utils.deterministic_newick(tree.to_ete()) for tree in dag.get_trees()) == set({utils.deterministic_newick(etetree2)}))

def test_twotrees():
    dag = hdag.history_dag_from_etes([etetree, etetree2], ['sequence'])
    dag.convert_to_collapsed()
    assert(dag.count_trees() == 1)
    assert({utils.deterministic_newick(tree.to_ete()) for tree in dag.get_trees()} == {utils.deterministic_newick(etetree2)})


def test_collapse():
    uncollapsed_dag = hdag.history_dag_from_etes(trees, ['sequence'])
    uncollapsed_dag.convert_to_collapsed()
    allcollapsedtrees = [utils.collapse_adjacent_sequences(tree) for tree in trees]
    collapsed_dag = hdag.history_dag_from_etes(allcollapsedtrees, ['sequence'])
    maybecollapsedtrees = [tree.to_ete() for tree in uncollapsed_dag.get_trees()]
    collapsedtrees = [tree.to_ete() for tree in collapsed_dag.get_trees()]
    assert all(utils.is_collapsed(tree) for tree in maybecollapsedtrees)
    n_before = uncollapsed_dag.count_trees()
    uncollapsed_dag.merge(collapsed_dag)
    assert n_before == uncollapsed_dag.count_trees()

def test_add_allowed_edges():
    # See that adding only edges that preserve parent labels preserves parsimony
    dag = hdag.history_dag_from_etes(trees, ['sequence'])
    dag.add_all_allowed_edges(preserve_parent_labels=True)
    c = dag.weight_count()
    assert min(c) == max(c)

    # See that adding only edges between nodes with different labels preserves collapse
    allcollapsedtrees = [utils.collapse_adjacent_sequences(tree) for tree in trees]
    collapsed_dag = hdag.history_dag_from_etes(allcollapsedtrees, ['sequence'])
    collapsed_dag.add_all_allowed_edges(adjacent_labels=False)
    assert all(parent.label != target.label for parent in collapsed_dag.postorder() for target in parent.children())
