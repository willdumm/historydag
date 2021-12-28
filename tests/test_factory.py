import historydag.dag as hdag
import historydag.utils as dagutils
from collections import Counter

newicklistlist = [
           [
               "((AA, CT)CG, (TA, CC)CG)CC;",
               "((AA, CT)CA, (TA, CC)CC)CC;"
           ],
           [
               "((CA, GG)CA, AA, (TT, (CC, GA)CC)CC)AA;",
               "((CA, GG)CA, AA, (TT, (CC, GA)CA)CA)AA;",
               "((CA, GG)CG, AA, (TT, (CC, GA)GC)GC)AG;"
           ],
]

dags = [hdag.history_dag_from_newicks(newicklist, ['name']) for newicklist in newicklistlist]
cdags = [dag.copy() for dag in dags]

for dag in cdags:
    dag.convert_to_collapsed()

def _testfactory(resultfunc, verify_func):
    for dag in dags:
        result = resultfunc(dag)
        verify_result = Counter([verify_func(tree) for tree in dag.get_trees()])
        assert result == verify_result

def test_parsimony():

    def parsimony(tree):
        tree.recompute_parents()
        return sum(dagutils.hamming_distance(list(node.parents)[0].label, node.label)
                   for node in hdag.postorder(tree)
                   if node.parents)

    _testfactory(lambda dag: dag.weight_count(), parsimony)

def test_copy():
    _testfactory(lambda dag: Counter(tree.to_newick() for tree in dag.copy().get_trees()), lambda tree: tree.to_newick())

def test_newicks():
    _testfactory(lambda dag: dag.to_newicks(), lambda tree: tree.to_newick())


