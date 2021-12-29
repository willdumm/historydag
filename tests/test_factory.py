import pickle
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

dags = [hdag.history_dag_from_newicks(newicklist, [], label_functions={'sequence': lambda n: n.name}) for newicklist in newicklistlist]

with open('sample_data/toy_trees_100_uncollapsed.p', 'rb') as fh:
    uncollapsed = pickle.load(fh)
# dags.append(hdag.history_dag_from_etes(uncollapsed, [], label_functions={'sequence': lambda n: n.sequence[0]}))
uncollapsed_newicks = [tree.write(format=1, features=['sequence'], format_root_node=True) for tree in uncollapsed]
dags.append(hdag.history_dag_from_newicks(uncollapsed_newicks, [], label_functions={'sequence': lambda n: n.sequence[0]}))
# dags.extend([dag.add_all_allowed_edges(new_from_root=False) for dag in dags])

cdags = dags
# cdags = [dag.copy() for dag in dags]
# for dag in cdags:
#     dag.convert_to_collapsed()

def _testfactory(resultfunc, verify_func, collapse_invariant=False, accum_func=Counter):
    for dag, cdag in zip(dags, cdags):
        # check dags
        result = resultfunc(dag)
        verify_result = accum_func([verify_func(tree) for tree in dag.get_trees()])
        assert result == verify_result

        # check cdags
        cresult = resultfunc(cdag)
        cverify_result = accum_func([verify_func(tree) for tree in cdag.get_trees()])
        assert result == verify_result

        # check they agree, if collapse_invariant.
        if collapse_invariant:
            assert result == cresult

def test_parsimony():
    # test parsimony counts without ete
    def parsimony(tree):
        tree.recompute_parents()
        return sum(dagutils.hamming_distance(list(node.parents)[0].label.sequence, node.label.sequence)
                   for node in hdag.postorder(tree)
                   if node.parents)

    _testfactory(lambda dag: dag.weight_count(), parsimony, collapse_invariant=True)

def test_parsimony_counts():
    # test parsimony counts using ete
    def parsimony(tree):
        etetree = tree.to_ete(features=['sequence'])
        print(etetree)
        return(sum(dagutils.hamming_distance(n.up.sequence, n.sequence) for n in etetree.iter_descendants()))

    _testfactory(lambda dag: dag.weight_count(), parsimony, collapse_invariant=True)

def test_copy():
    # Copying the DAG gives the same DAG back, or at least a DAG expressing
    # the same trees
    _testfactory(lambda dag: Counter(tree.to_newick() for tree in dag.copy().get_trees()), lambda tree: tree.to_newick())

def test_newicks():
    # See that the to_newicks method agrees with to_newick applied to all trees in DAG.
    kwargs = {"name_func": lambda n: n.label.sequence,
              "features": []}
    _testfactory(lambda dag: Counter(dag.to_newicks(**kwargs)), lambda tree: tree.to_newick(**kwargs))
    kwargs = {"name_func": lambda n: n.label.sequence,
              "features": ["sequence"]}
    _testfactory(lambda dag: Counter(dag.to_newicks(**kwargs)), lambda tree: tree.to_newick(**kwargs))
    kwargs = {"name_func": lambda n: '1',
              "features": ["sequence"]}
    _testfactory(lambda dag: Counter(dag.to_newicks(**kwargs)), lambda tree: tree.to_newick(**kwargs))
    kwargs = {"name_func": lambda n: '1',
              "features": None}
    _testfactory(lambda dag: Counter(dag.to_newicks(**kwargs)), lambda tree: tree.to_newick(**kwargs))
    kwargs = {"name_func": lambda n: '1',
              "features": []}
    _testfactory(lambda dag: Counter(dag.to_newicks(**kwargs)), lambda tree: tree.to_newick(**kwargs))

def test_verify_newicks():
    # See that the newick string output is the same as given by ete3
    kwargs = {"name_func": lambda n: n.label.sequence,
              "features": ['sequence']}
    def verify(tree):
        etetree = tree.to_ete(**kwargs)
        return(etetree.write(format=8, features=['sequence'], format_root_node=True))
    _testfactory(lambda dag: Counter(dag.to_newicks(**kwargs)), verify)


def test_collapsed_counts():
    def uncollapsed(tree):
        etetree = tree.to_ete(features=['sequence'])
        return(sum(n.up.sequence == n.sequence for n in etetree.iter_descendants()))

    _testfactory(lambda dag: dag.weight_count(edge_weight_func=lambda n1, n2: n1.label.sequence == n2.label.sequence), uncollapsed)


def test_min_weight():
    def parsimony(tree):
        tree.recompute_parents()
        return sum(dagutils.hamming_distance(list(node.parents)[0].label.sequence, node.label.sequence)
                   for node in hdag.postorder(tree)
                   if node.parents)
    _testfactory(lambda dag: dag.optimal_weight_annotate(), parsimony, accum_func=min, collapse_invariant=True)

def test_cm_counter():
    pass
