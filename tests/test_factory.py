import ete3
import pickle
import historydag.dag as hdag
import historydag.utils as dagutils
from collections import Counter


def deterministic_newick(tree: ete3.TreeNode) -> str:
    """For use in comparing ete3 TreeNodes with newick strings"""
    newtree = tree.copy()
    for node in newtree.traverse():
        node.name = 1
        node.children.sort(key=lambda node: node.sequence)
        node.dist = 1
    return newtree.write(format=1, features=["sequence"], format_root_node=True)


def deterministic_newick_topology(tree: ete3.TreeNode) -> str:
    """For use in comparing ete3 TreeNodes with newick strings, distinguishing only
    by topologies above leaves."""
    newtree = tree.copy()
    for node in newtree.traverse():
        node.name = node.sequence
        node.children.sort(
            key=lambda node: str(sorted(lf.name for lf in node.get_leaves()))
        )
        node.dist = 1
    return newtree.write(format=9)


newicklistlist = [
    ["((AA, CT)CG, (TA, CC)CG)CC;", "((AA, CT)CA, (TA, CC)CC)CC;"],
    [
        "((CA, GG)CA, AA, (TT, (CC, GA)CC)CC)AA;",
        "((CA, GG)CA, AA, (TT, (CC, GA)CA)CA)AA;",
        "((CA, GG)CG, AA, (TT, (CC, GA)GC)GC)AG;",
    ],
    ["((AA, CT)CG, (TA, CC)CG)CC;", "((AA, CT)CA, (TA, CC)CC)CC;"],
]

dags = [
    hdag.history_dag_from_newicks(
        newicklist, [], label_functions={"sequence": lambda n: n.name}
    )
    for newicklist in newicklistlist
]

with open("sample_data/toy_trees_100_uncollapsed.p", "rb") as fh:
    uncollapsed = pickle.load(fh)
for tree in uncollapsed:
    if len(tree.children) == 1:
        newchild = tree.copy()
        for child in newchild.get_children():
            newchild.remove_child(child)
        tree.add_child(newchild)
        assert newchild.is_leaf()

dags.append(
    hdag.history_dag_from_etes(
        uncollapsed[0:5], [], label_functions={"sequence": lambda n: n.sequence}
    )
)

compdags = [dag.copy() for dag in dags]
for dag in compdags:
    dag.add_all_allowed_edges()
dags.extend(compdags)

cdags = [dag.copy() for dag in dags]
for dag in cdags:
    dag.convert_to_collapsed()


def _testfactory(resultfunc, verify_func, collapse_invariant=False, accum_func=Counter):
    for dag, cdag in zip(dags, cdags):
        # check dags
        result = resultfunc(dag)
        verify_result = accum_func([verify_func(tree) for tree in dag.get_trees()])
        assert result == verify_result

        # check cdags
        cresult = resultfunc(cdag)
        cverify_result = accum_func([verify_func(tree) for tree in cdag.get_trees()])
        assert cresult == cverify_result

        # check they agree, if collapse_invariant.
        if collapse_invariant:
            assert result == cresult


def test_valid_dags():
    for dag in dags + cdags:
        # each edge is allowed:
        for node in dag.postorder():
            for clade in node.clades:
                for target in node.clades[clade].targets:
                    assert target.under_clade() == clade

        # each clade has a descendant edge:
        for node in dag.postorder():
            for clade in node.clades:
                assert len(node.clades[clade].targets) > 0

        # leaf labels are unique:
        leaf_labels = [node.label for node in dag.postorder() if len(node.clades) == 0]
        assert len(set(leaf_labels)) == len(leaf_labels)


def test_count_topologies():
    kwargs = dagutils.make_newickcountfuncs(internal_labels=False)
    for dag in dags:
        checkset = {
            tree.to_newick(
                name_func= lambda n: n.label.sequence if n.is_leaf() else '',
                features =  [],
                feature_funcs = {},
                )
            for tree in dag.get_trees()
        }
        print(checkset)
        assert dag.count_topologies() == len(checkset)


def test_parsimony():
    # test parsimony counts without ete
    def parsimony(tree):
        tree.recompute_parents()
        return sum(
            dagutils.wrapped_hamming_distance(list(node.parents)[0].label, node.label)
            for node in tree.postorder()
            if node.parents
        )

    _testfactory(lambda dag: dag.weight_count(), parsimony)


def test_parsimony_counts():
    # test parsimony counts using ete
    def parsimony(tree):
        etetree = tree.to_ete(features=["sequence"])
        return sum(
            dagutils.hamming_distance(n.up.sequence, n.sequence)
            for n in etetree.iter_descendants()
        )

    _testfactory(lambda dag: dag.weight_count(), parsimony)


def test_copy():
    # Copying the DAG gives the same DAG back, or at least a DAG expressing
    # the same trees
    _testfactory(
        lambda dag: Counter(tree.to_newick() for tree in dag.copy().get_trees()),
        lambda tree: tree.to_newick(),
    )


def test_newicks():
    # See that the to_newicks method agrees with to_newick applied to all trees in DAG.
    kwargs = {"name_func": lambda n: n.label.sequence, "features": []}
    _testfactory(
        lambda dag: Counter(dag.to_newicks(**kwargs)),
        lambda tree: tree.to_newick(**kwargs),
    )
    kwargs = {"name_func": lambda n: n.label.sequence, "features": ["sequence"]}
    _testfactory(
        lambda dag: Counter(dag.to_newicks(**kwargs)),
        lambda tree: tree.to_newick(**kwargs),
    )
    kwargs = {"name_func": lambda n: "1", "features": ["sequence"]}
    _testfactory(
        lambda dag: Counter(dag.to_newicks(**kwargs)),
        lambda tree: tree.to_newick(**kwargs),
    )
    kwargs = {"name_func": lambda n: "1", "features": None}
    _testfactory(
        lambda dag: Counter(dag.to_newicks(**kwargs)),
        lambda tree: tree.to_newick(**kwargs),
    )
    kwargs = {"name_func": lambda n: "1", "features": []}
    _testfactory(
        lambda dag: Counter(dag.to_newicks(**kwargs)),
        lambda tree: tree.to_newick(**kwargs),
    )


def test_verify_newicks():
    # See that the newick string output is the same as given by ete3
    kwargs = {"name_func": lambda n: n.label.sequence, "features": ["sequence"]}
    invkwargs = {"label_features": ["sequence"], "label_functions": {}}

    def verify(tree):
        etetree = tree.to_ete(**kwargs)
        cladetree = hdag.from_tree(etetree, **invkwargs)
        return cladetree.to_newick(**kwargs)

    _testfactory(lambda dag: Counter(dag.to_newicks(**kwargs)), verify)


def test_collapsed_counts():
    def uncollapsed(tree):
        # Returns the number of uncollapsed edges in the tree
        etetree = tree.to_ete(features=["sequence"])
        return sum(n.up.sequence == n.sequence for n in etetree.iter_descendants())

    _testfactory(
        lambda dag: dag.weight_count(
            edge_weight_func=dagutils.access_nodefield_default("sequence", False)(
                lambda s1, s2: s1 == s2
            )
        ),
        uncollapsed,
    )


def test_min_weight():
    def parsimony(tree):
        tree.recompute_parents()
        return sum(
            dagutils.wrapped_hamming_distance(list(node.parents)[0].label, node.label)
            for node in tree.postorder()
            if node.parents
        )

    _testfactory(
        lambda dag: dag.optimal_weight_annotate(),
        parsimony,
        accum_func=min,
        collapse_invariant=True,
    )


def test_count_trees():
    _testfactory(lambda dag: dag.count_trees(), lambda tree: 1, accum_func=sum)


def test_count_trees_expanded():
    for dag in dags + cdags:
        ndag = dag.copy()
        ndag.explode_nodes()
        assert (
            dag.count_trees(expand_func=dagutils.sequence_resolutions)
            == ndag.count_trees()
        )


def test_count_weights_expanded():
    for dag in dags + cdags:
        ndag = dag.copy()
        ndag.explode_nodes()
        assert dag.hamming_parsimony_count() == ndag.weight_counts_with_ambiguities()


def test_cm_counter():
    pass
