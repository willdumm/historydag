import ete3
import pickle
import historydag
import historydag.dag as hdag
import historydag.utils as dagutils
from collections import Counter, namedtuple
import pytest
import random
from historydag import parsimony_utils
from math import exp
from scipy.special import logsumexp


def normalize_counts(counter):
    n = len(list(counter.elements()))
    return ([num / n for _, num in counter.items()], (n / len(counter)) / n)


def is_close(f1, f2, tol=0.03):
    return abs(f1 - f2) < tol


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
    [
        "((CA, GG)CA, AT, (TT, (CC, GA)CC)CC)AA;",
        "((CA, GG)CA, AA, (TT, (CC, GA)CA)CA)AA;",
        "((CA, GG)CG, AA, (TT, (CC, GA)GC)GC)AG;",
    ],
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
    dag.make_complete()
dags.extend(compdags)

cdags = [dag.copy() for dag in dags]
for dag in cdags:
    dag.convert_to_collapsed()


def _testfactory(resultfunc, verify_func, collapse_invariant=False, accum_func=Counter):
    for dag, cdag in zip(dags, cdags):
        # check dags
        result = resultfunc(dag)
        verify_result = accum_func([verify_func(tree) for tree in dag.get_histories()])
        assert result == verify_result

        # check cdags
        cresult = resultfunc(cdag)
        cverify_result = accum_func(
            [verify_func(tree) for tree in cdag.get_histories()]
        )
        assert cresult == cverify_result

        # check they agree, if collapse_invariant.
        if collapse_invariant:
            assert result == cresult


def test_valid_dags():
    for dag in dags + cdags:
        dag._check_valid()
        dag.copy()._check_valid()
        # each edge is allowed:
        for node in dag.postorder():
            for clade in node.clades:
                for target in node.clades[clade].targets:
                    assert target.clade_union() == clade or node.is_ua_node()

        # each clade has a descendant edge:
        for node in dag.postorder():
            for clade in node.clades:
                assert len(node.clades[clade].targets) > 0

        # leaf labels are unique:
        leaf_labels = [node.label for node in dag.postorder() if len(node.clades) == 0]
        assert len(set(leaf_labels)) == len(leaf_labels)


def test_count_topologies():
    dagutils.make_newickcountfuncs(internal_labels=False)
    for dag in dags:
        checkset = {
            tree.to_newick(
                name_func=lambda n: n.label.sequence if n.is_leaf() else "",
                features=[],
                feature_funcs={},
            )
            for tree in dag.get_histories()
        }
        print(checkset)
        assert dag.count_topologies_fast() == len(checkset)


def test_count_topologies_equals_newicks():
    for dag in dags:
        assert dag.count_topologies_fast() == dag.count_topologies()


def test_parsimony():
    # test parsimony counts without ete
    def parsimony(tree):
        tree.recompute_parents()
        return sum(
            parsimony_utils.hamming_edge_weight(list(node.parents)[0], node)
            for node in tree.postorder()
            if node.parents
        )

    _testfactory(lambda dag: dag.weight_count(), parsimony)


def test_parsimony_counts():
    # test parsimony counts using ete
    def parsimony(tree):
        etetree = tree.to_ete(features=["sequence"])
        return sum(
            parsimony_utils.default_nt_transitions.weighted_hamming_distance(
                n.up.sequence, n.sequence
            )
            for n in etetree.iter_descendants()
        )

    _testfactory(lambda dag: dag.weight_count(), parsimony)


def test_copy():
    # Copying the DAG gives the same DAG back, or at least a DAG expressing
    # the same trees
    _testfactory(
        lambda dag: Counter(tree.to_newick() for tree in dag.copy().get_histories()),
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
            parsimony_utils.hamming_edge_weight(list(node.parents)[0], node)
            for node in tree.postorder()
            if node.parents
        )

    _testfactory(
        lambda dag: dag.optimal_weight_annotate(),
        parsimony,
        accum_func=min,
        collapse_invariant=True,
    )


def test_count_histories():
    _testfactory(lambda dag: dag.count_histories(), lambda tree: 1, accum_func=sum)


def test_count_histories_expanded():
    for dag in dags + cdags:
        ndag = dag.copy()
        ndag.explode_nodes()
        assert (
            dag.count_histories(
                expand_func=parsimony_utils.default_nt_transitions.ambiguity_map.get_sequence_resolution_func(
                    "sequence"
                )
            )
            == ndag.count_histories()
        )


def test_count_weights_expanded():
    for dag in dags + cdags:
        ndag = historydag.sequence_dag.SequenceHistoryDag.from_history_dag(dag.copy())
        odag = ndag.copy()
        ndag.explode_nodes()
        assert odag.hamming_parsimony_count() == ndag.weight_counts_with_ambiguities()


def test_topology_decompose():
    # make sure that trimming to a topology results in a DAG expressing exactly
    # the trees which have that topology.
    for collapse_leaves in [False, True]:
        kwargs = dagutils.make_newickcountfuncs(
            internal_labels=False, collapse_leaves=collapse_leaves
        )
        for dag in [dag.copy() for dag in dags]:
            nl = dag.weight_count(**kwargs)
            for idx, (topology, count) in enumerate(nl.items()):
                # print(topology, count, idx)
                trimdag = dag.copy()
                print(trimdag.weight_count(**kwargs))
                print(topology)
                trimdag.trim_topology(topology, collapse_leaves=collapse_leaves)
                assert trimdag.weight_count(**kwargs) == {topology: count}


def test_topology_count_collapse():
    dag = dags[0].copy()
    print(
        dag.weight_count(
            **dagutils.make_newickcountfuncs(
                internal_labels=False, collapse_leaves=True
            )
        )
    )
    assert dag.count_topologies(collapse_leaves=True) == 2


# this tests is each of the trees indexed are valid subtrees
# they should have exactly one edge descending from each node clade pair
def test_valid_subtrees():
    for history_dag in dags + cdags:
        for curr_dag_index in range(0, len(history_dag)):
            next_tree = history_dag[curr_dag_index]
            assert next_tree.to_newick() in history_dag.to_newicks()
            assert next_tree.is_history()


# this should check if the indexing algorithm accurately
# captures all possible subtrees of the dag
def test_indexing_comprehensive():
    with pytest.raises(Exception) as _:
        dags[5][-len(dags[5]) - 5]

    for history_dag in dags + cdags:
        # get all the possible dags using indexing
        all_dags_indexed = {None}
        for curr_dag_index in range(0, len(history_dag)):
            next_tree = history_dag[curr_dag_index]
            all_dags_indexed.add(next_tree.to_newick())
        print("len: " + str(len(history_dag)))
        print("number of indexed trees: " + str(len(all_dags_indexed)))
        all_dags_indexed.remove(None)

        # get the set of all actual dags from the get_histories
        all_dags_true = set(history_dag.to_newicks())

        print("actual number of subtrees: " + str(len(all_dags_true)))
        assert all_dags_true == all_dags_indexed

        # verify the lengths match
        assert len(history_dag) == len(all_dags_indexed)
        assert len(all_dags_indexed) == len(all_dags_true)

        # test the for each loop
        assert set(history_dag.to_newicks()) == set(
            {tree.to_newick() for tree in history_dag}
        )


def test_trim_fixedleaves():
    for dag in dags + cdags:
        reftree = dag.sample()
        kwarglist = [
            (dagutils.make_rfdistance_countfuncs(reftree, rooted=True), min),
            (dagutils.make_rfdistance_countfuncs(reftree, rooted=False), min),
        ]
        for kwargs, opt_func in kwarglist:
            dag = dag.copy()
            dag.make_complete()
            dag._check_valid()
            dag.recompute_parents()
            dag._check_valid()
            all_weights = dag.weight_count(**kwargs)
            optimal_weight = dag.trim_optimal_weight(**kwargs, optimal_func=opt_func)
            assert all_weights[optimal_weight] == dag.count_trees()
            dag._check_valid()
            dag.convert_to_collapsed()
            dag._check_valid()


def test_trim():
    kwarglist = [
        (parsimony_utils.hamming_distance_countfuncs, min),
        (dagutils.node_countfuncs, min),
    ]
    for dag in dags + cdags:
        for kwargs, opt_func in kwarglist:
            dag = dag.copy()
            dag.make_complete()
            dag._check_valid()
            dag.recompute_parents()
            dag._check_valid()
            all_weights = dag.weight_count(**kwargs)
            optimal_count = dag.count_optimal_histories(**kwargs, optimal_func=opt_func)
            optimal_weight = dag.trim_optimal_weight(**kwargs, optimal_func=opt_func)
            assert all_weights[optimal_weight] == dag.count_trees() == optimal_count
            dag._check_valid()
            dag.convert_to_collapsed()
            dag._check_valid()


def test_from_nodes():
    for dag in dags + cdags:
        cdag = dag.copy()
        cdag.make_complete()
        cdag.trim_optimal_weight()
        cdag._check_valid()
        wc = cdag.weight_count()
        ndag = hdag.history_dag_from_nodes(cdag.preorder())
        ndag._check_valid()
        ndag.trim_optimal_weight()
        ndag._check_valid()
        print(ndag.to_graphviz())
        assert wc == ndag.weight_count()


def test_sample_with_node():
    random.seed(1)
    dag = dags[-1]
    dag.make_uniform()
    node_to_count = dag.count_nodes()
    min_count = min(node_to_count.values())
    least_supported_nodes = [
        node for node, val in node_to_count.items() if val == min_count
    ]
    for node in least_supported_nodes:
        tree_samples = [dag.sample_with_node(node) for _ in range(min_count * 5)]
        tree_samples[0]._check_valid()
        tree_newicks = {tree.to_newick() for tree in tree_samples}
        # We sampled all trees possible containing the node
        assert len(tree_newicks) == min_count
        # All trees sampled contained the node
        assert all(node in set(tree.preorder()) for tree in tree_samples)
        # # trees containing the node were sampled uniformly
        # # (This is slow but seems to work)
        # norms, avg = normalize_counts(Counter(tree.to_newick() for tree in tree_samples))
        # print(norms)
        # assert all(is_close(norm, avg) for norm in norms)


def test_sample_with_edge():
    random.seed(1)
    dag = dags[-1]
    dag.recompute_parents()
    node_to_count = dag.count_nodes()
    min_count = min(node_to_count.values())
    least_supported_nodes = [
        node for node, val in node_to_count.items() if val == min_count
    ]
    node = least_supported_nodes[0]

    def edges(dag):
        eset = set()
        for node in dag.preorder():
            for child in node.children():
                eset.add((node, child))
        return eset

    for parent in node.parents:
        edge = (parent, node)
        tree_samples = [dag.sample_with_edge(edge) for _ in range(min_count * 5)]
        tree_samples[0]._check_valid()
        # We sampled all trees possible containing the node
        # All trees sampled contained the node
        assert all(edge in edges(tree) for tree in tree_samples)


def test_iter_covering_histories():
    for dag in dags + cdags:
        codag = dag.copy()
        codag.make_complete()
        trees = list(dag.iter_covering_histories())
        tdag = trees[0] | trees
        tdag.make_complete()
        assert tdag.weight_count() == codag.weight_count()


def test_iter_covering_histories_edges():
    for dag in dags + cdags:
        trees = list(dag.iter_covering_histories(cover_edges=True))
        tdag = trees[0] | trees
        assert tdag.weight_count() == dag.weight_count()
        assert set(tdag.to_newicks()) == set(dag.to_newicks())


def test_relabel():
    dag = dags[-1]
    Label = namedtuple("Label", ["sequence", "newthing"])
    ndag = dag.relabel(lambda n: Label(n.label.sequence, len(list(n.children()))))
    Label = namedtuple("Label", ["sequence"])
    odag = ndag.relabel(lambda n: Label(n.label.sequence))
    odag._check_valid()
    assert dag.weight_count() == odag.weight_count()


def test_add_label_fields():
    dag = dags[-1]
    old_fieldset = dag.get_label_type()._fields + tuple(["isLeaf", "originalLocation"])
    new_field_values = {n: [n.is_leaf(), "new location"] for n in dag.postorder()}
    ndag = dag.add_label_fields(["isLeaf", "originalLocation"], new_field_values)
    ndag._check_valid()
    new_fieldset = ndag.get_label_type()._fields
    assert old_fieldset == new_fieldset


def test_remove_label_fields():
    dag = dags[-1]
    old_fieldset = dag.get_label_type()._fields
    new_field_values = {n: [n.is_leaf()] for n in dag.postorder()}
    ndag = dag.add_label_fields(["added_field"], new_field_values)
    ndag._check_valid()
    # test removing a field that isn't there and one that is
    odag = dag.remove_label_fields(["removed_field", "added_field"])
    new_fieldset = odag.get_label_type()._fields
    odag._check_valid()
    assert old_fieldset == new_fieldset


def rooted_rf_distance(history1, history2):
    cladeset1 = {n.clade_union() for n in history1.preorder(skip_ua_node=True)}
    cladeset2 = {n.clade_union() for n in history2.preorder(skip_ua_node=True)}
    return len(cladeset1 ^ cladeset2)


def test_rf_rooted_distances():
    for dag in dags:
        ref_tree = dag.sample()
        weight_kwargs = dagutils.make_rfdistance_countfuncs(ref_tree, rooted=True)

        weight_to_self = ref_tree.optimal_weight_annotate(**weight_kwargs)
        if not (weight_to_self == 0):
            print("nonzero distance to self in this tree ^^: ", weight_to_self)
            assert False

        def rf_distance(intree):
            return rooted_rf_distance(intree, ref_tree)

        assert Counter(rf_distance(tree) for tree in dag) == dag.weight_count(
            **weight_kwargs
        )


def test_rf_unrooted_distances():
    for dag in reversed(dags):
        ref_tree = dag.sample()
        weight_kwargs = dagutils.make_rfdistance_countfuncs(ref_tree, rooted=False)
        ref_tree_ete = ref_tree.to_ete(features=["sequence"])
        for node in ref_tree_ete.traverse():
            node.name = node.sequence
        ref_taxa = {n.sequence for n in ref_tree_ete.get_leaves()}
        weight_to_self = ref_tree.optimal_weight_annotate(**weight_kwargs)
        if not (weight_to_self == 0):
            print(ref_tree_ete)
            print("nonzero distance to self in this tree ^^: ", weight_to_self)
            assert False

        def rf_distance(intree):
            intreeete = intree.to_ete(features=["sequence"])
            assert len(intreeete.children) != 1
            return ref_tree_ete.robinson_foulds(
                intreeete, attr_t1="sequence", attr_t2="sequence", unrooted_trees=True
            )[0]

        if Counter(rf_distance(tree) for tree in dag) != dag.weight_count(
            **weight_kwargs
        ):
            print("label format ", next(dag.postorder()).label)
            for tree in dag:
                thistree = tree.to_ete(features=["sequence"])
                tree_taxa = {n.sequence for n in thistree.get_leaves()}
                if tree_taxa != ref_taxa:
                    continue
                ref_dist = rf_distance(tree)
                comp_dist = tree.optimal_weight_annotate(**weight_kwargs)
                if ref_dist != comp_dist:
                    for node in thistree.get_leaves():
                        node.name = node.sequence
                        # node.name = str(namedict[node.sequence])
                    for node in ref_tree_ete.get_leaves():
                        node.name = node.sequence
                        # node.name = str(namedict[node.sequence])
                    print("reference tree:")
                    print(ref_tree_ete)
                    print("this tree:")
                    print(thistree)
                    print("correct RF: ", ref_dist)
                    print("computed RF: ", comp_dist)
                    assert False


def test_optimal_sum_rf_distance():
    for dag_idx, ref_dag in enumerate(dags):
        print("dagnum ", dag_idx)
        # let's just do this test for three trees in each dag:
        for tree_idx, tree in zip(range(3), ref_dag):
            print("treenum ", tree_idx)
            # First let's just make sure that when the ref_dag is just a single
            # tree, optimal_sum_rf_distance agrees with normal rf_distance.
            single_tree_dag = ref_dag[0]
            # Here we get all the distances between trees in 'single_tree_dag' and the
            # reference tree 'tree' (there's only one, since 'single_tree_dag'
            # only contains one tree:
            expected = single_tree_dag.count_rf_distances(tree, rooted=True)
            expected_sum = sum(expected.elements())
            calculated_sum = tree.optimal_sum_rf_distance(single_tree_dag)
            assert calculated_sum == expected_sum

            # Now let's try computing the summed rf distance on tree relative
            # to ref_dag...

            # Here we get all the distances between trees in 'dag' and the
            # reference tree 'tree':
            expected = dag.count_rf_distances(tree, rooted=True)
            # Here we sum all elements in the counter, with multiplicity:
            # in other words we sum all distances from trees in 'dag' to 'tree'
            expected_sum = sum(expected.elements())
            # This should calculate the sum RF distance from 'tree' to all
            # trees in 'dag':
            calculated_sum = tree.optimal_sum_rf_distance(dag)
            assert calculated_sum == expected_sum


def test_trim_range():
    for curr_dag in dags + cdags:
        history_dag = curr_dag.copy()
        print("history dag contains ", history_dag.count_trees(), " trees")
        counter = history_dag.weight_count()
        max_weight_passed = list(counter.keys())[int(len(counter.keys()) / 2)]
        min_weight_passed = list(counter.keys())[int(len(counter.keys()) / 2)] - 1

        print("weight count before trimming:")
        print(counter)
        print("max weight passed in:")
        print(max_weight_passed)
        trees_to_merge = [
            tree.copy()
            for tree in history_dag
            if (
                tree.optimal_weight_annotate() <= max_weight_passed
                and tree.optimal_weight_annotate() >= min_weight_passed
            )
        ]
        true_subdag = hdag.history_dag_from_clade_trees(trees_to_merge)

        history_dag.trim_within_range(
            min_weight=min_weight_passed, max_weight=max_weight_passed
        )
        print("weight count after trimming:")
        print(history_dag.weight_count())
        print("weight count expected:")
        print(true_subdag.weight_count())

        difference = set(history_dag.to_newicks()) - set(true_subdag.to_newicks())
        print(
            "number of subtrees included in tree after trimming but not after merging: "
            + str(len(difference))
        )
        difference2 = set(true_subdag.to_newicks()) - set(history_dag.to_newicks())
        print(
            "number of subtrees included in tree after merging but not trimming: "
            + str(len(difference2))
        )
        assert set(true_subdag.to_newicks()) == set(history_dag.to_newicks())


def test_trim_above():
    for curr_dag in dags + cdags:
        history_dag = curr_dag.copy()
        # print(history_dag.to_newicks())
        print("history dag contains ", history_dag.count_trees(), " trees")
        counter = history_dag.weight_count()
        min_weight_passed = list(counter.keys())[int(len(counter.keys()) / 2)]

        print("weight count before trimming:")
        print(counter)
        print("min weight passed in:")
        print(min_weight_passed)
        trees_to_merge = [
            tree.copy()
            for tree in history_dag
            if tree.optimal_weight_annotate() >= min_weight_passed
        ]
        true_subdag = hdag.history_dag_from_clade_trees(trees_to_merge)

        history_dag.trim_within_range(min_weight=min_weight_passed)
        print("weight count after trimming:")
        print(history_dag.weight_count())
        print("weight count expected:")
        print(true_subdag.weight_count())

        difference = set(history_dag.to_newicks()) - set(true_subdag.to_newicks())
        print(
            "number of subtrees included in tree after trimming but not after merging: "
            + str(len(difference))
        )
        difference2 = set(true_subdag.to_newicks()) - set(history_dag.to_newicks())
        print(
            "number of subtrees included in tree after merging but not trimming: "
            + str(len(difference2))
        )
        assert set(true_subdag.to_newicks()) == set(history_dag.to_newicks())


def test_trim_weight():
    for curr_dag in dags + cdags:
        history_dag = curr_dag.copy()
        print("history dag contains ", history_dag.count_trees(), " trees")
        counter = history_dag.weight_count()
        max_weight_passed = list(counter.keys())[int(len(counter.keys()) / 2)]

        print("weight count before trimming:")
        print(counter)
        print("max weight passed in:")
        print(max_weight_passed)
        trees_to_merge = [
            tree.copy()
            for tree in history_dag
            if tree.optimal_weight_annotate() <= max_weight_passed
        ]
        true_subdag = hdag.history_dag_from_clade_trees(trees_to_merge)

        history_dag.trim_within_range(max_weight=max_weight_passed)
        print("weight count after trimming:")
        print(history_dag.weight_count())
        print("weight count expected:")
        print(true_subdag.weight_count())

        difference = set(history_dag.to_newicks()) - set(true_subdag.to_newicks())
        print(
            "number of subtrees included in tree after trimming but not after merging: "
            + str(len(difference))
        )
        difference2 = set(true_subdag.to_newicks()) - set(history_dag.to_newicks())
        print(
            "number of subtrees included in tree after merging but not trimming: "
            + str(len(difference2))
        )
        assert set(true_subdag.to_newicks()) == set(history_dag.to_newicks())


def test_trim_filter():
    dag = dags[-1].copy()
    pars_d = parsimony_utils.hamming_distance_countfuncs
    node_d = hdag.utils.node_countfuncs

    # >>
    cdag = dag.copy()
    pars_f = hdag.utils.HistoryDagFilter(pars_d, min)
    node_f = hdag.utils.HistoryDagFilter(node_d, max)
    p_n_f = pars_f + node_f

    assert (cdag[pars_f][node_f]).weight_count(**p_n_f) == (cdag[p_n_f]).weight_count(
        **p_n_f
    )
    print("sequential trimming is the same as trimming with sum of filters")
    # >>
    rf_d = hdag.utils.make_rfdistance_countfuncs(dag[25])
    rf_f = hdag.utils.HistoryDagFilter(rf_d, min)

    pnf_f = p_n_f + rf_f
    assert (cdag[p_n_f][rf_f]).weight_count(**pnf_f) == (cdag[pnf_f]).weight_count(
        **pnf_f
    )
    print("sequential addition of Filters works")
    # >>
    lc_d = pnf_f.weight_funcs.linear_combination([1, 2, 0])
    lc_f = hdag.utils.HistoryDagFilter(lc_d, min)

    min_weight = cdag.optimal_weight_annotate(**lc_f)
    weights = cdag.weight_count(**pnf_f)
    min_weights = Counter(
        w
        for w in weights.elements()
        if sum(c * ww for c, ww in zip([1, 2, 0], w)) == min_weight
    )
    assert (cdag[lc_f]).weight_count(**pnf_f) == min_weights
    print("linear combination trimming works")

    # >>
    print(node_d)
    print(node_f)
    print(p_n_f)
    print(pnf_f)
    print(lc_f)


def test_weight_range_annotate():
    kwargs = parsimony_utils.hamming_distance_countfuncs
    for dag in dags:
        assert dag.weight_range_annotate(**kwargs) == (
            dag.optimal_weight_annotate(**kwargs, optimal_func=min),
            dag.optimal_weight_annotate(**kwargs, optimal_func=max),
        )


def test_sum_all_pair_rf_distance():
    dag = dags[-1]

    # check 0 on single-tree dag vs itself:
    assert dag[0].sum_rf_distances() == 0
    assert dag[0].sum_rf_distances(reference_dag=dag[0]) == 0

    # check matches single rf distance between two single-tree dags:
    udag = dag.unlabel()
    assert udag[0].sum_rf_distances(reference_dag=udag[-1]) == udag[
        0
    ].optimal_rf_distance(udag[-1])

    # check matches truth on whole DAG vs self:
    assert dag.sum_rf_distances() == sum(dag.count_sum_rf_distances(dag).elements())


def test_intersection():
    dag = dags[-1]
    dag1 = hdag.history_dag_from_histories(dag.sample() for _ in range(8)) | dag[0]
    dag2 = hdag.history_dag_from_histories(dag.sample() for _ in range(8)) | dag[0]
    idag = dag1 & dag2

    assert len(idag | dag1) == len(dag1)
    assert len(idag | dag2) == len(dag2)


def test_node_support():
    dag = dags[-1].copy()
    # first compare to uniform:
    dag.probability_annotate(lambda n1, n2: 1, log_probabilities=False)
    nd = dag.node_probabilities(log_probabilities=False)
    od = dag.count_nodes()
    hists = dag.count_histories()
    od = {n: count / hists for n, count in od.items()}
    for node in dag.postorder():
        assert is_close(nd[node], od[node], tol=0.0001)

    # now try log version for uniform:
    dag.probability_annotate(lambda n1, n2: 0, log_probabilities=True)
    nd = dag.node_probabilities(log_probabilities=True)
    print(nd[dag.dagroot])
    for node in dag.postorder():
        assert is_close(exp(nd[node]), od[node], tol=0.0001)

    # hamming parsimony weighted support:
    def edge_weight_func(n1, n2):
        if n1.is_ua_node():
            return 1
        else:
            return exp(-3 * parsimony_utils.hamming_edge_weight(n1, n2))

    kwargs = {
        "accum_func": dagutils.prod,
        "start_func": lambda n: 1,
        "edge_weight_func": edge_weight_func,
        "optimal_func": sum,
    }

    dag.probability_annotate(edge_weight_func, log_probabilities=False)
    conditional_edge_probs = dag.export_edge_probabilities()

    normalization_constant = dag.optimal_weight_annotate(**kwargs)
    check_kwargs = kwargs.copy()
    check_kwargs["edge_weight_func"] = lambda n1, n2: conditional_edge_probs[(n1, n2)]
    for tree in dag:
        assert is_close(
            tree.optimal_weight_annotate(**kwargs) / normalization_constant,
            tree.optimal_weight_annotate(**check_kwargs),
            tol=0.00001,
        )
    cdag = dag.copy()
    cdag.trim_optimal_weight()
    print(cdag.optimal_weight_annotate(**kwargs) / normalization_constant)

    # now hamming parsimony log version:
    def edge_weight_func(n1, n2):
        if n1.is_ua_node():
            return 0
        else:
            return -3 * parsimony_utils.hamming_edge_weight(n1, n2)

    kwargs = {
        "accum_func": sum,
        "start_func": lambda n: 0,
        "edge_weight_func": edge_weight_func,
        "optimal_func": logsumexp,
    }

    dag.probability_annotate(edge_weight_func, log_probabilities=True)
    conditional_edge_probs = dag.export_edge_probabilities()

    normalization_constant = dag.optimal_weight_annotate(**kwargs)
    check_kwargs = kwargs.copy()
    check_kwargs["edge_weight_func"] = lambda n1, n2: conditional_edge_probs[(n1, n2)]
    for tree in dag:
        assert is_close(
            tree.optimal_weight_annotate(**kwargs) - normalization_constant,
            tree.optimal_weight_annotate(**check_kwargs),
            tol=0.00001,
        )
