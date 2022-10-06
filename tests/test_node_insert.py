import historydag as hdag
import pickle

with open("sample_data/toy_trees.p", "rb") as f:
    trees = pickle.load(f)

dag = hdag.history_dag_from_etes(trees, ["sequence"])
dag.recompute_parents()

seq_len = len(next(dag.postorder()).label.sequence)
new_seq = "A" * seq_len


def test_inserting_leafnode_at_nearest_leaf():
    dag_with_leaf_added = dag.copy()
    dag_with_leaf_added.add_node_at_nearest_leaf("A" * seq_len)
    dag_with_leaf_added.recompute_parents()
    assert dag_with_leaf_added._check_valid()

    # check we successfully added one new node
    assert (
        len(list(dag_with_leaf_added.postorder())) == len(list(dag.postorder())) + 1
    ), "failed to add one node successfully"


def test_inserting_leafnode_everywhere():
    dag_with_leaf_added_everywhere = dag.copy()
    dag_with_leaf_added_everywhere.add_node_at_all_possible_places("A" * seq_len)
    dag_with_leaf_added_everywhere.recompute_parents()
    assert dag_with_leaf_added_everywhere._check_valid()

    num_leaf = len([x for x in dag.postorder() if x.is_leaf()])
    num_treeroots = len([x for x in dag.dagroot.children()])
    # expected number of nodes added:
    #   1 for dagroot,
    #   1 for the newly created node,
    #   1 for each leaf,
    #   2+number non-single-leaf clades for each internal nonroot node
    #   1+num_clades for each treeroot node
    expected_num_with_all_additions = (
        2
        + num_leaf
        + sum(
            [
                2
                + len(n.child_clades())
                - len([c for c in n.child_clades() if len(c) < 2])
                for n in dag.postorder()
                if not (n.is_leaf() or n.is_ua_node())
            ]
        )
        - num_treeroots
    )
    assert (
        len(list(dag_with_leaf_added_everywhere.postorder()))
        == expected_num_with_all_additions
    ), "failed to add expected set of new nodes"


def test_inserting_leafnode_at_closest_spot():
    dag_with_leaf_added_closer = dag.copy()
    dag_with_leaf_added_closer.insert_node("A" * seq_len)
    dag_with_leaf_added_closer.recompute_parents()
    assert dag_with_leaf_added_closer._check_valid()

    # check we successfully added one new node
    assert (
        len(list(dag_with_leaf_added_closer.postorder()))
        == len(list(dag.postorder())) + 1
    ), "failed to add one node successfully"
