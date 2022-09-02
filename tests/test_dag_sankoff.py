import pickle
import numpy as np
import historydag as hdag
import historydag.parsimony as dag_parsimony


def compare_dag_and_tree_parsimonies(
    dag, transition_weights=None, filter_min_score=True
):
    dag.recompute_parents()
    dag.convert_to_collapsed()

    # extract sample tree
    s = dag.sample()
    # convert to ete3.Tree format
    s_ete = s.to_ete()

    # compute cost vectors for sample tree in dag and in ete3.Tree format to compare
    a = dag_parsimony.sankoff_upward(
        s, transition_weights=transition_weights, filter_min_score=filter_min_score
    )
    b = dag_parsimony.sankoff_upward(s_ete, transition_weights=transition_weights)
    assert (
        a == b
    ), "Upward Sankoff on ete_Tree vs on the dag version of the tree produced different results"

    # calculate sequences for internal nodes using Sankoff for both formats of sample tree
    s_weight = dag_parsimony.sankoff_downward(
        s,
        compute_cvs=False,
        transition_weights=transition_weights,
        filter_min_score=filter_min_score,
    )
    s_ete = dag_parsimony.disambiguate(
        s_ete, compute_cvs=False, transition_weights=transition_weights
    )
    # convert ete3.Tree back to a HistoryDag object so as to compare, but keeping the data calculated using ete3 structure
    s_ete_as_dag = hdag.history_dag_from_etes([s_ete], label_features=["sequence"])

    # parsimony score depends on the choice of `transition_weights` arg
    if transition_weights is not None:
        s_ete_weight = s_ete_as_dag.optimal_weight_annotate(
            edge_weight_func=dag_parsimony.edge_weight_func_from_weight_matrix(
                s_ete_as_dag, transition_weights, dag_parsimony.bases
            )
        )
    else:
        s_ete_weight = s_ete_as_dag.optimal_weight_annotate()
    assert (
        s_weight == s_ete_weight
    ), "Downward sankoff on ete_Tree vs on the dag version of the tree produced different results"

    s_labels = set(n.label.sequence for n in s.postorder() if not n.is_root())
    s_ete_labels = set(
        n.label.sequence for n in s_ete_as_dag.postorder() if not n.is_root()
    )
    assert (
        len(s_ete_labels - s_labels) < 1
    ), "DAG Sankoff missed a label that occurs in the tree Sankoff."


def check_sankoff_on_dag(
    dag, expected_score, transition_weights=None, filter_min_score=True
):
    # perform upward sweep of sankoff to calculate overall parsimony score and assign cost vectors to internal nodes
    upward_pass_min_cost = dag_parsimony.sankoff_upward(
        dag, transition_weights=transition_weights, filter_min_score=filter_min_score
    )
    assert np.isclose(
        [upward_pass_min_cost], [expected_score]
    ), "Upward pass of Sankoff on dag did not yield expected score"

    # perform downward sweep of sankoff to calculate all possible internal node sequences.
    downward_pass_min_cost = dag_parsimony.sankoff_downward(
        dag,
        transition_weights=transition_weights,
        filter_min_score=filter_min_score,
        compute_cvs=False,
    )
    assert np.isclose(
        [downward_pass_min_cost], [expected_score]
    ), "Downward pass of Sankoff on dag did not yield expected score"

    assert (
        dag.count_trees() == dag.copy().count_trees()
    ), "Resulting DAG had invalid internal node assignments"


with open("sample_data/toy_trees.p", "rb") as f:
    ete_trees = pickle.load(f)
dg = hdag.history_dag_from_etes(ete_trees, ["sequence"])

tw_options = [
    (75, None),
    (
        93,
        np.array(
            [
                [0, 1, 2.5, 1, 1],
                [1, 0, 1, 2.5, 1],
                [2.5, 1, 0, 1, 1],
                [1, 2.5, 1, 0, 1],
                [1, 1, 1, 1, 0],
            ]
        ),
    ),
    (
        106,
        np.array(
            [
                [0, 1, 5, 1, 1],
                [1, 0, 1, 5, 1],
                [5, 1, 0, 1, 1],
                [1, 5, 1, 0, 1],
                [1, 1, 1, 1, 0],
            ]
        ),
    ),
]

for (w, tw) in tw_options:
    check_sankoff_on_dag(dg.copy(), w, transition_weights=tw, filter_min_score=False)
    check_sankoff_on_dag(dg.copy(), w, transition_weights=tw, filter_min_score=True)
    compare_dag_and_tree_parsimonies(
        dg.copy(), transition_weights=tw, filter_min_score=False
    )
    compare_dag_and_tree_parsimonies(
        dg.copy(), transition_weights=tw, filter_min_score=True
    )
