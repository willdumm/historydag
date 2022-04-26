from historydag.dag import (
    history_dag_from_newicks,
)

newicklistlist = [
    [
        "((AA, CT)CG, (TA, CC)CG)CC;",
        "((AA, CT)CA, (TA, CC)CC)CC;",
    ],
    [
        "((CA, GG)CA, AA, (TT, (CC, GA)CC)CC)AA;",
        "((CA, GG)CA, AA, (TT, (CC, GA)CA)CA)AA;",
        "((CA, GG)CG, AA, (TT, (CC, GA)GC)GC)AG;",
    ],
    [
        "((AA, CT)CG, (TA, CC)CG)CC;",
        "((AA, CT)CA, (TA, CC)CC)CC;",
    ],
]


# this should check if the indexing algorithm accurately
# captures all possible subtrees of the dag
# (this is the dag defined by newicklistlist)
def test_indexing_comprehensive():
    dags = [
        history_dag_from_newicks(
            newicklist, [], label_functions={"sequence": lambda n: n.name}
        )
        for newicklist in newicklistlist
    ]
    history_dag = history_dag[1]
    # get the set of all dags that were indexed
    all_dags_indexed = {None} # set of all the indexed dags
    curr_dag_index = 0
    while not history_dag[curr_dag_index] == None:
        next_tree = history_dag[curr_dag_index]
        curr_dag_index = curr_dag_index + 1
        all_dags_indexed.add(next_tree)
    
    # get the set of all dags from the get_trees
    all_dags_true_generator = history_dag.get_trees()
    all_dags_true = {None}

    for tree in all_dags_true_generator:
        all_dags_true.add(tree)

    assert all_dags_true == all_dags_indexed