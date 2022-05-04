from historydag.dag import history_dag_from_newicks

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

history_dag = [
    history_dag_from_newicks(
        newicklist, [], label_functions={"sequence": lambda n: n.name}
    )
    for newicklist in newicklistlist
][1]


# this tests is each of the trees indexed are valide subtrees
# they should have exactly one edge descending from each node clade pair
def test_valid_subtrees():
    # get the set of all dags that were indexed
    # all_dags_indexed = {None}  # set of all the indexed dags
    # curr_dag_index = 0
    # while not history_dag[curr_dag_index] is None:
    for curr_dag_index in range(0, len(history_dag)):
        next_tree = history_dag[curr_dag_index]
        print("in the while loop")
        print(next_tree)
        print(next_tree.to_newick())
        assert next_tree.to_newick() in history_dag.to_newicks()
        assert next_tree.is_clade_tree()
        # curr_dag_index = curr_dag_index + 1


# this should check if the indexing algorithm accurately
# captures all possible subtrees of the dag
# (this is the dag defined by newicklistlist)
def test_indexing_comprehensive():
    # get the set of all dags that were indexed
    all_dags_indexed = {None}  # set of all the indexed dags

    # get all the possible dags using indexingd
    for curr_dag_index in range(0, len(history_dag)):
        next_tree = history_dag[curr_dag_index]
        all_dags_indexed.add(next_tree.to_newick())
        print(curr_dag_index)
    print("len: " + str(len(history_dag)))
    print("number of indexed trees: " + str(len(all_dags_indexed)))

    # get the set of all actual dags from the get_trees
    all_dags_true_generator = history_dag.to_newicks()
    all_dags_true = {None}

    for tree in all_dags_true_generator:
        all_dags_true.add(tree)
    print("actual number of subtrees: " + str(len(all_dags_true)))
    assert all_dags_true == all_dags_indexed

    # verify the lengths match
    assert(len(history_dag) == len(all_dags_indexed) - 1)  # subtracting 1 because of the None added to the sets
    assert(len(all_dags_indexed) == len(all_dags_true))

    # test the for each loop
    for_each_indexed = {None}  # set of all the indexed dags
    for next_tree in history_dag:
        for_each_indexed.add(next_tree.to_newick())
    assert for_each_indexed == all_dags_true
