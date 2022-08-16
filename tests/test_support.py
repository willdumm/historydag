import ete3
import pickle
import historydag.dag as hdag


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
    ["((AA, CT)CG, (TA, CC)CG)CC;"],
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


def test_node_counts():
    print(f"Testing with {len(dags)} dags")
    for dag in dags:
        node2count = dag.count_nodes()
        for node in node2count.keys():
            ground_truth = sum(
                [node in set(tree.postorder()) for tree in dag.get_trees()]
            )
            # print(
            #     f"\t node2count[node] = {node2count[node]} \t ground_truth = {ground_truth}"
            # )
            assert node2count[node] == ground_truth


def test_collapsed_node_counts():
    print(f"Testing with {len(dags)} dags")
    for dag in dags:
        node2count = dag.count_nodes(collapse=True)
        print("new dag")
        # print(dag.to_graphviz())

        for node in node2count.keys():
            ground_truth = sum(
                [
                    node in set([n.under_clade() for n in tree.postorder()])
                    for tree in dag.get_trees()
                ]
            )
            print(
                f"node2count[node] = {node2count[node]} \t ground_truth = {ground_truth}\t {node}"
            )
            assert node2count[node] == ground_truth

from math import log
def test_most_supported_trees():
    print(f"Testing with {len(dags)} dags")
    for dag in dags:
        node2count = dag.count_nodes()
        total_trees = dag.count_trees()
        clade2support = {}
        for node, count in node2count.items():
            if node.under_clade() not in clade2support:
                clade2support[node.under_clade()] = 0
            clade2support[node.under_clade()] += count / total_trees

        print(dag.to_graphviz())
        best_sup = dag.most_supported_trees()
        print("Best support =", best_sup)
        print(f"\tnum trees before trim: {total_trees}")

        trees = dag.get_trees()
        print(f"\tnum trees after trim: {dag.count_trees()}")
        for tree in trees:
            tree_sup = 0
            for node in tree.postorder():
                tree_sup += log(clade2support[node.under_clade()])
            assert tree_sup == best_sup
            


# NOTE: Tests that edge2count contains edges -> count that are correct, but not that all
# edges are contained...
# def test_edge_counts():
#     print(f"Testing with {len(dags)} dags")
#     for dag in dags:
#         edge2count = dag.count_edges()

#         for parent, node in edge2count.keys():
#             ground_truth = 0
#             for tree in dag.get_trees():
#                 for curr_parent in tree.postorder():
#                     if curr_parent != parent:
#                         continue
#                     if node in curr_parent.children():
#                         ground_truth += 1
#                         break

#             print(
#                 f"count={edge2count[(parent, node)]}\tground_truth={ground_truth}\t{parent.label}\t{node.label}"
#             )
#             print()
#             assert edge2count[(parent, node)] == ground_truth