from historydag.dag import EdgeSet, from_tree
from test_historydag import newickstring2, newickstring3
import ete3

""" EdgeSet Tests """


def test_init():
    tree = ete3.Tree(newickstring2, format=1)
    dag = from_tree(tree, ["sequence"])
    tree1 = ete3.Tree(newickstring2, format=1)
    dag1 = from_tree(tree1, ["sequence"])
    EdgeSet()
    EdgeSet([dag.dagroot])
    EdgeSet([dag.dagroot], probs=[0.5])
    try:
        EdgeSet([dag.dagroot, dag1.dagroot])
        raise TypeError("EdgeSet init is allowing identical dags to be added")
    except TypeError:
        pass
    try:
        EdgeSet([dag.dagroot], [dag1.dagroot])
        raise TypeError("EdgeSet init is allowing multiple positional arguments")
    except TypeError:
        pass


def test_iter():
    tree = ete3.Tree(newickstring2, format=1)
    dag = from_tree(tree.children[0], ["sequence"])
    tree1 = ete3.Tree(newickstring3, format=1)
    dag1 = from_tree(tree1.children[0], ["sequence"])
    e2 = EdgeSet([next(dag.dagroot.children()), next(dag1.dagroot.children())])
    for target, weight, prob in e2:
        pass


def test_copy():
    tree = ete3.Tree(newickstring2, format=1)
    dag = from_tree(tree.children[0], ["sequence"])
    tree1 = ete3.Tree(newickstring3, format=1)
    dag1 = from_tree(tree1.children[0], ["sequence"])
    e2 = EdgeSet([next(dag.dagroot.children()), next(dag1.dagroot.children())])
    e2.shallowcopy()


def test_not_to_root():
    tree = ete3.Tree(newickstring2, format=1)
    dag = from_tree(tree.children[0], ["sequence"])
    tree1 = ete3.Tree(newickstring3, format=1)
    dag1 = from_tree(tree1.children[0], ["sequence"])
    e1 = EdgeSet([next(dag.dagroot.children()), next(dag1.dagroot.children())])
    try:
        EdgeSet([dag.dagroot, next(dag1.dagroot.children())])
        raise ValueError("Adding a root node to an edge set was allowed")
    except ValueError:
        pass
    try:
        e1.add_to_edgeset(dag.dagroot)
        raise ValueError("Adding a root node to an edge set was allowed")
    except ValueError:
        pass
