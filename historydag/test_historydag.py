from historydag import SdagNode, EdgeSet, from_tree, postorder, sdag_from_newicks
import ete3

""" SdagNode tests:"""

newickstring1 = (
    "(((4[&&NHX:name=4:sequence=C],(6[&&NHX:name=6:sequence=C],"
    "7[&&NHX:name=7:sequence=A])5[&&NHX:name=5:sequence=M])3[&&NHX:name=3:sequence=M],"
    "8[&&NHX:name=8:sequence=A],(11[&&NHX:name=11:sequence=A],"
    "10[&&NHX:name=10:sequence=G])9[&&NHX:name=9:sequence=R])"
    "2[&&NHX:name=2:sequence=R])1[&&NHX:name=1:sequence=G];"
)

newickstring2 = (
    "(((4[&&NHX:name=4:sequence=K],(6[&&NHX:name=6:sequence=J],"
    "7[&&NHX:name=7:sequence=I])5[&&NHX:name=5:sequence=H])3[&&NHX:name=3:sequence=G],"
    "8[&&NHX:name=8:sequence=F],(11[&&NHX:name=11:sequence=E],"
    "10[&&NHX:name=10:sequence=D])9[&&NHX:name=9:sequence=C])"
    "2[&&NHX:name=2:sequence=B])1[&&NHX:name=1:sequence=A];"
)

newickstring3 = (
    "(((4[&&NHX:name=4:sequence=K],(6[&&NHX:name=6:sequence=J],"
    "7[&&NHX:name=7:sequence=I])5[&&NHX:name=5:sequence=H])2[&&NHX:name=2:sequence=B],"
    "8[&&NHX:name=8:sequence=F],(11[&&NHX:name=11:sequence=E],"
    "10[&&NHX:name=10:sequence=D])9[&&NHX:name=9:sequence=C])"
    "3[&&NHX:name=3:sequence=G])1[&&NHX:name=1:sequence=A];"
)

namedict = {
    "A": 1,
    "B": 2,
    "C": 9,
    "D": 10,
    "E": 11,
    "F": 8,
    "G": 3,
    "H": 5,
    "I": 7,
    "J": 6,
    "K": 4,
    "root": "root",
}


def test_init():
    r = SdagNode("a")
    assert r.is_leaf() == True
    r = SdagNode(
        "a", clades={frozenset(["a", "b"]): EdgeSet(), frozenset(["c", "d"]): EdgeSet()}
    )
    assert r.is_leaf() == False
    s = SdagNode(
        "b",
        clades={frozenset(["a", "b"]): EdgeSet(), frozenset(["c", "d"]): EdgeSet([r])},
    )
    assert s.is_leaf() == False


def test_edge():
    r = SdagNode("a")
    r2 = SdagNode("b", {frozenset({"z", "y"}): EdgeSet(), frozenset({"a"}): EdgeSet()})
    s = SdagNode(
        "b", clades={frozenset(["a"]): EdgeSet(), frozenset(["c", "d"]): EdgeSet()}
    )
    s.add_edge(r)
    try:
        s.add_edge(r2)
        assert False
    except KeyError:
        pass


def test_from_tree():
    tree = ete3.Tree(newickstring2, format=1)
    print(tree.sequence)
    dag = from_tree(tree)
    G = dag.to_graphviz(namedict)
    return G


def test_postorder():
    tree = ete3.Tree(newickstring2, format=1)
    dag = from_tree(tree)
    assert [namedict[node.label] for node in postorder(dag)] == [
        4,
        6,
        7,
        5,
        3,
        8,
        11,
        10,
        9,
        2,
        1,
        "root",
    ]
    # print([namedict[node.label] for node in postorder(dag)])


def test_children():
    tree = ete3.Tree(newickstring2, format=1)
    dag = from_tree(tree)
    print([child.label for child in dag.children()])
    for child in dag.children():
        print([cc.label for cc in child.children()])
        for ccc in child.children():
            print([cccc.label for cccc in ccc.children()])


def test_merge():
    tree1 = ete3.Tree(newickstring2, format=1)
    dag1 = from_tree(tree1)
    tree2 = ete3.Tree(newickstring3, format=1)
    dag2 = from_tree(tree2)
    dag1.merge(dag2)
    return dag1.to_graphviz(namedict)


def test_weight():
    tree1 = ete3.Tree(newickstring2, format=1)
    dag1 = from_tree(tree1)
    tree2 = ete3.Tree(newickstring3, format=1)
    dag2 = from_tree(tree2)
    dag1.merge(dag2)
    return dag1.to_graphviz(namedict)
    assert dag1.weight() == 16


def test_internal_avg_parents():
    tree1 = ete3.Tree(newickstring2, format=1)
    dag1 = from_tree(tree1)
    tree2 = ete3.Tree(newickstring3, format=1)
    dag2 = from_tree(tree2)
    dag1.merge(dag2)
    return dag1.to_graphviz(namedict)
    assert dag1.internal_avg_parents() == 9 / 7


def test_sample():
    newicks = ["((a, b)b, c)c;", "((a, b)c, c)c;", "((a, b)a, c)c;", "((a, b)r, c)r;"]
    newicks = ["((1, 2)2, 3)3;", "((1, 2)3, 3)3;", "((1, 2)1, 3)3;", "((1, 2)4, 3)4;"]
    namedict = {str(x): x for x in range(5)}
    dag = sdag_from_newicks(newicks)
    sample = dag.sample()
    return sample.to_graphviz(namedict)


test_init()
test_edge()
G = test_from_tree()
test_postorder()
G = test_merge()
test_children()
G = test_weight()
test_internal_avg_parents()
G = test_sample()
G
