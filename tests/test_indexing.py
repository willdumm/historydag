from historydag.dag import (
    history_dag_from_newicks,
)

newickstring1 = (
    "((4[&&NHX:name=4:sequence=K],(6[&&NHX:name=6:sequence=S],"
    "7[&&NHX:name=7:sequence=T])5[&&NHX:name=5:sequence=M])3[&&NHX:name=3:sequence=M],"
    "8[&&NHX:name=8:sequence=W],(11[&&NHX:name=11:sequence=V],"
    "10[&&NHX:name=10:sequence=D])9[&&NHX:name=9:sequence=R])1[&&NHX:name=1:sequence=G];"
)

newickstring2 = (
    "((4[&&NHX:name=4:sequence=K],(6[&&NHX:name=6:sequence=S],"
    "7[&&NHX:name=7:sequence=T])5[&&NHX:name=5:sequence=H])3[&&NHX:name=3:sequence=G],"
    "8[&&NHX:name=8:sequence=W],(11[&&NHX:name=11:sequence=V],"
    "10[&&NHX:name=10:sequence=D])9[&&NHX:name=9:sequence=C])1[&&NHX:name=1:sequence=A];"
)

newickstring3 = (
    "((4[&&NHX:name=4:sequence=K],(6[&&NHX:name=6:sequence=S],"
    "7[&&NHX:name=7:sequence=T])5[&&NHX:name=5:sequence=H])2[&&NHX:name=2:sequence=B],"
    "8[&&NHX:name=8:sequence=W],(11[&&NHX:name=11:sequence=V],"
    "10[&&NHX:name=10:sequence=D])9[&&NHX:name=9:sequence=C])1[&&NHX:name=1:sequence=A];"
)


def test_preserve_attr():
    dag = history_dag_from_newicks(
        [newickstring1, newickstring2, newickstring3],
        ["sequence"],
        attr_func=lambda n: n.name,
    )
    assert all(n.attr for n in dag.preorder(skip_root=True))
    dag[2]
