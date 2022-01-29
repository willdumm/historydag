from historydag.utils import (
    hamming_distance,
    UALabel,
    hist,
    AddFuncDict,
    hamming_distance_countfuncs,
    prod,
)
from collections import Counter


def test_hamming_distance():
    assert hamming_distance("AGC", "AGC") == 0
    assert hamming_distance("ACC", "AGC") == 1
    try:
        hamming_distance("AG", "GCC")
    except ValueError:
        return
    raise ValueError(
        "hamming distance allowed comparison of sequences with different lengths."
    )


def test_ualabel():
    assert UALabel() == UALabel()
    assert UALabel() is not None
    assert UALabel() != 0


def test_hist():
    hist(Counter([1, 2, 3, 3, 3, 3, 4]))


def test_prod():
    assert prod([]) == 1


def test_dp_templatefuncs():
    templatedict = AddFuncDict(
        {
            "start_func": lambda n: 0,
            "edge_weight_func": lambda n, n1: int(n == n1),
            "accum_func": sum,
        },
        name="sum0",
    )
    templatedict1 = AddFuncDict(
        {
            "start_func": lambda n: 1,
            "edge_weight_func": lambda n, n1: int(n != n1),
            "accum_func": min,
        },
        name="min1",
    )
    template = templatedict + templatedict1

    startfunc, ewfunc, accumfunc = template.values()
    assert hamming_distance_countfuncs.names == "HammingParsimony"
    assert template.names == ("sum0", "min1")
    assert startfunc("doesnotmatter") == (0, 1)
    assert ewfunc("somenode", "someother") == (0, 1)
    assert accumfunc([(1, 2), (2, 3), (4, 5)]) == (7, 2)

    template1 = template + templatedict
    startfunc, ewfunc, accumfunc = template1.values()
    assert template1.names == ("sum0", "min1", "sum0")
    assert startfunc("doesnotmatter") == (0, 1, 0)
    assert ewfunc("somenode", "someother") == (0, 1, 0)
    assert accumfunc([(1, 2, 3), (2, 3, 4), (4, 5, 6)]) == (7, 2, 13)
