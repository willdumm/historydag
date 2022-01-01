from historydag.utils import *
from historydag.counterops import *
from collections import Counter

def test_hamming_distance():
    assert hamming_distance('AGC', 'AGC') == 0
    assert hamming_distance('ACC', 'AGC') == 1
    try:
        hamming_distance('AG', 'GCC')
    except ValueError:
        return
    raise ValueError("hamming distance allowed comparison of sequences with different lengths.")

def test_ualabel():
    assert UALabel() == UALabel()
    assert UALabel() != None
    assert UALabel() != 0

def test_hist():
    hist(Counter([1,2,3,3,3,3,4]))

def test_prod():
    assert prod([]) == 1
