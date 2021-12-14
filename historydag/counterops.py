from collections import Counter
import historydag.utils as utils
import operator


def prod(l: list):
    """Return product of elements of the input list.
    if passed list is empty, returns 1."""
    n = len(l)
    if n > 0:
        accum = l[0]
        if n > 1:
            for item in l[1:]:
                accum *= item
    else:
        accum = 1
    return accum

def counter_prod(counterlist, accumfunc):
    """Really a sort of cartesian product, which does accumfunc to keys and counts all the ways each result
    can be achieved using contents of counters in counterlist.
    accumfunc must be a function like sum which acts on a list of arbitrary length. Probably should return an
    object of the same type."""
    newc = Counter()
    for combi in utils.cartesian_product([c.items for c in counterlist]):
        weights, counts = [[t[i] for t in combi] for i in range(len(combi[0]))]
        newc.update({accumfunc(weights): prod(counts)})
    return newc

def counter_sum(counterlist, counter_type=Counter):
    """Sum a list of counters, like concatenating their representative lists"""
    newc = counter_type()
    for c in counterlist:
        newc += c
    return newc

def addweight(c, w, addfunc=operator.add):
    return Counter({addfunc(key, w): val for key, val in c.items()})
