from collections import Counter
import historydag.dag as hdag
import historydag.utils as utils

newicks = [
           ["((AA, CT)CG, (TA, CC)CG)CC;",
           "((AA, CT)CA, (TA, CC)CC)CC;"],
           ["((CA, GG)CA, AA, (TT, (CC, GA)CC)CC)AA;",
           "((CA, GG)CA, AA, (TT, (CC, GA)CA)CA)AA;",
           "((CA, GG)CG, AA, (TT, (CC, GA)GC)GC)AG;"]
]

abundance_dicts = [{
    'AA': 2,
    'CT': 1,
    'TA': 1,
    'CC': 3,
},
                   {
    'CA': 2,
    'GG': 1,
    'AA': 1,
    'TT': 1,
    'CC': 5,
    'GA': 2,
}]

dags = [hdag.history_dag_from_newicks(newicklist) for newicklist in newicks]
cdags = [dag.copy() for dag in dags]
for dag in cdags:
    dag.convert_to_collapsed()

test_parsimony_counters = [
    Counter({8:1, 6:1, 7:1, 5:1}),
    Counter({9: 1, 8: 1, 10: 1})
]

def test_parsimony_counts():
    assert [dag.get_weight_counts() for dag in dags] == test_parsimony_counters
    assert [dag.get_weight_counts() for dag in cdags] == test_parsimony_counters

test_uncollapsed_counters = [
    Counter({0: 2, 2: 2}),
    Counter({4: 1, 3: 1, 1: 1})
]

test_collapsed_counters = [
    Counter({0: 2, 1: 2}),
    Counter({2: 1, 3: 1, 0: 1})
]

def test_collapsed_counts():
    assert [dag.get_weight_counts(distance_func=lambda x, y: x == y) for dag in dags] == test_uncollapsed_counters
    assert [dag.get_weight_counts(distance_func=lambda x, y: x == y) for dag in cdags] == test_collapsed_counters


def test_min_weight():
    for dag in dags:
        dag.min_weight_annotate()
    for dag in cdags:
        dag.min_weight_annotate()
    assert [dag.min_weight_under for dag in dags] == [5, 8]
    assert [dag.min_weight_under for dag in cdags] == [5, 8]

def test_cm_counter():
    pass

