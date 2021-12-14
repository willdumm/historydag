import historydag.dag as hdag
import historydag.utils as dagutils
import ete3
import random
from collections import Counter


def get_samples(seed=3):
    random.seed(seed)
    bases = 'AGCT-'
    rootseq = 'A' * 5
    def tips(rootsequence, n):
        """
        Return n sequences with a random number of mutations from the rootsequence.
        """
        tipseqs = []
        rootsequence = list(rootsequence)
        while len(tipseqs) < n:
            mutindices = random.choices([i for i in range(len(rootsequence))], k=len(rootsequence))
            tipseq = rootsequence.copy()
            for index in mutindices:
                tipseq[index] = random.choice(bases)
            if ''.join(tipseq) not in tipseqs:
                tipseqs.append(''.join(tipseq))
        return(tipseqs)

    samplelist = []

    # We will build ten random trees on a fixed set of seven tip sequences
    # Interior nodes will have entirely ambiguous sequences '?????'
    tps = tips(rootseq, 7)
    for i in range(10):
        node = ete3.TreeNode()
        node.populate(7)
        # Assign sequences to nodes
        node.sequence = rootseq
        for ind, leaf in enumerate(node.get_leaves()):
            leaf.sequence = tps[ind]
        for node2 in node.get_descendants():
            if not node2.is_leaf():
                node2.sequence = '?' * 5
        samplelist.append(node)
    return samplelist

samplelist = get_samples(seed=3)

def test_sitewise_expansion():
    dag = hdag.history_dag_from_etes(samplelist)
    ambigset = set()
    for node in hdag.postorder(dag):
        if node.label != "DAG_root":
            ambigset.update({site for site, base in enumerate(node.label) if base not in dagutils.bases})
    print(ambigset)
    for site in ambigset:
        nodedict = {id(node): node for node in hdag.postorder(dag)}
        dag.expand_ambiguities(focus_site=site)
        for node in hdag.postorder(dag):
            if node.label != "DAG_root" and node.label[site] not in dagutils.bases:
                new = id(node) not in nodedict
                raise RuntimeError(f"There remains an ambiguous base {node.label[site]} at site {site} after expanding.\nnew node: {new}")
        dag.trim_min_weight(focus_site=site)

def test_dag_disambiguate():
    resolvedsamples = [dagutils.disambiguate_sitewise(tree) for tree in samplelist]
    resolvedsamples = [sample for sublist in resolvedsamples for sample in sublist]
    weights = Counter([dagutils.recalculate_ete_parsimony(tree) for tree in resolvedsamples])
    minweightsamples = [tree for tree in resolvedsamples if dagutils.recalculate_ete_parsimony(tree) == min(weights)]
    print("Weights of resolved samples")
    dagutils.hist(weights)


    dag = hdag.history_dag_from_etes(samplelist)
    dag.disambiguate_sitewise()
    for node in hdag.postorder(dag):
        if node.label != "DAG_root" and dagutils.is_ambiguous(node.label):
            raise RuntimeError(f"found ambiguous node label {node.label}\nwas removed: {node.removed}")
    print("Summary of DAG constructed from all disambiguations of samples:")
    ddag = hdag.history_dag_from_etes(resolvedsamples)
    ddag.summary()

    print("Summary of DAG constructed from ambiguous samples then disambiguated:")
    dag.summary()
    dagweights = dag.get_weight_counts()
    dag.trim_min_weight()
    trees = {dagutils.deterministic_newick(tree.to_ete()) for tree in dag.get_trees()}
    minweightnewicks = {dagutils.deterministic_newick(tree) for tree in minweightsamples}
    assert sum(1 for _ in hdag.postorder(dag)) == len({node for node in hdag.postorder(dag)})
    if not minweightnewicks < trees:
        print(minweightnewicks - trees)
        raise RuntimeError("Disambiguated DAG doesn't represent all min-weight disambiguations of the original ambiguous trees.")

# def test_sample_data_dag_disambiguate():
#     with open('sample_data/resolved_dag.p', 'rb') as fh:
#         rdag = hdag.deserialize(fh.read())

#     with open('sample_data/ambiguous_dag.p', 'rb') as fh:
#         adag = hdag.deserialize(fh.read())

#     rdag.summary()
#     adag.disambiguate_sitewise()
#     adag.summary()

#     rdag.trim_min_weight()
#     adag.trim_min_weight()
#     rdagtrees = {dagutils.deterministic_newick(tree.to_ete()) for tree in rdag.get_trees()}
#     adagtrees = {dagutils.deterministic_newick(tree.to_ete()) for tree in adag.get_trees()}
#     assert rdagtrees < adagtrees
