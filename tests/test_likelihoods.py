# import historydag.dag as hdag
# import gctree
# import pickle
# from historydag.utils import hamming_distance
# import numpy as np
# from scipy.special import logsumexp
# from multiset import FrozenMultiset
# from typing import Tuple
# from collections import Counter

# def add_distances(tree):
#     for node in tree.iter_descendants():
#         node.dist = hamming_distance(node.up.sequence, node.sequence)
#     tree.dist = 0

# def lltree(cm_list, p: np.float64, q: np.float64) -> Tuple[np.float64, np.ndarray]:
#     r"""Minimal adaptation of gctree.CollapsedTree.ll
#     Log likelihood of branching process parameters :math:`(p, q)` given tree topology :math:`T` and genotype abundances :math:`A`.
#     .. math::
#         \ell(p, q; T, A) = \log\mathbb{P}(T, A \mid p, q)
#     Args:
#         p: branching probability
#         q: mutation probability
#         build_cache: build cache from the bottom up. Normally this should be left to its default ``True``.
#     Returns:
#         Log likelihood :math:`\ell(p, q; T, A)` and its gradient :math:`\nabla\ell(p, q; T, A)`
#     """
#     if (
#         cm_list[0][0] == 0
#         and cm_list[0][1] == 1
#         and gctree.CollapsedTree._ll_genotype(cm_list[0][0], cm_list[0][1], p, q)[0]
#         == -np.inf
#     ):
#         # if unifurcation not possible under current model, add a
#         # psuedocount for the root
#         cm_list[0] = (1, cm_list[0][1])
#     # extract vector of function values and gradient components
#     logf_data = [gctree.CollapsedTree._ll_genotype(c, m, p, q) for c, m in cm_list]
#     logf = np.array([[x[0]] for x in logf_data]).sum()
#     grad_ll_genotype = np.array([x[1] for x in logf_data]).sum(axis=0)
#     return logf, grad_ll_genotype


# def llforest(
#     cm_list_list,
#     p: np.float64,
#     q: np.float64,
#     marginal: bool = True,
# ) -> Tuple[np.float64, np.ndarray]:
#     r"""Minimal adaptation of gctree.CollapsedForest.ll
#     Log likelihood of branching process parameters :math:`(p, q)` given tree topologies :math:`T_1, \dots, T_n` and corresponding genotype abundances vectors :math:`A_1, \dots, A_n` for each of :math:`n` trees in the forest.
#     If ``marginal=False`` (the default), compute the joint log likelihood
#     .. math::
#         \ell(p, q; T, A) = \sum_{i=1}^n\log\mathbb{P}(T_i, A_i \mid p, q),
#     otherwise compute the marginal log likelihood
#     .. math::
#         \ell(p, q; T, A) = \log\left(\sum_{i=1}^n\mathbb{P}(T_i, A_i \mid p, q)\right).
#     Args:
#         p: branching probability
#         q: mutation probability
#         marginal: compute the marginal likelihood over trees, otherwise compute the joint likelihood of trees
#     Returns:
#         Log likelihood :math:`\ell(p, q; T, A)` and its gradient :math:`\nabla\ell(p, q; T, A)`
#     """
#     c_max = max([t[0] for sublist in cm_list_list for t in sublist])
#     m_max = max([t[1] for sublist in cm_list_list for t in sublist])
#     gctree.CollapsedTree._build_ll_genotype_cache(c_max, m_max, p, q)
#     # we don't want to build the cache again in each tree
#     terms = [lltree(cm_list, p, q) for cm_list in cm_list_list]
#     ls = np.array([term[0] for term in terms])
#     grad_ls = np.array([term[1] for term in terms])
#     if marginal:
#         # we need to find the smallest derivative component for each
#         # coordinate, then subtract off to get positive things to logsumexp
#         grad_l = []
#         for j in range(len((p, q))):
#             i_prime = grad_ls[:, j].argmin()
#             grad_l.append(
#                 grad_ls[i_prime, j]
#                 + np.exp(
#                     logsumexp(ls - ls[i_prime], b=grad_ls[:, j] - grad_ls[i_prime, j])
#                     - logsumexp(ls - ls[i_prime])
#                 )
#             )
#         return (-np.log(len(ls)) + logsumexp(ls)), np.array(grad_l)
#     else:
#         return ls.sum(), grad_ls.sum(axis=0)


# def verify_likelihoods(self, abundance_dict, p, q, marginal=True):
#     """Abundance_dict is a map of node sequences to abundances"""
#     etetrees = [tree.to_ete() for tree in self.get_trees()]
#     for tree in etetrees:
#         for node in tree.traverse():
#             node.name = node.sequence
#             if node.name in abundance_dict and node.is_leaf():
#                 node.abundance = abundance_dict[node.name]
#             else:
#                 node.abundance = 0
#         add_distances(tree)
#     ctrees = [gctree.CollapsedTree(tree) for tree in etetrees]
#     cforest = gctree.CollapsedForest(ctrees)
#     return (
#         cforest.ll(p, q, marginal=marginal),
#         Counter(
#             [FrozenMultiset([tuple(ls) for ls in ctree._cm_list]) for ctree in ctrees]
#         ),
#     )


# with open("sample_data/50_toy_trees.p", "rb") as fh:
#     trees = pickle.load(fh)

# with open("sample_data/sample.counts", "r") as fh:
#     counts = {line.split(",")[0]: int(line.split(",")[1]) for line in fh}

# namedict = {node.sequence: node.name for tree in trees for node in tree.traverse()}

# abundance_dict = {
#     sequence: counts[name] for sequence, name in namedict.items() if name in counts
# }

# treeroot = trees[0].sequence

# dag = hdag.history_dag_from_etes(trees)
# dag.convert_to_collapsed()

# trees = list(dag.get_trees())
# trees = [tree.to_ete(namedict=namedict) for tree in trees]
# for tree in trees:
#     for node in tree.traverse():
#         if node.name in counts:
#             node.abundance = counts[node.name]
#         else:
#             node.abundance = 0
#         if not node.is_root():
#             node.dist = gctree.utils.hamming_distance(node.up.sequence, node.sequence)
#         else:
#             node.dist = 0
# ctrees = [gctree.CollapsedTree(tree) for tree in trees]
# forest = gctree.CollapsedForest(forest=ctrees)
# dag.add_abundances(
#     {sequence: counts[name] for sequence, name in namedict.items() if name in counts}
# )


# def test_bp_likelihood():
#     p, q = 0.4, 0.5
#     true_forest_ll, true_cmcounters = verify_likelihoods(dag, abundance_dict, p, q)
#     dag_cmcounter = dag.cmcounters()
#     cm_list_list = [[cm for cm in list(mset)] for mset in list(dag_cmcounter.elements())]
#     assert dag_cmcounter == true_cmcounters
#     dag_forest_ll = llforest(cm_list_list, p, q)
#     close = lambda a1, a2: np.isclose(a1, a2, atol=1e-10, rtol=0)
#     assert close(dag_forest_ll[0], true_forest_ll[0]) and all(close(dag_forest_ll[1], true_forest_ll[1]))
#     # ctrees_likelihoods = verify_likelihoods(dag, abundance_dict, p, q)[1]
#     # ctrees_likelihoods.sort()
#     # dag_likelihoods = list(dag.bp_loglikelihoods(p, q).elements())
#     # dag_likelihoods.sort()
#     # assert len(ctrees_likelihoods) == len(dag_likelihoods)
#     # assert ctrees_likelihoods == dag_likelihoods
