from historydag import utils
from historydag import dag as hdag
import ete3
from Bio.Data.IUPACData import ambiguous_dna_values

bases = "AGCT-"
ambiguous_dna_values.update({"?": "GATC-", "-": "-"})


def sequence_resolutions(sequence):
    """Iterates through possible disambiguations of sequence, recursively.
    Recursion-depth-limited by number of ambiguity codes in
    sequence, not sequence length.
    """

    def _sequence_resolutions(sequence, _accum=""):
        if sequence:
            for index, base in enumerate(sequence):
                if base in bases:
                    _accum += base
                else:
                    for newbase in ambiguous_dna_values[base]:
                        yield from _sequence_resolutions(
                            sequence[index + 1:], _accum=(_accum + newbase)
                        )
                    return
        yield _accum

    return _sequence_resolutions(sequence)


def disambiguate_sitewise(tree: ete3.TreeNode) -> ete3.TreeNode:
    """Resolve tree and return list of all possible resolutions"""
    code_vectors = {
        code: [
            0 if base in ambiguous_dna_values[code] else float("inf") for base in bases
        ]
        for code in ambiguous_dna_values
    }
    cost_adjust = {
        base: [int(not i == j) for j in range(5)] for i, base in enumerate(bases)
    }

    for node in tree.traverse(strategy="postorder"):

        def cvup(node, site):
            cv = code_vectors[node.sequence[site]].copy()
            if not node.is_leaf():
                for i in range(5):
                    for child in node.children:
                        cv[i] += min(
                            [
                                sum(v)
                                for v in zip(child.cvd[site], cost_adjust[bases[i]])
                            ]
                        )
            return cv

        # Make dictionary of cost vectors for each site
        node.cvd = {site: cvup(node, site) for site in range(len(node.sequence))}

    disambiguated = [tree.copy()]
    ambiguous = True
    while ambiguous:
        ambiguous = False
        treesindex = 0
        while treesindex < len(disambiguated):
            tree2 = disambiguated[treesindex]
            treesindex += 1
            for node in tree2.traverse(strategy="preorder"):
                ambiguous_sites = [
                    site for site, code in enumerate(node.sequence) if code not in bases
                ]
                if not ambiguous_sites:
                    continue
                else:
                    ambiguous = True
                    # Adjust cost vectors for ambiguous sites base on above
                    if not node.is_root():
                        for site in ambiguous_sites:
                            base_above = node.up.sequence[site]
                            node.cvd[site] = [
                                sum(v)
                                for v in zip(node.cvd[site], cost_adjust[base_above])
                            ]
                    option_dict = {site: "" for site in ambiguous_sites}
                    # Enumerate min-cost choices
                    for site in ambiguous_sites:
                        min_cost = min(node.cvd[site])
                        min_cost_sites = [
                            bases[i]
                            for i, val in enumerate(node.cvd[site])
                            if val == min_cost
                        ]
                        option_dict[site] = "".join(min_cost_sites)

                    sequences = list(utils._options(option_dict, node.sequence))
                    # Give this tree the first sequence, append copies with all
                    # others to disambiguated.
                    numseqs = len(sequences)
                    for idx, sequence in enumerate(sequences):
                        node.sequence = sequence
                        if idx < numseqs - 1:
                            disambiguated.append(tree2.copy())
                    break
    for tree in disambiguated:
        for node in tree.traverse():
            try:
                node.del_feature("cvd")
            except KeyError:
                pass
    return disambiguated


def disambiguate(tree: ete3.Tree, dist_func=utils.hamming_distance):
    """Resolve ambiguous bases using a two-pass Sankoff Algorithm on entire tree and entire sequence at each node.
    This does not disambiguate sitewise, so trees with many ambiguities may make this run very slowly.
    Returns a list of all possible disambiguations, minimizing the passed distance function dist_func.
    """
    # First pass of Sankoff: compute cost vectors
    for node in tree.traverse(strategy="postorder"):
        node.add_feature(
            "costs", [[seq, 0] for seq in sequence_resolutions(node.sequence)]
        )
        if not node.is_leaf():
            for seq_cost in node.costs:
                for child in node.children:
                    seq_cost[1] += min(
                        [
                            dist_func(seq_cost[0], child_seq) + child_cost
                            for child_seq, child_cost in child.costs
                        ]
                    )
    disambiguate_queue = [tree]
    disambiguated = []

    def incremental_disambiguate(ambig_tree):
        for node in ambig_tree.traverse(strategy="preorder"):
            if not utils.is_ambiguous(node.sequence):
                continue
            else:
                if not node.is_root():
                    node.costs = [
                        [
                            sequence,
                            cost + dist_func(node.up.sequence, sequence),
                        ]
                        for sequence, cost in node.costs
                    ]
                min_cost = min([cost for _, cost in node.costs])
                for resolved_sequence in [
                    sequence for sequence, cost in node.costs if cost == min_cost
                ]:
                    node.sequence = resolved_sequence
                    disambiguate_queue.append(ambig_tree.copy())
                return
        disambiguated.append(ambig_tree)

    while disambiguate_queue:
        incremental_disambiguate(disambiguate_queue.pop())
    for tree in disambiguated:
        for node in tree.traverse():
            try:
                node.del_feature("costs")
            except (AttributeError, KeyError):
                pass
    return disambiguated


newick_tree2 = (
    "((12[&&NHX:name=12:sequence=TT],"
    "(6[&&NHX:name=6:sequence=CG],"
    "7[&&NHX:name=7:sequence=AC])5[&&NHX:name=5:sequence=??])"
    "3[&&NHX:name=3:sequence=?T],8[&&NHX:name=8:sequence=AA],"
    "(11[&&NHX:name=11:sequence=AG],10[&&NHX:name=10:sequence=GT])"
    "9[&&NHX:name=9:sequence=?T])2[&&NHX:name=2:sequence=?T];"
)
tree2 = ete3.TreeNode(newick=newick_tree2, format=1)
dags = [
    hdag.history_dag_from_newicks(newicklist, ["sequence"])
    for newicklist in [[newick_tree2]]
]


def treeprint(tree: ete3.TreeNode):
    tree = tree.copy()
    for node in tree.traverse():
        node.name = node.sequence
    return tree.write(format=8)


def test_expand_ambiguities():

    for dag in dags:
        cdag = dag.copy()
        print(cdag.count_trees())
        cdag.explode_nodes()
        print(cdag.count_trees())
        cdag.trim_optimal_weight()
        print(cdag.count_trees())
        print(cdag.weight_count())
        checkset = {
            hdag.from_tree(tree, ["sequence"]).to_newick()
            for cladetree in dag.get_trees()
            for tree in disambiguate(cladetree.to_ete(features=["sequence"]))
        }
        print(len(checkset))
        assert checkset == {cladetree.to_newick() for cladetree in cdag.get_trees()}


#     newickset = {treeprint(tree) for tree in utils.disambiguate(tree2)}
#     correctset = {
#         "((((T)C,(C,A)C)C,A,(A,G)G)G);",
#         "((((T)C,(C,A)A)A,A,(A,G)A)A);",
#         "((((T)C,(C,A)C)C,A,(A,G)A)A);",
#     }
#     if not newickset == correctset:
#         missing = correctset - newickset
#         wrong = newickset - correctset
#         print(
#             f"\nDisambiguate function is missing {missing}\n"
#             f"and came up with these incorrect trees: {wrong}"
#         )
#         raise ValueError("Invalid Disambiguation")
