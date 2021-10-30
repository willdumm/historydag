import ete3
from Bio.Data.IUPACData import ambiguous_dna_values
from collections import Counter
import random

bases = "AGCT-"
ambiguous_dna_values.update({"?": "GATC-", "-": "-"})


def weight_function(func):
    """A wrapper to allow distance to label 'DAG_root' to be zero"""

    def wrapper(s1, s2):
        if s1 == "DAG_root" or s2 == "DAG_root":
            return 0
        else:
            return func(s1, s2)

    return wrapper


@weight_function
def hamming_distance(s1: str, s2: str) -> int:
    if len(s1) != len(s2):
        raise ValueError("Sequences must have the same length!")
    return sum(x != y for x, y in zip(s1, s2))

def is_ambiguous(sequence):
    return any(code not in bases for code in sequence)

def disambiguate(tree: ete3.Tree, random_state=None, dist_func=hamming_distance) -> ete3.Tree:
    """Resolve ambiguous bases using a two-pass Sankoff Algorithm on entire tree and entire sequence at each node.
    This does not disambiguate sitewise, so trees with many ambiguities may make this run very slowly.
    Returns a list of all possible disambiguations, minimizing the passed distance function dist_func.
    """
    if random_state is None:
        random.seed(tree.write(format=1))
    else:
        random.setstate(random_state)
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
            if not is_ambiguous(node.sequence):
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
                for resolved_sequence in [sequence for sequence, cost in node.costs if cost == min_cost]:
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


def cartesian_product(optionlist, accum=tuple()):
    """Takes a list of functions which each return a fresh generator
    on options at that site"""
    if optionlist:
        for term in optionlist[0]():
            yield from cartesian_product(optionlist[1:], accum=(accum + (term,)))
    else:
        yield accum


def _options(option_dict, sequence):
    """option_dict is keyed by site index, with iterables containing
    allowed bases as values"""
    if option_dict:
        site, choices = option_dict.popitem()
        for choice in choices:
            sequence = sequence[:site] + choice + sequence[site + 1 :]
            yield from _options(option_dict.copy(), sequence)
    else:
        yield sequence

def sequence_resolutions(sequence, _accum=""):
    """Iterates through possible disambiguations of sequence, recursively.
    Recursion-depth-limited by number of ambiguity codes in
    sequence, not sequence length.
    """
    if sequence:
        for index, base in enumerate(sequence):
            if base in bases:
                _accum += base
            else:
                for newbase in ambiguous_dna_values[base]:
                    yield from sequence_resolutions(
                        sequence[index + 1 :], _accum=(_accum + newbase)
                    )
                return
    yield _accum

def disambiguate_all(treelist):
    resolvedsamples = []
    for sample in treelist:
        resolvedsamples.extend(disambiguate(sample))
    return resolvedsamples


def recalculate_ete_parsimony(
    tree: ete3.TreeNode, distance_func=hamming_distance
) -> float:
    tree.dist = 0
    for node in tree.iter_descendants():
        node.dist = distance_func(node.sequence, node.up.sequence)
    return total_weight(tree)


def hist(c: Counter, samples=1):
    l = list(c.items())
    l.sort()
    print("Weight\t| Frequency\n------------------")
    for weight, freq in l:
        print(f"{weight}  \t| {freq if samples==1 else freq/samples}")


def total_weight(tree: ete3.TreeNode) -> float:
    return sum(node.dist for node in tree.traverse())


def collapse_adjacent_sequences(tree: ete3.TreeNode) -> ete3.TreeNode:
    """Collapse nonleaf nodes that have the same sequence"""
    # Need to keep doing this until the tree fully collapsed. See gctree for this!
    to_delete = []
    for node in tree.get_descendants():
        # This must stay invariably hamming distance, since it's measuring equality of strings
        if not node.is_leaf() and hamming_distance(node.up.sequence, node.sequence) == 0:
            to_delete.append(node)
    for node in to_delete:
        node.delete()
    return tree

def deterministic_newick(tree: ete3.TreeNode):
    """For use in comparing TreeNodes with newick strings"""
    newtree = tree.copy()
    for node in newtree.traverse():
        node.name = 1
        node.children.sort(key=lambda node: node.sequence)
    return newtree.write(format=1, features=['sequence'], format_root_node=True)

def is_collapsed(tree: ete3.TreeNode):
    return not any(node.sequence == node.up.sequence and not node.is_leaf() for node in tree.iter_descendants())

