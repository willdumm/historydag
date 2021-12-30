import ete3
from Bio.Data.IUPACData import ambiguous_dna_values
from collections import Counter
import random
from functools import wraps
from typing import List, Any

bases = "AGCT-"
ambiguous_dna_values.update({"?": "GATC-", "-": "-"})


def weight_function(func):
    """A decorator to allow distance to label None to be zero"""
    @wraps(func)
    def wrapper(s1, s2):
        if s1 is None or s2 is None:
            return 0
        else:
            return func(s1, s2)

    return wrapper

def explode_label(labelfield: str):
    """A decorator to make it easier to expand a Label by a certain field.

    Args:
        labelfield: the name of the field whose contents the wrapped function is expected to
            explode

    Returns:
        A decorator which converts a function which explodes a field value, into a function
            which explodes the whole label at that field."""
    def decorator(func):
        @wraps(func)
        def wrapfunc(label, *args, **kwargs):
            Label = type(label)
            d = label._asdict()
            for newval in func(d[labelfield], *args, **kwargs):
                d[labelfield] = newval
                yield Label(**d)
        return wrapfunc
    return decorator

@weight_function
def hamming_distance(s1: str, s2: str) -> int:
    if len(s1) != len(s2):
        raise ValueError("Sequences must have the same length!")
    return sum(x != y for x, y in zip(s1, s2))


def compare_site_func(site):
    
    @weight_function
    def dist_func(s1: str, s2: str) -> int:
        return int(s1[site] != s2[site])

    return dist_func

def is_ambiguous(sequence):
    return any(code not in bases for code in sequence)

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

@explode_label('sequence')
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
                            sequence[index + 1 :], _accum=(_accum + newbase)
                        )
                    return
        yield _accum
    return _sequence_resolutions(sequence)

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
    tree = tree.copy()
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
        node.dist = 1
    return newtree.write(format=1, features=['sequence'], format_root_node=True)

def is_collapsed(tree: ete3.TreeNode):
    return not any(node.sequence == node.up.sequence and not node.is_leaf() for node in tree.iter_descendants())


