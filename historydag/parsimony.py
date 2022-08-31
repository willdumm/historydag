"""A module implementing Sankoff Algorithm"""

import random
import ete3
import numpy as np
import Bio.Data.IUPACData
from historydag.dag import history_dag_from_clade_trees, history_dag_from_etes

bases = "AGCT-"
ambiguous_dna_values = Bio.Data.IUPACData.ambiguous_dna_values.copy()
ambiguous_dna_values.update({"?": "GATC-", "-": "-"})


def hamming_distance(seq1: str, seq2: str) -> int:
    r"""Hamming distance between two sequences of equal length.

    Args:
        seq1: sequence 1
        seq2: sequence 2
    """
    if len(seq1) != len(seq2):
        raise ValueError("sequence lengths do not match!")
    return sum(x != y for x, y in zip(seq1, seq2))


_yey = np.array(
    [
        [0, 1, 1, 1, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 1, 0],
    ]
)

# This is applicable even when diagonal entries in transition rate matrix are
# nonzero, since it is only a mask on allowable sites based on each base.
code_vectors = {
    code: np.array(
        [0 if base in ambiguous_dna_values[code] else float("inf") for base in bases]
    )
    for code in ambiguous_dna_values
}

ambiguous_codes_from_vecs = {
    tuple(0 if base in base_set else 1 for base in bases): code
    for code, base_set in ambiguous_dna_values.items()
}


def _get_adj_array(seq_len, transition_weights=None):
    if transition_weights is None:
        transition_weights = _yey
    else:
        transition_weights = np.array(transition_weights)

    if transition_weights.shape == (5, 5):
        adj_arr = np.array([transition_weights] * seq_len)
    elif transition_weights.shape == (seq_len, 5, 5):
        adj_arr = transition_weights
    else:
        raise RuntimeError(
            "Transition weight matrix must have shape (5, 5) or (sequence_length, 5, 5)."
        )
    return adj_arr


def sankoff_upward(tree, gap_as_char=False, transition_weights=None):
    """Compute Sankoff cost vectors at nodes in a postorder traversal, and
    return best possible parsimony score of the tree.

    Args:
        gap_as_char: if True, the gap character ``-`` will be treated as a fifth character. Otherwise,
            it will be treated the same as an ``N``.
        transition_weights: A 5x5 transition weight matrix, with base order `AGCT-`.
            Rows contain targeting weights. That is, the first row contains the transition weights
            from `A` to each possible target base. Alternatively, a sequence-length array of these
            transition weight matrices, if transition weights vary by-site. By default, a constant
            weight matrix will be used containing 1 in all off-diagonal positions, equivalent
            to Hamming parsimony.
    """
    if gap_as_char:

        def translate_base(char):
            return char

    else:

        def translate_base(char):
            if char == "-":
                return "N"
            else:
                return char

    adj_arr = _get_adj_array(len(tree.sequence), transition_weights=transition_weights)

    # First pass of Sankoff: compute cost vectors
    for node in tree.traverse(strategy="postorder"):
        node.add_feature(
            "cv",
            np.array(
                [code_vectors[translate_base(base)].copy() for base in node.sequence]
            ),
        )
        if not node.is_leaf():
            child_costs = []
            for child in node.children:
                stacked_child_cv = np.stack((child.cv,) * 5, axis=1)
                total_cost = adj_arr + stacked_child_cv
                child_costs.append(np.min(total_cost, axis=2))
            child_cost = np.sum(child_costs, axis=0)
            node.cv = (
                np.array([code_vectors[translate_base(base)] for base in node.sequence])
                + child_cost
            )
    return np.sum(np.min(tree.cv, axis=1))


def disambiguate(
    tree,
    compute_cvs=True,
    random_state=None,
    remove_cvs=False,
    adj_dist=False,
    gap_as_char=False,
    transition_weights=None,
    min_ambiguities=False,
):
    """Randomly resolve ambiguous bases using a two-pass Sankoff Algorithm on
    subtrees of consecutive ambiguity codes.

    Args:
        compute_cvs: If true, compute upward sankoff cost vectors. If ``sankoff_upward`` was
            already run on the tree, this may be skipped.
        random_state: A ``random`` module random state, returned by ``random.getstate()``. Output
            from this function is otherwise deterministic.
        remove_cvs: Remove sankoff cost vectors from tree nodes after disambiguation.
        adj_dist: Recompute hamming parsimony distances on tree after disambiguation, and store them
            in ``dist`` node attributes.
        gap_as_char: if True, the gap character ``-`` will be treated as a fifth character. Otherwise,
            it will be treated the same as an ``N``.
        transition_weights: A 5x5 transition weight matrix, with base order `AGCT-`.
            Rows contain targeting weights. That is, the first row contains the transition weights
            from `A` to each possible target base. Alternatively, a sequence-length array of these
            transition weight matrices, if transition weights vary by-site. By default, a constant
            weight matrix will be used containing 1 in all off-diagonal positions, equivalent
            to Hamming parsimony.
        min_ambiguities: If True, leaves ambiguities in reconstructed sequences, expressing which
            bases are possible at each site in a maximally parsimonious disambiguation of the given
            topology. In the history DAG paper, this is known as a strictly min-weight ambiguous labeling.
            Otherwise, sequences are resolved in one possible way, and include no ambiguities.
    """
    if random_state is None:
        random.seed(tree.write(format=1))
    else:
        random.setstate(random_state)

    seq_len = len(tree.sequence)
    adj_arr = _get_adj_array(len(tree.sequence), transition_weights=transition_weights)
    if compute_cvs:
        sankoff_upward(tree, gap_as_char=gap_as_char)
    # Second pass of Sankoff: choose bases
    preorder = list(tree.traverse(strategy="preorder"))
    for node in preorder:
        if min_ambiguities:
            adj_vec = node.cv != np.stack((node.cv.min(axis=1),) * 5, axis=1)
            new_seq = [
                ambiguous_codes_from_vecs[tuple(map(float, row))] for row in adj_vec
            ]
        else:
            base_indices = []
            for idx in range(seq_len):
                min_cost = min(node.cv[idx])
                base_index = random.choice(
                    [i for i, val in enumerate(node.cv[idx]) if val == min_cost]
                )
                base_indices.append(base_index)

            adj_vec = adj_arr[np.arange(seq_len), base_indices]
            new_seq = [bases[base_index] for base_index in base_indices]

        # Adjust child cost vectors
        for child in node.children:
            child.cv += adj_vec
        node.sequence = "".join(new_seq)

    if remove_cvs:
        for node in tree.traverse():
            try:
                node.del_feature("cv")
            except (AttributeError, KeyError):
                pass
    if adj_dist:
        tree.dist = 0
        for node in tree.iter_descendants():
            node.dist = hamming_distance(node.up.sequence, node.sequence)
    return tree


def load_fasta(fastapath):
    """Load a fasta file as a dictionary, with sequence ids as keys and
    sequences as values."""
    fasta_map = {}
    with open(fastapath, "r") as fh:
        seqid = None
        for line in fh:
            if line[0] == ">":
                seqid = line[1:].strip()
                if seqid in fasta_map:
                    raise ValueError(
                        "Duplicate records with matching identifier in fasta file"
                    )
                else:
                    fasta_map[seqid] = ""
            else:
                if seqid is None and line.strip():
                    raise ValueError(
                        "First non-blank line in fasta does not contain identifier"
                    )
                else:
                    fasta_map[seqid] += line.strip().upper()
    return fasta_map


def build_tree(
    newickstring,
    fasta_map,
    newickformat=1,
    reference_id=None,
    reference_sequence=None,
    ignore_internal_sequences=False,
):
    """Load an ete tree from a newick string, and add 'sequence' attributes
    from fasta.

    If internal node sequences aren't specified in the newick string and fasta data,
    internal node sequences will be fully ambiguous (contain repeated N's).

    Arguments:
        newickstring: a newick string
        fasta_map: a dictionary with sequence id keys matching node names in `newickstring`, and sequence values.
        newickformat: the ete format identifier for the passed newick string. See ete docs
        reference_id: key in `fasta_map` corresponding to the root sequence of the tree, if root sequence is fixed.
        reference_sequence: fixed root sequence of the tree, if root sequence is fixed
        ignore_internal_sequences: if True, sequences at non-leaf nodes specified in fasta_map
        and newickstring will be ignored, and all internal sequences will be fully ambiguous.
    """
    tree = ete3.Tree(newickstring, format=newickformat)
    # all fasta entries should be same length
    seq_len = len(next(iter(fasta_map.values())))
    ambig_seq = "?" * seq_len
    for node in tree.traverse():
        if node.is_root() and reference_sequence is not None:
            node.add_feature("sequence", reference_sequence)
        elif node.is_root() and reference_id is not None:
            node.add_feature("sequence", fasta_map[reference_id])
        elif (not node.is_leaf()) and ignore_internal_sequences:
            node.add_feature("sequence", ambig_seq)
        elif node.name in fasta_map:
            node.add_feature("sequence", fasta_map[node.name])
        else:
            node.add_feature("sequence", ambig_seq)
    return tree


def build_trees_from_files(newickfiles, fastafile, **kwargs):
    """Same as `build_tree`, but takes a list of filenames containing newick
    strings, and a filename for a fasta file, and returns a generator on
    trees."""
    fasta_map = load_fasta(fastafile)
    for newick in newickfiles:
        with open(newick, "r") as fh:
            newick = fh.read()
        yield build_tree(newick, fasta_map, **kwargs)


def parsimony_score(tree):
    """returns the parsimony score of a (disambiguated) ete tree.

    Tree must have 'sequence' attributes on all nodes.
    """
    return sum(
        hamming_distance(node.up.sequence, node.sequence)
        for node in tree.iter_descendants()
    )


def remove_invariant_sites(fasta_map):
    """Eliminate invariant characters in a fasta alignment and record the
    0-based indices of those variant sites."""
    # eliminate characters for which there's no diversity:
    informative_sites = [
        idx for idx, chars in enumerate(zip(*fasta_map.values())) if len(set(chars)) > 1
    ]
    newfasta = {
        key: "".join(oldseq[idx] for idx in informative_sites)
        for key, oldseq in fasta_map.items()
    }
    return newfasta, informative_sites


def parsimony_scores_from_topologies(
    newicks, fasta_map, gap_as_char=False, remove_invariants=False, **kwargs
):
    """returns a generator on parsimony scores of trees specified by newick
    strings and fasta. additional keyword arguments are passed to `build_tree`.

    Args:
        newicks: newick strings of trees whose parsimony scores will be computed
        fasta_map: fasta data as a dictionary, as output by ``load_fasta``
        gap_as_char: if True, gap characters `-` will be treated as a fifth character.
            Otherwise, they will be synonymous with `N`'s.
        remove_invariants: if True, removes invariant sites from the provided fasta_map
    """
    if remove_invariants:
        print("removing invariant sites...")
        newfasta = remove_invariant_sites(fasta_map)
        print("...finished removing invariant sites")
    else:
        newfasta = fasta_map
    yield from (
        sankoff_upward(build_tree(newick, newfasta, **kwargs), gap_as_char=gap_as_char)
        for newick in newicks
    )


def parsimony_scores_from_files(treefiles, fastafile, **kwargs):
    """returns the parsimony scores of trees specified by newick files and a
    fasta file.

    Arguments match `build_trees_from_files`. Additional keyword
    arguments are passed to ``parsimony_scores_from_topologies``.
    """

    def load_newick(npath):
        with open(npath, "r") as fh:
            return fh.read()

    return parsimony_scores_from_topologies(
        (load_newick(path) for path in treefiles), load_fasta(fastafile), **kwargs
    )


def build_dag_from_trees(trees):
    """Build a history DAG from trees containing a `sequence` attribute on all
    nodes.

    unifurcations in the provided trees will be deleted.
    """
    trees = [tree.copy() for tree in trees]
    for tree in trees:
        for node in tree.iter_descendants():
            if len(node.children) == 1:
                node.delete(prevent_nondicotomic=False)
        if len(tree.children) == 1:
            newchild = tree.add_child()
            newchild.add_feature("sequence", tree.sequence)
    return history_dag_from_etes(
        trees,
        ["sequence"],
    )


def summarize_dag(dag):
    """print summary information about the provided history DAG."""
    print("DAG contains")
    print("trees: ", dag.count_trees())
    print("nodes: ", len(list(dag.preorder())))
    print("edges: ", sum(len(list(node.children())) for node in dag.preorder()))
    print("parsimony scores: ", dag.weight_count())


def disambiguate_history(history):
    """A rather ugly way to disambiguate a history, by converting to ete, using
    the disambiguate method, then converting back to HistoryDag."""
    seq_len = len(next(history.get_leaves()).label.sequence)
    ambig_seq = "N" * seq_len
    etetree = history.to_ete(
        feature_funcs={
            "sequence": (lambda n: n.label.sequence if n.is_leaf() else ambig_seq)
        }
    )
    disambiguate(etetree, min_ambiguities=True)
    rhistory = history_dag_from_etes([etetree], ["sequence"])
    return rhistory


def treewise_sankoff_in_dag(dag, cover_edges=False):
    """Perform tree-wise sankoff to compute labels for all nodes in the DAG."""
    newdag = history_dag_from_clade_trees(
        disambiguate_history(history) for history in dag.iter_covering_histories(cover_edges=cover_edges)
    )
    newdag.explode_nodes()
    newdag.add_all_allowed_edges()
    newdag.trim_optimal_weight()
    newdag.convert_to_collapsed()
    return newdag
