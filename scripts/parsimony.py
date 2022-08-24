import random
import click
from collections import Counter
from itertools import product
import ete3
import historydag as hdag
import pickle
import numpy as np
import Bio.Data.IUPACData

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


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def _cli():
    """
    A collection of tools for calculating parsimony scores of newick trees, and
    using them to create a history DAG
    """
    pass


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
    """Compute Sankoff cost vectors at nodes in a postorder traversal,
    and return best possible parsimony score of the tree or DAG.

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

    if isinstance(tree, ete3.Tree):
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

    elif isinstance(tree, hdag.HistoryDag):
        adj_arr = _get_adj_array(len(next(tree.postorder()).label.sequence), transition_weights=transition_weights)

        def children_cost(child_cost_vectors):
            costs=[]
            for c in child_cost_vectors:
                cost = adj_arr + np.stack((c,) * 5, axis=1)
                costs.append(np.min(cost, axis=2))
            return np.sum(costs, axis=0)

        def leaf_func(n):
            return {
                "cost_vectors": [np.array([code_vectors[translate_base(base)].copy() for base in n.label.sequence])],
                "paired_children": [],
                "subtree_cost": 0
            }

        def accum_between_clade(clade_data):
            cost_vectors = []
            paired_children = []
            min_cost = float("inf")
            # iterate over each possible combination of edge choice across clades
            for choice in product(*clade_data):
                # compute every possible combination of cost vectors for the given edge choice 
                # (each child node has possibly multiple cost vectors that are all optimal)
                for cost_vector_combination in product(*[c._dp_data["cost_vectors"] for c in choice]):
                    cv = children_cost(cost_vector_combination)
                    cost = np.sum(np.min(cv, axis=1))
                    if cost < min_cost:
                        min_cost = cost
                        cost_vectors = [cv]
                        paired_children = [(c for c in choice)]
                    elif cost <= min_cost and not any([np.array_equal(cv, cv2) for cv2 in cost_vectors]):
                        cost_vectors.append(cv)
                        paired_children.append((c for c in choice))
            return {"cost_vectors": cost_vectors,
                    "paired_children": paired_children,
                    "subtree_cost": min_cost}
        tree.postorder_cladetree_accum(
                leaf_func=leaf_func,
                edge_func=lambda x, y: y,
                accum_within_clade=lambda x: x,
                accum_between_clade=accum_between_clade,
                accum_above_edge=lambda x, y: y)
        return next(tree.preorder())._dp_data["subtree_cost"]

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
    """Load a fasta file as a dictionary, with sequence ids as keys and sequences as values."""
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
    """Load an ete tree from a newick string, and add 'sequence' attributes from fasta.

    If internal node sequences aren't specified in the newick string and fasta data,
    internal node sequences will be fully ambiguous (contain repeated N's).

    Arguments:
        newickstring: a newick string
        fasta_map: a dictionary with sequence id keys matching node names in `newickstring`, and sequence values.
        newickformat: the ete format identifier for the passed newick string. See ete docs
        reference_id: key in `fasta_map` corresponding to the root sequence of the tree, if root sequence is fixed.
        reference_sequence: fixed root sequence of the tree, if root sequence is fixed
        ignore_internal_sequences: if True, sequences at non-leaf nodes specified in fasta_map and newickstring will be ignored, and all internal sequences will be fully ambiguous."""
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
    """Same as `build_tree`, but takes a list of filenames containing newick strings, and a filename for a fasta file, and returns a generator on trees"""
    fasta_map = load_fasta(fastafile)
    trees = []
    for newick in newickfiles:
        with open(newick, "r") as fh:
            newick = fh.read()
        yield build_tree(newick, fasta_map, **kwargs)


def parsimony_score(tree):
    """returns the parsimony score of a (disambiguated) ete tree.
    Tree must have 'sequence' attributes on all nodes."""
    return sum(
        hamming_distance(node.up.sequence, node.sequence)
        for node in tree.iter_descendants()
    )


def remove_invariant_sites(fasta_map):
    """Eliminate invariant characters in a fasta alignment"""
    # eliminate characters for which there's no diversity:
    informative_sites = [
        idx for idx, chars in enumerate(zip(*fasta_map.values())) if len(set(chars)) > 1
    ]
    newfasta = {
        key: "".join(oldseq[idx] for idx in informative_sites)
        for key, oldseq in fasta_map.items()
    }
    return newfasta


@_cli.command("remove-invariants")
@click.argument("in_fasta")
@click.argument("out_fasta")
def _cli_remove_invariant_sites(in_fasta, out_fasta):
    fasta_map = load_fasta(in_fasta)
    newfasta = remove_invariant_sites(fasta_map)
    with open(out_fasta, "w") as fh:
        for seqid, seq in newfasta.items():
            print(">" + seqid, file=fh)
            print(seq, file=fh)


def parsimony_scores_from_topologies(
    newicks, fasta_map, gap_as_char=False, remove_invariants=False, **kwargs
):
    """returns a generator on parsimony scores of trees specified by newick strings and fasta.
    additional keyword arguments are passed to `build_tree`.

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
    """returns the parsimony scores of trees specified by newick files and a fasta file.
    Arguments match `build_trees_from_files`. Additional keyword arguments are passed to
    ``parsimony_scores_from_topologies``."""

    def load_newick(npath):
        with open(npath, "r") as fh:
            return fh.read()

    return parsimony_scores_from_topologies(
        (load_newick(path) for path in treefiles), load_fasta(fastafile), **kwargs
    )


def build_dag_from_trees(trees):
    """Build a history DAG from trees containing a `sequence` attribute on all nodes.
    unifurcations in the provided trees will be deleted."""
    trees = [tree.copy() for tree in trees]
    for tree in trees:
        for node in tree.iter_descendants():
            if len(node.children) == 1:
                node.delete(prevent_nondicotomic=False)
        if len(tree.children) == 1:
            newchild = tree.add_child()
            newchild.add_feature("sequence", tree.sequence)
    return hdag.history_dag_from_etes(
        trees,
        ["sequence"],
    )


@_cli.command("build-trees")
@click.argument("treefiles", nargs=-1)
@click.option(
    "-f",
    "--fasta-file",
    required=True,
    help="Filename of a fasta file containing sequences appearing on nodes of newick tree",
)
@click.option(
    "-r",
    "--root-id",
    default=None,
    help="The fasta identifier of the fixed root of provided trees. May be omitted if there is no fixed root sequence.",
)
@click.option(
    "-F",
    "--newick-format",
    default=1,
    help="Newick format of the provided newick file. See http://etetoolkit.org/docs/latest/reference/reference_tree.html#ete3.TreeNode",
)
@click.option(
    "-i",
    "--include-internal-sequences",
    is_flag=True,
    help="include non-leaf node labels, and associated sequences in the fasta file.",
)
@click.option(
    "-g",
    "--gap-as-char",
    is_flag=True,
    help="Treat gap character `-` as a fifth character. Otherwise treated as ambiguous `N`.",
)
@click.option(
    "-a",
    "--preserve-ambiguities",
    is_flag=True,
    help="Do not disambiguate fully, but rather preserve ambiguities to express all maximally parsimonious assignments at each site.",
)
@click.option(
    "-o", "--outdir", default=".", help="Directory in which to write pickled trees."
)
@click.option(
    "-c",
    "--clean-trees",
    is_flag=True,
    help="remove cost vectors from tree, resulting in smaller pickled tree files",
)
def _cli_build_trees(
    treefiles,
    fasta_file,
    root_id,
    newick_format,
    include_internal_sequences,
    gap_as_char,
    preserve_ambiguities,
    outdir,
    clean_trees,
):
    trees = build_trees_from_files(
        treefiles,
        fasta_file,
        reference_id=root_id,
        ignore_internal_sequences=(not include_internal_sequences),
    )
    trees = (
        disambiguate(
            tree,
            gap_as_char=gap_as_char,
            min_ambiguities=preserve_ambiguities,
            remove_cvs=clean_trees,
        )
        for tree in trees
    )
    for idx, tree in enumerate(trees):
        print("saving tree")
        with open(outdir + f"tree_{idx}.p", "wb") as fh:
            fh.write(pickle.dumps(tree))
        print("saved tree")


@_cli.command("parsimony_scores")
@click.argument("treefiles", nargs=-1)
@click.option(
    "-f",
    "--fasta-file",
    required=True,
    help="Filename of a fasta file containing sequences appearing on nodes of newick tree",
)
@click.option(
    "-r",
    "--root-id",
    default=None,
    help="The fasta identifier of the fixed root of provided trees. May be omitted if there is no fixed root sequence.",
)
@click.option(
    "-F",
    "--newick-format",
    default=1,
    help="Newick format of the provided newick file. See http://etetoolkit.org/docs/latest/reference/reference_tree.html#ete3.TreeNode",
)
@click.option(
    "-i",
    "--include-internal-sequences",
    is_flag=True,
    help="include non-leaf node labels, and associated sequences in the fasta file.",
)
@click.option(
    "-d",
    "--save-to-dag",
    default=None,
    help="Combine loaded and disambiguated trees into a history DAG, and save pickled DAG to provided path.",
)
@click.option(
    "-g",
    "--gap-as-char",
    is_flag=True,
    help="Treat gap character `-` as a fifth character. Otherwise treated as ambiguous `N`.",
)
@click.option(
    "-a",
    "--preserve-ambiguities",
    is_flag=True,
    help="Do not disambiguate fully, but rather preserve ambiguities to express all maximally parsimonious assignments at each site.",
)
def _cli_parsimony_score_from_files(
    treefiles,
    fasta_file,
    root_id,
    newick_format,
    include_internal_sequences,
    save_to_dag,
    gap_as_char,
    preserve_ambiguities,
):
    """Print the parsimony score of one or more newick files"""
    pscore_gen = parsimony_scores_from_files(
        treefiles,
        fasta_file,
        reference_id=root_id,
        gap_as_char=gap_as_char,
        ignore_internal_sequences=(not include_internal_sequences),
        remove_invariants=True,
    )

    for score, treepath in zip(pscore_gen, treefiles):
        print(treepath)
        print(score)

    if save_to_dag is not None:
        trees = build_trees_from_files(
            treefiles,
            fasta_file,
            reference_id=root_id,
            ignore_internal_sequences=(not include_internal_sequences),
        )
        trees = (
            disambiguate(
                tree, gap_as_char=gap_as_char, min_ambiguities=preserve_ambiguities
            )
            for tree in trees
        )
        dag = build_dag_from_trees(trees)
        with open(save_to_dag, "wb") as fh:
            fh.write(pickle.dumps(dag))


def summarize_dag(dag):
    """print summary information about the provided history DAG."""
    print("DAG contains")
    print("trees: ", dag.count_trees())
    print("nodes: ", len(list(dag.preorder())))
    print("edges: ", sum(len(list(node.children())) for node in dag.preorder()))
    print("parsimony scores: ", dag.weight_count())


@_cli.command("summarize-dag")
@click.argument("dagpath")
def _cli_summarize_dag(dagpath):
    """print summary information about the provided history DAG."""
    with open(dagpath, "rb") as fh:
        dag = pickle.load(fh)
    summarize_dag(dag)


if __name__ == "__main__":
    _cli()
