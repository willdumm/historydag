"""A module implementing Sankoff Algorithm."""
import random
import ete3
import numpy as np
from itertools import product
from historydag.dag import (
    history_dag_from_histories,
    history_dag_from_etes,
    HistoryDag,
    utils,
)
from copy import deepcopy
import historydag.parsimony_utils as parsimony_utils


def replace_label_attr(original_label, list_of_replacements={}):
    """Generalizes :meth: ``_replace()`` for namedtuple datatype to replace
    multiple fields at once, and by string rather than as a keyword argument.

    Caveat: the keys of ``list_of_replacements`` dict should be existing fields in the namedtuple object for ``original_label``
    """
    fields = original_label._asdict()
    fields.update(list_of_replacements)
    return type(original_label)(**fields)


def make_weighted_hamming_count_funcs(
    transition_model=parsimony_utils.default_nt_transitions,
    allow_ambiguous_leaves=False,
    sequence_label="sequence",
):
    """Returns an :class:`AddFuncDict` for computing weighted parsimony.

    The returned ``AddFuncDict`` is for use with :class:`HistoryDag` objects whose nodes
    have unambiguous sequences stored in label attributes named ``sequence``.

    Args:

    Returns:
        :class:`AddFuncDict` object for computing weighted parsimony.
    """

    if allow_ambiguous_leaves:
        weight_func = transition_model.min_weighted_hamming_edge_weight(sequence_label)
    else:
        weight_func = transition_model.weighted_hamming_edge_weight(sequence_label)
    return utils.AddFuncDict(
        {
            "start_func": lambda n: 0,
            "edge_weight_func": weight_func,
            "accum_func": sum,
        },
        name="WeightedParsimony",
    )


def sankoff_postorder_iter_accum(
    postorder_iter, node_clade_function, child_node_function
):
    """this is a re-take of :meth:`postorder_history_accum` that is altered so
    that it does not require a complete DAG, and simplified for the specific
    Sankoff algorithm.

    Args:
        postorder_iter: iterable of nodes to traverse.
        node_clade_function: function to combine results for a given clade.
        child_node_function: function that is applied to children for a given node.
    """
    if any(postorder_iter):
        for node in postorder_iter:
            if node.is_leaf():
                node._dp_data = {
                    "cost_vectors": child_node_function(node),
                    "subtree_cost": 0,
                }
            else:
                node._dp_data = node_clade_function(
                    [
                        [
                            child_node_function(target)
                            for target in node.children(clade=clade)
                        ]
                        for clade in node.clades
                    ]
                )
        return node._dp_data
    return {"cost_vectors": [], "subtree_cost": 0}


def sankoff_upward(
    node_list,
    seq_len,
    sequence_attr_name="sequence",
    transition_model=parsimony_utils.default_nt_transitions,
    use_internal_node_sequences=False,
):
    """Compute Sankoff cost vectors at nodes in a postorder traversal, and
    return best possible parsimony score of the tree.

    Args:
        node_list: An iterator for the set of nodes to compute Sankoff on. Can be of type ``ete3.TreeNode``
            or of type ``HistoryDag``, or else a list of nodes that respects postorder traversal. This last
            option is useful when using the Sankoff algorithm to compute node labels for a subset of the
            internal nodes of a history DAG, as only the nodes in the list will be relabelled. However it
            does assume that the remaining internal nodes are in fact equipped with a sequence.
        seq_len: The length of sequence for each leaf node.
        sequence_attr_name: the field name of the label attribute that is used to store sequences on nodes.
            Default value is set to 'sequence'
        transition_model: The type of ``TransitionModel`` to use for Sankoff. See docstring for more information.
        use_internal_node_sequences: (used when tree is of type ``ete3.TreeNode``) If True, then compute the
            transition cost for sequences assigned to internal nodes.
    """

    adj_arr = transition_model.get_adjacency_array(seq_len)
    if isinstance(node_list, ete3.TreeNode):
        # First pass of Sankoff: compute cost vectors
        for node in node_list.traverse(strategy="postorder"):
            node.add_feature(
                "cost_vector",
                np.array(
                    [
                        transition_model.mask_vectors[base].copy()
                        for base in getattr(node, sequence_attr_name)
                    ]
                ),
            )
            if not node.is_leaf():
                child_costs = []
                for child in node.children:
                    stacked_child_cv = np.stack(
                        (child.cost_vector,) * len(transition_model.bases), axis=1
                    )
                    total_cost = adj_arr + stacked_child_cv
                    child_costs.append(np.min(total_cost, axis=2))
                child_cost = np.sum(child_costs, axis=0)
                node.cost_vector = child_cost
            if use_internal_node_sequences:
                node.cost_vector += np.array(
                    [
                        transition_model.mask_vectors[base]
                        for base in getattr(node, sequence_attr_name)
                    ]
                )
        return np.sum(np.min(node_list.cost_vector, axis=1))

    elif isinstance(node_list, HistoryDag):
        node_list = list(node_list.postorder())
    if isinstance(node_list, list):
        sequence_attr_idx = node_list[0].label._fields.index(sequence_attr_name)
        max_transition_cost = np.amax(adj_arr) * seq_len

        def children_cost(child_cost_vectors):
            costs = []
            for c in child_cost_vectors:
                cost = adj_arr + np.stack((c,) * len(transition_model.bases), axis=1)
                costs.append(np.min(cost, axis=2))
            return np.sum(costs, axis=0)

        def cost_vector(node):
            if node.is_leaf():
                return [
                    np.array(
                        [
                            transition_model.mask_vectors[base].copy()
                            for base in node.label[sequence_attr_idx]
                        ]
                    )
                ]
            elif isinstance(node._dp_data, dict):
                return node._dp_data["cost_vectors"]
            else:
                return [
                    np.array(
                        [
                            transition_model.mask_vectors[base].copy()
                            for base in node.label[sequence_attr_idx]
                        ]
                    )
                ]

        def accum_between_clade(list_of_clade_cvs):
            cost_vectors = []
            min_cost = float("inf")
            for choice in product(*list_of_clade_cvs):
                for cost_vector_combination in product(*[c for c in choice]):
                    cv = children_cost(cost_vector_combination)
                    cost = np.sum(np.min(cv, axis=1))
                    if (cost + max_transition_cost) < min_cost:
                        min_cost = cost
                        cost_vectors = [cv]
                    elif cost <= (min_cost + max_transition_cost) and not any(
                        [np.array_equal(cv, other_cv) for other_cv in cost_vectors]
                    ):
                        min_cost = min(cost, min_cost)
                        cost_vectors.append(cv)
            return {"cost_vectors": cost_vectors, "subtree_cost": min_cost}

        compute_val = sankoff_postorder_iter_accum(
            node_list, accum_between_clade, cost_vector
        )
        return compute_val["subtree_cost"]
    else:
        return 0


def sankoff_downward(
    dag,
    partial_node_list=None,
    sequence_attr_name="sequence",
    compute_cvs=True,
    transition_model=parsimony_utils.default_nt_transitions,
    trim=True,
):
    """Assign sequences to internal nodes of dag using a weighted Sankoff
    algorithm by exploding all possible labelings associated to each internal
    node based on its subtrees.

    Args:
        partial_node_list: If None, all internal node sequences will be re-computed. If a list of nodes
            is provided, only sequences on nodes from that list will be updated. This list must be in
            post-order.
        sequence_attr_name: The field name of the label attribute that is used to store sequences on nodes.
            Default value is set to 'sequence'
        compute_cvs: If true, compute upward sankoff cost vectors. If ``sankoff_upward`` was
            already run on the tree/dag, this may be skipped.
        transition_model: The type of ``TransitionModel`` to use for Sankoff. See docstring for more information.
        trim: If False, the history DAG will not be trimmed to express only maximally parsimonious
            histories after Sankoff.
    """

    sequence_attr_idx = next(dag.get_leaves()).label._fields.index(sequence_attr_name)
    seq_len = len(next(dag.get_leaves()).label[sequence_attr_idx])
    if partial_node_list is None:
        partial_node_list = list(dag.postorder())
    if compute_cvs:
        sankoff_upward(
            partial_node_list,
            seq_len,
            sequence_attr_name=sequence_attr_name,
            transition_model=transition_model,
        )
    # save the field names/types of the label datatype for this dag
    adj_arr = transition_model.get_adjacency_array(seq_len)

    def transition_cost(seq):
        return adj_arr[
            np.arange(len(seq)), [transition_model.base_indices[s] for s in seq]
        ]

    def compute_sequence_data(cost_vector):
        """Compute all possible sequences that minimize transition costs as given by cost_vector.
        Returns: a list of tuples, each tuple containing:
                  - a sequence
                  - an adjacency array of costs for that sequence, and
                  - the minimum cost associated to the cost vector
        """
        all_base_indices = [[]]
        min_cost = sum(np.min(cost_vector, axis=1))
        for idx in range(seq_len):
            min_cost_indices = np.where(cost_vector[idx] == cost_vector[idx].min())[0]
            all_base_indices = [
                base_idx + [i]
                for base_idx in all_base_indices
                for i in min_cost_indices
            ]
        adj_vec = [
            adj_arr[np.arange(seq_len), base_indices]
            for base_indices in all_base_indices
        ]
        new_sequence = [
            "".join([transition_model.bases[base_index] for base_index in base_indices])
            for base_indices in all_base_indices
        ]
        return list(zip(new_sequence, adj_vec, [min_cost] * len(new_sequence)))

    dag_nodes = {}
    # downward pass of Sankoff: find and assign sequence labels to each internal node
    for node in reversed(partial_node_list):
        if not (node.is_leaf() or node.is_ua_node()):
            node_data = {k: v for k, v in node._dp_data.items()}
            node_copies = {}
            node_children = set(node.children())
            node_parents = set(node.parents)
            for p in node_parents:
                if p.is_ua_node():
                    new_seq_data = [
                        y
                        for cv in node._dp_data["cost_vectors"]
                        for y in compute_sequence_data(cv)
                    ]
                else:
                    new_seq_data = [
                        y
                        for cv in node._dp_data["cost_vectors"]
                        for y in compute_sequence_data(
                            cv + transition_cost(p.label[sequence_attr_idx])
                        )
                    ]

                # only keep those node/parent/cost_vector choices that
                # achieve a minimal cost for the node/parent choice
                min_val = new_seq_data[0][-1]
                if len(new_seq_data) > 1:
                    min_val = min(*list(zip(*new_seq_data))[-1])
                for nsd in new_seq_data:
                    if (nsd[-1] <= min_val) and (nsd[0] not in node_copies):
                        new_node = node.empty_copy()
                        new_node.label = replace_label_attr(
                            node.label, {sequence_attr_name: nsd[0]}
                        )
                        new_node._dp_data = deepcopy(node_data)
                        node_copies[nsd[0]] = new_node
            for c in node_children:
                c.parents.remove(node)
            for p in node_parents:
                p.remove_edge_by_clade_and_id(node, node.clade_union())
            # add all new copies of current node(with alt sequence labels) into the dag
            for new_sequence, new_node in node_copies.items():
                if new_node in dag_nodes:
                    new_node = dag_nodes[new_node]
                for c in node_children:
                    new_node.add_edge(c)
                    c.parents.add(new_node)
                for parent in node_parents:
                    parent.add_edge(new_node)
                new_node.parents.update(node_parents)
                dag_nodes[new_node] = new_node

    dag.recompute_parents()
    # still need to trim the dag since the final addition of all
    # parents/children to new nodes can yield suboptimal choices

    weight_func = utils.access_nodefield_default(sequence_attr_name, 0)(
        transition_model.weighted_hamming_distance
    )
    if trim:
        optimal_weight = dag.trim_optimal_weight(edge_weight_func=weight_func)
    else:
        optimal_weight = dag.optimal_weight_annotate(edge_weight_func=weight_func)
    return optimal_weight


def disambiguate(
    tree,
    compute_cvs=True,
    random_state=None,
    remove_cvs=False,
    adj_dist=False,
    transition_model=parsimony_utils.default_nt_transitions,
    min_ambiguities=False,
    use_internal_node_sequences=False,
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
        min_ambiguities: If True, leaves ambiguities in reconstructed sequences, expressing which
            bases are possible at each site in a maximally parsimonious disambiguation of the given
            topology. In the history DAG paper, this is known as a strictly min-weight ambiguous labeling.
            Otherwise, sequences are resolved in one possible way, and include no ambiguities.
    """
    if random_state is None:
        random.seed(tree.write(format=1))
    else:
        random.setstate(random_state)

    seq_len = len(next(tree.iter_leaves()).sequence)
    adj_arr = transition_model.get_adjacency_array(seq_len)
    if compute_cvs:
        sankoff_upward(
            tree,
            len(tree.sequence),
            transition_model=transition_model,
            use_internal_node_sequences=use_internal_node_sequences,
        )
    # Second pass of Sankoff: choose bases
    preorder = list(tree.traverse(strategy="preorder"))
    for node in preorder:
        if min_ambiguities:
            adj_vec = node.cost_vector != np.stack(
                (node.cost_vector.min(axis=1),) * len(transition_model.bases), axis=1
            )
            new_seq = [
                transition_model.get_ambiguity_from_tuple(tuple(map(float, row)))
                for row in adj_vec
            ]
        else:
            base_indices = []
            for idx in range(seq_len):
                min_cost = min(node.cost_vector[idx])
                base_index = random.choice(
                    [
                        i
                        for i, val in enumerate(node.cost_vector[idx])
                        if val == min_cost
                    ]
                )
                base_indices.append(base_index)

            adj_vec = adj_arr[np.arange(seq_len), base_indices]
            new_seq = [
                transition_model.bases[base_index] for base_index in base_indices
            ]

        # Adjust child cost vectors
        for child in node.children:
            child.cost_vector += adj_vec
        node.sequence = "".join(new_seq)

    if remove_cvs:
        for node in tree.traverse():
            try:
                node.del_feature("cost_vector")
            except (AttributeError, KeyError):
                pass
    if adj_dist:
        tree.dist = 0
        for node in tree.iter_descendants():
            node.dist = transition_model.weighted_hamming_distance(
                node.up.sequence, node.sequence
            )
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


def parsimony_score(tree, transition_model=parsimony_utils.default_nt_transitions):
    """returns the parsimony score of a (disambiguated) ete tree.

    Tree must have 'sequence' attributes on all nodes.
    """
    return sum(
        transition_model.weighted_hamming_distance(node.up.sequence, node.sequence)
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
    newdag = history_dag_from_histories(
        disambiguate_history(history)
        for history in dag.iter_covering_histories(cover_edges=cover_edges)
    )
    newdag.explode_nodes()
    newdag.make_complete()
    newdag.trim_optimal_weight()
    newdag.convert_to_collapsed()
    return newdag
