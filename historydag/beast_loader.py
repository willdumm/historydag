import historydag as hdag
from warnings import warn
from functools import lru_cache
import dendropy
import xml.etree.ElementTree as ET
from historydag.parsimony import ambiguous_dna_values


ambiguous_dna_values = ambiguous_dna_values.copy()
# Set '-' to be equivalent to 'N'.
ambiguous_dna_values.update({"?": "GATC", "-": "GATC"})
character_lookup = {
    frozenset(char_set): character
    for character, char_set in ambiguous_dna_values.items()
}


def dag_from_beast_trees(
    beast_xml_file,
    beast_output_file,
    reference_sequence=None,
    mask_ambiguous_sites=True,
):
    """A convenience method to build a dag out of the output from
    :meth:`load_beast_trees`."""
    dp_trees = load_beast_trees(
        beast_xml_file,
        beast_output_file,
        reference_sequence=reference_sequence,
        mask_ambiguous_sites=mask_ambiguous_sites,
    )
    dag = hdag.history_dag_from_trees(
        [tree.seed_node for tree in dp_trees],
        [],
        label_functions={
            "compact_genome": lambda n: n.cg,
        },
        attr_func=lambda n: {"name": (n.taxon.label if n.is_leaf() else "internal")},
        child_node_func=dendropy.Node.child_nodes,
        leaf_node_func=dendropy.Node.leaf_iter,
    )
    return hdag.mutation_annotated_dag.CGHistoryDag.from_history_dag(dag)


def load_beast_trees(
    beast_xml_file,
    beast_output_file,
    reference_sequence=None,
    mask_ambiguous_sites=True,
):
    """Load trees from BEAST output.

    Loads trees from BEAST output, in which each node has a `history_all` attribute
    containing the mutations inferred along that node's parent branch.

    Args:
        beast_xml_file: The xml input file to BEAST
        beast_output_file: The .trees output file from BEAST
        reference_sequence: If provided, a reference sequence which will be used for all
            compact genomes. By default, uses the ancestral sequence of the first tree.
        mask_ambiguous_sites: If True, ignore mutations for all sites whose observed set
            of characters is a subset of {N, -, ?} (recommended).

    Returns:
        A :class:`dendropy.TreeList` containing the trees output by BEAST. Each tree has:
        * ancestral sequence attribute on each tree object, containing the complete reference
            for that tree
        * cg attribute on all nodes, containing a compact genome relative to the reference
            sequence
        * mut attribute on all nodes containing a list of mutations on parent branch, in
            order of occurrence
    """
    # get alignment from xml:
    _etree = ET.parse("clade_13.GTR.xml")
    _alignment = _etree.getroot().find("alignment")
    unmasked_fasta = {
        a[0].attrib["idref"].strip(): a[0].tail.strip() for a in _alignment
    }
    masked_sites = {
        i
        for i in range(len(next(iter(unmasked_fasta.values()))))
        if len({seq[i] for seq in unmasked_fasta.values()} - {"N", "?"}) == 0
    }

    def mask_sequence(unmasked):
        return "".join(char for i, char in enumerate(unmasked) if i not in masked_sites)

    fasta = {key: mask_sequence(val) for key, val in unmasked_fasta.items()}

    # dendropy doesn't parse nested lists correctly in metadata, so we load the
    # trees with raw comment strings using `extract_comment_metadata`
    dp_trees = dendropy.TreeList.get(
        path="clade_13.GTR.history.trees",
        schema="nexus",
        extract_comment_metadata=False,
    )

    for tree in dp_trees:
        for node in tree.postorder_node_iter():
            node.muts = list(_comment_parser(node.comments))
        tree.ancestral_sequence = _recover_reference(tree, fasta)

    if reference_sequence is None:
        reference_sequence = dp_trees[0].ancestral_sequence

    def compute_cgs(tree):
        if mask_ambiguous_sites:
            extra_masked_sites = {
                i
                for i in range(len(next(iter(fasta.values()))))
                if len({seq[i] for seq in fasta.values()} - {"N", "?", "-"}) == 0
            }

            def cg_transform(cg):
                return cg.mask_sites(extra_masked_sites, one_based=False)

        else:

            def cg_transform(cg):
                return cg

        ancestral_cg = hdag.compact_genome.compact_genome_from_sequence(
            tree.ancestral_sequence, reference_sequence
        )

        @lru_cache(maxsize=(2 * len(dp_trees[0].nodes())))
        def compute_cg(node):
            if node.parent_node is None:
                # base case: node is a root node
                parent_cg_mut = ancestral_cg
            else:
                parent_cg_mut = compute_cg(node.parent_node)
            return parent_cg_mut.apply_muts(node.muts)

        for node in tree.preorder_node_iter():
            node.cg = cg_transform(compute_cg(node))

    for tree in dp_trees:
        compute_cgs(tree)

    return dp_trees


def _comment_parser(node_comments):
    if len(node_comments) == 0:
        yield from ()
        return
    elif len(node_comments) == 1:
        comment_string = node_comments[0]
    else:
        raise ValueError("node_comments has more than one element" + str(node_comments))
    if "history_all=" not in comment_string:
        yield from ()
        return
    else:
        mutations_string = comment_string.split("history_all=")[-1]
        stripped_mutations_list = mutations_string[2:-2]
        if stripped_mutations_list:
            mutations_list = stripped_mutations_list.split("},{")
            for mut in mutations_list:
                try:
                    idx_str, _, from_base, to_base = mut.split(",")
                except ValueError:
                    raise ValueError("comment_parser failed on: " + str(node_comments))
                assert to_base in "AGCT"
                yield from_base + idx_str + to_base
        else:
            yield from ()
            return


def _recover_reference(tree, fasta):
    sequence_dict = {}

    def mut_upward_child(c_node):
        for mut in reversed(c_node.muts):
            upbase = mut[0]
            downbase = mut[-1]
            site = int(mut[1:-1]) - 1
            if downbase not in ambiguous_dna_values[sequence_dict[c_node][site]]:
                warn("child base doesn't match mut base")
            sequence_dict[c_node][site] = upbase

    for node in tree.postorder_node_iter():
        if node.is_leaf():
            sequence_dict[node] = list(fasta[node.taxon.label])
        else:
            children = node.child_nodes()
            mut_upward_child(children[0])
            sequence = sequence_dict[children[0]]
            for child in children[1:]:
                mut_upward_child(child)
                for site, (obase, nbase) in enumerate(
                    zip(sequence, sequence_dict[child])
                ):
                    intersection = frozenset(ambiguous_dna_values[obase]) & frozenset(
                        ambiguous_dna_values[nbase]
                    )
                    if len(intersection) == 0:
                        warn(
                            "conflicting base found between children, using base from first child"
                        )
                    else:
                        sequence[site] = character_lookup[intersection]
            sequence_dict[node] = sequence
    mut_upward_child(tree.seed_node)
    return "".join(sequence_dict[tree.seed_node])
