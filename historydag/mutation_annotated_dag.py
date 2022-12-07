"""This module allows the loading and manipulation of Larch mutation annotated
DAG protobuf files.

The resulting history DAG contains labels with 'compact genomes', and a
'refseq' attribute describing a reference sequence and set of mutations
relative to the reference.
"""

import functools
from frozendict import frozendict
from historydag.dag import HistoryDag, HistoryDagNode, UANode, EdgeSet
import historydag.utils
from historydag.utils import Weight
from historydag.compact_genome import (
    CompactGenome,
    compact_genome_from_sequence,
    wrapped_cg_hamming_distance,
    cg_diff,
)
import historydag.dag_pb2 as dpb
import json
from typing import NamedTuple

_pb_nuc_lookup = {0: "A", 1: "C", 2: "G", 3: "T"}
_pb_nuc_codes = {nuc: code for code, nuc in _pb_nuc_lookup.items()}


def _pb_mut_to_str(mut):
    """Unpack protobuf-encoded mutation into 1-indexed mutations string."""
    return (
        _pb_nuc_lookup[mut.par_nuc] + str(mut.position) + _pb_nuc_lookup[mut.mut_nuc[0]]
    )


class HDagJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, CompactGenome):
            return dict(obj.mutations)
        elif isinstance(obj, frozendict):
            return dict(obj)
        elif isinstance(obj, frozenset):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


class CGHistoryDag(HistoryDag):
    """A HistoryDag subclass with node labels containing CompactGenome objects.

    The constructor for this class requires that each node label contain a 'compact_genome'
    field, which is expected to hold a :class:`compact_genome.CompactGenome` object.

    A HistoryDag containing 'sequence' node label fields may be automatically converted to
    this subclass by calling the class method :meth:`CGHistoryDag.from_dag`, providing the
    HistoryDag object to be converted, and the reference sequence to the keyword argument
    'reference'.

    This subclass provides specialized methods for interfacing with Larch's MADAG protobuf format
    """

    _required_label_fields = {
        "compact_genome": [
            (
                ("sequence",),
                lambda n, reference, **kwargs: compact_genome_from_sequence(
                    n.label.sequence, reference, **kwargs
                ),
            )
        ]
    }

    # #### Overridden Methods ####

    def summary(self):
        HistoryDag.summary(self)
        min_pars, max_pars = self.weight_range_annotate(**compact_genome_hamming_distance_countfuncs)
        print(f"Parsimony score range {min_pars} to {max_pars}")

    def weight_count(
        self,
        *args,
        edge_weight_func=wrapped_cg_hamming_distance,
        **kwargs,
    ):
        """See :meth:`historydag.HistoryDag.weight_count`"""
        return super().weight_count(*args, edge_weight_func=edge_weight_func, **kwargs)

    def optimal_weight_annotate(
        self, *args, edge_weight_func=wrapped_cg_hamming_distance, **kwargs
    ) -> Weight:
        """See :meth:`historydag.HistoryDag.optimal_weight_annotate`"""
        return super().optimal_weight_annotate(
            *args, edge_weight_func=edge_weight_func, **kwargs
        )

    def trim_optimal_weight(
        self,
        *args,
        edge_weight_func=wrapped_cg_hamming_distance,
        **kwargs,
    ) -> Weight:
        """See :meth:`historydag.HistoryDag.trim_optimal_weight`"""
        return super().trim_optimal_weight(
            *args, edge_weight_func=edge_weight_func, **kwargs
        )

    def trim_within_range(
        self,
        *args,
        edge_weight_func=wrapped_cg_hamming_distance,
        **kwargs,
    ):
        """See :meth:`historydag.HistoryDag.trim_within_range`"""
        return super().trim_within_range(
            *args, edge_weight_func=edge_weight_func, **kwargs
        )

    def trim_below_weight(
        self,
        *args,
        edge_weight_func=wrapped_cg_hamming_distance,
        **kwargs,
    ):
        """See :meth:`historydag.HistoryDag.trim_below_weight`"""
        return super().trim_below_weight(
            *args, edge_weight_func=edge_weight_func, **kwargs
        )

    def insert_node(
        self,
        *args,
        dist=wrapped_cg_hamming_distance,
        **kwargs,
    ):
        """See :meth:`historydag.HistoryDag.insert_node`"""
        return super().insert_node(*args, dist=dist, **kwargs)

    def hamming_parsimony_count(self):
        """See :meth:`historydag.sequence_dag.SequenceHistoryDag.hamming_parsim
        ony_count`"""
        return self.weight_count(**compact_genome_hamming_distance_countfuncs)

    # #### CGHistoryDag-Specific Methods ####

    def to_protobuf(self, leaf_data_func=None):
        """convert a DAG with compact genome data on each node, to a MAD
        protobuf with mutation information on edges.

        Args:
            dag: the history DAG to be converted
            leaf_data_func: a function taking a DAG node and returning a string to store
                in the protobuf node_name field `condensed_leaves` of leaf nodes
        """

        refseq = next(self.preorder(skip_ua_node=True)).label.compact_genome.reference
        empty_cg = CompactGenome(dict(), refseq)

        def mut_func(pnode, cnode):
            if pnode.is_ua_node():
                parent_seq = empty_cg
            else:
                parent_seq = pnode.label.compact_genome
            return cg_diff(parent_seq, child.label.compact_genome)

        def key_func(cladeitem):
            clade, _ = cladeitem
            return sorted(
                sorted(idx for idx in label.compact_genome.mutations) for label in clade
            )

        node_dict = {}
        data = dpb.data()
        for idx, node in enumerate(self.postorder()):
            node_dict[node] = idx
            node_name = data.node_names.add()
            node_name.node_id = idx
            if leaf_data_func is not None:
                if node.is_leaf():
                    node_name.condensed_leaves.append(leaf_data_func(node))

        for node in self.postorder():
            for cladeidx, (clade, edgeset) in enumerate(
                sorted(node.clades.items(), key=key_func)
            ):
                for child in edgeset.targets:
                    edge = data.edges.add()
                    edge.parent_node = node_dict[node]
                    edge.parent_clade = cladeidx
                    edge.child_node = node_dict[child]
                    for par_nuc, child_nuc, idx in mut_func(node, child):
                        mut = edge.edge_mutations.add()
                        mut.position = idx
                        mut.par_nuc = _pb_nuc_codes[par_nuc.upper()]
                        mut.mut_nuc.append(_pb_nuc_codes[child_nuc.upper()])
        data.reference_seq = self.get_reference_sequence()
        data.reference_id = (
            self.attr["refseqid"] if "refseqid" in self.attr else "unknown_seqid"
        )
        return data

    def to_protobuf_file(self, filename, leaf_data_func=None):
        """Write this CGHistoryDag to a Mutation Annotated DAG protobuf for use
        with Larch."""
        data = self.to_protobuf(leaf_data_func=leaf_data_func)
        with open(filename, "wb") as fh:
            fh.write(data.SerializeToString())

    def flatten(self, sort_compact_genomes=False):
        """return a dictionary containing four keys:

        * `refseq` is a list containing the reference sequence id, and the reference sequence
          (the implied sequence on the UA node)
        * `compact_genome_list` is a list of compact genomes, where each compact genome is a
          list of nested lists `[seq_idx, [old_base, new_base]]` where `seq_idx` is (1-indexed)
          nucleotide sequence site. If sort_compact_genomes is True, compact genomes and `compact_genome_list` are sorted.
        * `node_list` is a list of `[label_idx, clade_list]` pairs, where
            * `label_idx` is the index of the node's compact genome in `compact_genome_list`, and
            * `clade_list` is a list of lists of `compact_genome_list` indices, encoding sets of child clades.

        * `edge_list` is a list of triples `[parent_idx, child_idx, clade_idx]`, where
            * `parent_idx` is the index of the edge's parent node in `node_list`,
            * `child_idx` is the index of the edge's child node in `node_list`, and
            * `clade_idx` is the index of the clade in the parent node's `clade_list` from which this edge descends.
        """
        compact_genome_list = []
        node_list = []
        edge_list = []
        node_indices = {}
        cg_indices = {}

        def get_child_clades(node):
            return [
                frozenset(cg_indices[label] for label in clade) for clade in node.clades
            ]

        def get_compact_genome(node):
            if node.is_ua_node():
                return []
            else:
                ret = [
                    [idx, list(bases)]
                    for idx, bases in node.label.compact_genome.mutations.items()
                ]

            if sort_compact_genomes:
                ret.sort()
            return ret

        for node in self.postorder():
            node_cg = get_compact_genome(node)
            if node.label not in cg_indices:
                cg_indices[node.label] = len(compact_genome_list)
                compact_genome_list.append(node_cg)

        if sort_compact_genomes:
            cgindexlist = sorted(enumerate(compact_genome_list), key=lambda t: t[1])
            compact_genome_list = [cg for _, cg in cgindexlist]
            # the rearrangement is a bijection of indices
            indexmap = {
                oldidx: newidx for newidx, (oldidx, _) in enumerate(cgindexlist)
            }
            for key in cg_indices:
                cg_indices[key] = indexmap[cg_indices[key]]

        for node_idx, node in enumerate(self.postorder()):
            node_indices[id(node)] = node_idx
            node_list.append((cg_indices[node.label], get_child_clades(node)))
            for clade_idx, (clade, eset) in enumerate(node.clades.items()):
                for child in eset.targets:
                    edge_list.append((node_idx, node_indices[id(child)], clade_idx))

        if "refseq" in self.attr:
            refseqid = self.attr["refseq"]
        else:
            refseqid = "unknown_seqid"
        return {
            "refseq": (refseqid, self.get_reference_sequence()),
            "compact_genomes": compact_genome_list,
            "nodes": node_list,
            "edges": edge_list,
        }

    def test_equal(self, other):
        """Test whether two history DAGs are equal.

        Compares sorted JSON representation.
        """
        flatdag1 = self.flatten()
        flatdag2 = other.flatten()
        cg_list1 = flatdag1["compact_genomes"]
        cg_list2 = flatdag2["compact_genomes"]

        def get_edge_set(flatdag):
            edgelist = flatdag["edges"]
            nodelist = flatdag["nodes"]

            def convert_flatnode(flatnode):
                label_idx, clade_list = flatnode
                clades = frozenset(
                    frozenset(label_idx_list) for label_idx_list in clade_list
                )
                return (label_idx, clades)

            nodelist = [convert_flatnode(node) for node in nodelist]
            return frozenset(
                (nodelist[p_idx], nodelist[c_idx]) for p_idx, c_idx, _ in edgelist
            )

        return cg_list1 == cg_list2 and get_edge_set(flatdag1) == get_edge_set(flatdag2)

    def get_reference_sequence(self):
        """Return the reference sequence for this CGHistoryDag.

        This is the sequence with respect to which all node label
        CompactGenomes record mutations.
        """
        return next(self.preorder(skip_ua_node=True)).label.compact_genome.reference

    def _check_valid(self, *args, **kwargs):
        super()._check_valid(*args, **kwargs)
        reference = self.get_reference_sequence()
        for node in self.preorder(skip_ua_node=True):
            if node.label.compact_genome.reference != reference:
                raise ValueError(
                    "Multiple compact genome reference sequences found in node label CompactGenomes."
                )

    def to_json(self, sort_compact_genomes=False):
        """Write this history DAG to a JSON object."""
        return json.dumps(
            self.flatten(sort_compact_genomes=sort_compact_genomes), cls=HDagJSONEncoder
        )

    def to_json_file(self, filename, sort_compact_genomes=False):
        """Write this history DAG to a JSON file."""
        with open(filename, "w") as fh:
            fh.write(self.to_json(sort_compact_genomes=sort_compact_genomes))


def load_json_file(filename):
    """Load a Mutation Annotated DAG stored in a JSON file and return a
    CGHistoryDag."""
    with open(filename, "r") as fh:
        json_dict = json.load(fh)
    return unflatten(json_dict)


def unflatten(flat_dag):
    """Takes a dictionary like that returned by flatten, and returns a
    HistoryDag."""
    refseqid, reference = flat_dag["refseq"]
    compact_genome_list = [
        CompactGenome({idx: tuple(bases) for idx, bases in flat_cg}, reference)
        for flat_cg in flat_dag["compact_genomes"]
    ]
    node_list = flat_dag["nodes"]
    edge_list = flat_dag["edges"]
    Label = NamedTuple("Label", [("compact_genome", CompactGenome)])

    def unpack_cladelabellists(cladelabelsetlist):
        return [
            frozenset(Label(compact_genome_list[cg_idx]) for cg_idx in idx_clade)
            for idx_clade in cladelabelsetlist
        ]

    node_postorder = []
    # a list of (node, [(clade, eset), ...]) tuples
    for cg_idx, cladelabellists in node_list:
        clade_eset_list = [
            (clade, EdgeSet()) for clade in unpack_cladelabellists(cladelabellists)
        ]
        if len(clade_eset_list) == 1:
            # This must be the UA node
            label = historydag.utils.UALabel()
        else:
            label = Label(compact_genome_list[cg_idx])
        try:
            node = HistoryDagNode(label, dict(clade_eset_list), attr=None)
        except ValueError:
            node = UANode(clade_eset_list[0][1])
        node_postorder.append((node, clade_eset_list))

    # adjust UA node label
    node_postorder[-1][0].label = historydag.utils.UALabel()

    # Add edges
    for parent_idx, child_idx, clade_idx in edge_list:
        node_postorder[parent_idx][1][clade_idx][1].add_to_edgeset(
            node_postorder[child_idx][0]
        )

    # UA node is last in postorder
    dag = CGHistoryDag(node_postorder[-1][0])
    dag.attr["refseq"] = refseqid
    # This shouldn't be necessary, but appears to be
    dag.recompute_parents()
    return dag


def load_MAD_protobuf(pbdata):
    """Convert a MAD protobuf to a CGHistoryDag with compact genomes in the
    `compact_genome` label attribute."""
    # use HistoryDag.__setstate__ to make this happen
    # all of a node's parent edges
    reference = pbdata.reference_seq
    parent_edges = {node.node_id: [] for node in pbdata.node_names}
    # a list of list of a node's child edges
    child_edges = {node.node_id: [] for node in pbdata.node_names}
    for edge in pbdata.edges:
        parent_edges[edge.child_node].append(edge)
        child_edges[edge.parent_node].append(edge)

    # now each node id is in parent_edges and child_edges as a key,
    # fix the UA node's compact genome (could be done in function but this
    # asserts only one node has no parent edges)
    (ua_node_id,) = [
        node_id for node_id, eset in parent_edges.items() if len(eset) == 0
    ]

    @functools.lru_cache(maxsize=None)
    def get_node_compact_genome(node_id):
        if node_id == ua_node_id:
            return CompactGenome(frozendict(), reference)
        else:
            edge = parent_edges[node_id][0]
            parent_seq = get_node_compact_genome(edge.parent_node)
            str_mutations = tuple(_pb_mut_to_str(mut) for mut in edge.edge_mutations)
            return parent_seq.apply_muts(str_mutations)

    label_list = []
    label_dict = {}

    for node_record in pbdata.node_names:
        cg = get_node_compact_genome(node_record.node_id)
        if cg in label_dict:
            cg_idx = label_dict[cg]
        else:
            cg_idx = len(label_list)
            label_dict[cg] = cg_idx
            label_list.append(cg)

    # now build clade unions by dynamic programming:
    @functools.lru_cache(maxsize=None)
    def get_clade_union(node_id):
        if len(child_edges[node_id]) == 0:
            # it's a leaf node
            return frozenset({label_dict[get_node_compact_genome(node_id)]})
        else:
            return frozenset(
                {
                    label
                    for child_edge in child_edges[node_id]
                    for label in get_clade_union(child_edge.child_node)
                }
            )

    def get_child_clades(node_id):
        return tuple(
            get_clade_union(child_edge.child_node)
            for child_edge in child_edges[node_id]
        )
        # maybe we need this??
        # return frozenset({get_clade_union(child_edge.child_node) for child_edge in child_edges[node_id]})

    # order node_ids in postordering
    visited = set()

    def traverse(node_id):
        visited.add(node_id)
        child_ids = [edge.child_node for edge in child_edges[node_id]]
        if len(child_ids) > 0:
            for child_id in child_ids:
                if child_id not in visited:
                    yield from traverse(child_id)
        yield node_id

    id_postorder = list(traverse(ua_node_id))
    # Start building DAG data
    node_index_d = {node_id: idx for idx, node_id in enumerate(id_postorder)}
    node_list = [
        (
            label_dict[get_node_compact_genome(node_id)],
            get_child_clades(node_id),
            {"node_id": node_id},
        )
        for node_id in id_postorder
    ]

    edge_list = [
        (node_index_d[edge.parent_node], node_index_d[edge.child_node], 0, 1)
        for edge in pbdata.edges
    ]
    # fix label list
    label_list = [(item,) for item in label_list]
    label_list.append(None)
    ua_node = list(node_list[-1])
    ua_node[0] = len(label_list) - 1
    node_list[-1] = tuple(ua_node)
    dag = HistoryDag(UANode(EdgeSet()))
    dag.__setstate__(
        {
            "label_fields": ("compact_genome",),
            "label_list": label_list,
            "node_list": node_list,
            "edge_list": edge_list,
            "attr": {"refseqid": pbdata.reference_id},
        }
    )
    return CGHistoryDag.from_history_dag(dag)


def load_MAD_protobuf_file(filename):
    """Load a mutation annotated DAG protobuf file and return a
    CGHistoryDag."""
    with open(filename, "rb") as fh:
        pb_data = dpb.data()
        pb_data.ParseFromString(fh.read())
    return load_MAD_protobuf(pb_data)


compact_genome_hamming_distance_countfuncs = historydag.utils.AddFuncDict(
    {
        "start_func": lambda n: 0,
        "edge_weight_func": wrapped_cg_hamming_distance,
        "accum_func": sum,
    },
    name="HammingParsimony",
)
"""Provides functions to count hamming distance parsimony when sequences are
stored as CompactGenomes.
For use with :meth:`historydag.CGHistoryDag.weight_count`."""
