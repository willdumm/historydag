from historydag import HistoryDag
from functools import lru_cache
import historydag.utils
from historydag.utils import Weight


def nonleaf_sequence_resolutions(node):
    if node.is_leaf():
        yield node.label
    else:
        yield from historydag.utils.sequence_resolutions(node.label)


@lru_cache(maxsize=20000)
def _ambiguous_hamming_distance(node1, node2):
    return sum(
        pbase not in historydag.utils.ambiguous_dna_values[cbase]
        for pbase, cbase in zip(node1.label.sequence, node2.label.sequence)
    )


def ambiguous_hamming_distance(node1, node2):
    """Returns the hamming distance between node1.label.sequence and
    node2.label.sequence.

    If node2 is a leaf node, then its sequence may be ambiguous, and the
    minimum possible distance will be returned.
    """
    if node1.is_ua_node():
        return 0
    if node2.is_leaf():
        return _ambiguous_hamming_distance(node1, node2)
    else:
        return historydag.utils.hamming_distance(
            node1.label.sequence, node2.label.sequence
        )


leaf_ambiguous_hamming_distance_countfuncs = historydag.utils.AddFuncDict(
    {
        "start_func": lambda n: 0,
        "edge_weight_func": ambiguous_hamming_distance,
        "accum_func": sum,
    },
    name="HammingParsimony",
)
"""Provides functions to count hamming distance parsimony when leaf sequences
may be ambiguous.

For use with :meth:`historydag.AmbiguousLeafSequenceHistoryDag.weight_count`.
"""


class SequenceHistoryDag(HistoryDag):
    """A HistoryDag subclass with node labels containing full, unambiguous
    nucleotide sequences.

    The constructor for this class requires that each node label contain a 'sequence'
    field, which is expected to hold an unambiguous nucleotide sequence.

    A HistoryDag containing 'compact_genome' node label fields which contain
    :class:`compact_genome.CompactGenome` objects may be automatically converted to
    this subclass by calling the class method :meth:`SequenceHistoryDag.from_dag`, providing the
    HistoryDag object to be converted.
    """

    _required_label_fields = {
        "sequence": [
            (("compact_genome",), lambda node: node.label.compact_genome.to_sequence())
        ]
    }

    def hamming_parsimony_count(self):
        """Count the hamming parsimony scores of all trees in the history DAG.

        Returns a Counter with integer keys.
        """
        return self.weight_count(**historydag.utils.hamming_distance_countfuncs)

    def summary(self):
        HistoryDag.summary(self)
        min_pars, max_pars = self.weight_range_annotate(
            **historydag.utils.hamming_distance_countfuncs
        )
        print(f"Parsimony score range {min_pars} to {max_pars}")


class AmbiguousLeafSequenceHistoryDag(SequenceHistoryDag):
    """A HistoryDag subclass with node labels containing full nucleotide
    sequences.

    The constructor for this class requires that each node label contain a 'sequence'
    field, which is expected to hold an unambiguous nucleotide sequence if the node
    is internal. The sequence may be ambiguous for leaf nodes.

    A HistoryDag containing 'compact_genome' node label fields which contain
    :class:`compact_genome.CompactGenome` objects may be automatically converted to
    this subclass by calling the class method :meth:`SequenceHistoryDag.from_dag`, providing the
    HistoryDag object to be converted.
    """

    # #### Overridden Methods ####

    def weight_count(
        self,
        *args,
        edge_weight_func=ambiguous_hamming_distance,
        **kwargs,
    ):
        """See :meth:`historydag.HistoryDag.weight_count`"""
        return super().weight_count(*args, edge_weight_func=edge_weight_func, **kwargs)

    def optimal_weight_annotate(
        self, *args, edge_weight_func=ambiguous_hamming_distance, **kwargs
    ) -> Weight:
        """See :meth:`historydag.HistoryDag.optimal_weight_annotate`"""
        return super().optimal_weight_annotate(
            *args, edge_weight_func=edge_weight_func, **kwargs
        )

    def trim_optimal_weight(
        self,
        *args,
        edge_weight_func=ambiguous_hamming_distance,
        **kwargs,
    ) -> Weight:
        """See :meth:`historydag.HistoryDag.trim_optimal_weight`"""
        return super().trim_optimal_weight(
            *args, edge_weight_func=edge_weight_func, **kwargs
        )

    def trim_within_range(
        self,
        *args,
        edge_weight_func=ambiguous_hamming_distance,
        **kwargs,
    ):
        """See :meth:`historydag.HistoryDag.trim_within_range`"""
        return super().trim_within_range(
            *args, edge_weight_func=edge_weight_func, **kwargs
        )

    def trim_below_weight(
        self,
        *args,
        edge_weight_func=ambiguous_hamming_distance,
        **kwargs,
    ):
        """See :meth:`historydag.HistoryDag.trim_below_weight`"""
        return super().trim_below_weight(
            *args, edge_weight_func=edge_weight_func, **kwargs
        )

    def insert_node(
        self,
        *args,
        dist=ambiguous_hamming_distance,
        **kwargs,
    ):
        """See :meth:`historydag.HistoryDag.insert_node`"""
        return super().insert_node(*args, dist=dist, **kwargs)

    def hamming_parsimony_count(self):
        """See :meth:`historydag.sequence_dag.SequenceHistoryDag.hamming_parsim
        ony_count`"""
        return self.weight_count(**leaf_ambiguous_hamming_distance_countfuncs)

    def summary(self):
        HistoryDag.summary(self)
        min_pars, max_pars = self.weight_range_annotate(
            **leaf_ambiguous_hamming_distance_countfuncs
        )
        print(f"Parsimony score range {min_pars} to {max_pars}")
