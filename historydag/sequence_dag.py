from historydag import HistoryDag
import historydag.parsimony_utils as parsimony_utils
from historydag.utils import Weight


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
        return self.weight_count(**parsimony_utils.hamming_distance_countfuncs)

    def summary(self):
        HistoryDag.summary(self)
        min_pars, max_pars = self.weight_range_annotate(
            **parsimony_utils.hamming_distance_countfuncs
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
        edge_weight_func=parsimony_utils.hamming_edge_weight_ambiguous_leaves,
        **kwargs,
    ):
        """See :meth:`historydag.HistoryDag.weight_count`"""
        return super().weight_count(*args, edge_weight_func=edge_weight_func, **kwargs)

    def optimal_weight_annotate(
        self,
        *args,
        edge_weight_func=parsimony_utils.hamming_edge_weight_ambiguous_leaves,
        **kwargs,
    ) -> Weight:
        """See :meth:`historydag.HistoryDag.optimal_weight_annotate`"""
        return super().optimal_weight_annotate(
            *args, edge_weight_func=edge_weight_func, **kwargs
        )

    def trim_optimal_weight(
        self,
        *args,
        edge_weight_func=parsimony_utils.hamming_edge_weight_ambiguous_leaves,
        **kwargs,
    ) -> Weight:
        """See :meth:`historydag.HistoryDag.trim_optimal_weight`"""
        return super().trim_optimal_weight(
            *args, edge_weight_func=edge_weight_func, **kwargs
        )

    def trim_within_range(
        self,
        *args,
        edge_weight_func=parsimony_utils.hamming_edge_weight_ambiguous_leaves,
        **kwargs,
    ):
        """See :meth:`historydag.HistoryDag.trim_within_range`"""
        return super().trim_within_range(
            *args, edge_weight_func=edge_weight_func, **kwargs
        )

    def trim_below_weight(
        self,
        *args,
        edge_weight_func=parsimony_utils.hamming_edge_weight_ambiguous_leaves,
        **kwargs,
    ):
        """See :meth:`historydag.HistoryDag.trim_below_weight`"""
        return super().trim_below_weight(
            *args, edge_weight_func=edge_weight_func, **kwargs
        )

    def insert_node(
        self,
        *args,
        dist=parsimony_utils.hamming_edge_weight_ambiguous_leaves,
        **kwargs,
    ):
        """See :meth:`historydag.HistoryDag.insert_node`"""
        return super().insert_node(*args, dist=dist, **kwargs)

    def hamming_parsimony_count(self):
        """See :meth:`historydag.sequence_dag.SequenceHistoryDag.hamming_parsim
        ony_count`"""
        return self.weight_count(
            **parsimony_utils.leaf_ambiguous_hamming_distance_countfuncs
        )

    def summary(self):
        HistoryDag.summary(self)
        min_pars, max_pars = self.weight_range_annotate(
            **parsimony_utils.leaf_ambiguous_hamming_distance_countfuncs
        )
        print(f"Parsimony score range {min_pars} to {max_pars}")
