from historydag import HistoryDag
import historydag.parsimony_utils as parsimony_utils
from historydag.utils import UALabel


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

    _default_args = dict(parsimony_utils.hamming_distance_countfuncs) | {
        "start_func": (lambda n: 0),
        "edge_func": lambda l1, l2: (
            0
            if isinstance(l1, UALabel)
            else parsimony_utils.default_nt_transitions.weighted_hamming_distance(
                l1.sequence, l2.sequence
            )
        ),
        "expand_func": parsimony_utils.default_nt_transitions.ambiguity_map.get_sequence_resolution_func(
            "sequence"
        ),
        "optimal_func": min,
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

    _default_args = dict(parsimony_utils.leaf_ambiguous_hamming_distance_countfuncs) | {
        "start_func": (lambda n: 0),
        "edge_func": lambda l1, l2: (
            0
            if isinstance(l1, UALabel)
            else parsimony_utils.default_nt_transitions.weighted_hamming_distance(
                l1.sequence, l2.sequence
            )
        ),
        "expand_func": parsimony_utils.default_nt_transitions.ambiguity_map.get_sequence_resolution_func(
            "sequence"
        ),
        "optimal_func": min,
    }

    def hamming_parsimony_count(self):
        """Count the hamming parsimony scores of all trees in the history DAG.

        Returns a Counter with integer keys.
        """
        return self.weight_count(
            **parsimony_utils.leaf_ambiguous_hamming_distance_countfuncs
        )

    def summary(self):
        HistoryDag.summary(self)
        min_pars, max_pars = self.weight_range_annotate(
            **parsimony_utils.leaf_ambiguous_hamming_distance_countfuncs
        )
        print(f"Parsimony score range {min_pars} to {max_pars}")
