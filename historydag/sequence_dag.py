from historydag import HistoryDag
import historydag.utils


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
