import numpy as np
import historydag.utils as utils
import Bio.Data.IUPACData
from frozendict import frozendict
from typing import (
    Generator,
)


class AmbiguityMap:
    """Implements a bijection between a subset of the power set of an alphabet
    of bases, and an expanded alphabet of ambiguity codes.

    To look up the set of bases represented by an ambiguity code, use the object like a dictionary.
    To look up an ambiguity code representing a set of bases, use the ``reversed`` attribute.

    Args:
        ambiguity_character_map: A mapping from ambiguity codes to collections of bases represented by that code.
        bases: A collection of bases. If not provided, this will be inferred from the ambiguity_character_map.
    """

    def __init__(self, ambiguity_character_map, bases=None):
        ambiguous_values = {
            char: frozenset(bases) for char, bases in ambiguity_character_map.items()
        }
        if bases is None:
            self.bases = frozenset(
                base for bset in self.ambiguous_values for base in bset
            )
        else:
            self.bases = frozenset(bases)

        ambiguous_values.update({base: frozenset({base}) for base in self.bases})
        self.ambiguous_values = frozendict(ambiguous_values)
        self.reversed = ReversedAmbiguityMap(
            {bases: char for char, bases in self.ambiguous_values.items()}
        )
        self.uninformative_chars = frozenset(
            char
            for char, base_set in self.ambiguous_values.items()
            if base_set == self.bases
        )

    def __getitem__(self, key):
        try:
            return self.ambiguous_values[key]
        except KeyError:
            raise KeyError(f"{key} is not a valid ambiguity code for this map.")

    def __iter__(self):
        return self.ambiguous_values.__iter__()

    def items(self):
        return self.ambiguous_values.items()

    def is_ambiguous(self, sequence: str) -> bool:
        """Returns whether the provided sequence contains IUPAC nucleotide
        ambiguity codes."""
        return any(code not in self.bases for code in sequence)

    def sequence_resolutions(self, sequence: str) -> Generator[str, None, None]:
        """Iterates through possible disambiguations of sequence, recursively.

        Recursion-depth-limited by number of ambiguity codes in
        sequence, not sequence length.
        """

        def _sequence_resolutions(sequence, _accum=""):
            if sequence:
                for index, base in enumerate(sequence):
                    if base in self.bases:
                        _accum += base
                    else:
                        for newbase in self.ambiguous_values[base]:
                            yield from _sequence_resolutions(
                                sequence[index + 1 :], _accum=(_accum + newbase)
                            )
                        return
            yield _accum

        return _sequence_resolutions(sequence)

    def get_sequence_resolution_func(self, field_name):
        """Returns a function which takes a Label, and returns a generator on
        labels containing all possible resolutions of the sequence in that
        node's label's field_name attribute."""

        @utils.explode_label(field_name)
        def sequence_resolutions(sequence: str) -> Generator[str, None, None]:
            return self.sequence_resolutions(sequence)

        return sequence_resolutions

    def sequence_resolution_count(self, sequence: str) -> int:
        """Count the number of possible sequence resolutions Equivalent to the
        length of the list returned by :meth:`sequence_resolutions`."""
        base_options = [
            len(self.ambiguous_values[base])
            for base in sequence
            if base in self.ambiguous_values
        ]
        return utils.prod(base_options)

    def get_sequence_resolution_count_func(self, field_name):
        @utils.access_field(field_name)
        def sequence_resolution_count(sequence) -> int:
            return self.sequence_resolution_count(sequence)

        return sequence_resolution_count


class ReversedAmbiguityMap(frozendict):
    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            raise KeyError(f"No ambiguity code is defined for the set of bases {key}")


_ambiguous_dna_values_gap_as_char = Bio.Data.IUPACData.ambiguous_dna_values.copy()
_ambiguous_dna_values_gap_as_char.update({"?": "GATC-", "-": "-"})
_ambiguous_dna_values = Bio.Data.IUPACData.ambiguous_dna_values.copy()
_ambiguous_dna_values.update({"?": "GATC", "-": "GATC"})

standard_aa_ambiguity_map = AmbiguityMap(_ambiguous_dna_values, "AGCT")
standard_aa_ambiguity_map_gap_as_char = AmbiguityMap(
    _ambiguous_dna_values_gap_as_char, "AGCT-"
)


class TransitionAlphabet:
    def __init__(
        self, bases=tuple("AGCT"), transition_weights=None, ambiguity_map=None
    ):
        self.bases = tuple(bases)
        self.base_indices = frozendict({base: idx for idx, base in enumerate(bases)})
        self.yey = np.array(
            [[i != j for i in range(len(self.bases))] for j in range(len(self.bases))]
        )
        if transition_weights is None:
            self.transition_weights = self.yey
        else:
            if len(transition_weights) != len(self.bases) or len(
                transition_weights[0]
            ) != len(self.bases):
                raise ValueError(
                    "transition_weights must be a nxn matrix, with n=len(bases)"
                )
            self.transition_weights = transition_weights
        if ambiguity_map is None:
            if self.bases == tuple("AGCT"):
                self.ambiguity_map = standard_aa_ambiguity_map
            elif self.bases == tuple("AGCT-"):
                self.ambiguity_map = standard_aa_ambiguity_map_gap_as_char
            else:
                self.ambiguity_map = AmbiguityMap({}, self.bases)
        else:
            if not isinstance(ambiguity_map, AmbiguityMap):
                raise ValueError(
                    "ambiguity_map, if provided, must be a historydag.parsimony.AmbiguityMap object"
                )
            self.ambiguity_map = ambiguity_map

        # This is applicable even when diagonal entries in transition rate matrix are
        # nonzero, since it is only a mask on allowable sites based on each base.
        self.mask_vectors = {
            code: np.array(
                [
                    0 if base in self.ambiguity_map[code] else float("inf")
                    for base in self.bases
                ]
            )
            for code in self.ambiguity_map
        }

    def get_adjacency_array(self, seq_len):
        return np.array([self.transition_weights] * seq_len)

    def get_ambiguity_from_tuple(self, tup):
        """Retrieve an ambiguity code encoded by tup, which encodes a set of
        bases with a tuple of 0's and 1's.

        For example, with ``bases='AGCT'``, ``(0, 1, 1, 1)`` would
        return `A`.
        """
        return self.ambiguity_map.reversed[
            frozenset(self.bases[i] for i, flag in enumerate(tup) if flag == 0)
        ]

    def character_distance(self, parent_char, child_char, site=None):
        return self.transition_weights[self.base_indices[parent_char]][
            self.base_indices[child_char]
        ]

    def weighted_hamming_distance(self, parent_seq, child_seq):
        if len(parent_seq) != len(child_seq):
            raise ValueError("sequence lengths do not match!")
        return sum(
            self.character_distance(pchar, cchar, site=(idx + 1))
            for idx, (pchar, cchar) in enumerate(zip(parent_seq, child_seq))
        )

    def weighted_cg_hamming_distance(self, parent_cg, child_cg):
        if parent_cg.reference != child_cg.reference:
            raise ValueError("Reference sequences do not match!")
        s1 = set(parent_cg.mutations.keys())
        s2 = set(child_cg.mutations.keys())
        return sum(
            self.character_distance(
                parent_cg.get_site(site), child_cg.get_site(site), site=site
            )
            for site in s1 | s2
        )

    def min_character_distance(self, parent_char, child_char, site=None):
        """Allowing child_char to be ambiguous, returns the minimum possible
        transition weight between parent_char and child_char."""
        child_set = self.ambiguity_map[child_char]
        p_idx = self.base_indices[parent_char]
        return min(
            self.transition_weights[p_idx][self.base_indices[cbase]]
            for cbase in child_set
        )

    def min_weighted_hamming_distance(self, parent_seq, child_seq):
        """Assuming the child_seq may contain ambiguous characters, returns the
        minimum possible transition weight between parent_seq and child_seq."""
        if len(parent_seq) != len(child_seq):
            raise ValueError("sequence lengths do not match!")
        return sum(
            self.min_character_distance(pchar, cchar, site=(idx + 1))
            for idx, (pchar, cchar) in enumerate(zip(parent_seq, child_seq))
        )

    def min_weighted_cg_hamming_distance(self, parent_cg, child_cg):
        if parent_cg.reference != child_cg.reference:
            raise ValueError("Reference sequences do not match!")
        s1 = set(parent_cg.mutations.keys())
        s2 = set(child_cg.mutations.keys())
        return sum(
            self.min_character_distance(
                parent_cg.get_site(site), child_cg.get_site(site), site=site
            )
            for site in s1 | s2
        )

    def weighted_hamming_edge_weight(self, field_name):
        """Returns a function for computing weighted hamming distance between
        two nodes' sequences.

        Args:
            field_name: The name of the node label field which contains sequences.

        Returns:
            A function accepting two :class:`HistoryDagNode` objects ``n1`` and ``n2`` and returning
            a float: the transition cost from ``n1.label.<field_name>`` to ``n2.label.<field_name>``, or 0 if
            n1 is the UA node.
        """

        @utils.access_nodefield_default(field_name, 0)
        def edge_weight(parent, child):
            return self.weighted_hamming_distance(parent, child)

        return edge_weight

    def weighted_cg_hamming_edge_weight(self, field_name):
        """Returns a function for computing weighted hamming distance between
        two nodes' compact genomes.

        Args:
            field_name: The name of the node label field which contains compact genomes.

        Returns:
            A function accepting two :class:`HistoryDagNode` objects ``n1`` and ``n2`` and returning
            a float: the transition cost from ``n1.label.<field_name>`` to ``n2.label.<field_name>``, or 0 if
            n1 is the UA node.
        """

        @utils.access_nodefield_default(field_name, 0)
        def edge_weight(parent, child):
            return self.weighted_cg_hamming_distance(parent, child)

        return edge_weight

    def min_weighted_hamming_edge_weight(self, field_name):
        """Returns a function for computing weighted hamming distance between
        two nodes' sequences.

        If the child node is a leaf node, and its sequence contains ambiguities, the minimum possible
        transition cost from parent to child will be returned.

        Args:
            field_name: The name of the node label field which contains sequences.

        Returns:
            A function accepting two :class:`HistoryDagNode` objects ``n1`` and ``n2`` and returning
            a float: the transition cost from ``n1.label.<field_name>`` to ``n2.label.<field_name>``, or 0 if
            n1 is the UA node.
        """

        def edge_weight(parent, child):
            if parent.is_ua_node():
                return 0
            elif child.is_leaf():
                return self.min_weighted_hamming_distance(
                    getattr(parent.label, field_name), getattr(child.label, field_name)
                )
            else:
                return self.weighted_hamming_distance(
                    getattr(parent.label, field_name), getattr(child.label, field_name)
                )

        return edge_weight

    def min_weighted_cg_hamming_edge_weight(self, field_name):
        """Returns a function for computing weighted hamming distance between
        two nodes' compact genomes.

        If the child node is a leaf node, and its compact genome contains ambiguities, the minimum possible
        transition cost from parent to child will be returned by this function.

        Args:
            field_name: The name of the node label field which contains compact genomes.

        Returns:
            A function accepting two :class:`HistoryDagNode` objects ``n1`` and ``n2`` and returning
            a float: the transition cost from ``n1.label.<field_name>`` to ``n2.label.<field_name>``, or 0 if
            n1 is the UA node.
        """

        def edge_weight(parent, child):
            if parent.is_ua_node():
                return 0
            elif child.is_leaf():
                return self.min_weighted_cg_hamming_distance(
                    getattr(parent.label, field_name), getattr(child.label, field_name)
                )
            else:
                return self.weighted_cg_hamming_distance(
                    getattr(parent.label, field_name), getattr(child.label, field_name)
                )

        return edge_weight

    def get_weighted_cg_parsimony_countfuncs(
        self, field_name, leaf_ambiguities=False, name="WeightedParsimony"
    ):
        if leaf_ambiguities:
            edge_weight = self.min_weighted_cg_hamming_edge_weight(field_name)
        else:
            edge_weight = self.weighted_cg_hamming_edge_weight(field_name)
        return utils.AddFuncDict(
            {
                "start_func": lambda n: 0,
                "edge_weight_func": edge_weight,
                "accum_func": sum,
            },
            name=name,
        )

    def get_weighted_parsimony_countfuncs(
        self, field_name, leaf_ambiguities=False, name="WeightedParsimony"
    ):
        if leaf_ambiguities:
            edge_weight = self.min_weighted_hamming_edge_weight(field_name)
        else:
            edge_weight = self.weighted_hamming_edge_weight(field_name)
        return utils.AddFuncDict(
            {
                "start_func": lambda n: 0,
                "edge_weight_func": edge_weight,
                "accum_func": sum,
            },
            name=name,
        )


class UnitTransitionAlphabet(TransitionAlphabet):
    """A subclass of :class:`TransitionAlphabet` to be used when all
    transitions have unit cost."""

    def __init__(self, bases="AGCT", ambiguity_map=None):
        super().__init__(bases=bases, ambiguity_map=None)

    def character_distance(self, parent_char, child_char, site=None):
        return int(parent_char != child_char)

    def min_character_distance(self, parent_char, child_char, site=None):
        return int(parent_char not in self.ambiguity_map[child_char])

    def get_weighted_cg_parsimony_countfuncs(
        self, field_name, leaf_ambiguities=False, name="HammingParsimony"
    ):
        return super().get_weighted_cg_parsimony_countfuncs(
            field_name, leaf_ambiguities=leaf_ambiguities, name=name
        )

    def get_weighted_parsimony_countfuncs(
        self, field_name, leaf_ambiguities=False, name="HammingParsimony"
    ):
        return super().get_weighted_parsimony_countfuncs(
            field_name, leaf_ambiguities=leaf_ambiguities, name=name
        )


class VariableTransitionAlphabet(TransitionAlphabet):
    """A subclass of :class:`TransitionAlphabet` to be used when transition
    costs depend on site."""

    def __init__(self, bases="AGCT", transition_matrix=None, ambiguity_map=None):
        assert transition_matrix.shape[1:] == (len(bases), len(bases))
        super().__init__(bases=bases, ambiguity_map=None)
        self.transition_weights = None
        self.sitewise_transition_matrix = transition_matrix
        self._seq_len

    def character_distance(self, parent_char, child_char, site):
        return self.sitewise_transition_matrix[site - 1][parent_char][child_char]

    def min_character_distance(self, parent_char, child_char, site):
        """Allowing child_char to be ambiguous, returns the minimum possible
        transition weight between parent_char and child_char."""
        child_set = self.ambiguity_map[child_char]
        p_idx = self.base_indices[parent_char]
        return min(
            self.transition_weights[site - 1][p_idx][self.base_indices[cbase]]
            for cbase in child_set
        )

    def get_adjacency_array(self, seq_len):
        if seq_len != self._seq_len:
            raise ValueError(
                f"VariableTransitionAlphabet instance supports sequence length of {self._seq_len}"
            )
        return self.sitewise_transition_matrix


default_aa_transitions = UnitTransitionAlphabet(bases="AGCT")
default_aa_gaps_transitions = UnitTransitionAlphabet(bases="AGCT-")

hamming_edge_weight = default_aa_transitions.weighted_hamming_edge_weight("sequence")
hamming_edge_weight_ambiguous_leaves = (
    default_aa_transitions.min_weighted_hamming_edge_weight("sequence")
)

hamming_cg_edge_weight = default_aa_transitions.weighted_cg_hamming_edge_weight(
    "compact_genome"
)
hamming_cg_edge_weight_ambiguous_leaves = (
    default_aa_transitions.min_weighted_cg_hamming_edge_weight("compact_genome")
)

hamming_distance_countfuncs = utils.AddFuncDict(
    {
        "start_func": lambda n: 0,
        "edge_weight_func": hamming_edge_weight,
        "accum_func": sum,
    },
    name="HammingParsimony",
)
"""Provides functions to count hamming distance parsimony when leaf sequences
may be ambiguous.
For use with :meth:`historydag.AmbiguousLeafSequenceHistoryDag.weight_count`."""

leaf_ambiguous_hamming_distance_countfuncs = utils.AddFuncDict(
    {
        "start_func": lambda n: 0,
        "edge_weight_func": hamming_edge_weight_ambiguous_leaves,
        "accum_func": sum,
    },
    name="HammingParsimony",
)
"""Provides functions to count hamming distance parsimony when leaf sequences
may be ambiguous.
For use with :meth:`historydag.AmbiguousLeafSequenceHistoryDag.weight_count`."""


compact_genome_hamming_distance_countfuncs = (
    default_aa_transitions.get_weighted_cg_parsimony_countfuncs(
        "compact_genome", leaf_ambiguities=False
    )
)
"""Provides functions to count hamming distance parsimony when sequences are
stored as CompactGenomes.
For use with :meth:`historydag.CGHistoryDag.weight_count`."""


leaf_ambiguous_compact_genome_hamming_distance_countfuncs = (
    default_aa_transitions.get_weighted_cg_parsimony_countfuncs(
        "compact_genome", leaf_ambiguities=True
    )
)
"""Provides functions to count hamming distance parsimony when leaf compact genomes
may be ambiguous.
For use with :meth:`historydag.AmbiguousLeafCGHistoryDag.weight_count`."""
