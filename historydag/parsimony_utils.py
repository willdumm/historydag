"""This module provides tools for describing and computing parsimony and
weighted parsimony, and for describing allowed characters and ambiguity codes.

:class:`AmbiguityMap` stores a collection of ``Characters`` (for example the strings
``'A'``, ``'G'``, ``'C'``, and ``'T'``) that are considered valid states, as well as
a mapping between other ``Characters`` ('ambiguity codes') and subsets of these allowed
characters. This type allows forward- and backward-lookup of ambiguity codes.

:class:`TransitionModel` describes transition weights between an ordered collection of
``Characters``, and provides methods for computing weighted parsimony. Functions in
``historydag.parsimony`` accept ``TransitionModel`` objects to customize the implementation
of Sankoff, for example. This class has two subclasses: :class:`UnitTransitionModel`,
which describes unit transition costs between non-identical ``Characters``,
and :class:`SitewiseTransitionModel`, which allows transition costs to depend on the
location in a sequence in which a transition occurs.
"""
import numpy as np
import historydag.utils as utils
from historydag.utils import AddFuncDict, Label
import Bio.Data.IUPACData
from frozendict import frozendict
from typing import (
    Generator,
    Tuple,
    Callable,
    Iterable,
    Sequence,
    Any,
    Mapping,
    TYPE_CHECKING,
)

Character = Any
CharacterSequence = Sequence[Character]

if TYPE_CHECKING:
    from historydag.dag import HistoryDagNode
    from historydag.compact_genome import CompactGenome


class AmbiguityMap:
    """Implements a bijection between a subset of the power set of an alphabet
    of bases, and an expanded alphabet of ambiguity codes.

    To look up the set of bases represented by an ambiguity code, use the object like a dictionary.
    To look up an ambiguity code representing a set of bases, use the ``reversed`` attribute.

    Args:
        ambiguity_character_map: A mapping from ambiguity codes to collections of bases represented by that code.
        bases: A collection of bases. If not provided, this will be inferred from the ambiguity_character_map.
    """

    def __init__(
        self,
        ambiguity_character_map: Mapping[Character, Iterable[Character]],
        bases: Iterable[Character] = None,
    ):
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

    def is_ambiguous(self, sequence: CharacterSequence) -> bool:
        """Returns whether the provided sequence contains any non-base
        characters (such as ambiguity codes)."""
        return any(code not in self.bases for code in sequence)

    def sequence_resolutions(
        self, sequence: CharacterSequence
    ) -> Generator[CharacterSequence, None, None]:
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

    def get_sequence_resolution_func(
        self, field_name: str
    ) -> Callable[[Label], Generator[Label, None, None]]:
        """Returns a function which takes a Label, and returns a generator on
        labels containing all possible resolutions of the sequence in that
        node's label's ``field_name`` attribute."""

        @utils.explode_label(field_name)
        def sequence_resolutions(
            sequence: CharacterSequence,
        ) -> Generator[CharacterSequence, None, None]:
            return self.sequence_resolutions(sequence)

        return sequence_resolutions

    def sequence_resolution_count(self, sequence: CharacterSequence) -> int:
        """Count the number of possible sequence resolutions.

        Equivalent to the number of items yielded by
        :meth:`sequence_resolutions`.
        """
        base_options = [
            len(self.ambiguous_values[base])
            for base in sequence
            if base in self.ambiguous_values
        ]
        return utils.prod(base_options)

    def get_sequence_resolution_count_func(
        self, field_name: str
    ) -> Callable[[Label], int]:
        """Returns a function taking a Label and returning the number of
        resolutions for the sequence held in the label's ``field_name``
        attribute."""

        @utils.access_field(field_name)
        def sequence_resolution_count(sequence) -> int:
            return self.sequence_resolution_count(sequence)

        return sequence_resolution_count


class ReversedAmbiguityMap(frozendict):
    """A subclass of frozendict for holding the reversed map in a
    :class:`AmbiguityMap` instance."""

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            raise KeyError(f"No ambiguity code is defined for the set of bases {key}")


_ambiguous_dna_values_gap_as_char = Bio.Data.IUPACData.ambiguous_dna_values.copy()
_ambiguous_dna_values_gap_as_char.update({"?": "GATC-", "-": "-"})
_ambiguous_dna_values = Bio.Data.IUPACData.ambiguous_dna_values.copy()
_ambiguous_dna_values.update({"?": "GATC", "-": "GATC"})

standard_nt_ambiguity_map = AmbiguityMap(_ambiguous_dna_values, "AGCT")
"""The standard IUPAC nucleotide ambiguity map, in which 'N', '?', and '-' are
all considered fully ambiguous."""

standard_nt_ambiguity_map_gap_as_char = AmbiguityMap(
    _ambiguous_dna_values_gap_as_char, "AGCT-"
)
"""The standard IUPAC nucleotide ambiguity map, in which '-' is treated as a
fifth base, '?' is fully ambiguous, and the ambiguity 'N' excludes '-'."""


class TransitionModel:
    """A class describing a collection of states, and the transition costs
    between them, for weighted parsimony.

    In addition to them methods defined below, there are also attributes which are created by
    the constructor, which should be ensured to be correct in any subclass of TransitionModel:

    * ``base_indices`` is a dictionary mapping bases in ``self.bases`` to their indices
    * ``mask_vectors`` is a dictionary mapping ambiguity codes (including non-ambiguous bases)
    to vectors of floats which are 0 at indices compatible with the ambiguity code, and infinity
    otherwise.
    * ``bases`` is a tuple recording the correct order of unambiguous bases

    Args:
        bases: An ordered collection of valid character states
        transition_weights: A matrix describing transition costs between bases, with index order (from_base, to_base)
        ambiguity_map: An :class:`AmbiguityMap` object describing ambiguity codes. If not provided, the default
            mapping depends on the provided bases. If provided bases are ``AGCT``, then
            :class:`standard_nt_ambiguity_map` is used. If bases are ``AGCT-``, then
            :class:`standard_nt_ambiguity_map_gap_as_char` is used. Otherwise, it is assumed that no ambiguity
            codes are allowed. To override these defaults, provide an :class:`AmbiguityMap` explicitly.
    """

    def __init__(
        self,
        bases: CharacterSequence = tuple("AGCT"),
        transition_weights: Sequence[Sequence[float]] = None,
        ambiguity_map: AmbiguityMap = None,
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
            if set(self.bases) == set("AGCT"):
                self.ambiguity_map = standard_nt_ambiguity_map
            elif set(self.bases) == set("AGCT-"):
                self.ambiguity_map = standard_nt_ambiguity_map_gap_as_char
            else:
                self.ambiguity_map = AmbiguityMap({}, self.bases)
        else:
            if not isinstance(ambiguity_map, AmbiguityMap):
                raise ValueError(
                    "ambiguity_map, if provided, must be a historydag.parsimony.AmbiguityMap object"
                )
            if ambiguity_map.bases != set(self.bases):
                raise ValueError(
                    "ambiguity_map.bases does not match bases provided to TransitionModel constructor"
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

    def get_adjacency_array(self, seq_len: int) -> np.array:
        """Return an adjacency array for a sequence of length ``seq_len``.

        This is a matrix containing transition costs, with index order
        (site, from_base, to_base)
        """
        return np.array([self.transition_weights] * seq_len)

    def get_ambiguity_from_tuple(self, tup: Tuple) -> Character:
        """Retrieve an ambiguity code encoded by tup, which encodes a set of
        bases with a tuple of 0's and 1's.

        For example, if ``self.bases`` is 'AGCT'``, then ``(0, 1, 1,
        1)`` would return `A`.
        """
        return self.ambiguity_map.reversed[
            frozenset(self.bases[i] for i, flag in enumerate(tup) if flag == 0)
        ]

    def character_distance(
        self, parent_char: Character, child_char: Character, site: int = None
    ) -> float:
        """Return the transition cost from parent_char to child_char, two
        unambiguous characters.

        keyword argument ``site`` is ignored in this base class.
        """
        return self.transition_weights[self.base_indices[parent_char]][
            self.base_indices[child_char]
        ]

    def weighted_hamming_distance(
        self, parent_seq: CharacterSequence, child_seq: CharacterSequence
    ) -> float:
        """Return the sum of sitewise transition costs, from parent_seq to
        child_seq.

        Both parent_seq and child_seq are expected to be unambiguous.
        """
        if len(parent_seq) != len(child_seq):
            raise ValueError("sequence lengths do not match!")
        return sum(
            self.character_distance(pchar, cchar, site=(idx + 1))
            for idx, (pchar, cchar) in enumerate(zip(parent_seq, child_seq))
        )

    def weighted_cg_hamming_distance(
        self, parent_cg: "CompactGenome", child_cg: "CompactGenome"
    ) -> float:
        """Return the sum of sitewise transition costs, from parent_cg to
        child_cg.

        Both parent_seq and child_seq are expected to be unambiguous,
        but sites where parent_cg and child_cg both match their
        reference sequence are ignored, so this method is not suitable
        for weighted parsimony for transition matrices that contain
        nonzero entries along the diagonal.
        """
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

    def min_character_distance(
        self, parent_char: Character, child_char: Character, site=None
    ) -> float:
        """Allowing child_char to be ambiguous, returns the minimum possible
        transition weight between parent_char and child_char.

        Keyword argument ``site`` is ignored in this base class.
        """
        child_set = self.ambiguity_map[child_char]
        p_idx = self.base_indices[parent_char]
        return min(
            self.transition_weights[p_idx][self.base_indices[cbase]]
            for cbase in child_set
        )

    def min_weighted_hamming_distance(
        self, parent_seq: CharacterSequence, child_seq: CharacterSequence
    ) -> float:
        """Assuming the child_seq may contain ambiguous characters, returns the
        minimum possible transition weight between parent_seq and child_seq."""
        if len(parent_seq) != len(child_seq):
            raise ValueError("sequence lengths do not match!")
        return sum(
            self.min_character_distance(pchar, cchar, site=(idx + 1))
            for idx, (pchar, cchar) in enumerate(zip(parent_seq, child_seq))
        )

    def min_weighted_cg_hamming_distance(
        self, parent_cg: "CompactGenome", child_cg: "CompactGenome"
    ) -> float:
        """Return the sum of sitewise transition costs, from parent_cg to
        child_cg.

        child_cg may contain ambiguous characters, and in this case
        those sites will contribute the minimum possible transition cost
        to the returned value.

        Sites where parent_cg and child_cg both match their reference
        sequence are ignored, so this method is not suitable for
        weighted parsimony for transition matrices that contain nonzero
        entries along the diagonal.
        """
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

    def weighted_hamming_edge_weight(
        self, field_name: str
    ) -> Callable[["HistoryDagNode", "HistoryDagNode"], float]:
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

    def weighted_cg_hamming_edge_weight(
        self, field_name: str
    ) -> Callable[["HistoryDagNode", "HistoryDagNode"], float]:
        """Returns a function for computing weighted hamming distance between
        two nodes' compact genomes.

        Args:
            field_name: The name of the node label field which contains compact genomes.

        Returns:
            A function accepting two :class:`HistoryDagNode` objects ``n1`` and ``n2`` and returning
            a float: the transition cost from ``n1.label.<field_name>`` to ``n2.label.<field_name>``, or 0 if
            n1 is the UA node.

        Sites where parent_cg and child_cg
        both match their reference sequence are ignored, so this method is not suitable for weighted parsimony
        for transition matrices that contain nonzero entries along the diagonal.
        """

        @utils.access_nodefield_default(field_name, 0)
        def edge_weight(parent, child):
            return self.weighted_cg_hamming_distance(parent, child)

        return edge_weight

    def min_weighted_hamming_edge_weight(
        self, field_name: str
    ) -> Callable[["HistoryDagNode", "HistoryDagNode"], float]:
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

    def min_weighted_cg_hamming_edge_weight(
        self, field_name: str
    ) -> Callable[["HistoryDagNode", "HistoryDagNode"], float]:
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
        self,
        field_name: str,
        leaf_ambiguities: bool = False,
        name: str = "WeightedParsimony",
    ) -> AddFuncDict:
        """Create a :class:`historydag.utils.AddFuncDict` object for counting
        weighted parsimony in a HistoryDag with labels containing compact
        genomes.

        Args:
            field_name: the label field name in which compact genomes can be found
            leaf_ambiguities: if True, leaf compact genomes are permitted to contain ambiguity codes
            name: the name for the returned AddFuncDict object
        """
        if leaf_ambiguities:
            edge_weight = self.min_weighted_cg_hamming_edge_weight(field_name)
        else:
            edge_weight = self.weighted_cg_hamming_edge_weight(field_name)
        return AddFuncDict(
            {
                "start_func": lambda n: 0,
                "edge_weight_func": edge_weight,
                "accum_func": sum,
            },
            name=name,
        )

    def get_weighted_parsimony_countfuncs(
        self,
        field_name: str,
        leaf_ambiguities: bool = False,
        name: str = "WeightedParsimony",
    ):
        """Create a :class:`historydag.utils.AddFuncDict` object for counting
        weighted parsimony in a HistoryDag with labels containing sequences.

        Args:
            field_name: the label field name in which sequences can be found
            leaf_ambiguities: if True, leaf sequences are permitted to contain ambiguity codes
            name: the name for the returned AddFuncDict object
        """
        if leaf_ambiguities:
            edge_weight = self.min_weighted_hamming_edge_weight(field_name)
        else:
            edge_weight = self.weighted_hamming_edge_weight(field_name)
        return AddFuncDict(
            {
                "start_func": lambda n: 0,
                "edge_weight_func": edge_weight,
                "accum_func": sum,
            },
            name=name,
        )


class UnitTransitionModel(TransitionModel):
    """A subclass of :class:`TransitionModel` to be used when all non-identical
    transitions have unit cost.

    Args:
        bases: An ordered container of allowed character states
        ambiguity_map: A :class:`AmbiguityMap` object describing allowed ambiguity codes.
            If not provided, a default will be chosen by the same method as the constructor for
            :class:`TransitionModel`.
    """

    def __init__(
        self, bases: CharacterSequence = "AGCT", ambiguity_map: AmbiguityMap = None
    ):
        super().__init__(bases=bases, ambiguity_map=None)

    def character_distance(
        self, parent_char: Character, child_char: Character, site: int = None
    ) -> int:
        """Return the transition cost from parent_char to child_char, two
        unambiguous characters.

        keyword argument ``site`` is ignored in this subclass.
        """
        return int(parent_char != child_char)

    def min_character_distance(
        self, parent_char: Character, child_char: Character, site: int = None
    ) -> int:
        """Allowing child_char to be ambiguous, returns the minimum possible
        transition weight between parent_char and child_char.

        Keyword argument ``site`` is ignored in this subclass.
        """
        if parent_char == child_char:
            return 0  # TODO do we really want this?
        else:
            return int(parent_char not in self.ambiguity_map[child_char])

    def get_weighted_cg_parsimony_countfuncs(
        self,
        field_name: str,
        leaf_ambiguities: AmbiguityMap = False,
        name: str = "HammingParsimony",
    ) -> AddFuncDict:
        """Create a :class:`historydag.utils.AddFuncDict` object for counting
        parsimony in a HistoryDag with labels containing compact genomes.

        Args:
            field_name: the label field name in which compact genomes can be found
            leaf_ambiguities: if True, leaf compact genomes are permitted to contain ambiguity codes
            name: the name for the returned AddFuncDict object
        """
        return super().get_weighted_cg_parsimony_countfuncs(
            field_name, leaf_ambiguities=leaf_ambiguities, name=name
        )

    def get_weighted_parsimony_countfuncs(
        self,
        field_name: str,
        leaf_ambiguities: AmbiguityMap = False,
        name: str = "HammingParsimony",
    ) -> AddFuncDict:
        """Create a :class:`historydag.utils.AddFuncDict` object for counting
        weighted parsimony in a HistoryDag with labels containing sequences.

        Args:
            field_name: the label field name in which sequences can be found
            leaf_ambiguities: if True, leaf sequences are permitted to contain ambiguity codes
            name: the name for the returned AddFuncDict object
        """
        return super().get_weighted_parsimony_countfuncs(
            field_name, leaf_ambiguities=leaf_ambiguities, name=name
        )


class SitewiseTransitionModel(TransitionModel):
    """A subclass of :class:`TransitionModel` to be used when transition costs
    depend on site.

    Args:
        bases: An ordered container of allowed character states
        transition_matrix: A matrix of transition costs between provided bases, in which
            index order is (site, from_base, to_base)
        ambiguity_map: An :class:`AmbiguityMap` object describing ambiguous characters. If not provided
            a default will be chosen as in the constructor for :class:`TransitionModel`.
    """

    def __init__(
        self,
        bases: CharacterSequence = "AGCT",
        transition_matrix: Sequence[Sequence[Sequence[float]]] = None,
        ambiguity_map: AmbiguityMap = None,
    ):
        assert transition_matrix.shape[1:] == (len(bases), len(bases))
        super().__init__(bases=bases, ambiguity_map=None)
        self.transition_weights = None
        self.sitewise_transition_matrix = transition_matrix
        self._seq_len = len(self.sitewise_transition_matrix)

    def character_distance(
        self, parent_char: Character, child_char: Character, site: int
    ) -> float:
        """Returns the transition cost from ``parent_char`` to ``child_char``,
        assuming each character is at the one-based ``site`` in their
        respective sequences.

        Both parent_char and child_char are expected to be unambiguous.
        """
        return self.sitewise_transition_matrix[site - 1][parent_char][child_char]

    def min_character_distance(
        self, parent_char: Character, child_char: Character, site: int
    ) -> float:
        """Allowing child_char to be ambiguous, returns the minimum possible
        transition weight between parent_char and child_char, given that these
        characters are found at the (one-based) site in their respective
        sequences."""
        child_set = self.ambiguity_map[child_char]
        p_idx = self.base_indices[parent_char]
        return min(
            self.transition_weights[site - 1][p_idx][self.base_indices[cbase]]
            for cbase in child_set
        )

    def get_adjacency_array(self, seq_len):
        """Return an adjacency array for a sequence of length ``seq_len``.

        This is a matrix containing transition costs, with index order
        (site, from_base, to_base)
        """
        if seq_len != self._seq_len:
            raise ValueError(
                f"SitewiseTransitionModel instance supports sequence length of {self._seq_len}"
            )
        return self.sitewise_transition_matrix


default_nt_transitions = UnitTransitionModel(bases="AGCT")
"""A transition model for nucleotides using unit transition weights, and the
standard IUPAC ambiguity codes, with base order ``AGCT`` and ``N``, ``?`` and
``-`` treated as fully ambiguous characters."""

default_nt_gaps_transitions = UnitTransitionModel(bases="AGCT-")
"""A transition model for nucleotides using unit transition weights, and the
standard IUPAC nucleotide ambiguity map, with '-' is treated as a fifth base,
'?' fully ambiguous, and the ambiguity 'N' excluding only '-'."""

hamming_edge_weight = default_nt_transitions.weighted_hamming_edge_weight("sequence")
"""An edge weight function accepting (parent, child) pairs of HistoryDagNodes,
and returning the hamming distance between the sequences stored in the
``sequence`` attributes of their labels."""
hamming_edge_weight_ambiguous_leaves = (
    default_nt_transitions.min_weighted_hamming_edge_weight("sequence")
)
"""An edge weight function accepting (parent, child) pairs of HistoryDagNodes,
and returning the hamming distance between the sequences stored in the
``sequence`` attributes of their labels.

This edge weight function allows leaf nodes to have ambiguous sequences.
"""

hamming_cg_edge_weight = default_nt_transitions.weighted_cg_hamming_edge_weight(
    "compact_genome"
)
"""An edge weight function accepting (parent, child) pairs of HistoryDagNodes,
and returning the hamming distance between the compact genomes stored in the
``compact_genome`` attributes of their labels."""

hamming_cg_edge_weight_ambiguous_leaves = (
    default_nt_transitions.min_weighted_cg_hamming_edge_weight("compact_genome")
)
"""An edge weight function accepting (parent, child) pairs of HistoryDagNodes,
and returning the hamming distance between the compact genomes stored in the
``compact_genome`` attributes of their labels.

This edge weight function allows leaf nodes to have ambiguous compact
genomes.
"""

hamming_distance_countfuncs = default_nt_transitions.get_weighted_parsimony_countfuncs(
    "sequence", leaf_ambiguities=False
)
"""Provides functions to count hamming distance parsimony when leaf sequences
may be ambiguous.

For use with :meth:`historydag.AmbiguousLeafSequenceHistoryDag.weight_count`.
"""

leaf_ambiguous_hamming_distance_countfuncs = (
    default_nt_transitions.get_weighted_parsimony_countfuncs(
        "sequence", leaf_ambiguities=True
    )
)
"""Provides functions to count hamming distance parsimony when leaf sequences
may be ambiguous.

For use with :meth:`historydag.AmbiguousLeafSequenceHistoryDag.weight_count`.
"""


compact_genome_hamming_distance_countfuncs = (
    default_nt_transitions.get_weighted_cg_parsimony_countfuncs(
        "compact_genome", leaf_ambiguities=False
    )
)
"""Provides functions to count hamming distance parsimony when sequences are
stored as CompactGenomes.

For use with :meth:`historydag.CGHistoryDag.weight_count`.
"""


leaf_ambiguous_compact_genome_hamming_distance_countfuncs = (
    default_nt_transitions.get_weighted_cg_parsimony_countfuncs(
        "compact_genome", leaf_ambiguities=True
    )
)
"""Provides functions to count hamming distance parsimony when leaf compact
genomes may be ambiguous.

For use with :meth:`historydag.AmbiguousLeafCGHistoryDag.weight_count`.
"""
