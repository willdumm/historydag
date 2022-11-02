from frozendict import frozendict
from typing import Dict, Sequence
from warnings import warn
import historydag.utils


class CompactGenome:
    """A collection of mutations relative to a reference sequence.

    Args:
        mutations: The difference between the reference and this sequence, expressed
            in a dictionary, in which keys are one-based sequence indices, and values
            are (reference base, new base) pairs.
        reference: The reference sequence
    """

    def __init__(self, mutations: Dict, reference: str):
        self.reference = reference
        self.mutations = frozendict(mutations)

    def __hash__(self):
        return hash(self.mutations)

    def __eq__(self, other):
        return (self.mutations, self.reference) == (other.mutations, other.reference)

    def __repr__(self):
        return f"CompactGenome({self.mutations}, '{self.reference}')"

    def __str__(self):
        return f"CompactGenome[{', '.join(self.mutations_as_strings())}]"

    def mutations_as_strings(self):
        """Return mutations as a tuple of strings of the format '<reference
        base><index><new base>', sorted by index."""
        return tuple(
            (startbase + str(index) + endbase)
            for index, (startbase, endbase) in sorted(
                self.mutations.items(), key=lambda t: t[0]
            )
        )

    def mutate(self, mutstring: str, reverse: bool = False):
        """Apply a mutstring such as 'A110G' to this compact genome.

        In this example, A is the old base, G is the new base, and 110 is the 1-based
        index of the mutation in the sequence. Returns the new
        CompactGenome, and prints a warning if the old base doesn't
        match the recorded old base in this compact genome.

        Args:
            mutstring: The mutation to apply
            reverse: Apply the mutation in reverse, such as when the provided mutation
                describes how to achieve this CompactGenome from the desired CompactGenome.
        Returns:
            The new CompactGenome
        """
        oldbase = mutstring[0]
        newbase = mutstring[-1]
        if reverse:
            oldbase, newbase = newbase, oldbase
        idx = int(mutstring[1:-1])
        if idx in self.mutations:
            if self.mutations[idx][0] == newbase:
                return CompactGenome(self.mutations.delete(idx), self.reference)
            else:
                if self.mutations[idx][1] != oldbase:
                    warn("recorded old base in sequence doesn't match old base")
                return CompactGenome(
                    self.mutations.set(idx, (self.mutations[idx][0], newbase)),
                    self.reference,
                )
        else:
            return CompactGenome(
                self.mutations.set(idx, (oldbase, newbase)), self.reference
            )

    def apply_muts(self, muts: Sequence[str], reverse: bool = False):
        """Apply a sequence of mutstrings like 'A110G' to this compact genome.

        In this example, A is the old base, G is the new base, and 110 is the 1-based
        index of the mutation in the sequence. Returns the new
        CompactGenome, and prints a warning if the old base doesn't
        match the recorded old base in this compact genome.

        Args:
            muts: The mutations to apply
            reverse: Apply the mutations in reverse, such as when the provided mutations
                describe how to achieve this CompactGenome from the desired CompactGenome.

        Returns:
            The new CompactGenome
        """
        newcg = self
        for mut in muts:
            newcg = newcg.mutate(mut, reverse=reverse)
        return newcg

    def to_sequence(self):
        """Convert this CompactGenome to a full nucleotide sequence."""
        newseq = []
        newseq = list(self.reference)
        for idx, (ref_base, newbase) in self.mutations.items():
            if ref_base != newseq[idx - 1]:
                print(
                    "CompactGenome.to_sequence warning: reference base doesn't match cg reference base"
                )
            newseq[idx - 1] = newbase
        return "".join(newseq)


def compact_genome_from_sequence(sequence: str, reference: str):
    """Create a CompactGenome from a sequence and a reference sequence.

    Args:
        sequence: the sequence to be represented by a CompactGenome
        reference: the reference sequence for the CompactGenome
    """
    cg = {
        zero_idx + 1: (old_base, new_base)
        for zero_idx, (old_base, new_base) in enumerate(zip(reference, sequence))
        if old_base != new_base
    }
    return CompactGenome(cg, reference)


def cg_hamming_distance(seq1: CompactGenome, seq2: CompactGenome):
    """An implementation of hamming distance on compact genomes."""
    if seq1.reference != seq2.reference:
        raise ValueError("Reference sequences do not match!")
    s1 = set(seq1.mutations.keys())
    s2 = set(seq2.mutations.keys())
    return (
        len(s1 - s2)
        + len(s2 - s1)
        + len(
            [1 for idx in s1 & s2 if seq1.mutations[idx][1] != seq2.mutations[idx][1]]
        )
    )


@historydag.utils.access_nodefield_default("compact_genome", 0)
def wrapped_cg_hamming_distance(s1, s2) -> int:
    """The sitewise sum of base differences between sequence field contents of
    two nodes.

    Takes two HistoryDagNodes as arguments.

    If l1 or l2 is a UANode, returns 0.
    """
    return cg_hamming_distance(s1, s2)


def cg_diff(parent_cg: CompactGenome, child_cg: CompactGenome):
    """Yields mutations in the format (parent_nuc, child_nuc, sequence_index)
    distinguishing two compact genomes, such that applying the resulting
    mutations to `parent_cg` would yield `child_cg`"""
    keys = set(parent_cg.mutations.keys()) | set(child_cg.mutations.keys())
    for key in keys:
        if key in parent_cg.mutations:
            parent_base = parent_cg.mutations[key][1]
        else:
            parent_base = child_cg.mutations[key][0]
        if key in child_cg.mutations:
            new_base = child_cg.mutations[key][1]
        else:
            new_base = parent_cg.mutations[key][0]
        if parent_base != new_base:
            yield (parent_base, new_base, key)
