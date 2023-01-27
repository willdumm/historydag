from frozendict import frozendict
from typing import Dict, Sequence
from warnings import warn


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

    def __le__(self, other: object) -> bool:
        if isinstance(other, CompactGenome):
            return sorted(self.mutations.items()) <= sorted(other.mutations.items())
        else:
            raise NotImplementedError

    def __lt__(self, other: object) -> bool:
        if isinstance(other, CompactGenome):
            return sorted(self.mutations.items()) < sorted(other.mutations.items())
        else:
            raise NotImplementedError

    def __gt__(self, other: object) -> bool:
        return not self.__le__(other)

    def __ge__(self, other: object) -> bool:
        return not self.__lt__(other)

    def __repr__(self):
        return (
            f"CompactGenome({self.mutations},"
            f" <reference sequence str with id:{id(self.reference)}>)"
        )

    def __str__(self):
        return f"CompactGenome[{', '.join(self.mutations_as_strings())}]"

    def get_site(self, site):
        """Get the base at the provided (one-based) site index."""
        mut = self.mutations.get(site)
        if mut is None:
            return self.reference[site - 1]
        else:
            return mut[-1]

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
        ref_base = self.reference[idx - 1]
        idx_present = idx in self.mutations
        if idx_present:
            old_recorded_base = self.mutations[idx][1]
        else:
            old_recorded_base = ref_base

        if oldbase != old_recorded_base:
            warn("recorded old base in sequence doesn't match old base")
        if ref_base == newbase:
            if idx_present:
                return CompactGenome(self.mutations.delete(idx), self.reference)
            else:
                return self
        return CompactGenome(
            self.mutations.set(idx, (ref_base, newbase)), self.reference
        )

    def apply_muts(self, muts: Sequence[str], reverse: bool = False):
        """Apply a sequence of mutstrings like 'A110G' to this compact genome.

        In this example, A is the old base, G is the new base, and 110 is the 1-based
        index of the mutation in the sequence. Returns the new
        CompactGenome, and prints a warning if the old base doesn't
        match the recorded old base in this compact genome.

        Args:
            muts: The mutations to apply, in the order they should be applied
            reverse: Apply the mutations in reverse, such as when the provided mutations
                describe how to achieve this CompactGenome from the desired CompactGenome.
                If True, the mutations in `muts` will also be applied in reversed order.

        Returns:
            The new CompactGenome
        """
        newcg = self
        if reverse:
            mod_func = reversed
        else:

            def mod_func(seq):
                yield from seq

        for mut in mod_func(muts):
            newcg = newcg.mutate(mut, reverse=reverse)
        return newcg

    def to_sequence(self):
        """Convert this CompactGenome to a full nucleotide sequence."""
        newseq = []
        newseq = list(self.reference)
        for idx, (ref_base, newbase) in self.mutations.items():
            if ref_base != newseq[idx - 1]:
                warn(
                    "CompactGenome.to_sequence warning: reference base doesn't match cg reference base"
                )
            newseq[idx - 1] = newbase
        return "".join(newseq)

    def mask_sites(self, sites, one_based=True):
        """Remove any mutations on sites in `sites`, leaving the reference
        sequence unchanged.

        Args:
            sites: A collection of sites to be masked
            one_based: If True, the provided sites will be interpreted as one-based sites. Otherwise,
                they will be interpreted as 0-based sites.
        """
        sites = set(sites)
        if one_based:

            def site_translate(site):
                return site

        else:

            def site_translate(site):
                return site - 1

        return CompactGenome(
            {
                site: data
                for site, data in self.mutations.items()
                if site_translate(site) not in sites
            },
            self.reference,
        )


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


def cg_diff(parent_cg: CompactGenome, child_cg: CompactGenome):
    """Yields mutations in the format (parent_nuc, child_nuc, sequence_index)
    distinguishing two compact genomes, such that applying the resulting
    mutations to `parent_cg` would yield `child_cg`"""
    keys = set(parent_cg.mutations.keys()) | set(child_cg.mutations.keys())
    for key in keys:
        parent_base = parent_cg.get_site(key)
        child_base = child_cg.get_site(key)
        if parent_base != child_base:
            yield (parent_base, child_base, key)
