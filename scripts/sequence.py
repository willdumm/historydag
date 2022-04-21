from frozendict import frozendict

"""functions for using a frozendict to represent a sequence, or 'compact genome' object, with sequence indices as keys,
and (old_base, new_base) tuples as values. Applying mutations can at times overwrite mutations already present in a
compact genome. The goal is to store only those mutations relative a fixed reference sequence, that are necessary to describe the state of 'this' sequence."""

def mutate(seq_dict, mutstring, reverse=False):
    """Apply a mutstring such as 'A110G' to the collapsed genome `seq_dict`.
    A is the old base, G is the new base, and 110 is the 1-based index of
    the mutation in the sequence."""
    oldbase = mutstring[0]
    newbase = mutstring[-1]
    if reverse:
        oldbase, newbase = newbase, oldbase
    idx = int(mutstring[1:-1])
    if idx in seq_dict:
        if seq_dict[idx][0] == newbase:
            return seq_dict.delete(idx)
        else:
            if seq_dict[idx][1] != oldbase:
                print("warning: recorded old base in sequence doesn't match old base")
            return seq_dict.set(idx, (seq_dict[idx][0], newbase))
    else:
        return seq_dict.set(idx, (oldbase, newbase))

def distance(seq1, seq2):
    """An implementation of hamming distance on collapsed genomes"""
    s1 = set(seq1.keys())
    s2 = set(seq2.keys())
    return (len(s1 - s2)
            + len(s2 - s1)
            + len([1 for idx in s1 & s2 if seq1[idx][1] != seq2[idx][1]]))
