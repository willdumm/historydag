import historydag.compact_genome as compact_genome
from historydag.compact_genome import CompactGenome
from historydag.parsimony_utils import default_aa_transitions


def test_sequence_cg_convert():
    seqs = [
        "AAAA",
        "TAAT",
        "CTGA",
        "TGCA",
    ]
    for refseq in seqs:
        for seq in seqs:
            cg = compact_genome.compact_genome_from_sequence(seq, refseq)
            reseq = cg.to_sequence()
            if reseq != seq:
                print("\nUnmatched reconstructed sequence:")
                print("ref sequence:", refseq)
                print("sequence:", seq)
                print("cg:", cg)
                print("reconstructed sequence:", reseq)
                assert False


def test_cg_diff():
    refseq = "C" * 1000
    cgs = [
        CompactGenome({287: ("C", "G")}, refseq),
        CompactGenome({287: ("C", "G"), 318: ("C", "A"), 495: ("C", "T")}, refseq),
        CompactGenome(
            {287: ("C", "G"), 80: ("C", "T"), 257: ("C", "G"), 591: ("C", "A")}, refseq
        ),
        CompactGenome(
            {
                287: ("C", "G"),
                191: ("C", "G"),
                492: ("C", "G"),
                612: ("C", "G"),
                654: ("C", "G"),
            },
            refseq,
        ),
        CompactGenome({287: ("C", "G"), 318: ("C", "A"), 495: ("C", "T")}, refseq),
    ]
    for parent_cg in cgs:
        for child_cg in cgs:
            assert (
                parent_cg.apply_muts(
                    str(p) + str(key) + str(c)
                    for p, c, key in compact_genome.cg_diff(parent_cg, child_cg)
                )
                == child_cg
            )
            assert all(
                par_nuc != child_nuc
                for par_nuc, child_nuc, idx in compact_genome.cg_diff(
                    parent_cg, child_cg
                )
            )


def test_ambiguous_cg_distance():
    reference_seq = "AAAAAAAN"

    s1 = compact_genome.CompactGenome({1: ("A", "T"), 2: ("A", "G")}, reference_seq)
    s2 = compact_genome.CompactGenome({1: ("A", "T"), 2: ("A", "N")}, reference_seq)
    s3 = compact_genome.CompactGenome({1: ("A", "T")}, reference_seq)
    s4 = compact_genome.CompactGenome({8: ("N", "C")}, reference_seq)

    assert default_aa_transitions.min_weighted_cg_hamming_distance(s1, s2) == 0
    assert default_aa_transitions.min_weighted_cg_hamming_distance(s1, s3) == 1
    assert default_aa_transitions.min_weighted_cg_hamming_distance(s1, s4) == 3
    assert default_aa_transitions.min_weighted_cg_hamming_distance(s4, s1) == 2
    assert default_aa_transitions.min_weighted_cg_hamming_distance(s3, s2) == 0
    assert default_aa_transitions.min_weighted_cg_hamming_distance(s4, s2) == 1
    assert default_aa_transitions.min_weighted_cg_hamming_distance(s4, s3) == 1
