import historydag.compact_genome as compact_genome
from frozendict import frozendict


def _test_sequence_cg_convert():
    seqs = [
        "AAAA",
        "TAAT",
        "CTGA",
        "TGCA",
    ]
    for refseq in seqs:
        for seq in seqs:
            cg = compact_genome.sequence_to_cg(seq, refseq)
            reseq = cg.to_sequence()
            if reseq != seq:
                print("\nUnmatched reconstructed sequence:")
                print("ref sequence:", refseq)
                print("sequence:", seq)
                print("cg:", cg)
                print("reconstructed sequence:", reseq)
                assert False


def _test_cg_diff():
    cgs = [
        frozendict({287: ("C", "G")}),
        frozendict({287: ("C", "G"), 318: ("C", "A"), 495: ("C", "T")}),
        frozendict({287: ("C", "G"), 80: ("A", "C"), 257: ("C", "G"), 591: ("G", "A")}),
        frozendict(
            {
                287: ("C", "G"),
                191: ("A", "G"),
                492: ("C", "G"),
                612: ("C", "G"),
                654: ("A", "G"),
            }
        ),
        frozendict({287: ("C", "G"), 318: ("C", "A"), 495: ("C", "T")}),
    ]
    for parent_cg in cgs:
        for child_cg in cgs:
            assert (
                parent_cg.apply_muts(
                    compact_genome.str_mut_from_tups(
                        compact_genome.cg_diff(parent_cg, child_cg)
                    )
                )
                == child_cg
            )
            assert all(
                par_nuc != child_nuc
                for par_nuc, child_nuc, idx in compact_genome.cg_diff(
                    parent_cg, child_cg
                )
            )
