import historydag.parsimony_utils as parsimony_utils
from historydag.utils import AddFuncDict
from math import log


def _JC_branch_length_raw(seq_len, mutated_sites):
    p = mutated_sites / seq_len
    return -(0.75) * log(1 - ((4 * p) / 3))


def _JC_log_edge_weight_raw(seq_len, mutated_sites):
    if mutated_sites == 0:
        return 0
    else:
        N = seq_len
        m = mutated_sites
        return m * (log(m) - log(3 * N)) + (N - m) * (log(N - m) - log(N))


def JC_branch_length(seq1, seq2):
    return _JC_branch_length_raw(
        len(seq1),
        parsimony_utils.default_nt_transitions.weighted_hamming_distance(seq1, seq2),
    )


def JC_log_edge_weight(seq1, seq2):
    mutated_sites = parsimony_utils.default_nt_transitions.weighted_hamming_distance(
        seq1, seq2
    )
    return _JC_log_edge_weight_raw(len(seq1), mutated_sites)


def JC_cg_branch_length(seq1, seq2):
    return _JC_branch_length_raw(
        len(seq1.reference),
        parsimony_utils.default_nt_transitions.weighted_cg_hamming_distance(seq1, seq2),
    )


def JC_cg_log_edge_weight(seq1, seq2):
    mutated_sites = parsimony_utils.default_nt_transitions.weighted_cg_hamming_distance(
        seq1, seq2
    )
    return _JC_log_edge_weight_raw(len(seq1.reference), mutated_sites)


JC_cg_branch_length_countfuncs = AddFuncDict(
    {
        "start_func": lambda n: 0,
        "edge_weight_func": lambda n1, n2: 0
        if n1.is_ua_node()
        else JC_cg_branch_length(n1.label.compact_genome, n2.label.compact_genome),
        "accum_func": sum,
    },
    name="JukesCantorBranchLength",
)

JC_cg_log_countfuncs = AddFuncDict(
    {
        "start_func": lambda n: 0,
        "edge_weight_func": lambda n1, n2: 0
        if n1.is_ua_node()
        else JC_cg_log_edge_weight(n1.label.compact_genome, n2.label.compact_genome),
        "accum_func": sum,
    },
    name="JukesCantorLogLikelihood",
)

JC_branch_length_countfuncs = AddFuncDict(
    {
        "start_func": lambda n: 0,
        "edge_weight_func": lambda n1, n2: 0
        if n1.is_ua_node()
        else JC_branch_length(n1.label.sequence, n2.label.sequence),
        "accum_func": sum,
    },
    name="JukesCantorBranchLength",
)

JC_log_countfuncs = AddFuncDict(
    {
        "start_func": lambda n: 0,
        "edge_weight_func": lambda n1, n2: 0
        if n1.is_ua_node()
        else JC_log_edge_weight(n1.label.sequence, n2.label.sequence),
        "accum_func": sum,
    },
    name="JukesCantorLogLikelihood",
)
