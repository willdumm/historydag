from historydag.mutation_annotated_dag import load_MAD_protobuf_file
from historydag.sequence_dag import SequenceHistoryDag


def test_load_protobuf():
    dag = load_MAD_protobuf_file("sample_data/full_dag.pb")
    dag._check_valid()


def test_convert_cgdag_seqdag():
    dag = load_MAD_protobuf_file("sample_data/full_dag.pb")
    sdag = SequenceHistoryDag.from_history_dag(dag)
    sdag._check_valid()
