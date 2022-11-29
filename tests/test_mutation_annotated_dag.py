from historydag.mutation_annotated_dag import (
    load_MAD_protobuf_file,
    load_json_file,
)
from historydag.sequence_dag import SequenceHistoryDag

pbdag = load_MAD_protobuf_file("sample_data/full_dag.pb")


def test_load_protobuf():
    dag = load_MAD_protobuf_file("sample_data/full_dag.pb")
    dag._check_valid()
    test_filename = "_test_write_pb.pb"
    dag.to_protobuf_file(test_filename)
    ndag = load_MAD_protobuf_file(test_filename)
    ndag._check_valid()
    ndag.convert_to_collapsed()
    assert dag.weight_count() == ndag.weight_count()


def test_load_json():
    dag = pbdag.copy()
    dag._check_valid()
    test_filename = "_test_write_json.json"
    dag.to_json_file(test_filename)
    ndag = load_json_file(test_filename)
    assert dag.test_equal(ndag)
    ndag._check_valid()
    assert dag.weight_count() == ndag.weight_count()


def test_weight_count():
    sdag = SequenceHistoryDag.from_history_dag(pbdag.copy())
    cdag = pbdag.copy()
    assert cdag.weight_count() == sdag.weight_count()
    assert cdag.optimal_weight_annotate() == sdag.optimal_weight_annotate()
    assert cdag.trim_optimal_weight() == sdag.trim_optimal_weight()
