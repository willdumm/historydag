# Remember to add any additional functions/modules to docs/api.rst
from .dag import (  # noqa
    HistoryDag,
    HistoryDagNode,
    from_tree,
    empty_node,
    from_newick,
    history_dag_from_newicks,
    history_dag_from_etes,
    history_dag_from_histories,
    history_dag_from_trees,
)

from . import _version

from . import (  # noqa
    utils,
    parsimony_utils,
    parsimony,
    mutation_annotated_dag,
    sequence_dag,
    compact_genome,
)

try:
    # requires dendropy
    from . import beast_loader  # noqa
except ModuleNotFoundError:
    pass

__version__ = _version.get_versions()["version"]

# patching deprecated call paths here to avoid circular import in utils:
utils.sequence_resolutions_count = utils._deprecate_message(
    "`utils.sequence_resolutions_count` deprecated. Use the `get_sequence_resolution_count_func` method from an "
    "appropriate `parsimony_utils.AmbiguityMap` object`"
)(
    parsimony_utils.standard_nt_ambiguity_map.get_sequence_resolution_count_func(
        "sequence"
    )
)

utils.sequence_resolutions = utils._deprecate_message(
    "`utils.sequence_resolutions` deprecated. Use the `get_sequence_resolution_func` method from an "
    "appropriate `parsimony_utils.AmbiguityMap` object`"
)(parsimony_utils.standard_nt_ambiguity_map.get_sequence_resolution_func("sequence"))
