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
)

from . import _version

from . import (  # noqa
    utils,
    parsimony,
    mutation_annotated_dag,
    sequence_dag,
    compact_genome,
)

__version__ = _version.get_versions()["version"]
