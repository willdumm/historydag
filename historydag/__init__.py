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
