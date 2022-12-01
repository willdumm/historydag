.. currentmodule:: historydag

Description
-----------

This page documents the API for the ``historydag`` package.
The fundamental data structure implemented here is the :class:`HistoryDag`.
This data structure provides efficient storage for collections of trees with
internal node labels. In this package, we refer to such a tree as a _history_.
A history can be represented as a tree-shaped :class:`HistoryDag` object.

This package provides functions for:

* creating histories from tree data,
* merging histories together to create history DAGs,
* doing efficient computation on collections of histories stored in history DAGs, and
* accessing histories contained in a history DAG, and exporting them to other
  tree formats.

Classes
-------

Top level classes, promoted from the ``dag`` module.

.. autosummary::
    :toctree: stubs

    HistoryDag
    HistoryDagNode

Functions
---------

Top level functions, promoted from the ``dag`` module.

.. autosummary::
    :toctree: stubs

    from_tree
    empty_node
    from_newick
    history_dag_from_newicks
    history_dag_from_etes
    history_dag_from_histories

Modules
-------

.. autosummary::
    :toctree: stubs

    dag
    sequence_dag
    mutation_annotated_dag
    utils
    parsimony
    compact_genome
    counterops
