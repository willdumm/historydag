"""A module providing the class HistoryDag, and supporting functions."""

import pickle
import graphviz as gv
import ete3
import random
from typing import (
    List,
    Callable,
    Any,
    Mapping,
    Generator,
    Iterable,
    Set,
    Optional,
    Tuple,
    NamedTuple,
    Dict,
    FrozenSet,
    Union,
    Sequence,
)
from collections import Counter
from copy import deepcopy

from historydag import utils
from historydag.utils import Weight, Label, UALabel, prod
from historydag.counterops import counter_sum, counter_prod


def _clade_union_dict(nodeseq: Sequence["HistoryDagNode"]) -> Dict:
    clade_dict: Dict[FrozenSet[Label], List[HistoryDagNode]] = {}
    for node in nodeseq:
        clade_union = node.clade_union()
        if clade_union not in clade_dict:
            clade_dict[clade_union] = []
        clade_dict[node.clade_union()].append(node)
    return clade_dict


class HistoryDagNode:
    r"""A recursive representation of a history DAG object.

    - a dictionary keyed by clades (frozensets) containing EdgeSet objects
    - a label, which is a namedtuple.
    """

    def __init__(self, label: Label, clades: dict, attr: Any):
        self.clades = clades
        # If passed a nonempty dictionary, need to add self to children's parents
        self.label = label
        self.parents: Set[HistoryDagNode] = set()
        self.attr = attr
        self._dp_data: Any = None
        if self.clades:
            for _, edgeset in self.clades.items():
                edgeset.parent = self
            for child in self.children():
                child.parents.add(self)

        if len(self.clades) == 1:
            raise ValueError(
                "Internal nodes (those which are not the DAG UA root node) "
                "may not have exactly one child clade; Unifurcations cannot be expressed "
                "in the history DAG."
            )

    def __repr__(self) -> str:
        return str((self.label, set(self.clades.keys())))

    def __hash__(self) -> int:
        return hash((self.label, self.child_clades()))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, HistoryDagNode):
            return (self.label, self.child_clades()) == (
                other.label,
                other.child_clades(),
            )
        else:
            raise NotImplementedError

    def __le__(self, other: object) -> bool:
        if isinstance(other, HistoryDagNode):
            return (self.label, self.sorted_child_clades()) <= (
                other.label,
                other.sorted_child_clades(),
            )
        else:
            raise NotImplementedError

    def __lt__(self, other: object) -> bool:
        if isinstance(other, HistoryDagNode):
            return (self.label, self.sorted_child_clades()) < (
                other.label,
                other.sorted_child_clades(),
            )
        else:
            raise NotImplementedError

    def __gt__(self, other: object) -> bool:
        return not self.__le__(other)

    def __ge__(self, other: object) -> bool:
        return not self.__lt__(other)

    def empty_copy(self) -> "HistoryDagNode":
        """Returns a HistoryDagNode object with the same clades, label, and
        attr dictionary, but no descendant edges."""
        return HistoryDagNode(
            self.label, {clade: EdgeSet() for clade in self.clades}, deepcopy(self.attr)
        )

    def node_self(self) -> "HistoryDagNode":
        """Deprecated name for :meth:`empty_copy`"""
        return self.empty_copy()

    def clade_union(self) -> FrozenSet[Label]:
        r"""Returns the union of this node's child clades (or a set containing
        only the node label, for leaf nodes.)"""
        if self.is_leaf():
            return frozenset([self.label])
        else:
            return frozenset().union(*self.clades.keys())

    def under_clade(self) -> FrozenSet[Label]:
        """Deprecated name for :meth:`clade_union`"""
        return self.clade_union()

    def is_leaf(self) -> bool:
        """Returns whether this is a leaf node."""
        return not bool(self.clades)

    def is_ua_node(self) -> bool:
        """Returns whether this is the source node in the DAG, from which all
        others are reachable."""
        return False

    def is_root(self) -> bool:
        """Deprecated name for :meth:`is_ua_node`"""
        return self.is_ua_node()

    def child_clades(self) -> frozenset:
        """Returns the node's child clades, or a frozenset containing a
        frozenset if this node is a UANode."""
        return frozenset(self.clades.keys())

    def partitions(self) -> frozenset:
        """Deprecated name for :meth:`child_clades`"""
        return self.child_clades()

    def sorted_child_clades(self) -> tuple:
        """Returns the node's child clades as a sorted tuple containing leaf
        labels in sorted tuples."""
        return tuple(sorted([tuple(sorted(clade)) for clade in self.clades.keys()]))

    def sorted_partitions(self) -> tuple:
        """Deprecated name for :meth:`sorted_child_clades`"""
        return self.sorted_child_clades()

    def children(
        self, clade: Set[Label] = None
    ) -> Generator["HistoryDagNode", None, None]:
        r"""Returns generator object containing child nodes.

        Args:
            clade: If clade is provided, returns generator object of edge targets from that
        clade. If no clade is provided, generator includes all children of self.
        """
        if clade is None:
            return (
                target for clade in self.clades for target, _, _ in self.clades[clade]
            )
        else:
            return (child for child, _, _ in self.clades[clade])

    def add_edge(
        self,
        target: "HistoryDagNode",
        weight: Weight = 0,
        prob: float = None,
        prob_norm: bool = True,
    ) -> bool:
        r"""Adds edge, if allowed and not already present.

        Returns whether edge was added.
        """
        # target clades must union to a clade of self
        key = frozenset() if self.is_ua_node() else target.clade_union()
        if key not in self.clades:
            raise KeyError("Target clades' union is not a clade of this parent node: ")
        else:
            target.parents.add(self)
            return self.clades[key].add_to_edgeset(
                target,
                weight=weight,
                prob=prob,
                prob_norm=prob_norm,
            )

    def _get_subhistory_by_subid(self, subid: int) -> "HistoryDagNode":
        r"""Returns the subtree below the current HistoryDagNode corresponding
        to the given index."""
        if self.is_leaf():  # base case - the node is a leaf
            return self
        else:
            history = self.empty_copy()

            # get the subtree for each of the clades
            for clade, eset in self.clades.items():
                # get the sum of subtrees of the edges for this clade
                num_subtrees = 0  # is this the right way to get the number of edges?
                for child, weight, _ in eset:
                    num_subtrees = num_subtrees + child._dp_data
                curr_index = subid % num_subtrees

                # find the edge corresponding to the curr_index
                for child, weight, _ in eset:
                    if curr_index >= child._dp_data:
                        curr_index = curr_index - child._dp_data
                    else:
                        # add this edge to the tree somehow
                        history.clades[clade].add_to_edgeset(
                            child._get_subhistory_by_subid(curr_index)
                        )
                        break

                subid = subid / num_subtrees
        return history

    def remove_edge_by_clade_and_id(self, target: "HistoryDagNode", clade: frozenset):
        key: frozenset
        if self.is_ua_node():
            key = frozenset()
        else:
            key = clade
        self.clades[key].remove_from_edgeset_byid(target)

    def remove_node(self, nodedict: Dict["HistoryDagNode", "HistoryDagNode"] = {}):
        r"""Recursively removes node self and any orphaned children from dag.

        May not work on root. Does not check to make sure that parent
        clade still has descendant edges.
        """
        if self in nodedict:
            nodedict.pop(self)
        for child in self.children():
            if self in child.parents:
                child.parents.remove(self)
            if not child.parents:
                child.remove_node(nodedict=nodedict)
        for parent in self.parents:
            parent.remove_edge_by_clade_and_id(self, self.clade_union())
        self.removed = True

    def _sample(self, edge_selector=lambda n: True) -> "HistoryDagNode":
        r"""Samples a history (a sub-history DAG containing the root and all
        leaf nodes).

        Returns a new HistoryDagNode object.
        """
        sample = self.empty_copy()
        for clade, eset in self.clades.items():
            mask = [edge_selector((self, target)) for target in eset.targets]
            sampled_target, target_weight = eset.sample(mask=mask)
            sampled_target_subsample = sampled_target._sample(
                edge_selector=edge_selector
            )
            sampled_target_subsample.parents = set([self])
            sample.clades[clade].add_to_edgeset(
                sampled_target_subsample,
                weight=target_weight,
            )
        return sample

    def _get_subhistories(self) -> Generator["HistoryDagNode", None, None]:
        r"""Return a generator to iterate through all trees expressed by the
        DAG."""

        def genexp_func(clade):
            # Return generator expression of all possible choices of tree
            # structure from dag below clade
            def f():
                eset = self.clades[clade]
                return (
                    (clade, targettree, i)
                    for i, target in enumerate(eset.targets)
                    for targettree in target._get_subhistories()
                )

            return f

        optionlist = [genexp_func(clade) for clade in self.clades]

        # TODO is this duplicated code?
        for option in utils.cartesian_product(optionlist):
            tree = self.empty_copy()
            for clade, targettree, index in option:
                tree.clades[clade].add_to_edgeset(
                    targettree,
                    weight=self.clades[clade].weights[index],
                )
            yield tree

    def _newick_label(
        self,
        name_func: Callable[["HistoryDagNode"], str] = (lambda n: "unnamed"),
        features: Iterable[str] = None,
        feature_funcs: Mapping[str, Callable[["HistoryDagNode"], str]] = {},
    ) -> str:
        """Return an extended newick format node label.

        Args:
            name_func: A function which maps nodes to names
            features: A list of label fields to be recorded in extended newick format
            feature_funcs: A dictionary keyed by extended newick field names containing
                functions which map nodes to field values. These override fields named
                in `features`, if a key in `feature_funcs` is also contained in `features`.

        Returns:
            A string which can be used as a node name in a newick string.
            For example, `namefuncresult[&&NHX:feature1=val1:feature2=val2]`
        """
        if self.is_ua_node():
            return self.label  # type: ignore
        else:
            if features is None:
                features = self.label._fields
            # Use dict to avoid duplicate fields
            nameval_dict = {
                name: val
                for name, val in self.label._asdict().items()
                if name in features
            }
            nameval_dict.update({name: f(self) for name, f in feature_funcs.items()})
            featurestr = ":".join(f"{name}={val}" for name, val in nameval_dict.items())
            return name_func(self) + (f"[&&NHX:{featurestr}]" if featurestr else "")


class UANode(HistoryDagNode):
    r"""A universal ancestor node, the root node of a HistoryDag."""

    def __init__(self, targetnodes: "EdgeSet"):
        self.label = UALabel()
        # an empty frozenset is not used as a key in any other node
        self.targetnodes = targetnodes
        self.clades = {frozenset(): targetnodes}
        self.parents = set()
        self.attr = dict()
        for child in self.children():
            child.parents.add(self)

    def empty_copy(self) -> "UANode":
        """Returns a UANode object with the same clades and label, but no
        descendant edges."""
        newnode = UANode(EdgeSet())
        newnode.attr = deepcopy(self.attr)
        return newnode

    def is_ua_node(self) -> bool:
        """Returns whether this is the source node in the DAG, from which all
        others are reachable."""
        return True


class HistoryDag:
    r"""An object to represent a collection of internally labeled trees. A
    wrapper object to contain exposed HistoryDag methods and point to a
    HistoryDagNode root.

    Args:
        dagroot: The root node of the history DAG
        attr: An attribute to contain data which will be preserved by copying (default and empty dict)
    """

    def __init__(self, dagroot: HistoryDagNode, attr: Any = {}):
        assert isinstance(dagroot, UANode)  # for typing
        self.attr = attr
        self.dagroot = dagroot
        self.current = 0

    def __eq__(self, other: object) -> bool:
        # Eventually this can be done by comparing bytestrings, but we need
        # some sorting to be done first, to ensure two dags that represent
        # identical trees return True. TODO
        raise NotImplementedError

    def __getitem__(self, key) -> "HistoryDag":
        r"""Returns the sub-history below the current history dag corresponding
        to the given index."""
        length = self.count_histories()
        if key < 0:
            key = length + key
        if isinstance(key, slice) or not type(key) == int:
            raise TypeError(f"History DAG indices must be integers, not {type(key)}")
        if not (key >= 0 and key < length):
            raise IndexError
        self.count_histories()
        return HistoryDag(self.dagroot._get_subhistory_by_subid(key))

    def __len__(self) -> int:
        return self.count_histories()

    def __or__(self, other) -> "HistoryDag":
        newdag = self.copy()
        newdag.merge(other)
        return newdag

    def __ior__(self, other) -> "HistoryDag":
        self.merge(other)
        return self

    def __ror__(self, other) -> "HistoryDag":
        return other | self

    def __getstate__(self) -> Dict:
        r"""Converts HistoryDag to a bytestring-serializable dictionary.

        Since a HistoryDag is a recursive data structure, and contains label
        types defined in function scope, modifications must be made for pickling.

        Returns:
            A dictionary containing:
            * label_fields: The names of label fields.
            * label_list: labels used in nodes, without duplicates. Indices are
                mapped to nodes in node_list
            * node_list: node tuples containing
                (node label index in label_list, tuple of frozensets of leaf label indices, node.attr).
            * edge_list: a tuple for each edge:
                    (origin node index, target node index, edge weight, edge probability)
        """
        label_fields = list(self.dagroot.children())[0].label._fields
        label_list: List[Optional[Tuple]] = []
        node_list: List[Tuple] = []
        edge_list: List[Tuple] = []
        label_indices: Dict[Label, int] = {}
        node_indices = {node: idx for idx, node in enumerate(self.postorder())}

        def cladesets(node):
            return tuple(
                frozenset({label_indices[label] for label in clade})
                for clade in node.clades
            )

        for node in self.postorder():
            if node.label not in label_indices:
                label_indices[node.label] = len(label_list)
                label_list.append(None if node.is_ua_node() else tuple(node.label))
                assert (
                    label_list[label_indices[node.label]] == node.label
                    or node.is_ua_node()
                )
            node_list.append((label_indices[node.label], cladesets(node), node.attr))
            node_idx = len(node_list) - 1
            for eset in node.clades.values():
                for idx, target in enumerate(eset.targets):
                    edge_list.append(
                        (
                            node_idx,
                            node_indices[target],
                            eset.weights[idx],
                            eset.probs[idx],
                        )
                    )
        serial_dict = {
            "label_fields": label_fields,
            "label_list": label_list,
            "node_list": node_list,
            "edge_list": edge_list,
            "attr": self.attr,
        }
        return serial_dict

    def __setstate__(self, serial_dict):
        """Rebuilds a HistoryDagNode using a serial_dict output by
        __getstate__"""
        label_list: List[Tuple] = serial_dict["label_list"]
        node_list: List[Tuple] = serial_dict["node_list"]
        edge_list: List[Tuple[int, int, float, float]] = serial_dict["edge_list"]
        label_fields: Tuple[str] = serial_dict["label_fields"]
        Label = NamedTuple("Label", [(label, any) for label in label_fields])  # type: ignore

        def unpack_labels(labelset):
            res = frozenset({Label(*label_list[idx]) for idx in labelset})
            return res

        node_postorder = [
            UANode(EdgeSet())
            if label_list[labelidx] is None
            else HistoryDagNode(
                (Label(*label_list[labelidx])),
                {unpack_labels(clade): EdgeSet() for clade in clades},
                attr,
            )
            for labelidx, clades, attr in node_list
        ]
        # Last node in list is root
        for origin_idx, target_idx, weight, prob in edge_list:
            node_postorder[origin_idx].add_edge(
                node_postorder[target_idx], weight=weight, prob=prob, prob_norm=False
            )
        self.dagroot = node_postorder[-1]
        self.attr = serial_dict["attr"]

    def _check_valid(self) -> bool:
        """Check that this HistoryDag complies with all the conditions of the
        definition."""
        # Traversal checks if a node has been visited by its id, which makes it
        # suitable for these checks.
        po = list(self.postorder())
        node_set = set(po)

        # ***Node instances are unique (And therefore leaves are uniquely labeled also):
        if len(po) != len(node_set):
            raise ValueError("Node instances are not unique")

        # ***All nodes are reachable from the UA node: this is proven by the
        # structure of the postorder traversal; if a node is visited, then it's
        # reachable by following directed edges downward. (parent sets aren't
        # used in the traversal)

        for node in po:
            if not node.is_ua_node():
                for clade, eset in node.clades.items():
                    for child in eset.targets:
                        # ***Parent clade equals child clade union for all edges:
                        if child.clade_union() != clade:
                            raise ValueError(
                                "Parent clade does not equal child clade union"
                            )

        for node in po:
            for clade, eset in node.clades.items():
                # ***At least one edge descends from each node-clade pair:
                if len(eset.targets) == 0:
                    raise ValueError("Found a clade with no child edges")
                # ...and there are no duplicate children:
                if len(eset.targets) != len(set(eset.targets)):
                    raise ValueError(
                        "Duplicate child edges found descending from the same clade"
                    )
                # ...and the eset._targetset set is correct
                if eset._targetset != set(eset.targets):
                    raise ValueError("eset._targetset doesn't match eset.targets")

        parents = {node: [] for node in po}
        for node in po:
            for child in node.children():
                parents[child].append(node)
        for node in po:
            # ... and parent sets are correct:
            if node.parents != set(parents[node]):
                raise ValueError("Found an incorrect parent set")
            # ... and there are no duplicate parents:
            if len(parents[node]) != len(set(parents[node])):
                raise ValueError("Found duplicate parents")

        return True

    def serialize(self) -> bytes:
        return pickle.dumps(self.__getstate__())

    def get_histories(self) -> Generator["HistoryDag", None, None]:
        """Return a generator containing all internally labeled trees in the
        history DAG.

        Note that each history is a history DAG, containing a UA node.

        The order of these histories does not necessarily match the order of
        indexing. That is, ``dag.get_histories()`` and ``history for history in
        dag`` will result in different orderings. ``get_histories`` should
        be slightly faster, but possibly more memory intensive.
        """
        for history in self.dagroot._get_subhistories():
            yield HistoryDag(history)

    def get_trees(self) -> Generator["HistoryDag", None, None]:
        """Deprecated name for :meth:`get_histories`"""
        return self.get_histories()

    def get_leaves(self) -> Generator["HistoryDag", None, None]:
        """Return a generator containing all leaf nodes in the history DAG."""
        return (node for node in self.postorder() if node.is_leaf())

    def num_nodes(self) -> int:
        """Return the number of nodes in the DAG, not counting the UA node."""
        return sum(1 for _ in self.preorder(skip_ua_node=True))

    def num_leaves(self) -> int:
        """Return the number of leaf nodes in the DAG."""
        return sum(1 for _ in self.get_leaves())

    def sample(self, edge_selector=lambda e: True) -> "HistoryDag":
        r"""Samples a history from the history DAG. (A history is a sub-history
        DAG containing the root and all leaf nodes) For reproducibility, set
        ``random.seed`` before sampling.

        When there is an option, edges pointing to nodes on which `selection_func` is True
        will always be chosen.

        Returns a new HistoryDag object.
        """
        return HistoryDag(self.dagroot._sample(edge_selector=edge_selector))

    def nodes_above_node(self, node) -> Set[HistoryDagNode]:
        """Return a set of nodes from which the passed node is reachable along
        directed edges."""
        self.recompute_parents()
        mask_true = set()
        nodequeue = {node}
        while len(nodequeue) > 0:
            curr_node = nodequeue.pop()
            if curr_node not in mask_true:
                nodequeue.update(curr_node.parents)
                mask_true.add(curr_node)
        return mask_true

    def sample_with_node(self, node) -> "HistoryDag":
        """Samples a history which contains ``node`` from the history DAG.

        Sampling is likely unbiased from the distribution of trees in
        the DAG, conditioned on each sampled tree containing the passed
        node. However, if unbiased sampling from the conditional
        distribution is important, this should be tested.
        """

        mask_true = self.nodes_above_node(node)

        def edge_selector(edge):
            return edge[-1] in mask_true

        return self.sample(edge_selector=edge_selector)

    def sample_with_edge(self, edge) -> "HistoryDag":
        """Samples a history which contains ``edge`` (a tuple of
        HistoryDagNodes) from the history DAG.

        Sampling is likely unbiased from the distribution of trees in
        the DAG, conditioned on each sampled tree containing the passed
        edge. However, if unbiased sampling from the conditional
        distribution is important, this should be tested.
        """
        mask_true = self.nodes_above_node(edge[0])

        def edge_selector(inedge):
            return inedge[-1] in mask_true or inedge == edge

        return self.sample(edge_selector=edge_selector)

    def iter_covering_histories(
        self, cover_edges=False
    ) -> Generator["HistoryDag", None, None]:
        """Samples a sequence of histories which together contain all nodes in
        the history DAG.

        Histories are sampled using :meth:`sample_with_node`, starting
        with the nodes which are contained in the fewest of the DAG's
        histories. The sequence of trees is therefore non-deterministic
        unless ``random.seed`` is set.
        """
        node_counts = self.count_nodes()
        node_list = sorted(node_counts.keys(), key=lambda n: node_counts[n])
        visited = set()
        if cover_edges:
            part_list = [
                (parent, child) for parent in node_list for child in parent.children()
            ]
            sample_func = self.sample_with_edge

            def update_visited(tree):
                visited.update(
                    set(
                        (parent, child)
                        for parent in tree.preorder()
                        for child in parent.children()
                    )
                )

        else:
            part_list = node_list
            sample_func = self.sample_with_node

            def update_visited(tree):
                visited.update(set(tree.preorder()))

        for part in part_list:
            if part not in visited:
                tree = sample_func(part)
                olen = len(visited)
                update_visited(tree)
                # At least part must have been added.
                assert len(visited) > olen
                yield tree

    def unlabel(self) -> "HistoryDag":
        """Sets all internal node labels to be identical, and merges nodes so
        that all histories in the DAG have unique topologies."""

        newdag = self.copy()
        model_label = next(self.preorder(skip_ua_node=True)).label
        # initialize empty/default value for each item in model_label
        field_values = tuple(type(item)() for item in model_label)
        internal_label = type(model_label)(*field_values)
        for node in newdag.preorder(skip_ua_node=True):
            if not node.is_leaf():
                node.label = internal_label

        # Use merging method to eliminate duplicate nodes, by starting with
        # a subdag with no duplicate nodes.
        ret = newdag.sample()
        ret.merge(newdag)
        return ret

    def relabel(self, relabel_func: Callable[[HistoryDagNode], Label]) -> "HistoryDag":
        """Return a new HistoryDag with labels modified according to a provided
        function.

        `relabel_func` should take a node and return the new label
        appropriate for that node.
        """

        leaf_label_dict = {leaf.label: relabel_func(leaf) for leaf in self.get_leaves()}
        if len(leaf_label_dict) != len(set(leaf_label_dict.keys())):
            raise RuntimeError(
                "relabeling function maps multiple leaf nodes to the same new label"
            )

        def remove_abundance_clade(old_clade):
            return frozenset(leaf_label_dict[old_label] for old_label in old_clade)

        def remove_abundance_node(old_node):
            if old_node.is_ua_node():
                return UANode(
                    EdgeSet(
                        [
                            remove_abundance_node(old_child)
                            for old_child in old_node.children()
                        ]
                    )
                )
            else:
                clades = {
                    remove_abundance_clade(old_clade): EdgeSet(
                        [
                            remove_abundance_node(old_child)
                            for old_child in old_eset.targets
                        ],
                        weights=old_eset.weights,
                        probs=old_eset.probs,
                    )
                    for old_clade, old_eset in old_node.clades.items()
                }
                return HistoryDagNode(relabel_func(old_node), clades, None)

        newdag = HistoryDag(remove_abundance_node(self.dagroot))
        # do any necessary collapsing
        newdag = newdag.sample() | newdag
        return newdag

    def is_history(self) -> bool:
        """Returns whether history DAG is a history.

        That is, each node-clade pair has exactly one descendant edge.
        """
        for node in self.postorder():
            for clade, eset in node.clades.items():
                if len(eset.targets) != 1:
                    return False
        return True

    def is_clade_tree(self) -> bool:
        """Deprecated name for :meth:`is_history`"""
        return self.is_history()

    def copy(self) -> "HistoryDag":
        """Uses bytestring serialization, and is guaranteed to copy:

        * node labels
        * node attr attributes
        * edge weights
        * edge probabilities

        However, other object attributes will not be copied.
        """
        return pickle.loads(pickle.dumps(self))

    def merge(self, trees: Union["HistoryDag", Sequence["HistoryDag"]]):
        r"""Graph union this history DAG with all those in a list of history
        DAGs."""
        if isinstance(trees, HistoryDag):
            trees = [trees]

        selforder = self.postorder()
        nodedict = {n: n for n in selforder}

        for other in trees:
            if not self.dagroot == other.dagroot:
                raise ValueError(
                    f"The given HistoryDag must be a root node on identical taxa.\n{self.dagroot}\nvs\n{other.dagroot}"
                )
            otherorder = other.postorder()
            # hash and __eq__ are implemented for nodes, but we need to retrieve
            # the actual instance that's the same as a proposed node-to-add:
            for n in otherorder:
                if n in nodedict:
                    pnode = nodedict[n]
                else:
                    pnode = n.empty_copy()
                    nodedict[n] = pnode

                for _, edgeset in n.clades.items():
                    for child, weight, _ in edgeset:
                        pnode.add_edge(nodedict[child], weight=weight)

    def add_all_allowed_edges(self, *args, **kwargs) -> int:
        """Provided as a deprecated synonym for :meth:``make_complete``."""
        return self.make_complete(*args, **kwargs)

    def make_complete(
        self,
        new_from_root: bool = True,
        adjacent_labels: bool = True,
        preserve_parent_labels: bool = False,
    ) -> int:
        r"""Add all allowed edges to the DAG.

        Args:
            new_from_root: If False, no edges will be added that start at the DAG root.
                Useful when attempting to constrain root label.
            adjacent_labels: If False, no edges will be added between nodes with the same
                labels. Useful when attempting to maintain the history DAG in a 'collapsed'
                state.
            preserve_parent_labels: If True, ensures that for any edge added between a
                parent and child node, the parent node label was already among the original
                parent labels of the child node. This ensures that parsimony score is preserved.

        Returns:
            The number of edges added to the history DAG
        """
        n_added = 0
        if preserve_parent_labels is True:
            self.recompute_parents()
            uplabels = {
                node: {parent.label for parent in node.parents}
                for node in self.postorder()
            }

        clade_dict = _clade_union_dict(self.preorder(skip_ua_node=True))
        clade_dict[self.dagroot.clade_union()] = []

        for node in self.postorder():
            if new_from_root is False and node.is_ua_node():
                continue
            else:
                for clade in node.clades:
                    for target in clade_dict[clade]:
                        if adjacent_labels is False and target.label == node.label:
                            continue
                        elif (
                            preserve_parent_labels is True
                            and node.label not in uplabels[target]
                        ):
                            continue
                        else:
                            n_added += node.add_edge(target)
        return n_added

    @utils._history_method
    def to_newick(
        self,
        name_func: Callable[[HistoryDagNode], str] = lambda n: "unnamed",
        features: Optional[List[str]] = None,
        feature_funcs: Mapping[str, Callable[[HistoryDagNode], str]] = {},
    ) -> str:
        r"""Converts a history to extended newick format. Supports arbitrary
        node names and a sequence feature. For use on a history DAG which is a
        history.

        For extracting newick representations of trees in a general history DAG, see
        :meth:`HistoryDag.to_newicks`.

        Args:
            name_func: A map from nodes to newick node names
            features: A list of label field names to be included in extended newick data.
                If `None`, all label fields will be included. To include none of them,
                pass an empty list.
            feature_funcs: A dictionary keyed by extended newick field names, containing
                functions specifying how to populate that field for each node.

        Returns:
            A newick string. If `features` is an empty list, and feature_funcs is empty,
                then this will be a standard newick string. Otherwise, it will have ete3's
                extended newick format.
        """

        def newick(node):
            if node.is_leaf():
                return node._newick_label(
                    name_func, features=features, feature_funcs=feature_funcs
                )
            else:
                childnewicks = sorted([newick(node2) for node2 in node.children()])
                return (
                    "("
                    + ",".join(childnewicks)
                    + ")"
                    + node._newick_label(
                        name_func, features=features, feature_funcs=feature_funcs
                    )
                )

        return newick(next(self.dagroot.children())) + ";"

    @utils._history_method
    def to_ete(
        self,
        name_func: Callable[[HistoryDagNode], str] = lambda n: "unnamed",
        features: Optional[List[str]] = None,
        feature_funcs: Mapping[str, Callable[[HistoryDagNode], str]] = {},
    ) -> ete3.TreeNode:
        """Convert a history DAG which is a history to an ete tree.

        Args:
            name_func: A map from nodes to newick node names
            features: A list of label field names to be included in extended newick data.
                If `None`, all label fields will be included. To include none of them,
                pass an empty list.
            feature_funcs: A dictionary keyed by extended newick field names, containing
                functions specifying how to populate that field for each node.

        Returns:
            An ete3 Tree with the same topology as self, and node names and attributes
            as specified.
        """
        # First build a dictionary of ete3 nodes keyed by HDagNodes.
        if features is None:
            labelfeatures = list(
                list(self.dagroot.children())[0].label._asdict().keys()
            )
        else:
            labelfeatures = features

        def etenode(node: HistoryDagNode) -> ete3.TreeNode:
            newnode = ete3.TreeNode()
            newnode.name = name_func(node)
            for feature in labelfeatures:
                newnode.add_feature(feature, getattr(node.label, feature))
            for feature, func in feature_funcs.items():
                newnode.add_feature(feature, func(node))
            return newnode

        nodedict = {node: etenode(node) for node in self.preorder(skip_ua_node=True)}

        for node in nodedict:
            for target in node.children():
                nodedict[node].add_child(child=nodedict[target])

        # Since self is cladetree, dagroot can have only one child
        return nodedict[list(self.dagroot.children())[0]]

    def to_graphviz(
        self,
        labelfunc: Optional[Callable[[HistoryDagNode], str]] = None,
        namedict: Mapping[Label, str] = {},
        show_child_clades: bool = True,
        show_partitions: bool = None,
    ) -> gv.Digraph:
        r"""Converts history DAG to graphviz (dot format) Digraph object.

        Args:
            labelfunc: A function to label nodes. If None, nodes will be labeled by
                their DAG node labels, or their label hash if label data is too large.
            namedict: A dictionary from node labels to label strings. Labelfunc will be
                used instead, if both are provided.
            show_child_clades: Whether to include child clades in output.
            show_partitions: Deprecated alias for show_child_clades.
        """
        if show_partitions is not None:
            show_child_clades = show_partitions

        def labeller(label):
            if label in namedict:
                return str(namedict[label])
            elif len(str(tuple(label))) < 11:
                return str(tuple(label))
            else:
                return str(hash(label))

        def taxa(clade):
            ls = [labeller(taxon) for taxon in clade]
            ls.sort()
            return ",".join(ls)

        if labelfunc is None:
            labelfunc = utils.ignore_uanode("UA_node")(lambda n: labeller(n.label))

        G = gv.Digraph("labeled partition DAG", node_attr={"shape": "record"})
        for node in self.postorder():
            if node.is_leaf() or show_child_clades is False:
                G.node(str(id(node)), f"<label> {labelfunc(node)}")
            else:
                splits = "|".join(
                    [f"<{taxa(clade)}> {taxa(clade)}" for clade in node.clades]
                )
                G.node(str(id(node)), f"{{ <label> {labelfunc(node)} |{{{splits}}} }}")
            for clade in node.clades:
                for target, weight, prob in node.clades[clade]:
                    label = ""
                    if prob < 1.0:
                        label += f"p:{prob:.2f}"
                    if weight > 0.0:
                        label += f"w:{weight}"
                    G.edge(
                        f"{id(node)}:{taxa(clade) if show_child_clades else 'label'}:s",
                        f"{id(target)}:n",
                        label=label,
                    )
        return G

    def internal_avg_parents(self) -> float:
        r"""Returns the average number of parents among internal nodes.

        A simple measure of similarity between the trees that the DAG
        expresses. However, keep in mind that two trees with the same
        topology but different labels would be considered entirely
        unalike by this measure.
        """
        nonleaf_parents = (len(n.parents) for n in self.postorder() if not n.is_leaf())
        n = 0
        cumsum = 0
        for sum in nonleaf_parents:
            n += 1
            cumsum += sum
        # Exclude root:
        return cumsum / float(n - 1)

    def make_uniform(self):
        """Adjust edge probabilities so that the DAG expresses a uniform
        distribution on expressed trees.

        The probability assigned to each edge below a clade is
        proportional to the number of subtrees possible below that edge.
        """
        self.count_histories()
        for node in self.postorder():
            for clade, eset in node.clades.items():
                for i, target in enumerate(eset.targets):
                    eset.probs[i] = target._dp_data

    def explode_nodes(
        self,
        expand_func: Callable[[Label], Iterable[Label]] = utils.sequence_resolutions,
        expandable_func: Callable[[Label], bool] = None,
    ) -> int:
        r"""Explode nodes according to a provided function. Adds copies of each
        node to the DAG with exploded labels, but with the same parents and
        children as the original node.

        Args:
            expand_func: A function that takes a node label, and returns an iterable
                containing 'exploded' or 'disambiguated' labels corresponding to the original.
                The wrapper :meth:`utils.explode_label` is provided to make such a function
                easy to write.
            expandable_func: A function that takes a node label, and returns whether the
                iterable returned by calling expand_func on that label would contain more
                than one item.

        Returns:
            The number of new nodes added to the history DAG.
        """

        if expandable_func is None:

            def is_ambiguous(label):
                # Check if expand_func(label) has at least two items, without
                # exhausting the (arbitrarily expensive) generator
                return len(list(zip([1, 2], expand_func(label)))) > 1

        else:
            is_ambiguous = expandable_func

        self.recompute_parents()
        nodedict = {node: node for node in self.postorder()}
        nodeorder = list(self.postorder())
        new_nodes = set()
        for node in nodeorder:
            if not node.is_ua_node() and is_ambiguous(node.label):
                if node.is_leaf():
                    raise ValueError(
                        "Passed expand_func would explode a leaf node. "
                        "Leaf nodes may not be exploded."
                    )
                for resolution in expand_func(node.label):
                    newnodetemp = node.empty_copy()
                    newnodetemp.label = resolution
                    if newnodetemp in nodedict:
                        newnode = nodedict[newnodetemp]
                    else:
                        newnode = newnodetemp
                        nodedict[newnode] = newnode
                        new_nodes.add(newnode)
                    # Add all edges into and out of node to newnode
                    for target in node.children():
                        newnode.add_edge(target)
                    for parent in sorted(node.parents):
                        parent.add_edge(newnode)
                # Delete old node
                node.remove_node(nodedict=nodedict)
        return len(new_nodes)

    def leaf_path_uncertainty_dag(self, leaf_label):
        """Compute the DAG of possible paths leading to `leaf_label`.

        Args:
            leaf_label: The node label of the leaf of interest

        Returns:
            parent_dictionary: A dictionary keyed by node labels, with sets
                of possible parent node labels.
        """
        parent_dictionary = {
            node.label: set()
            for node in self.dagroot.children()
            if leaf_label in node.clade_union()
        }

        for node in self.preorder(skip_ua_node=True):
            for clade, eset in node.clades.items():
                if leaf_label in clade:
                    for cnode in eset.targets:
                        if cnode.label not in parent_dictionary:
                            parent_dictionary[cnode.label] = set()
                        # exclude self loops in label space
                        if node.label != cnode.label:
                            parent_dictionary[cnode.label].add(node.label)

        return parent_dictionary

    def leaf_path_uncertainty_graphviz_collapse(
        self, leaf_label, edge2count, total_trees
    ):
        """send output of leaf_path_uncertainty_dag to graphviz for rendering.

        Args:
            leaf_label: The node label of the leaf of interest
            edge2count: A map of edges in the form of node pairs (parent, child) to their count
                in the history DAG
            total_trees: The total number of trees contained in the history DAG

        Returns:
            The graphviz DAG object, and a dictionary mapping node names to labels
        """
        G = gv.Digraph("Path DAG to leaf", node_attr={})
        parent_d = self.leaf_path_uncertainty_dag(leaf_label)
        label_ids = {key: str(idnum) for idnum, key in enumerate(parent_d)}

        for key in parent_d:
            if key == leaf_label:
                G.node(label_ids[key], shape="octagon")
            elif len(parent_d[key]) == 0:
                G.node(label_ids[key], shape="invtriangle")
            else:
                G.node(label_ids[key])

        for child_label, parentset in parent_d.items():
            for parent_label in parentset:
                if child_label == parent_label:  # skip self-edges
                    continue
                support = 0
                for parent, child in edge2count.keys():
                    if (
                        parent.label == parent_label
                        and child.label == child_label
                        and (
                            leaf_label in child.clade_union()
                            or (child.label == leaf_label and child.is_leaf())
                        )
                    ):
                        support += edge2count[(parent, child)]
                # Shifts color pallete to less extreme lower bouund
                color = f"0.0000 {support/total_trees * 0.9 + 0.1} 1.000"
                G.edge(
                    label_ids[parent_label],
                    label_ids[child_label],
                    penwidth="5",
                    color=color,
                    label=f"{support/total_trees:.2}",
                    weight=f"{support/total_trees}",
                )
        return G, {idnum: child_label for child_label, idnum in label_ids.items()}

    def leaf_path_uncertainty_graphviz(self, leaf_label):
        """send output of leaf_path_uncertainty_dag to graphviz for rendering.

        Returns:
            The graphviz DAG object, and a dictionary mapping node names to labels
        """
        G = gv.Digraph("Path DAG to leaf", node_attr={})
        parent_d = self.leaf_path_uncertainty_dag(leaf_label)
        label_ids = {key: str(idnum) for idnum, key in enumerate(parent_d)}
        for key in parent_d:
            if key == leaf_label:
                G.node(label_ids[key], shape="octagon")
            elif len(parent_d[key]) == 0:
                G.node(label_ids[key], shape="invtriangle")
            else:
                G.node(label_ids[key])
        for key, parentset in parent_d.items():
            for parent in parentset:
                G.edge(
                    label_ids[parent],
                    label_ids[key],
                    label="",
                )
        return G, {idnum: key for key, idnum in label_ids.items()}

    def leaf_path_uncertainty_dag_no_collapse(self, leaf_label):
        """Compute the DAG of possible paths leading to `leaf_label`.

        Args:
            leaf_label: The node label of the leaf of interest

        Returns:
            parent_dictionary: A dictionary keyed by hDAG nodes, with sets
                of possible parent nodes.
        """
        parent_dictionary = {
            node: set()
            for node in self.dagroot.children()
            if leaf_label in node.clade_union()
            or (node.is_leaf() and node.label == leaf_label)
        }

        for node in self.preorder(skip_ua_node=True):
            for clade, eset in node.clades.items():
                if leaf_label in clade:
                    for cnode in eset.targets:
                        if cnode not in parent_dictionary:
                            parent_dictionary[cnode] = set()
                        if node != cnode:
                            parent_dictionary[cnode].add(node)

        return parent_dictionary

    def leaf_path_uncertainty_graphviz_no_collapse(
        self, leaf_label, edge2count, total_trees
    ):
        """sends output of leaf_path_uncertainty_dag_no_collapse to graphviz
        for rendering.

        Args:
            leaf_label: The node label of the leaf of interest
            edge2count: A map of edges in the form of node pairs (parent, child) to their count
                in the history DAG
            total_trees: The total number of trees contained in the history DAG

        Returns:
            The graphviz DAG object, and a dictionary mapping node names to labels
        """
        G = gv.Digraph("Path DAG to leaf", node_attr={})
        parent_d = self.leaf_path_uncertainty_dag_no_collapse(leaf_label)
        node_ids = {key: str(idnum) for idnum, key in enumerate(parent_d)}

        for node in parent_d:
            if node.is_leaf():
                G.node(node_ids[node], shape="octagon")
            elif len(parent_d[node]) == 0:
                G.node(node_ids[node], shape="invtriangle")
            else:
                G.node(node_ids[node])

        for child, parentset in parent_d.items():
            for parent in parentset:
                support = edge2count[(parent, child)]
                # Shifts color pallete to less extreme lower bouund
                color = f"0.0000 {support/total_trees * 0.9 + 0.1} 1.000"
                G.edge(
                    node_ids[parent],
                    node_ids[child],
                    penwidth="5",
                    color=color,
                    label=f"{support/total_trees:.2}",
                    weight=f"{support/total_trees}",
                )
        return G, {idnum: child_label for child_label, idnum in node_ids.items()}

    def summary(self):
        """Print nicely formatted summary about the history DAG."""
        print(f"Nodes:\t{sum(1 for _ in self.postorder())}")
        print(f"Trees:\t{self.count_histories()}")
        utils.hist(self.weight_counts_with_ambiguities())

    def label_uncertainty_summary(self):
        """Print information about internal nodes which have the same child
        clades but different labels."""
        duplicates = list(
            Counter(
                node.child_clades()
                for node in self.preorder(skip_ua_node=True)
                if not node.is_leaf()
            ).values()
        )
        print(
            "Mean unique labels per unique child clade set:",
            sum(duplicates) / len(duplicates),
        )
        print("Maximum duplication:", max(duplicates))
        print(
            "Counts of duplication numbers by unique child clade set:",
            Counter(duplicates),
        )

    # ######## Abstract dp method and derivatives: ########

    def postorder_history_accum(
        self,
        leaf_func: Callable[["HistoryDagNode"], Weight],
        edge_func: Callable[["HistoryDagNode", "HistoryDagNode"], Weight],
        accum_within_clade: Callable[[List[Weight]], Weight],
        accum_between_clade: Callable[[List[Weight]], Weight],
        accum_above_edge: Optional[Callable[[Weight, Weight], Weight]] = None,
    ) -> Weight:
        """A template method for leaf-to-root dynamic programming.

        Intermediate computations are stored in a `_dp_data` attribute on each node.
        Note that a `Weight` can be whatever you like, such as integers, Counters,
        strings, or dictionaries.

        Args:
            leaf_func: A function to assign weights to leaf nodes
            edge_func: A function to assign weights to edges. The parent node will
                always be the first argument.
            accum_within_clade: A function which accumulates a list of weights of subtrees
                below a single clade. That is, the weights are for alternative trees.
            accum_between_clade: A function which accumulates a list of weights of subtrees
                below different clades. That is, the weights are for different parts of the
                same tree.
            accum_above_edge: A function which adds the weight for a subtree to the weight
                of the edge above it. If `None`, this function will be inferred from
                `accum_between_clade`. The edge weight is the second argument.

        Returns:
            The resulting weight computed for the History DAG UA (root) node.
        """
        if accum_above_edge is None:

            def default_accum_above_edge(subtree_weight, edge_weight):
                return accum_between_clade([subtree_weight, edge_weight])

            accum_above_edge = default_accum_above_edge

        for node in self.postorder():
            if node.is_leaf():
                node._dp_data = leaf_func(node)
            else:
                node._dp_data = accum_between_clade(
                    [
                        accum_within_clade(
                            [
                                accum_above_edge(
                                    target._dp_data,
                                    edge_func(node, target),
                                )
                                for target in node.children(clade=clade)
                            ]
                        )
                        for clade in node.clades
                    ]
                )
        return self.dagroot._dp_data

    def postorder_cladetree_accum(self, *args, **kwargs) -> Weight:
        """Deprecated name for :meth:`postorder_history_accum`"""
        return self.postorder_history_accum(*args, **kwargs)

    def optimal_weight_annotate(
        self,
        start_func: Callable[["HistoryDagNode"], Weight] = lambda n: 0,
        edge_weight_func: Callable[
            ["HistoryDagNode", "HistoryDagNode"], Weight
        ] = utils.wrapped_hamming_distance,
        accum_func: Callable[[List[Weight]], Weight] = sum,
        optimal_func: Callable[[List[Weight]], Weight] = min,
    ) -> Weight:
        r"""A template method for finding the optimal tree weight in the DAG.
        Dynamically annotates each node in the DAG with the optimal weight of a
        clade sub-tree beneath it, so that the DAG root node is annotated with
        the optimal weight of a history in the DAG.

        Args:
            start_func: A function which assigns starting weights to leaves.
            edge_weight_func: A function which assigns weights to DAG edges based on the
                parent node and the child node, in that order.
            accum_func: A function which takes a list of weights of different parts of a
                tree, and returns a weight, like sum.
            optimal_func: A function which takes a list of weights and returns the optimal
                one, like min.

        Returns:
            The optimal weight of a tree under the DAG UA node.
        """
        return self.postorder_history_accum(
            start_func,
            edge_weight_func,
            optimal_func,
            accum_func,
        )

    def weight_count(
        self,
        start_func: Callable[["HistoryDagNode"], Weight] = lambda n: 0,
        edge_weight_func: Callable[
            ["HistoryDagNode", "HistoryDagNode"], Weight
        ] = utils.wrapped_hamming_distance,
        accum_func: Callable[[List[Weight]], Weight] = sum,
    ):
        r"""A template method for counting weights of trees expressed in the
        history DAG.

        Weights must be hashable, but may otherwise be of arbitrary type.

        Args:
            start_func: A function which assigns a weight to each leaf node
            edge_func: A function which assigns a weight to pairs of labels, with the
                parent node label the first argument
            accum_func: A way to 'add' a list of weights together

        Returns:
            A Counter keyed by weights.
        """
        return self.postorder_history_accum(
            lambda n: Counter([start_func(n)]),
            lambda n1, n2: Counter([edge_weight_func(n1, n2)]),
            counter_sum,
            lambda x: counter_prod(x, accum_func),
        )

    def hamming_parsimony_count(self):
        """Count the hamming parsimony scores of all trees in the history DAG.

        Returns a Counter with integer keys.
        """
        return self.weight_count(**utils.hamming_distance_countfuncs)

    def to_newicks(self, **kwargs):
        """Returns a list of extended newick strings formed with label fields.

        Arguments are passed to :meth:`utils.make_newickcountfuncs`.
        Arguments are the same as for
        :meth:`historydag.HistoryDag.to_newick`.
        """

        newicks = self.weight_count(**utils.make_newickcountfuncs(**kwargs)).elements()
        return [newick[1:-1] + ";" for newick in newicks]

    def count_topologies(self, collapse_leaves: bool = False) -> int:
        """Counts the number of unique topologies in the history DAG. This is
        achieved by counting the number of unique newick strings with only
        leaves labeled.

        :meth:`count_histories` gives the total number of unique trees in the DAG, taking
        into account internal node labels.

        For large DAGs, this method is prohibitively slow. Use :meth:``count_topologies_fast`` instead.

        Args:
            collapse_leaves: By default, topologies are counted as-is in the DAG. However,
                even if the DAG is collapsed by label, edges above leaf nodes will not be collapsed.
                if `collapse_leaves` is True, then the number of unique topologies with all
                leaf-adjacent edges collapsed will be counted. Assumes that the DAG is collapsed
                with :meth:`HistoryDag.convert_to_collapsed`.

        Returns:
            The number of topologies in the history DAG
        """
        kwargs = utils.make_newickcountfuncs(
            internal_labels=False, collapse_leaves=collapse_leaves
        )
        return len(self.weight_count(**kwargs))

    def count_topologies_fast(self) -> int:
        """Counts the number of unique topologies in the history DAG.

        This is achieved by creating a new history DAG in which all
        internal nodes have matching labels.

        This is only guaranteed to match the output of ``count_topologies``
        if the DAG has all allowed edges added.
        """
        return self.unlabel().count_histories()

    def count_trees(self):
        """Deprecated name for :meth:`count_histories`"""
        return self.count_histories()

    def count_histories(
        self,
        expand_func: Optional[Callable[[Label], List[Label]]] = None,
        expand_count_func: Callable[[Label], int] = lambda ls: 1,
    ):
        r"""Annotates each node in the DAG with the number of clade sub-trees
        underneath.

        Args:
            expand_func: A function which takes a label and returns a list of labels, for
                example disambiguations of an ambiguous sequence. If provided, this method
                will count at least the number of histories that would be in the DAG,
                if :meth:`explode_nodes` were called with the same `expand_func`.
            expand_count_func: A function which takes a label and returns an integer value
                corresponding to the number of 'disambiguations' of that label. If provided,
                `expand_func` will be used to find this value.

        Returns:
            The total number of unique complete trees below the root node. If `expand_func`
            or `expand_count_func` is provided, the complete trees being counted are not
            guaranteed to be unique.
        """
        if expand_func is not None:

            def expand_count_func(label):
                return len(list(expand_func(label)))

        return self.postorder_history_accum(
            lambda n: 1,
            lambda parent, child: expand_count_func(child.label),
            sum,
            prod,
        )

    def count_nodes(self, collapse=False) -> Dict[HistoryDagNode, int]:
        """Counts the number of trees each node takes part in.

        Args:
            collapse: A flag that when set to true, treats nodes as clade unions and
                ignores label information. Then, the returned dictionary is keyed by
                clade union sets.

        Returns:
            A dicitonary mapping each node in the DAG to the number of trees
            that it takes part in.
        """
        node2count = {}
        node2stats = {}

        self.count_histories()
        reverse_postorder = reversed(list(self.postorder()))
        for node in reverse_postorder:
            below = node._dp_data
            curr_clade = node.clade_union()

            if node.is_ua_node():
                above = 1
            else:
                above = 0
                for parent in node.parents:
                    above_parent = node2stats[parent][0]
                    below_parent = 1
                    for clade in parent.clades:
                        # Skip clade covered by node of interest
                        if clade == curr_clade or parent.is_ua_node():
                            continue
                        below_clade = 0
                        for sib in parent.children(clade=clade):
                            below_clade += sib._dp_data
                        below_parent *= below_clade

                    above += above_parent * below_parent

            node2count[node] = above * below
            node2stats[node] = [above, below]

        collapsed_n2c = {}
        if collapse:
            for node in node2count.keys():
                clade = node.clade_union()
                if clade not in collapsed_n2c:
                    collapsed_n2c[clade] = 0

                collapsed_n2c[clade] += node2count[node]
            return collapsed_n2c
        else:
            return node2count
        return node2count

    def count_edges(
        self, collapsed=False
    ) -> Dict[Tuple[HistoryDagNode, HistoryDagNode], int]:
        """Counts the number of trees each edge takes part in.

        Returns:
            A dicitonary mapping each edge in the DAG to the number of trees
            that it takes part in.
        """
        edge2count = {}
        node2stats = {}

        self.count_histories()
        reverse_postorder = reversed(list(self.postorder()))
        for node in reverse_postorder:
            below = node._dp_data
            curr_clade = node.clade_union()

            if node.is_ua_node():
                above = 1
            else:
                above = 0
                for parent in node.parents:
                    above_parent = node2stats[parent][0]
                    below_parent = 1
                    for clade in parent.clades:
                        # Skip clade covered by node of interest
                        if clade == curr_clade or parent.is_ua_node():
                            continue
                        below_clade = 0
                        for sib in parent.children(clade=clade):
                            below_clade += sib._dp_data
                        below_parent *= below_clade

                    above += above_parent * below_parent

                    edge2count[(parent, node)] = (above_parent * below_parent) * below
            node2stats[node] = [above, below]

        e2c = {}
        if collapsed:
            for (parent, child), count in edge2count.items():
                parent_cu = parent.clade_union()
                child_cu = child.clade_union()
                if (parent_cu, child_cu) not in e2c:
                    e2c[(parent_cu, child_cu)] = 0
                e2c[(parent_cu, child_cu)] += count
            return e2c

        return edge2count

    def most_supported_trees(self):
        """Trims the DAG to only express the trees that have the highest
        support."""
        node2count = self.count_nodes()
        total_trees = self.count_histories()
        clade2support = {}
        for node, count in node2count.items():
            if node.clade_union() not in clade2support:
                clade2support[node.clade_union()] = 0
            clade2support[node.clade_union()] += count / total_trees

        from math import log

        self.trim_optimal_weight(
            start_func=lambda n: 0,
            edge_weight_func=lambda n1, n2: log(clade2support[n2.clade_union()]),
            accum_func=lambda weights: sum([w for w in weights]),
            optimal_func=max,
        )

        return self.dagroot._dp_data

    def count_paths_to_leaf(
        self,
        leaf_label,
        expand_func: Optional[Callable[[Label], List[Label]]] = None,
        expand_count_func: Callable[[Label], int] = lambda ls: 1,
    ):
        r"""Annotates each node in the DAG with the number of paths to
        ``leaf_label`` underneath.

        Args:
            leaf_label: The label of the leaf node of interest
            expand_func: A function which takes a label and returns a list of labels, for
                example disambiguations of an ambiguous sequence. If provided, this method
                will count at least the number of histories that would be in the DAG,
                if :meth:`explode_nodes` were called with the same `expand_func`.
            expand_count_func: A function which takes a label and returns an integer value
                corresponding to the number of 'disambiguations' of that label. If provided,
                `expand_func` will be used to find this value.

        Returns:
            The total number of unique paths to the leaf node of interest. If `expand_func`
            or `expand_count_func` is provided, the paths being counted are not guaranteed
            to be unique.
        """
        if expand_func is not None:

            def expand_count_func(label):
                return len(list(expand_func(label)))

        return self.postorder_history_accum(
            lambda n: 1 if n.label == leaf_label else 0,
            lambda parent, child: 0,
            sum,
            sum,
        )

    def weight_counts_with_ambiguities(
        self,
        start_func: Callable[["HistoryDagNode"], Weight] = lambda n: 0,
        edge_func: Callable[[Label, Label], Weight] = lambda l1, l2: (
            0 if isinstance(l1, UALabel) else utils.hamming_distance(l1.sequence, l2.sequence)  # type: ignore
        ),
        accum_func: Callable[[List[Weight]], Weight] = sum,
        expand_func: Callable[[Label], Iterable[Label]] = utils.sequence_resolutions,
    ):
        r"""Template method for counting tree weights in the DAG, with exploded
        labels. Like :meth:`weight_counts`, but creates dictionaries of Counter
        objects at each node, keyed by possible sequences at that node.
        Analogous to :meth:`count_histories` with `expand_func` provided.

        Weights must be hashable.

        Args:
            start_func: A function which assigns a weight to each leaf node
            edge_func: A function which assigns a weight to pairs of labels, with the
                parent node label the first argument. Must correctly handle the UA
                node label which is a UALabel instead of a namedtuple.
            accum_func: A way to 'add' a list of weights together
            expand_func: A function which takes a label and returns a list of labels, such
                as disambiguations of an ambiguous sequence.

        Returns:
            A Counter keyed by weights.
            The total number of trees will be greater than count_histories(), as these are
            possible disambiguations of trees. These disambiguations may not be unique,
            but if two are the same, they come from different subtrees of the DAG.
        """

        def wrapped_expand_func(label, is_ua_node):
            if is_ua_node:
                return [label]
            else:
                return expand_func(label)

        # The old direct implementation not using postorder_history_accum was
        # more straightforward, and may be significantly faster.
        def leaf_func(node):
            return {
                label: Counter({start_func(node): 1})
                for label in expand_func(node.label)
            }

        def edge_weight_func(parent, child):
            # This will handle 'adding' child node counts to the edge, so we
            # have accum_above_edge just return this result.
            return {
                label: counter_sum(
                    [
                        counter_prod(
                            [target_wc, Counter({edge_func(label, childlabel): 1})],
                            accum_func,
                        )
                        for childlabel, target_wc in child._dp_data.items()
                    ]
                )
                for childlabel, target_wc in child._dp_data.items()
                for label in wrapped_expand_func(parent.label, parent.is_ua_node())
            }

        def accum_within_clade(dictlist):
            keys = dictlist[0].keys()
            return {key: counter_sum([d[key] for d in dictlist]) for key in keys}

        def accum_between_clade(dictlist):
            keys = dictlist[0].keys()
            return {
                key: counter_prod([d[key] for d in dictlist], accum_func)
                for key in keys
            }

        return list(
            self.postorder_history_accum(
                leaf_func,
                edge_weight_func,
                accum_within_clade,
                accum_between_clade,
                accum_above_edge=lambda n, e: e,
            ).values()
        )[0]

    # ######## End Abstract DP method derivatives ########

    def trim_optimal_weight(
        self,
        start_func: Callable[["HistoryDagNode"], Weight] = lambda n: 0,
        edge_weight_func: Callable[
            [HistoryDagNode, HistoryDagNode], Weight
        ] = utils.wrapped_hamming_distance,
        accum_func: Callable[[List[Weight]], Weight] = sum,
        optimal_func: Callable[[List[Weight]], Weight] = min,
        eq_func: Callable[[Weight, Weight], bool] = lambda w1, w2: w1 == w2,
    ) -> Weight:
        """Trims the DAG to only express trees with optimal weight. This is
        guaranteed to be possible when edge_weight_func depends only on the
        labels of an edge's parent and child node.

        Requires that weights are of a type that supports reliable equality
        testing. In particular, floats are not recommended. Instead, consider
        defining weights to be a precursor type, and define `optimal_func` to
        choose the one whose corresponding float is maximized/minimized.

        If floats must be used, a Numpy type may help.

        Args:
            start_func: A function which assigns starting weights to leaves.
            edge_weight_func: A function which assigns weights to DAG edges based on the
                parent node and the child node, in that order.
            accum_func: A function which takes a list of weights of different parts of a tree,
                and returns a weight, like sum.
            optimal_func: A function which takes a list of weights and returns the optimal
                one, like min.
            eq_func: A function which tests equality, taking a pair of weights and returning a bool.
        """
        opt_weight = self.optimal_weight_annotate(
            start_func=start_func,
            edge_weight_func=edge_weight_func,
            accum_func=accum_func,
            optimal_func=optimal_func,
        )
        for node in self.preorder():
            for clade, eset in node.clades.items():
                weightlist = [
                    (
                        accum_func([target._dp_data, edge_weight_func(node, target)]),
                        target,
                        index,
                    )
                    for index, target in enumerate(eset.targets)
                ]
                optimalweight = optimal_func([weight for weight, _, _ in weightlist])
                for weight, target, index in weightlist:
                    if not eq_func(weight, optimalweight):
                        eset.remove_from_edgeset_byid(target)
                n = len(eset.targets)
                if n == 0:
                    raise ValueError(
                        f"Value returned by ``optimal_func`` {optimal_func} is not in the "
                        f"list of weights passed to that function, according to eq_func {eq_func}"
                    )
                eset.set_edge_stats(probs=[1.0 / n] * n)
        self.recompute_parents()
        return opt_weight

    def get_topologies(self, collapse_leaves: bool = False) -> List[str]:
        """Return a list of pseudo-newick representations of topologies in the
        history DAG.

        The newicks returned are not well-formed, and are for use with
        :meth:`HistoryDag.trim_topology`. Otherwise, this method would be equivalent to
        :meth:`HistoryDag.to_newicks` with keyword arguments ``internal_labels=False`` and
        ``collapsed_leaves`` as desired.

        Args:
            collapse_leaves: Whether to collapse leaf-adjacent edges between nodes with
                matching labels

        Returns:
            A list of strings, each representing a topology present in the history DAG.
        """
        kwargs = utils.make_newickcountfuncs(
            internal_labels=False, collapse_leaves=collapse_leaves
        )
        return list(self.weight_count(**kwargs).keys())

    def trim_topology(self, topology: str, collapse_leaves: bool = False):
        """Trims the history DAG to express only trees matching the provided
        topology.

        Args:
            topology: A string like one output by :meth:`HistoryDag.get_topologies`
            collapse_leaves: must match the same argument provided to :meth:`HistoryDag.get_topologies`
                when creating the string passed as ``topology``.
        """

        def min_func(newicks: List[str]) -> str:
            # Each newick in presented to min_func will be well-formed, since
            # it will consist of a subtree newick added to a parent edge's
            # newick.
            for newick in newicks:
                if newick in topology:
                    return newick
            if newicks:
                return "(;)"
            else:
                raise ValueError("min_func() arg is an empty sequence")

        self.trim_optimal_weight(
            **utils.make_newickcountfuncs(
                internal_labels=False, collapse_leaves=collapse_leaves
            ),
            optimal_func=min_func,
        )

    def recompute_parents(self):
        """Repopulate ``HistoryDagNode.parent`` attributes."""
        for node in self.postorder():
            node.parents = set()
        for node in self.postorder():
            for child in node.children():
                child.parents.add(node)

    def convert_to_collapsed(self):
        r"""Rebuilds the DAG so that no edge connects two nodes with the same
        label, unless one is a leaf node.

        The resulting DAG should express at least the collapsed
        histories present in the original.
        """

        self.recompute_parents()
        nodes = list(self.postorder())
        nodedict = {node: node for node in nodes}
        edgequeue = [
            [parent, target] for parent in nodes for target in parent.children()
        ]

        while edgequeue:
            parent, child = edgequeue.pop()
            clade = child.clade_union()
            if (
                parent.label == child.label
                and parent in nodedict
                and child in nodedict
                and not child.is_leaf()
            ):
                parent_clade_edges = len(parent.clades[clade].targets)
                new_parent_clades = (
                    frozenset(parent.clades.keys()) - {clade}
                ) | frozenset(child.clades.keys())
                newparenttemp = empty_node(
                    parent.label, new_parent_clades, deepcopy(parent.attr)
                )
                if newparenttemp in nodedict:
                    newparent = nodedict[newparenttemp]
                else:
                    newparent = newparenttemp
                    nodedict[newparent] = newparent
                # Add parents of parent to newparent
                for grandparent in parent.parents:
                    grandparent.add_edge(newparent)  # check parents logic
                    edgequeue.append([grandparent, newparent])
                # Add children of other clades to newparent
                for otherclade in parent.clades:
                    if otherclade != clade:
                        for othertarget in parent.clades[otherclade].targets:
                            newparent.add_edge(othertarget)
                            edgequeue.append([newparent, othertarget])
                # Add children of old child to newparent
                for grandchild in child.children():
                    newparent.add_edge(grandchild)
                    edgequeue.append([newparent, grandchild])
                # Remove the edge we were fixing from old parent
                parent.remove_edge_by_clade_and_id(child, clade)
                # Clean up the DAG:
                # Delete old parent if it is no longer a valid node
                if parent_clade_edges == 1:
                    # Remove old parent as child of all of its parents
                    # no need for recursion here, all of its parents had
                    # edges added to new parent from the same clade.
                    upclade = parent.clade_union()
                    for grandparent in sorted(parent.parents):
                        grandparent.remove_edge_by_clade_and_id(parent, upclade)
                    for child2 in parent.children():
                        child2.parents.remove(parent)
                        if not child2.parents:
                            child2.remove_node(nodedict=nodedict)
                    nodedict.pop(parent)

                # Remove child, if child no longer has parents
                if parent in child.parents:
                    child.parents.remove(parent)
                if not child.parents:
                    # This recursively removes children of child too, if necessary
                    child.remove_node(nodedict=nodedict)
        self.recompute_parents()

    def add_node_at_nearest_leaf(
        self,
        new_leaf_id,
        id_name: str = "sequence",
        distance_func: Callable[
            [HistoryDagNode, HistoryDagNode], Weight
        ] = utils.wrapped_hamming_distance,
    ):
        """Inserts a sequence into the dag in such a way that every tree in the
        dag now contains that node.

        This method adds the new sequence as a sibling of the leaf
        node that achieves the minimum distance to the new sequence. The
        resulting dag has nodecount incremented by 1, and potentially
        many more trees, since it is re-built using
        :meth:``history_dag_from_nodes`` which adds all allowed edges.
        """
        postorder = list(self.postorder())
        if not any(
            [
                new_leaf_id == getattr(n.label, id_name)
                for n in postorder
                if not n.is_ua_node()
            ]
        ):

            # make sure all connections are correctly built before manipulating dag
            self.recompute_parents()

            # create a new node to hold the new_leaf_id
            new_leaf = empty_node(
                postorder[0].label._replace(**{id_name: new_leaf_id}), {}, None
            )

            # find closest leaf neighbor(s)
            leaf_nodes = [n for n in postorder if n.is_leaf()]
            leaf_dists = [distance_func(n, new_leaf) for n in leaf_nodes]
            nearest_leaf = leaf_nodes[leaf_dists.index(min(leaf_dists))]

            dagnodes = {new_leaf: new_leaf}
            # iterate over nodes in the dag to create a new copy
            for node in postorder:
                # if the node is an ancestor of the neighboring leaf node, add the new_node as descendant
                if (nearest_leaf.label in node.clade_union()) and (
                    nearest_leaf != node
                ):
                    oldnode = node
                    # check first if the new_node should be added as a separate clade/child pair
                    if node in list(nearest_leaf.parents):
                        node.clades[frozenset([new_leaf.label])] = EdgeSet([new_leaf])
                    else:
                        (old_clade, old_edgeset) = [
                            (c, e)
                            for c, e in node.clades.items()
                            if nearest_leaf.label in c
                        ][0]
                        new_clade = frozenset(old_clade | {new_leaf.label})
                        node.clades.pop(old_clade)
                        node.clades[new_clade] = old_edgeset
                    dagnodes[oldnode] = node
                else:
                    dagnodes[node] = node

        for node in self.postorder():
            for clade, edgeset in node.clades.items():
                edgeset.set_targets([dagnodes[n] for n in edgeset.targets])
        self.recompute_parents()

    def add_node_at_all_possible_places(self, new_leaf_id, id_name: str = "sequence"):
        """Inserts a sequence into the dag such that every tree in the dag now
        contains that new node.

        This method adds the new node as a leaf node by connecting it as
        a child of every non-leaf node in the original dag. The
        resulting dag has one new node corresponding to the added
        sequence as well as copies of all internal nodes corresponding
        to parents (and more ancestral nodes) to the added sequence.
        """
        postorder = list(self.postorder())
        if not any(
            [
                new_leaf_id == getattr(n.label, id_name)
                for n in postorder
                if not n.is_ua_node()
            ]
        ):
            # make sure all connections are correctly built before manipulating dag
            self.recompute_parents()

            # create a new node corresponding to new_sequence
            new_leaf = empty_node(
                postorder[0].label._replace(**{id_name: new_leaf_id}), {}, None
            )

            dagnodes = {new_leaf: new_leaf}
            for node in postorder:
                if not (node.is_leaf() or node.is_ua_node()):
                    # create a copy of the node that has new_leaf as a direct child
                    clades = [c for c in node.clades] + [frozenset([new_leaf.label])]
                    node_copy_as_parent = empty_node(node.label, clades, None)
                    dagnodes[node_copy_as_parent] = node_copy_as_parent

                    # for each child clade, create a copy of the node as an ancestor
                    # of new_leaf through that clade
                    for clade in node.clades:
                        if len(clade) > 1:
                            clades = [
                                c if c != clade else frozenset(clade | {new_leaf.label})
                                for c in node.clades
                            ]
                            node_copy_as_ancestor = empty_node(node.label, clades, None)
                            dagnodes[node_copy_as_ancestor] = node_copy_as_ancestor

                    # if the current node is internal to any trees in the dag, then keep
                    # a copy of the node that does not have new_node as a descendant
                    if not any([p.is_ua_node() for p in node.parents]):
                        nodecopy = node.empty_copy()
                        dagnodes[nodecopy] = nodecopy
                else:
                    nodecopy = node.empty_copy()
                    dagnodes[nodecopy] = nodecopy
            self.__init__(history_dag_from_nodes(dict(dagnodes)).dagroot)

    def insert_node(
        self,
        new_leaf_id,
        id_name: str = "sequence",
        dist: Callable[
            [HistoryDagNode, HistoryDagNode], Weight
        ] = utils.wrapped_hamming_distance,
    ):
        self.recompute_parents()
        postorder = list(self.postorder())
        new_node = empty_node(
            postorder[0].label._replace(**{id_name: new_leaf_id}), {}, None
        )

        def insert_node_as_sibling(node, sib):
            updated_nodes = {node: node}
            for parent in sib.parents:
                if node.label not in parent.clade_union():
                    old_parent = parent
                    parent.clades[frozenset([node.label])] = EdgeSet([node])
                    updated_nodes[old_parent] = parent
                    for ancestor in self.postorder_above(parent):
                        old_ancestor = ancestor
                        for clade, edgeset in old_ancestor.clades.items():
                            if (sib.label in clade) and not (node.label in clade):
                                new_clade = frozenset(clade | {node.label})
                                ancestor.clades.pop(clade)
                                ancestor.clades[new_clade] = EdgeSet(
                                    [
                                        t
                                        if t not in updated_nodes
                                        else updated_nodes[t]
                                        for t, _, _ in edgeset
                                    ]
                                )
                                break
                        updated_nodes[old_ancestor] = ancestor
            return updated_nodes

        def find_min_dist_nodes(new_node, node_set):
            return_set = []
            min_dist = float("inf")
            for node in node_set:
                if not node.is_root():
                    this_dist = dist(node, new_node)
                    if this_dist < min_dist:
                        return_set = [node]
                        min_dist = this_dist
                    elif this_dist <= min_dist:
                        return_set.append(node)
            return return_set

        def incompatible(n1, n2):
            if n1.is_root() or n2.is_root():
                return False
            if any([n1.clade_union() <= B for B in n2.child_clades()]):
                return False
            if any([n2.clade_union() <= B for B in n1.child_clades()]):
                return False
            if (
                (n1.clade_union() == n2.clade_union())
                or any(
                    [
                        len([B for B in n1.clades if len(A.intersection(B)) > 0]) == 2
                        for A in n2.clades
                    ]
                )
                or any(
                    [
                        len([B for B in n2.clades if len(A.intersection(B)) > 0]) == 2
                        for A in n1.clades
                    ]
                )
            ):
                return True
            return False

        def incompatible_set(node, nodeset):
            for leaf in node.clade_union():
                pathdag = [
                    n
                    for n in self.leaf_path_uncertainty_dag(leaf)
                    if (n in nodeset and incompatible(n, node))
                ]
                yield from pathdag

        incompatible_nodes_so_far = set(postorder)
        changed_nodes = {}
        while len(incompatible_nodes_so_far) > 0:
            min_dist_nodes = find_min_dist_nodes(new_node, incompatible_nodes_so_far)
            for other_node in min_dist_nodes:
                if not other_node.is_leaf():
                    for child in incompatible_nodes_so_far.intersection(
                        other_node.children()
                    ):
                        changed_nodes.update(insert_node_as_sibling(new_node, child))
                        incompatible_nodes_so_far = set(
                            incompatible_set(child, incompatible_nodes_so_far)
                        )

        for node in self.postorder():
            for clade, edgeset in node.clades.items():
                edgeset.set_targets(
                    [
                        n if n not in changed_nodes else changed_nodes[n]
                        for n in edgeset.targets
                    ]
                )
        self.recompute_parents()

    def postorder_above(self, node_as_leaf):
        def follow_up(node):
            for p in node.parents:
                yield from follow_up(p)
            yield node

        ancestor_set = set(follow_up(node_as_leaf))
        visited = set()

        def traverse(node: HistoryDagNode):
            visited.add(id(node))
            for child in node.children():
                if (not id(child) in visited) and (node_as_leaf.label in ancestor_set):
                    yield from traverse(child)
            yield node

        yield from traverse(self.dagroot)

    # ######## DAG Traversal Methods ########

    def postorder(
        self, include_root: bool = True
    ) -> Generator[HistoryDagNode, None, None]:
        """Recursive postorder traversal of the history DAG.

        Returns:
            Generator on nodes
        """
        visited = set()

        def traverse(node: HistoryDagNode):
            visited.add(id(node))
            if not node.is_leaf():
                for child in node.children():
                    if not id(child) in visited:
                        yield from traverse(child)
            yield node

        yield from traverse(self.dagroot)

    def preorder(
        self, skip_ua_node=False, skip_root=None
    ) -> Generator[HistoryDagNode, None, None]:
        """Recursive postorder traversal of the history DAG.

        Careful! This is not guaranteed to visit a parent node before any of its children.
        for that, need reverse postorder traversal.

        If skip_ua_node is passed, the universal ancestor node will be skipped.
        skip_root is provided as a backwards-compatible synonym of skip_ua_node.

        Returns:
            Generator on nodes
        """
        if skip_root is not None:
            skip_ua_node = skip_root

        visited = set()

        def traverse(node: HistoryDagNode):
            visited.add(id(node))
            yield node
            if not node.is_leaf():
                for child in node.children():
                    if not id(child) in visited:
                        yield from traverse(child)

        gen = traverse(self.dagroot)
        if skip_ua_node:
            next(gen)
        yield from gen


class EdgeSet:
    r"""A container class for edge target nodes, and associated probabilities
    and weights.

    Goal: associate targets (edges) with arbitrary parameters, but support
    set-like operations like lookup and enforce that elements are unique.
    """

    def __init__(
        self,
        *args,
        weights: Optional[List[float]] = None,
        probs: Optional[List[float]] = None,
    ):
        r"""Takes no arguments, or an ordered iterable containing target
        nodes."""
        if len(args) == 0:
            targets = []
        elif len(args) == 1:
            targets = args[0]
        else:
            raise TypeError(
                f"__init__() takes 0 or 1 positional arguments but {len(args)} were given."
            )
        self.set_targets(targets, weights, probs)

    def __iter__(self):
        return (
            (self.targets[i], self.weights[i], self.probs[i])
            for i in range(len(self.targets))
        )

    def set_targets(self, targets, weights=None, probs=None):
        """Set the target nodes of this node.

        If no weights or probabilities are provided, then these will be
        set to 0 and 1/n, respectively.
        """
        n = len(targets)
        if len(set(targets)) != n:
            raise ValueError(
                f"duplicate target nodes provided: {len(set(targets))} out of {len(targets)} unique."
            )

        self.targets = targets
        self._targetset = set(targets)
        if weights is None:
            weights = [0] * n
        if probs is None:
            if n == 0:
                probs = []
            else:
                probs = [float(1) / n] * n
        self.set_edge_stats(weights, probs)

    def set_edge_stats(self, weights=None, probs=None):
        """Set the edge weights and/or probabilities of this EdgeSet."""
        n = len(self.targets)
        if weights is not None:
            if len(weights) != n:
                raise ValueError(
                    "length of provided weights list must match number of target nodes"
                )
            self.weights = weights
        if probs is not None:
            if len(probs) != n:
                raise ValueError(
                    "length of provided probabilities list must match number of target nodes"
                )
            self.probs = probs

    def shallowcopy(self) -> "EdgeSet":
        """Return an identical EdgeSet object, which points to the same target
        nodes."""
        return EdgeSet(
            [node for node in self.targets],
            weights=self.weights.copy(),
            probs=self.probs.copy(),
        )

    def remove_from_edgeset_byid(self, target_node):
        idlist = [id(target) for target in self.targets]
        if id(target_node) in idlist:
            idx_to_remove = idlist.index(id(target_node))
            self.targets.pop(idx_to_remove)
            self.probs.pop(idx_to_remove)
            self.weights.pop(idx_to_remove)
            self._targetset = set(self.targets)

    def sample(self, mask=None) -> Tuple[HistoryDagNode, float]:
        """Returns a randomly sampled child edge, and its corresponding weight.

        When possible, only edges pointing to child nodes on which
        ``selection_function`` evaluates to True will be sampled.
        """
        if sum(mask) == 0:
            weights = self.probs
        else:
            weights = [factor * prob for factor, prob in zip(mask, self.probs)]
        index = random.choices(list(range(len(self.targets))), weights=weights, k=1)[0]
        return (self.targets[index], self.weights[index])

    def add_to_edgeset(self, target, weight=0, prob=None, prob_norm=True) -> bool:
        """Add a target node to the EdgeSet.

        currently does nothing if edge is already present. Also does nothing
        if the target node has one child clade, and parent node is not the DAG root.

        Args:
            target: target node
            weight: edge weight
            prob: edge probability. If not provided, edge probability will be
                1 / n where n is the number of edges in the edgeset.
            prob_norm: if True, probability vector will be normalized.

        Returns:
            Whether an edge was added
        """
        if target.is_ua_node():
            raise ValueError(
                "Edges that target UA nodes are not allowed. "
                f"Target node has label {target.label} and therefore "
                "is assumed to be the DAG UA root node."
            )
        elif target in self._targetset:
            return False
        else:
            self._targetset.add(target)
            self.targets.append(target)
            self.weights.append(weight)

            if prob is None:
                prob = 1.0 / len(self.targets)
            if prob_norm:
                self.probs = list(
                    map(lambda x: x * (1 - prob) / sum(self.probs), self.probs)
                )
            self.probs.append(prob)
            return True


# ######## DAG Creation Functions ########


def empty_node(
    label: Label, clades: Iterable[FrozenSet[Label]], attr: Any = None
) -> HistoryDagNode:
    """Return a HistoryDagNode with the given label and clades, with no
    children."""
    return HistoryDagNode(label, {clade: EdgeSet() for clade in clades}, attr)


def from_tree(
    tree: ete3.TreeNode,
    label_features: List[str],
    label_functions: Mapping[str, Callable[[ete3.TreeNode], Any]] = {},
    attr_func: Callable[[ete3.TreeNode], Any] = lambda n: None,
) -> HistoryDag:
    """Build a full tree from an ete3 TreeNode.

    Args:
        tree: ete3 tree to be converted to HistoryDag history
        label_features: node attribute names to be used to distinguish nodes. Field names
            provided in `label_functions` will take precedence.
        label_functions: dictionary keyed by additional label field names, containing
            functions mapping tree nodes to intended label field value.
        attr_func: function to populate HistoryDag node `attr` attribute,
            which is not used to distinguish nodes, and may be overwritten
            by `attr` of another node with the same label and child clades.

    Returns:
        HistoryDag object, which has the same topology as the input tree, with the required
        UA node added as a new root.
    """
    # see https://stackoverflow.com/questions/50298582/why-does-python-asyncio-loop-call-soon-overwrite-data
    # or https://stackoverflow.com/questions/25670516/strange-overwriting-occurring-when-using-lambda-functions-as-dict-values
    # for why we can't just use lambda funcs defined in dict comprehension.
    def getnamefunc(name):
        def getter(node):
            return getattr(node, name)

        return getter

    feature_maps = {name: getnamefunc(name) for name in label_features}
    feature_maps.update(label_functions)
    Label = NamedTuple("Label", [(label, Any) for label in feature_maps.keys()])  # type: ignore

    def node_label(n: ete3.TreeNode):
        # This should not fail silently! Only DAG UA node is allowed to have
        # default (None) label values.
        return Label(**{name: f(n) for name, f in feature_maps.items()})

    def leaf_names(r: ete3.TreeNode):
        return frozenset((node_label(node) for node in r.get_leaves()))

    def _unrooted_from_tree(tree):
        dag = HistoryDagNode(
            node_label(tree),
            {
                leaf_names(child): EdgeSet(
                    [_unrooted_from_tree(child)], weights=[child.dist]
                )
                for child in tree.children
            },
            attr_func(tree),
        )
        return dag

    # Check for unique leaf labels:
    if len(list(tree.get_leaves())) != len(leaf_names(tree)):
        raise ValueError(
            "This tree's leaves are not labeled uniquely. Check your tree, "
            "or modify the label fields so that leaves are unique.\n"
        )

    # Checking for unifurcation is handled in HistoryDagNode.__init__.

    dag = _unrooted_from_tree(tree)
    dagroot = UANode(EdgeSet([dag], weights=[tree.dist]))
    return HistoryDag(dagroot)


def from_newick(
    tree: str,
    label_features: List[str],
    label_functions: Mapping[str, Callable[[ete3.TreeNode], Any]] = {},
    newick_format=8,
    attr_func: Callable[[ete3.TreeNode], Any] = lambda n: None,
) -> HistoryDag:
    """Make a history DAG using a newick string. Internally, utilizes newick
    parsing features provided by ete3, then calls :meth:`from_tree` on the
    resulting ete3.Tree object.

    Args:
        tree: newick string representation of a tree. May contain extended node data
            in 'extended newick format' used by ete3.
        label_features: (passed to :meth:`from_tree`) list of features to be used as label
            fields in resulting history DAG.  'name' refers to the node name string in the
            standard newick format. See ete3 docs for more details.
        newick_format: ete3 format number of passed newick string. See ete3 docs for details.
        label_functions: (passed to :meth:`from_tree`)
        attr_func: (passed to :meth:`from_tree`)

    Returns:
        HistoryDag object, which has the same topology as the input newick tree, with the
        required UA node added as a new root.
    """
    etetree = ete3.Tree(tree, format=newick_format)
    # from_tree checks that leaves are labeled uniquely. If this function is
    # ever rewritten to avoid ete newick parsing, we'd need to do that here.
    return from_tree(
        etetree, label_features, label_functions=label_functions, attr_func=attr_func
    )


def history_dag_from_newicks(
    newicklist: List[str],
    label_features: List[str],
    label_functions: Mapping[str, Callable[[ete3.TreeNode], Any]] = {},
    attr_func: Callable[[ete3.TreeNode], Any] = lambda n: None,
    newick_format=1,
) -> HistoryDag:
    """Build a history DAG from a list of newick strings.

    See :meth:`from_newick` for argument details.
    """
    return history_dag_from_histories(
        [
            from_newick(
                tree,
                label_features,
                label_functions=label_functions,
                attr_func=attr_func,
                newick_format=newick_format,
            )
            for tree in newicklist
        ]
    )


def history_dag_from_etes(
    treelist: List[ete3.TreeNode],
    label_features: List[str],
    label_functions: Mapping[str, Callable[[ete3.TreeNode], Any]] = {},
    attr_func: Callable[[ete3.TreeNode], Any] = lambda n: None,
) -> HistoryDag:
    """Build a history DAG from a list of ete3 Trees.

    See :meth:`from_tree` for argument details.
    """
    return history_dag_from_histories(
        [
            from_tree(
                tree,
                label_features,
                label_functions=label_functions,
                attr_func=attr_func,
            )
            for tree in treelist
        ]
    )


def history_dag_from_histories(treelist: Sequence[HistoryDag]) -> HistoryDag:
    """Build a history DAG from a list of history DAGs which are histories."""
    dag = next(iter(treelist))
    dag.merge(treelist)
    return dag


def history_dag_from_clade_trees(*args, **kwargs) -> HistoryDag:
    """Deprecated name for :meth:`history_dag_from_histories`"""
    return history_dag_from_histories(*args, **kwargs)


def history_dag_from_nodes(nodes: Sequence[HistoryDagNode]) -> HistoryDag:
    """Take an iterable containing HistoryDagNodes, and build a HistoryDag from
    those nodes."""
    # use dictionary to preserve order
    nodes = {node.empty_copy(): node for node in nodes}
    # check for UA node in passed set, and recover if present:
    ua_node = UANode(EdgeSet())
    if ua_node in nodes:
        ua_node = nodes[ua_node].empty_copy()
    nodes.pop(ua_node)
    clade_dict = _clade_union_dict(nodes.keys())
    edge_dict = {
        node: [child for clade in node.clades for child in clade_dict[clade]]
        for node in nodes
    }
    children = {node: "" for _, children in edge_dict.items() for node in children}
    source_nodes = set(nodes) - set(children.keys())
    edge_dict[ua_node] = list(source_nodes)

    for node, children in edge_dict.items():
        for child in children:
            node.add_edge(child)

    return HistoryDag(ua_node)
