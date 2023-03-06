from math import exp
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
    Dict,
    FrozenSet,
)
from copy import deepcopy
from historydag import utils
from historydag.utils import Weight, Label, UALabel


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

    def is_history_root(self) -> bool:
        """Return whether node is a root of any histories in the DAG."""
        return any(n.is_ua_node() for n in self.parents)

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
                num_subtrees = 0
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

    def to_ete_recursive(
        self,
        name_func: Callable[["HistoryDagNode"], str] = lambda n: "unnamed",
        feature_funcs: Mapping[str, Callable[["HistoryDagNode"], str]] = {},
        sort_func=lambda seq: seq,
    ) -> ete3.TreeNode:
        """Convert a history DAG node which is part of a history to an ete
        tree.

        Args:
            name_func: A map from nodes to newick node names
            feature_funcs: A dictionary keyed by extended newick field names, containing
                functions specifying how to populate that field for each node.

        Returns:
            An ete3 Tree with the same topology as the subhistory below self,
            and node names and attributes as specified.
        """
        node = ete3.TreeNode()
        node.name = name_func(self)
        for feature, func in feature_funcs.items():
            node.add_feature(feature, func(self))
        for child in self.children():
            node.add_child(child.to_ete_recursive(name_func, feature_funcs, sort_func))
        return node

    def _sample(
        self, edge_selector=lambda n: True, log_probabilities=False
    ) -> "HistoryDagNode":
        r"""Samples a history (a sub-history DAG containing the root and all
        leaf nodes).

        Returns a new HistoryDagNode object.
        """
        sample = self.empty_copy()
        for clade, eset in self.clades.items():
            mask = [edge_selector((self, target)) for target in eset.targets]
            sampled_target, target_weight = eset.sample(
                mask=mask, log_probabilities=log_probabilities
            )
            sampled_target_subsample = sampled_target._sample(
                edge_selector=edge_selector,
                log_probabilities=log_probabilities,
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

    def sample(
        self, mask=None, log_probabilities=False
    ) -> Tuple[HistoryDagNode, float]:
        """Returns a randomly sampled child edge, and its corresponding weight.

        When possible, only edges pointing to child nodes on which
        ``selection_function`` evaluates to True will be sampled.
        """
        if log_probabilities:
            weights = [exp(weight) for weight in self.probs]
        else:
            weights = self.probs

        if mask is None or sum(mask) == 0:
            pass
        else:
            weights = [factor * prob for factor, prob in zip(mask, weights)]

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


def empty_node(
    label: Label, clades: Iterable[FrozenSet[Label]], attr: Any = None
) -> HistoryDagNode:
    """Return a HistoryDagNode with the given label and clades, with no
    children."""
    return HistoryDagNode(label, {clade: EdgeSet() for clade in clades}, attr)
