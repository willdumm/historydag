"""A module providing the class HistoryDag, and supporting functions."""

import pickle
from math import log
import graphviz as gv
import ete3
import random
import warnings
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
from collections import Counter, namedtuple
from copy import deepcopy
from historydag import utils
from historydag.utils import Weight, Label, UALabel, prod
from historydag.counterops import counter_sum, counter_prod
import historydag.parsimony_utils as parsimony_utils


class IntersectionError(ValueError):
    pass


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


    Subclassing HistoryDag:
    HistoryDag may be subclassed without overriding __init__, by defining a `_required_label_fields` class variable
    for any subclasses.

    The value of `_required_label_fields` should be a dictionary keyed by label fields that are expected by methods
    of the subclass. Each dictionary entry shall be of the form `required_field: [(from_fields, conversion_func), ...]`, where
    the dict value is a list of tuples, with each `conversion_func` a function mapping `HistoryDagNode`s to the value of
    that node label's `required_field` field, and `from_fields` a tuple containing all label fields expected by that function.

    Keyword arguments passed to :meth:`HistoryDag.from_history_dag` will be passed to conversion functions provided in the
    appropriate subclass's `_required_label_fields` attribute. Be sure to document each subclass, including available
    conversion functions and their keywords, in each subclass's docstring.
    """
    _required_label_fields = dict()

    @classmethod
    def from_history_dag(
        cls, dag: "HistoryDag", label_fields: Sequence[str] = None, **kwargs
    ):
        """Converts HistoryDag instances between subclasses of HistoryDag. No
        copy is performed, so the passed `dag` will in general be modified.

        Args:
            dag: A HistoryDag (or subclass) instance
            label_fields: A list specifying the order of label fields in node labels on the resulting HistoryDag
            kwargs: Any additional arguments required for label conversions. For details, see the class docstring
                for the subclass into which the conversion is taking place.

        Returns:
            The converted HistoryDag object, carrying the type from which this static method was called.
            After conversion to the new HistoryDag subclass ``to_cls``, the following will be true about node labels:

            * If passed ``label_fields`` is None, then existing label fields will be preserved, except that missing
              required label fields will be recovered if possible, and the existing label fields used to recover
              them will be omitted. Recovered label fields will appear before the existing label fields.
            * If passed ``label_fields`` is not None, then it must include all fields expected in node labels
              in the converted history DAG object, otherwise an exception will be raised.
            * Converted node label field order will match the order of passed ``label_fields``.
            * All label fields passed in ``label_fields`` will be included
              in converted node labels, if possible. Otherwise, an exception will be raised.
        """
        if label_fields is not None:
            label_fields = list(label_fields)
        required_fields_set = set(cls._required_label_fields.keys())
        if label_fields is not None and (
            not required_fields_set.issubset(set(label_fields))
        ):
            # This would be handled by __init__ anyway, but this prevents
            # any changes from being applied first and provides better
            # error message
            raise ValueError(
                "If passed, `label_fields` must contain all required label fields"
                f"for {cls.__name__}: {', '.join(required_fields_set)}"
            )
        if label_fields == dag.label_fields or (
            label_fields is None and required_fields_set.issubset(set(dag.label_fields))
        ):
            # No label modification is needed
            return cls(dag.dagroot, dag.attr)

        def get_existing_field(fieldname: str):
            def get_field(node: HistoryDagNode, **kwargs):
                return getattr(node.label, fieldname)

            return get_field

        def raise_unable_error(fieldname: str):
            message = ""
            if label_fields is not None and fieldname in label_fields:
                message += f" Label field '{fieldname}' is not present in existing label fields."
            else:
                message += (
                    f"Unable to convert {dag.__class__.__name__} with label fields {dag.label_fields} to {cls.__name__}"
                    f", which requires label field '{fieldname}'."
                )
            if fieldname in required_fields_set:
                message += (
                    " Automatic conversion from label fields"
                    f" {' or '.join([str(converttuple[0]) for converttuple in cls._required_label_fields[fieldname]])}"
                    " is supported."
                )
            raise TypeError(message)

        precursor_fields = set()

        def find_conversion_func(fieldname: str):
            if fieldname in dag.label_fields:
                return (fieldname, get_existing_field(fieldname))
            elif fieldname in required_fields_set:
                for from_fields, conversion_func in cls._required_label_fields[
                    fieldname
                ]:
                    if set(from_fields).issubset(set(dag.label_fields)):
                        precursor_fields.update(from_fields)
                        return (fieldname, conversion_func)
            raise_unable_error(fieldname)

        convert_funcs = []

        if label_fields is None:
            # keep all existing fields, except those used to recover missing
            # required fields:
            added_fields = set()
            for field in cls._required_label_fields.keys():
                if field not in dag.label_fields and field not in added_fields:
                    convert_funcs.append(find_conversion_func(field))
                    added_fields.add(field)
            for field in dag.label_fields:
                if field not in added_fields and field not in precursor_fields:
                    convert_funcs.append(find_conversion_func(field))
                    added_fields.add(field)
        else:
            for field in label_fields:
                convert_funcs.append(find_conversion_func(field))

        Label = NamedTuple(
            "Label", [(converttuple[0], Any) for converttuple in convert_funcs]
        )

        def relabel_func(node):
            labeldata = [
                converttuple[1](node, **kwargs) for converttuple in convert_funcs
            ]
            return Label(*labeldata)

        newdag = dag.relabel(relabel_func, relax_type=True)
        return cls(newdag.dagroot, dag.attr)

    def __init__(self, dagroot: HistoryDagNode, attr: Any = {}):
        assert isinstance(dagroot, UANode)
        self.attr = attr
        self.dagroot = dagroot
        try:
            self.label_fields = next(self.dagroot.children()).label._fields
        except StopIteration:
            self.label_fields = tuple()
        for field in self.__class__._required_label_fields:
            if field not in self.label_fields:
                raise TypeError(
                    f"An instance of {self.__class__.__name__} must have node labels containing a '{field}' field."
                )

    def __eq__(self, other: object) -> bool:
        # Eventually this can be done by comparing bytestrings, but we need
        # some sorting to be done first, to ensure two dags that represent
        # identical trees return True. TODO
        raise NotImplementedError

    def __getitem__(self, key) -> "HistoryDag":
        r"""Returns the history (tree-shaped sub-history DAG) in the current
        history dag corresponding to the given index.

        Alternatively, if ``key`` is a :class:`utils.HistoryDagFilter`
        object, the returned history DAG is the result of
        ``dag.trim_optimal_weight(**key)`` where ``dag`` is a copy of
        ``self``.
        """
        if isinstance(key, utils.HistoryDagFilter):
            dag = self.copy()
            dag.trim_optimal_weight(**key)
            return dag
        elif isinstance(key, int):
            length = self.count_histories()
            if key < 0:
                key = length + key
            if not (key >= 0 and key < length):
                raise IndexError
            self.count_histories()
            return self.__class__(self.dagroot._get_subhistory_by_subid(key))
        else:
            raise TypeError(
                f"History DAG indices must be integers or utils.HistoryDagFilter"
                f" objects, not {type(key)}"
            )

    def get_label_type(self) -> type:
        """Return the type for labels on this dag's nodes."""
        return type(next(self.dagroot.children()).label)

    def trim_within_range(
        self,
        min_weight=None,
        max_weight=None,
        start_func: Callable[["HistoryDagNode"], Weight] = lambda n: 0,
        edge_weight_func: Callable[
            [HistoryDagNode, HistoryDagNode], Weight
        ] = parsimony_utils.hamming_edge_weight,
        min_possible_weight=-float("inf"),
        max_possible_weight=float("inf"),
    ):
        if max_weight is not None:
            self.trim_below_weight(
                max_weight, start_func, edge_weight_func, min_possible_weight
            )

        if min_weight is not None:
            self.trim_below_weight(
                -min_weight,
                lambda n: -start_func(n),
                lambda n1, n2: -edge_weight_func(n1, n2),
                -max_possible_weight,
            )

    def trim_below_weight(
        self,
        max_weight,
        start_func: Callable[["HistoryDagNode"], Weight] = lambda n: 0,
        edge_weight_func: Callable[
            [HistoryDagNode, HistoryDagNode], Weight
        ] = parsimony_utils.hamming_edge_weight,
        min_possible_weight=-float("inf"),
    ):
        """Trim the dag to contain at least all the histories within the
        specified weight range.

        Supports totally ordered weights, accumulated by addition. A
        weight type must implement all ordering operators properly, as
        well as + and -, and addition and subtraction must respect the
        ordering. That is, if a < b, then a + c < b + c for any c
        (including negative c)
        """

        def trim_node(node):
            if node.is_leaf():  # base case - the node is a leaf
                return
            else:
                node_min_weight = node._dp_data  # minimum weight of subtree under node
                for clade, eset in node.clades.items():
                    weightlist = []
                    for target in eset.targets:
                        edgeweight = edge_weight_func(node, target)
                        weightlist.append(
                            (target._dp_data + edgeweight, edgeweight, target)
                        )

                    # By assuming a minimum weight edge is chosen for all other
                    # clades, we compute the maximum weight of a subtree below this
                    # clade
                    min_weight_under_clade = min(
                        minweight for minweight, _, _ in weightlist
                    )
                    # The sum of minimum scores beneath all other clades is
                    # quantity in parentheses:
                    max_weight_allowed_clade = node.maxweight - (
                        node_min_weight - min_weight_under_clade
                    )

                    to_keep = []
                    for (
                        minweight,
                        edgeweight,
                        target,
                    ) in (
                        weightlist
                    ):  # this is looping through all the edges under clade
                        if minweight <= max_weight_allowed_clade:
                            targetmax = max_weight_allowed_clade - edgeweight
                            target.maxweight = max(target.maxweight, targetmax)
                            to_keep.append(target)
                    eset.set_targets(to_keep)

        self.optimal_weight_annotate(
            start_func=start_func, edge_weight_func=edge_weight_func
        )

        nl = list(reversed(list(self.postorder())))

        for node in nl:
            node.maxweight = min_possible_weight
        self.dagroot.maxweight = max_weight

        for node in nl:
            trim_node(node)

        self.recompute_parents()

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

    def __iand__(self, other) -> "HistoryDag":
        if not isinstance(other, HistoryDag):
            raise TypeError(
                f"Unsupported operand types for &: 'HistoryDag' and '{type(other)}'"
            )
        else:
            self.history_intersect(other)
            return self

    def __and__(self, other) -> "HistoryDag":
        cdag = self.copy()
        cdag &= other
        return cdag

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
        self.label_fields = label_fields
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
                                "Parent clade does not equal child clade union: "
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
        """Return a generator containing all histories in the history DAG.

        Note that each history is a tree-shaped history DAG, containing a UA node,
        which exists as a subgraph of the history DAG.

        The order of these histories does not necessarily match the order of
        indexing. That is, ``dag.get_histories()`` and ``history for history in
        dag`` will result in different orderings. ``get_histories`` should
        be slightly faster, but possibly more memory intensive.
        """
        for history in self.dagroot._get_subhistories():
            yield self.__class__(history)

    def get_trees(self) -> Generator["HistoryDag", None, None]:
        """Deprecated name for :meth:`get_histories`"""
        return self.get_histories()

    def get_leaves(self) -> Generator["HistoryDagNode", None, None]:
        """Return a generator containing all leaf nodes in the history DAG."""
        return self.find_nodes(HistoryDagNode.is_leaf)

    def get_edges(
        self, skip_ua_node=False
    ) -> Generator[Tuple["HistoryDagNode", "HistoryDagNode"], None, None]:
        """Return a generator containing all edges in the history DAG, as
        parent, child node tuples.

        Edges' parent nodes will be in preorder.
        """
        for parent in self.preorder(skip_ua_node=skip_ua_node):
            for child in parent.children():
                yield (parent, child)

    def num_edges(self, skip_ua_node=False) -> int:
        """Return the number of edges in the DAG, including edges descending
        from the UA node, unless skip_ua_node is True."""
        return sum(1 for _ in self.get_edges(skip_ua_node=skip_ua_node))

    def num_nodes(self) -> int:
        """Return the number of nodes in the DAG, not counting the UA node."""
        return sum(1 for _ in self.preorder(skip_ua_node=True))

    def num_leaves(self) -> int:
        """Return the number of leaf nodes in the DAG."""
        return sum(1 for _ in self.get_leaves())

    def find_nodes(
        self, filter_func: Callable[[HistoryDagNode], bool]
    ) -> Generator["HistoryDagNode", None, None]:
        """Return a generator on (non-UA) nodes for which ``filter_func``
        evaluates to True."""
        for node in self.preorder(skip_ua_node=True):
            if filter_func(node):
                yield node

    def find_node(
        self, filter_func: Callable[[HistoryDagNode], bool]
    ) -> HistoryDagNode:
        """Return the first (non-UA) node for which ``filter_func`` evaluates
        to True."""
        try:
            return next(self.find_nodes(filter_func))
        except StopIteration:
            raise ValueError("No matching node found.")

    def sample(self, edge_selector=lambda e: True) -> "HistoryDag":
        r"""Samples a history from the history DAG. (A history is a sub-history
        DAG containing the root and all leaf nodes) For reproducibility, set
        ``random.seed`` before sampling.

        When there is an option, edges pointing to nodes on which `selection_func` is True
        will always be chosen.

        Returns a new HistoryDag object.
        """
        return self.__class__(self.dagroot._sample(edge_selector=edge_selector))

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

        newdag = HistoryDag.from_history_dag(self.copy())
        model_label = next(self.preorder(skip_ua_node=True)).label
        # initialize empty/default value for each item in model_label
        # Use placeholder Ellipsis, since None could have interpretation
        # in context of label field type.
        field_values = tuple(Ellipsis for _ in model_label)
        internal_label = type(model_label)(*field_values)
        for node in newdag.preorder(skip_ua_node=True):
            if not node.is_leaf():
                node.label = internal_label

        # Use merging method to eliminate duplicate nodes, by starting with
        # a subdag with no duplicate nodes.
        ret = newdag.sample()
        ret.merge(newdag)
        return ret

    def relabel(
        self, relabel_func: Callable[[HistoryDagNode], Label], relax_type=False
    ) -> "HistoryDag":
        """Return a new HistoryDag with labels modified according to a provided
        function.

        Args:
            relabel_func: A function which takes a node and returns the new label
                appropriate for that node. The relabel_func should return a consistent
                NamedTuple type with name Label. That is, all returned labels
                should have matching `_fields` attribute.
                No two leaf nodes may be mapped to the same new label.
            relax_type: Whether to require the returned HistoryDag to be of the same subclass as self.
                If True, the returned HistoryDag will be of the abstract type `HistoryDag`
        """

        leaf_label_dict = {leaf.label: relabel_func(leaf) for leaf in self.get_leaves()}
        if len(leaf_label_dict) != len(set(leaf_label_dict.keys())):
            raise RuntimeError(
                "relabeling function maps multiple leaf nodes to the same new label"
            )

        def relabel_clade(old_clade):
            return frozenset(leaf_label_dict[old_label] for old_label in old_clade)

        def relabel_node(old_node):
            if old_node.is_ua_node():
                return UANode(
                    EdgeSet(
                        [relabel_node(old_child) for old_child in old_node.children()]
                    )
                )
            else:
                clades = {
                    relabel_clade(old_clade): EdgeSet(
                        [relabel_node(old_child) for old_child in old_eset.targets],
                        weights=old_eset.weights,
                        probs=old_eset.probs,
                    )
                    for old_clade, old_eset in old_node.clades.items()
                }
                return HistoryDagNode(relabel_func(old_node), clades, None)

        if relax_type:
            newdag = HistoryDag(relabel_node(self.dagroot))
        else:
            newdag = self.__class__(relabel_node(self.dagroot))
        # do any necessary collapsing
        newdag = newdag.sample() | newdag
        return newdag

    def add_label_fields(self, new_field_names=[], new_field_values=lambda n: []):
        """Adds list of new fields to each node's label in the DAG.

        Args:
            new_field_names: A list of strings consisting of the names of the new fields to add.
            new_field_values: A callable object that takes a node and returns the ordered list
                of values for each new field name to assign to that node.
        """
        old_label = self.get_label_type()
        if any(field_name in old_label._fields for field_name in new_field_names):
            raise ValueError("One or more field names are already found in the DAG")
        new_label = namedtuple("new_label", old_label._fields + tuple(new_field_names))

        def add_fields(node):
            updated_fields = [x for x in node.label] + new_field_values[node]
            return new_label(*updated_fields)

        return self.relabel(add_fields)

    def remove_label_fields(self, fields_to_remove=[]):
        """Removes a list of fields from each node's label in the DAG.

        Args:
            fields_to_remove: A list of strings consisting of the names of the new fields to remove.
        """
        old_label = self.get_label_type()
        if len(set(old_label._fields) - set(fields_to_remove)) < 1:
            return self.unlabel()

        field_indices_to_keep = [
            i for i, x in enumerate(old_label._fields) if x not in fields_to_remove
        ]
        new_label = namedtuple(
            "new_label", tuple([old_label._fields[i] for i in field_indices_to_keep])
        )

        def update_fields(node):
            updated_fields = [node.label[i] for i in field_indices_to_keep]
            return new_label(*updated_fields)

        return self.relabel(update_fields)

    def update_label_fields(self, field_names, new_field_values):
        """Changes label field values to values returned by the function
        new_field_values.

        Args:
            field_names: A list of strings containing names of label fields whose contents are to be modified
            new_field_values: A function taking a node and returning a tuple of field values whose order matches field_names
        """
        label_type = self.get_label_type()
        field_dict = {field: index for index, field in enumerate(label_type._fields)}
        try:
            update_indices = (field_dict[field] for field in field_names)
        except KeyError:
            raise KeyError(
                "One of the field names you provided does not appear on node labels."
            )

        def update_fields(node):
            old_data = list(node.label)
            new_data = new_field_values(node)
            for idx, new_val in zip(update_indices, new_data):
                old_data[idx] = new_val
            return label_type(*old_data)

        return self.relabel(update_fields)

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

    def history_intersect(self, reference_dag: "HistoryDag", key=lambda n: n):
        """Modify this HistoryDag to contain only the histories which are also
        contained in ``reference_dag``.

        Args:
            reference_dag: The history DAG with which this one will be intersected. ``reference_dag``
                will not be modified.
            key: A function accepting a node and returning a value which will be used to compare
                nodes.
        """

        count_funcs = utils.edge_difference_funcs(reference_dag, key=key)

        min_weight = self.trim_optimal_weight(
            optimal_func=min,
            **count_funcs,
        )
        if min_weight > 0:
            raise IntersectionError(
                "Provided history DAGs have no histories in common,"
                " and a history DAG must contain at least one history."
            )

    def shared_history_count(self, reference_dag: "HistoryDag", key=lambda n: n) -> int:
        """Count the histories which are also contained in ``reference_dag``.

        Args:
            reference_dag: The history DAG with which this one will be intersected. ``reference_dag``
                will not be modified.
            key: A function accepting a node and returning a value which will be used to compare
                nodes.
        Returns:
            The number of histories shared between this history DAG and the reference.
        """
        count_funcs = utils.edge_difference_funcs(reference_dag, key=key)
        optimal_count = self.count_optimal_histories(**count_funcs, optimal_func=min)
        if optimal_count.state > 0:
            # There are no histories whose edges are all in reference_dag
            return 0
        else:
            return int(optimal_count)

    def merge(self, trees: Union["HistoryDag", Sequence["HistoryDag"]]):
        r"""Graph union this history DAG with all those in a list of history
        DAGs."""
        if isinstance(trees, HistoryDag):
            trees = [trees]

        selforder = self.postorder()
        nodedict = {n: n for n in selforder}

        for other in trees:
            if not self.label_fields == other.label_fields:
                raise ValueError(
                    f"The given HistoryDag must contain identical label fields.\n{self.label_fields}\nvs\n{other.label_fields}"
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
        """Provided as a deprecated synonym for :meth:`make_complete`."""
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
        expand_func: Callable[
            [Label], Iterable[Label]
        ] = parsimony_utils.default_aa_transitions.ambiguity_map.get_sequence_resolution_func(
            "sequence"
        ),
        expand_node_func: Callable[[HistoryDagNode], Iterable[Label]] = None,
        expandable_func: Callable[[Label], bool] = None,
    ) -> int:
        r"""Explode nodes according to a provided function. Adds copies of each
        node to the DAG with exploded labels, but with the same parents and
        children as the original node.

        Args:
            expand_func: (Deprecated) A function that takes a node label, and returns an iterable
                containing 'exploded' or 'disambiguated' labels corresponding to the original.
                The wrapper :meth:`utils.explode_label` is provided to make such a function
                easy to write.
            expand_node_func: A function that takes a node and returns an iterable
                containing 'exploded' or 'disambiguated' labels corresponding to the
                node. If provided, expand_func will be ignored.
            expandable_func: A function that takes a node label, and returns whether the
                iterable returned by calling expand_func on that label would contain more
                than one item.

        Returns:
            The number of new nodes added to the history DAG.
        """

        if expand_node_func is None:

            def expand_node_func(node):
                return expand_func(node.label)

        if expandable_func is None:

            def is_ambiguous(node):
                # Check if expand_func(label) has at least two items, without
                # exhausting the (arbitrarily expensive) generator
                return len(list(zip([1, 2], expand_node_func(node)))) > 1

        else:

            def is_ambiguous(node):
                return expandable_func(node.label)

        self.recompute_parents()
        nodedict = {node: node for node in self.postorder()}
        nodeorder = list(self.postorder())
        new_nodes = set()
        for node in nodeorder:
            if not node.is_ua_node() and not node.is_leaf() and is_ambiguous(node):
                for resolution in expand_node_func(node):
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

    def leaf_path_uncertainty_dag(self, terminal_node, node_data_func=lambda n: n):
        """Create a DAG of possible paths leading to `terminal_node`

        Args:
            terminal_node: The returned path DAG will contain all paths from the
                UA node ending at this node.
            node_data_func: A function accepting a HistoryDagNode and returning
                data for the corresponding node in the path dag. Return type must
                be a valid dictionary key.
        Returns:
            child_dictionary: A dictionary keyed by return values of
            `node_data_func`, with each value a dictionary keyed by child nodes,
            with edge supports as values.
        """
        self.recompute_parents()
        edge_counts = self.count_edges()
        child_dictionary = {node_data_func(terminal_node): dict()}

        for node in self.postorder_above(terminal_node, skip_ua_node=True):
            # Traversal has not visited node, or any of its children yet!
            node_key = node_data_func(node)
            child_dictionary[node_key] = dict()
            for parent in node.parents:
                if not parent.is_ua_node():
                    parent_key = node_data_func(parent)
                    parent_entry = child_dictionary[parent_key]
                    if node_key not in parent_entry:
                        parent_entry[node_key] = edge_counts[(parent, node)]
                    else:
                        parent_entry[node_key] += edge_counts[(parent, node)]

        return child_dictionary

    def leaf_path_uncertainty_graphviz(
        self,
        terminal_node,
        node_data_func=lambda n: n,
        node_label_func=lambda n: str(n.label),
    ):
        """Create a graphviz DAG of possible paths leading to `terminal_node`

        Args:
            terminal_node: The returned path DAG will contain all paths from the
                UA node ending at this node.
            node_data_func: A function accepting a HistoryDagNode and returning
                data for the corresponding node in the path dag. Return type must
                be a valid dictionary key.
            node_label_func: A function accepting an object of the type returned
                by `node_data_func`, and returning a label to be displayed on the
                corresponding node.
        """
        total_trees = self.count_histories()
        G = gv.Digraph("Path DAG to leaf", node_attr={})
        child_d = self.leaf_path_uncertainty_dag(
            terminal_node, node_data_func=node_data_func
        )

        label_ids = {key: str(idnum) for idnum, key in enumerate(child_d)}
        source_nodes = {node_data_func(child) for child in self.dagroot.children()}

        for node in child_d:
            if node in source_nodes:
                G.node(
                    label_ids[node], label=node_label_func(node), shape="invtriangle"
                )
            elif len(child_d[node]) == 0:
                G.node(label_ids[node], label=node_label_func(node), shape="octagon")
            else:
                G.node(label_ids[node], label=node_label_func(node))

        for parent, child_edge_d in child_d.items():
            for child, support in child_edge_d.items():
                if parent == child:  # skip self-edges
                    continue
                # Shifts color pallete to less extreme lower bouund
                color = f"0.0000 {support/total_trees * 0.9 + 0.1} 1.000"
                G.edge(
                    label_ids[parent],
                    label_ids[child],
                    penwidth="5",
                    color=color,
                    label=f"{support/total_trees:.2}",
                    weight=f"{support/total_trees}",
                )
        return G

    def summary(self):
        """Print nicely formatted summary about the history DAG."""
        print(type(self))
        print(f"Nodes:\t{self.num_nodes()}")
        print(f"Edges:\t{self.num_edges()}")
        print(f"Histories:\t{self.count_histories()}")
        print(f"Unique leaves in DAG:\t{self.num_leaves()}")
        print(
            f"Average number of parents of internal nodes:\t{self.internal_avg_parents()}"
        )

        print("\nIn histories in the DAG:")
        min_leaves, max_leaves = self.weight_range_annotate(
            edge_weight_func=lambda n1, n2: n2.is_leaf()
        )
        print(f"Leaf node count range: {min_leaves} to {max_leaves}")
        min_nodes, max_nodes = self.weight_range_annotate(**utils.node_countfuncs)
        print(f"Total node count range: {min_nodes} to {max_nodes}")
        print(f"Average pairwise RF distance:\t{self.average_pairwise_rf_distance()}")

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
        """Deprecated name for :meth:`HistoryDag.postorder_history_accum`"""
        return self.postorder_history_accum(*args, **kwargs)

    def optimal_weight_annotate(
        self,
        start_func: Callable[["HistoryDagNode"], Weight] = lambda n: 0,
        edge_weight_func: Callable[
            ["HistoryDagNode", "HistoryDagNode"], Weight
        ] = parsimony_utils.hamming_edge_weight,
        accum_func: Callable[[List[Weight]], Weight] = sum,
        optimal_func: Callable[[List[Weight]], Weight] = min,
        **kwargs,
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

    def count_optimal_histories(
        self,
        start_func: Callable[["HistoryDagNode"], Weight] = lambda n: 0,
        edge_weight_func: Callable[
            ["HistoryDagNode", "HistoryDagNode"], Weight
        ] = parsimony_utils.hamming_edge_weight,
        accum_func: Callable[[List[Weight]], Weight] = sum,
        optimal_func: Callable[[List[Weight]], Weight] = min,
        eq_func: Callable[[Weight, Weight], bool] = lambda w1, w2: w1 == w2,
        **kwargs,
    ):
        """Count the number of histories which would be left if the DAG were
        trimmed.

        That is, how many histories would be left if :meth:`HistoryDag.trim_optimal_weight`
        were called with the same arguments?

        Args:
            All arguments are the same as :meth:`HistoryDag.trim_optimal_weight`.

        Returns:
            A :class:`utils.IntState` object containing the number of optimal histories
            in the DAG, with ``state`` attribute containing their (optimal) weight.

            As a side-effect, each node's ``_dp_data`` attribute is populated with
            IntState objects containing the number of optimal sub-histories rooted at that
            node, and the weight of those sub-histories.
        """

        def _start_func(node):
            return utils.IntState(1, state=start_func(node))

        def _edge_weight_func(parent, child):
            return utils.IntState(1, state=edge_weight_func(parent, child))

        def _between_clade_accum(clade_weight_list):
            return utils.IntState(
                prod(clade_weight_list),
                state=accum_func([el.state for el in clade_weight_list]),
            )

        def _within_clade_accum(subtree_weight_list):
            optimal_weight = optimal_func([el.state for el in subtree_weight_list])
            count = sum(
                el for el in subtree_weight_list if eq_func(optimal_weight, el.state)
            )
            return utils.IntState(count, state=optimal_weight)

        return self.postorder_history_accum(
            _start_func,
            _edge_weight_func,
            _within_clade_accum,
            _between_clade_accum,
        )

    def weight_count(
        self,
        start_func: Callable[["HistoryDagNode"], Weight] = lambda n: 0,
        edge_weight_func: Callable[
            ["HistoryDagNode", "HistoryDagNode"], Weight
        ] = parsimony_utils.hamming_edge_weight,
        accum_func: Callable[[List[Weight]], Weight] = sum,
        **kwargs,
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

    def weight_range_annotate(
        self,
        edge_weight_func: Callable[["HistoryDagNode", "HistoryDagNode"], Weight],
        start_func: Callable[["HistoryDagNode"], Weight] = lambda n: 0,
        accum_func: Callable[[List[Weight]], Weight] = sum,
        min_func: Callable[[List[Weight]], Weight] = min,
        max_func: Callable[[List[Weight]], Weight] = max,
        **kwargs,
    ):
        """Computes the minimum and maximum weight of any history in the
        history DAG.

        As a side-effect, this method also stores in each node's ``_dp_data`` attribute
        a tuple containing the minimum and maximum weights of any sub-history beneath that node.

        Args:
            edge_func: A function which assigns a weight to pairs of labels, with the
                parent node label the first argument
            start_func: A function which assigns a weight to each leaf node
            accum_func: A way to 'add' a list of weights together
            min_func: A function which takes a list of weights and returns their "minimum"
            max_func: A function which takes a list of weights and returns their "maximum"

        Returns:
            A tuple containing the minimum and maximum weight of any history in the history DAG.
        """
        single_kwarg = utils.AddFuncDict(
            {
                "start_func": start_func,
                "edge_weight_func": edge_weight_func,
                "accum_func": accum_func,
            },
            name="Weight",
        )

        pair_kwarg = single_kwarg + single_kwarg

        def accum_within_clade(weight_list):
            return (
                min_func([t[0] for t in weight_list]),
                max_func([t[1] for t in weight_list]),
            )

        return self.optimal_weight_annotate(
            optimal_func=accum_within_clade, **pair_kwarg
        )

    def hamming_parsimony_count(self):
        """Deprecated in favor of
        :meth:`sequence_dag.SequenceHistoryDag.hamming_parsimony_count`."""
        warnings.warn(
            "`HistoryDag.hamming_parsimony_count` is deprecated in favor of"
            " `sequence_dag.SequenceHistoryDag.hamming_parsimony_count`."
        )
        return self.weight_count()

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

        For large DAGs, this method is prohibitively slow. Use :meth:`count_topologies_fast` instead.

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
            A dictionary mapping each node in the DAG to the number of trees
            that it takes part in.
        """
        node2count = {}
        node2stats = {}

        self.count_histories()
        self.recompute_parents()
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

        if collapse:
            collapsed_n2c = {}
            for node in node2count.keys():
                clade = node.clade_union()
                if clade not in collapsed_n2c:
                    collapsed_n2c[clade] = 0

                collapsed_n2c[clade] += node2count[node]
            return collapsed_n2c
        else:
            return node2count

    def count_edges(
        self, collapsed=False
    ) -> Dict[Tuple[HistoryDagNode, HistoryDagNode], int]:
        """Counts the number of trees each edge takes part in.

        Returns:
            A dictionary mapping each edge in the DAG to the number of trees
            that it takes part in.
        """
        edge2count = {}
        node2stats = {}

        self.count_histories()
        self.recompute_parents()
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
            0
            if isinstance(l1, UALabel)
            else parsimony_utils.default_aa_transitions.weighted_hamming_distance(
                l1.sequence, l2.sequence
            )
        ),
        accum_func: Callable[[List[Weight]], Weight] = sum,
        expand_func: Callable[
            [Label], Iterable[Label]
        ] = parsimony_utils.default_aa_transitions.ambiguity_map.get_sequence_resolution_func(
            "sequence"
        ),
    ):
        r"""Template method for counting tree weights in the DAG, with exploded
        labels. Like :meth:`HistoryDag.weight_count`, but creates dictionaries
        of Counter objects at each node, keyed by possible sequences at that
        node. Analogous to :meth:`HistoryDag.count_histories` with
        `expand_func` provided.

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

    def optimal_sum_rf_distance(
        self,
        reference_dag: "HistoryDag",
        optimal_func: Callable[[List[Weight]], Weight] = min,
    ):
        """Returns the optimal (min or max) summed rooted RF distance to all
        histories in the reference DAG.

        The given history must be on the same taxa as all trees in the DAG.
        Since computing reference splits is expensive, it is better to use
        :meth:``optimal_weight_annotate`` and :meth:``utils.make_rfdistance_countfuncs``
        instead of making multiple calls to this method with the same reference
        history DAG.
        """
        kwargs = utils.sum_rfdistance_funcs(reference_dag)
        return self.optimal_weight_annotate(**kwargs, optimal_func=optimal_func)

    def trim_optimal_sum_rf_distance(
        self,
        reference_dag: "HistoryDag",
        optimal_func: Callable[[List[Weight]], Weight] = min,
    ):
        """Trims the DAG to contain only histories with the optimal (min or
        max) sum rooted RF distance to the given reference DAG.

        Trimming to the minimum sum RF distance is equivalent to finding 'median' topologies,
        and trimming to maximum sum rf distance is equivalent to finding topological outliers.

        The given history must be on the same taxa as all trees in the DAG.
        Since computing reference splits is expensive, it is better to use
        :meth:``trim_optimal_weight`` and :meth:``utils.sum_rfdistance_funcs``
        instead of making multiple calls to this method with the same reference
        history.
        """
        kwargs = utils.sum_rfdistance_funcs(reference_dag)
        return self.trim_optimal_weight(**kwargs, optimal_func=optimal_func)

    def optimal_rf_distance(
        self,
        history: "HistoryDag",
        rooted: bool = False,
        optimal_func: Callable[[List[Weight]], Weight] = min,
    ):
        """Returns the optimal (min or max) RF distance to a given history.

        The given history must be on the same taxa as all trees in the DAG.
        Since computing reference splits is expensive, it is better to use
        :meth:`optimal_weight_annotate` and :meth:`utils.make_rfdistance_countfuncs`
        instead of making multiple calls to this method with the same reference
        history.
        """
        kwargs = utils.make_rfdistance_countfuncs(history, rooted=rooted)
        return self.optimal_weight_annotate(**kwargs, optimal_func=optimal_func)

    def count_rf_distances(self, history: "HistoryDag", rooted: bool = False):
        """Returns a Counter containing all RF distances to a given history.

        The given history must be on the same taxa as all trees in the DAG.

        Since computing reference splits is expensive, it is better to use
        :meth:`weight_count` and :meth:`utils.make_rfdistance_countfuncs`
        instead of making multiple calls to this method with the same reference
        history.
        """
        kwargs = utils.make_rfdistance_countfuncs(history, rooted=rooted)
        return self.weight_count(**kwargs)

    def count_sum_rf_distances(self, reference_dag: "HistoryDag", rooted: bool = False):
        """Returns a Counter containing all sum RF distances to a given
        reference DAG.

        The given history DAG must be on the same taxa as all trees in the DAG.

        Since computing reference splits is expensive, it is better to use
        :meth:`weight_count` and :meth:`utils.sum_rfdistance_funcs`
        instead of making multiple calls to this method with the same reference
        history DAG.
        """
        kwargs = utils.sum_rfdistance_funcs(reference_dag)
        return self.weight_count(**kwargs)

    # ######## End Abstract DP method derivatives ########

    def sum_rf_distances(self, reference_dag: "HistoryDag" = None):
        r"""Computes the sum of all Robinson-Foulds distances between a history
        in this DAG and a history in the reference DAG.

        This is rooted RF distance.

        If T is the set of histories in the reference DAG, and T' is the set of histories in
        this DAG, then the returned sum is:

        .. math::

            \sum_{t\in T} \sum_{t'\in T'} d(t, t')

        That is, since RF distance is symmetric, when T = T' (such as when ``reference_dag=None``),
        or when the intersection of T and T' is nonempty, some distances are counted twice.

        Args:
            reference_dag: If None, the sum of pairwise distances between histories in this DAG
                is computed. If provided, the sum is over pairs containing one history in this DAG and
                one from ``reference_dag``.

        Returns:
            An integer sum of RF distances.
        """

        def get_data(dag):
            n_histories = dag.count_histories()
            N = dag.count_nodes(collapse=True)
            try:
                N.pop(frozenset())
            except KeyError:
                pass

            clade_count_sum = sum(N.values())
            return (n_histories, N, clade_count_sum)

        n_histories_prime, N_prime, clade_count_sum_prime = get_data(self)

        if reference_dag is None:
            n_histories, N, clade_count_sum = (
                n_histories_prime,
                N_prime,
                clade_count_sum_prime,
            )
        else:
            n_histories, N, clade_count_sum = get_data(reference_dag)

        intersection_term = sum(
            count_prime * N[clade]
            for clade, count_prime in N_prime.items()
            if clade in N
        )

        return (
            n_histories * clade_count_sum_prime
            + n_histories_prime * clade_count_sum
            - 2 * intersection_term
        )

    def average_pairwise_rf_distance(
        self, reference_dag: "HistoryDag" = None, non_identical=True
    ):
        """Return the average Robinson-Foulds distance between pairs of
        histories.

        Args:
            reference_dag: A history DAG from which to take the second history in
                each pair. If None, ``self`` will be used as the reference.
            non_identical: If True, mean divisor will be the number of non-identical pairs.

        Returns:
            The average rf-distance between pairs of histories, where the first history
            comes from this DAG, and the second comes from ``reference_dag``. The normalization
            constant is the product of the number of histories in the two DAGs, unless
            ``non_identical`` is True, in which case the number of histories which appear
            in both DAGs is subtracted from this constant.
        """
        sum_pairwise_distance = self.sum_rf_distances(reference_dag=reference_dag)
        if reference_dag is None:
            # ignore the diagonal in the distance matrix, since it contains
            # zeros:
            n1 = self.count_histories()
            n2 = n1

            def compute_intersection_size():
                return n1

        else:
            n1 = self.count_histories()
            n2 = reference_dag.count_histories()

            def compute_intersection_size():
                return self.shared_history_count(reference_dag)

        if non_identical:
            normalize_num = (n1 * n2) - compute_intersection_size()
        else:
            normalize_num = n1 * n2
        return sum_pairwise_distance / max(1, normalize_num)

    def trim_optimal_weight(
        self,
        start_func: Callable[["HistoryDagNode"], Weight] = lambda n: 0,
        edge_weight_func: Callable[
            [HistoryDagNode, HistoryDagNode], Weight
        ] = parsimony_utils.hamming_edge_weight,
        accum_func: Callable[[List[Weight]], Weight] = sum,
        optimal_func: Callable[[List[Weight]], Weight] = min,
        # max_weight: Weight = None,
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
        ] = parsimony_utils.hamming_edge_weight,
    ):
        """Inserts a sequence into the DAG.

        Sequence will be inserted as a child of the dagnode(s)
        realizing the minimum overall distance between sequences, and also added
        to the dag as a child of other nodes in such a way as to guarantee
        that every tree in the DAG now contains the new sequence.

        The choice of other nodes is computed by looking at the set of
        nodes that are `incompatible` with the first minimizing node.
        For a full description of this, see the docstring for the method-local
        function ``incompatible``.
        """
        # all nodes in the dag except for the UA
        postorder = list(self.postorder())[:-1]
        if not any([new_leaf_id == getattr(n.label, id_name) for n in postorder]):

            def insert_node_as_sibling(node, sib):
                altered_nodes = {}

                def follow_up(node, old_node_cu):
                    for p in node.parents:
                        if old_node_cu in p.clades:
                            old_p = p
                            p_clade_union = p.clade_union()
                            edgeset = p.clades.pop(old_node_cu)
                            p.clades[node.clade_union()] = edgeset
                            altered_nodes[old_p] = p
                            follow_up(p, p_clade_union)

                for parent in sib.parents:
                    if node.label not in parent.clade_union():
                        old_parent = parent
                        parent_clade_union = parent.clade_union()
                        parent.clades[frozenset([node.label])] = EdgeSet([node])
                        altered_nodes[old_parent] = parent
                        follow_up(parent, parent_clade_union)

                return altered_nodes

            def insert_node_as_child(node, parent):
                altered_nodes = {}

                def follow_up(node, old_node_cu):
                    for p in node.parents:
                        if old_node_cu in p.clades:
                            old_p = p
                            p_clade_union = p.clade_union()
                            edgeset = p.clades.pop(old_node_cu)
                            p.clades[node.clade_union()] = edgeset
                            altered_nodes[old_p] = p
                            follow_up(p, p_clade_union)

                if node.label not in parent.clade_union():
                    old_parent = parent
                    parent_clade_union = parent.clade_union()
                    parent.clades[frozenset([node.label])] = EdgeSet([node])
                    altered_nodes[old_parent] = parent
                    follow_up(parent, parent_clade_union)

                return altered_nodes

            def find_min_dist_nodes(new_node, node_set, dist):
                """Finds the set of nodes in arg `node_set` that realize the
                minimum distance to `new_node` (sort so that leaf nodes are at
                the end of the list)"""
                return_set = []
                min_dist = float("inf")
                for node in node_set:
                    this_dist = dist(node, new_node)
                    if this_dist < min_dist:
                        return_set = [(node.is_leaf(), node)]
                        min_dist = this_dist
                    elif this_dist <= min_dist:
                        return_set.append((node.is_leaf(), node))
                return list(zip(*sorted(return_set)))[1]

            def incompatible(n1, n2):
                """Checks if nodes n1 and n2 are `incompatible` in the sense
                that, based on their clade sets, they cannot both come from the
                same tree in the DAG.

                Note that, just because 2 nodes might be compatible does
                not mean that they actually are in the same tree. Merely
                that they could be.
                """
                if n1.is_root() or n2.is_root():
                    return False
                if n1 == n2:
                    return False
                if any([n1.clade_union() <= B for B in n2.child_clades()]):
                    return False
                if any([n2.clade_union() <= B for B in n1.child_clades()]):
                    return False
                return True

            leaf_dict = {n.label: n for n in self.get_leaves()}

            def incompatible_set(node, nodeset):
                """Returns the set of all nodes incompatible to `node` that
                satisfy the conditions:

                1. incompatible nodes lie in the path between the leaf
                nodes reachable by arg `node` and the UA,
                2. only the subset of incompatible nodes that are also in
                the set of nodes arg `nodeset`
                """
                self.recompute_parents()
                for leaf_label in node.clade_union():
                    yield from (
                        n
                        for n in self.postorder_above(
                            leaf_dict[leaf_label], recompute_parents=False
                        )
                        if (n in nodeset and incompatible(n, node))
                    )

            self.recompute_parents()
            new_node = empty_node(
                postorder[0].label._replace(**{id_name: new_leaf_id}), {}, None
            )
            changed_nodes = {}
            incompatible_nodes_so_far = set(postorder)
            while len(incompatible_nodes_so_far) > 0:
                min_dist_nodes = find_min_dist_nodes(
                    new_node, incompatible_nodes_so_far, dist
                )
                for other_node in min_dist_nodes:
                    if other_node in incompatible_nodes_so_far:
                        if other_node.is_leaf():
                            if len(changed_nodes) < 1:
                                changed_nodes.update(
                                    insert_node_as_sibling(new_node, other_node)
                                )
                                incompatible_nodes_so_far = set()
                        else:
                            incompatible_nodes_so_far.remove(other_node)
                            incompatible_nodes_so_far = set(
                                incompatible_set(other_node, incompatible_nodes_so_far)
                            )
                            changed_nodes.update(
                                insert_node_as_child(new_node, other_node)
                            )
                            incompatible_nodes_so_far = [
                                x if x not in changed_nodes else changed_nodes[x]
                                for x in incompatible_nodes_so_far
                            ]

            for clade, edgeset in self.dagroot.clades.items():
                edgeset.set_targets(
                    [
                        n if n not in changed_nodes else changed_nodes[n]
                        for n in edgeset.targets
                    ]
                )

    # ######## DAG Traversal Methods ########

    def postorder_above(
        self, terminal_node, skip_ua_node=False, recompute_parents=True
    ):
        """Recursive postorder traversal of all ancestors of a (possibly
        internal) node. This traversal is postorder with respect to reversed
        edge directions. With respect to standard edge directions (pointing
        towards leaves), the traversal order guarantees that all of a node's
        parents will be visited before the node itself.

        Args:
            terminal_node: The node whose ancestors should be included in the
                traversal. This must actually be a node in `self`, not simply
                compare equal to a node in `self`.
            skip_ua_node: If True, the UA node will not be included in the traversal
            recompute_parents: If False, node parent sets will not be recomputed.
                This makes many repeated calls to postorder_above much faster.

        Returns:
            Generator on nodes that lie on any path between node_as_leaf and UA node
        """
        if recompute_parents:
            self.recompute_parents()
        visited = set()

        def traverse(node: HistoryDagNode):
            visited.add(id(node))
            for parent in node.parents:
                if not id(parent) in visited:
                    if (not skip_ua_node) or (not parent.is_ua_node()):
                        yield from traverse(parent)
            yield node

        yield from traverse(terminal_node)

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
    treeroot: ete3.TreeNode,
    label_features: List[str],
    label_functions: Mapping[str, Callable[[ete3.TreeNode], Any]] = {},
    attr_func: Callable[[ete3.TreeNode], Any] = lambda n: None,
    child_node_func: Callable[
        [ete3.TreeNode], Sequence[ete3.TreeNode]
    ] = lambda n: n.children,
    leaf_node_func: Callable[
        [ete3.TreeNode], Sequence[ete3.TreeNode]
    ] = ete3.TreeNode.get_leaves,
    edge_weight_func: Callable[[ete3.TreeNode], Any] = lambda n: 1,
) -> HistoryDag:
    """Build a tree-shaped :meth:`historydag.HistoryDag` (a 'history') object
    from the provided tree data.

    Default arguments are suitable for loading a :class:`ete3.Tree`, but by providing
    appropriate `child_node_func` and `leaf_node_func`, any data structure implementing
    a tree can be used.

    Args:
        treeroot: The root node of a tree to be converted to HistoryDag history
        label_features: tree node attribute names to be used as HistoryDagNode label fields.
            Each attribute name must be accessible by ``getattr(treenode, name)``.
            Field names provided in `label_functions` will take precedence.
        label_functions: dictionary keyed by additional label field names, containing
            functions mapping tree nodes to intended label field values.
        attr_func: function to populate HistoryDag node `attr` attribute,
            which is not used to distinguish nodes, and may be overwritten
            by `attr` of another node with the same label and child clades.
        child_node_func: function taking a tree node and returning an iterable
            containing the node's children. By default, accesses node's
            `children` attribute.
        leaf_node_func: function accepting a tree node and returning an iterable
            containing the leaf nodes accessible from that node.
        edge_weight_func: function accepting a tree node and returning the weight
            of that node's parent edge.

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
        return frozenset(node_label(node) for node in leaf_node_func(r))

    def _unrooted_from_tree(treeroot):
        dag = HistoryDagNode(
            node_label(treeroot),
            {
                leaf_names(child): EdgeSet(
                    [_unrooted_from_tree(child)], weights=[edge_weight_func(child)]
                )
                for child in child_node_func(treeroot)
            },
            attr_func(treeroot),
        )
        return dag

    # Check for unique leaf labels:
    if len(list(leaf_node_func(treeroot))) != len(leaf_names(treeroot)):
        raise ValueError(
            "This tree's leaves are not labeled uniquely. Check your tree, "
            "or modify the label fields so that leaves are unique.\n"
        )

    # Checking for unifurcation is handled in HistoryDagNode.__init__.

    dag = _unrooted_from_tree(treeroot)
    dagroot = UANode(EdgeSet([dag], weights=[edge_weight_func(treeroot)]))
    return HistoryDag(dagroot)


def history_dag_from_trees(
    treelist: List[ete3.TreeNode],
    label_features: List[str],
    **kwargs,
):
    """Create a :class:`historydag.HistoryDag` from a list of trees.

    Default arguments are suitable for loading lists of :class:`ete3.Tree`s, but
    any tree data structure can be used by providing appropriate functions to
    `child_node_func` and `leaf_node_func` keyword arguments.

    Args:
        treelist: List of root nodes of input trees.
        label_features: tree node attribute names to be used as HistoryDagNode label fields.
            Each attribute name must be accessible by ``getattr(treenode, name)``.
            Field names provided in `label_functions` keyword argument will take precedence.
        kwargs: Passed to :meth:`historydag.from_tree`. See docstring
            for that method for argument details

    Returns:
        :class:`historydag.HistoryDag` constructed from input trees.
    """
    return history_dag_from_histories(
        [from_tree(treeroot, label_features, **kwargs) for treeroot in treelist]
    )


def history_dag_from_etes(*args, **kwargs) -> HistoryDag:
    """Deprecated name for :meth:`historydag.history_dag_from_trees`"""
    return history_dag_from_trees(*args, **kwargs)


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
