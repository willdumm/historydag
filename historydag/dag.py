import pickle
import operator
from functools import reduce
import math
import graphviz as gv
from historydag import utils
from historydag.utils import UALabel
import ete3
import random
from typing import List, Callable, Any, TypeVar, Mapping, Generator, Iterable, Set
from collections import Counter, namedtuple
from numbers import Number

# from gctree import CollapsedTree
from multiset import FrozenMultiset
from historydag.counterops import counter_sum, counter_prod, addweight, prod

Weight = TypeVar("Weight")
Label = TypeVar("Label")

class HistoryDagNode:
    r"""A recursive representation of a history DAG object
    - a dictionary keyed by clades (frozensets) containing EdgeSet objects
    - a label, which is a namedtuple.
    """

    def __init__(self, label: Label, clades: dict = {}, attr: Any = None):
        assert isinstance(label, tuple) or isinstance(label, UALabel)
        self.clades = clades
        # If passed a nonempty dictionary, need to add self to children's parents
        self.label = label
        self.parents = set()
        self.attr = attr
        if self.clades:
            for _, edgeset in self.clades.items():
                edgeset.parent = self
            for child in self.children():
                child.parents.add(self)

        if not self.is_root() and len(self.clades) == 1:
            raise ValueError(f"Internal nodes (those which are not the DAG UA root node) "
                             f"may not have exactly one child clade; Unifurcations cannot be expressed "
                             f"in the history DAG. A HistoryDagNode with {label} and clades {set(clades.keys())} is not allowed.")

    def __repr__(self) -> str:
        return str((self.label, set(self.clades.keys())))

    def __hash__(self) -> int:
        return hash((self.label, self.partitions()))

    def __eq__(self, other: "HistoryDagNode") -> bool:
        # return hash(self) == hash(other)
        return ((self.label, self.partitions()) == (other.label, other.partitions()))

    def node_self(self) -> "HistoryDagNode":
        """Returns a HistoryDagNode object with the same clades and label, but no descendant edges."""
        return HistoryDagNode(self.label, {clade: EdgeSet() for clade in self.clades})

    def under_clade(self) -> frozenset:
        r"""Returns the union of this node's child clades"""
        if self.is_leaf():
            return frozenset([self.label])
        else:
            return frozenset().union(*self.clades.keys())

    def is_leaf(self) -> bool:
        """Returns whether this is a leaf node."""
        return not bool(self.clades)

    def is_root(self) -> bool:
        """Returns whether this is a DAG root node, or UA (universal ancestor) node."""
        return isinstance(self.label, UALabel)

    def partitions(self) -> frozenset:
        """Returns the node's child clades"""
        return frozenset(self.clades.keys())

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
        r"""Adds edge, if not already present and allowed. Returns whether edge was added."""
        # target clades must union to a clade of self
        key = target.under_clade()
        if key not in self.clades:
            raise KeyError(
                "Target clades' union is not a clade of this parent node: "
                + str(key)
                + " not in "
                + str(self.clades)
            )
        else:
            target.parents.add(self)
            return self.clades[key].add_to_edgeset(
                target,
                weight=weight,
                prob=prob,
                prob_norm=prob_norm,
            )

    def remove_node(self, nodedict: Mapping[int, "HistoryDagNode"] = {}):
        r"""Recursively removes node self and any orphaned children from dag.
        May not work on root.
        Does not check to make sure that parent clade still has descendant edges."""
        if hash(self) in nodedict:
            nodedict.pop(hash(self))
        for child in self.children():
            if self in child.parents:
                child.parents.remove(self)
            if not child.parents:
                child.remove_node(nodedict=nodedict)
        for parent in self.parents:
            parent.clades[self.under_clade()].remove_from_edgeset_byid(self)
            self.removed = True

    def _sample(self) -> "HistoryDagNode":
        r"""Samples a sub-history-DAG that is also a tree containing the root and
        all leaf nodes. Returns a new HistoryDagNode object."""
        sample = self.node_self()
        for clade, eset in self.clades.items():
            sampled_target, target_weight = eset.sample()
            sample.clades[clade].add_to_edgeset(
                sampled_target._sample(),
                weight=target_weight,
            )
        return sample

    def _get_trees(self) -> Generator["HistoryDagNode", None, None]:
        r"""Return a generator to iterate through all trees expressed by the DAG."""

        def genexp_func(clade):
            # Return generator expression of all possible choices of tree
            # structure from dag below clade
            def f():
                eset = self.clades[clade]
                return (
                    (clade, targettree, i)
                    for i, target in enumerate(eset.targets)
                    for targettree in target._get_trees()
                )

            return f

        optionlist = [genexp_func(clade) for clade in self.clades]

        # TODO is this duplicated code?
        for option in utils.cartesian_product(optionlist):
            tree = self.node_self()
            for clade, targettree, index in option:
                tree.clades[clade].add_to_edgeset(
                    targettree,
                    weight=self.clades[clade].weights[index],
                )
            yield tree

    @utils.ignore_ualabel('UA_Node')
    def _newick_label(
        self,
        name_func: Callable[["HistoryDagNode"], str] = (lambda n: "unnamed"),
        features: List[str] = None,
        feature_funcs: Mapping[str, Callable[["HistoryDagNode"], str]] = {},
    ) -> str:
        if features is None:
            features = self.label._fields
        # Use dict to avoid duplicate fields
        nameval_dict = {
            name: val for name, val in self.label._asdict().items() if name in features
        }
        nameval_dict.update({name: f(self) for name, f in feature_funcs.items()})
        featurestr = ":".join(f"{name}={val}" for name, val in nameval_dict.items())
        return name_func(self) + (f"[&&NHX:{featurestr}]" if featurestr else "")


class HistoryDag(object):
    r"""A wrapper object to contain user-exposed HistoryDag methods and point to a HistoryDagNode root"""

    def __init__(self, dagroot):
        self.dagroot = dagroot

    def __eq__(self, other):
        # Eventually this can be done by comparing bytestrings, but we need
        # some sorting to be done first, to ensure two dags that represent
        # identical trees return True.
        raise NotImplementedError

    def get_trees(self, *args, **kwargs) -> Generator["HistoryDag", None, None]:
        """Iterate through trees in the history DAG"""
        for cladetree in self.dagroot._get_trees(*args, **kwargs):
            yield HistoryDag(cladetree)

    def sample(self, *args, **kwargs):
        return HistoryDag(self.dagroot._sample(*args, **kwargs))

    def copy(self):
        """Uses bytestring serialization, and is guaranteed to copy:

        * node labels
        * node attr attributes
        * edge weights
        * edge probabilities

        However, other object attributes will not be copied.
        """
        return deserialize(self.serialize())

    def merge(self, other: "HistoryDag"):
        r"""performs post order traversal to add node and all of its children,
        without introducing duplicate nodes in self."""
        if not hash(self.dagroot) == hash(other.dagroot):
            raise ValueError(
                f"The given HistoryDag must be a root node on identical taxa.\n{self.dagroot}\nvs\n{other.dagroot}"
            )
        selforder = self.postorder()
        otherorder = other.postorder()
        hashdict = {hash(n): n for n in selforder}
        for n in otherorder:
            if hash(n) in hashdict:
                pnode = hashdict[hash(n)]
            else:
                pnode = n.node_self()
                hashdict[hash(n)] = pnode

            for _, edgeset in n.clades.items():
                for child, weight, _ in edgeset:
                    pnode.add_edge(hashdict[hash(child)], weight=weight)

    def add_all_allowed_edges(
        self, new_from_root=True, adjacent_labels=True, preserve_parent_labels=False
    ):
        r"""Add all allowed edges to the DAG, returning the number that were added.
        if new_from_root is False, no edges are added that start at DAG root.
        This is useful to enforce a single ancestral sequence.
        If adjacent_labels is False, no edges will be added between nodes with the same labels.
        preserve_parent_labels was to show something was true....?"""
        n_added = 0
        clade_dict = {node.under_clade(): [] for node in self.postorder()}
        if preserve_parent_labels is True:
            self.recompute_parents()
            uplabels = {
                node: {parent.label for parent in node.parents}
                for node in self.postorder()
            }
        for node in self.postorder():
            clade_dict[node.under_clade()].append(node)
        for node in self.postorder():
            if new_from_root is False and node.is_root():
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

    def to_newick(
        self,
        name_func=lambda n: "unnamed",
        features=None,
        feature_funcs={},
    ):
        r"""Converts to extended newick format with arbitrary node names and a
        sequence feature. For use on tree-shaped DAG.
        If fixed_order is True, with same namedict, two trees' newick representations are
        equal iff the two trees have the same topology and node sequences.
        Should be usable for equality in this sense on general DAG too, with include_root
        True
        TODO refactor and add dp option"""

        def newick(node):
            if node.is_leaf():
                return node._newick_label(
                    name_func, features=features, feature_funcs={}
                )
            else:
                childnewicks = sorted([newick(node2) for node2 in node.children()])
                return (
                    "("
                    + ",".join(childnewicks)
                    + ")"
                    + node._newick_label(name_func, features=features, feature_funcs={})
                )

        return newick(next(self.dagroot.children())) + ";"

    def to_ete(self, name_func=lambda n: "unnamed", features=None):
        return ete3.TreeNode(
            newick=self.to_newick(name_func=name_func, features=features), format=1
        )

    def to_graphviz(self, labelfunc=None, namedict={}, show_partitions=True):
        r"""Converts to graphviz Digraph object. Namedict must associate sequences
        of all leaf nodes to a name
        """

        def taxa(clade):
            l = [labeller(taxon) for taxon in clade]
            l.sort()
            return ",".join(l)

        @utils.ignore_ualabel('UA_node')
        def labeller(label):
            if label in namedict:
                return str(namedict[label])
            elif len(str(tuple(label))) < 11:
                return str(tuple(label))
            else:
                return str(hash(label))

        if labelfunc is None:

            def labelfunc(node):
                return labeller(node.label)

        G = gv.Digraph("labeled partition DAG", node_attr={"shape": "record"})
        for node in self.postorder():
            if node.is_leaf() or show_partitions is False:
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
                        f"{id(node)}:{taxa(clade) if show_partitions else 'label'}:s",
                        f"{id(target)}:n",
                        label=label,
                    )
        return G

    def internal_avg_parents(self):
        r"""Returns the average number of parents among internal nodes
        A simple measure of similarity of trees that the DAG expresses."""
        nonleaf_parents = (len(n.parents) for n in self.postorder() if not n.is_leaf())
        n = 0
        cumsum = 0
        for sum in nonleaf_parents:
            n += 1
            cumsum += sum
        # Exclude root:
        return cumsum / float(n - 1)

    def make_uniform(self):
        """Adjust edge probabilities so that the DAG expresses a uniform distribution on expressed trees"""
        self.count_trees()
        for node in self.postorder():
            for clade, eset in node.clades.items():
                for i, target in enumerate(eset.targets):
                    eset.probs[i] = target._dp_data


    def explode_nodes(self, expand_func: Callable[[Label], Iterable[Label]]=utils.sequence_resolutions, expandable_func: Callable[[Label], bool]=None):
        r"""Explode nodes according to a provided function, adding copies of each node to the DAG with
        exploded labels, but with the same parents and children as the original node.
        
        Args:
            expand_func: A function that takes a node label, and returns an iterable containing
                'exploded' or 'disambiguated' labels corresponding to the original.
                The wrapper :meth:`utils.explode_label` is provided to make such a function easy to write.
            expandable_func: A function that takes a node label, and returns whether the iterable returned by calling expand_func on that label would contain more than one item.
        """

        if expandable_func is None:
            def is_ambiguous(label):
                # Check if expand_func(label) has at least two items, without
                # exhausting the (arbitrarily expensive) generator
                return len(list(zip([1,2], expand_func(label)))) > 1
        else:
            is_ambiguous = expandable_func

        self.recompute_parents()
        nodedict = {hash(node): node for node in self.postorder()}
        nodeorder = list(self.postorder())
        new_nodes = set()
        for node in nodeorder:
            if not node.is_root() and is_ambiguous(node.label):
                if node.is_leaf():
                    raise ValueError("Passed expand_func would explode a leaf node. "
                                     "Leaf nodes may not be exploded.")
                clades = frozenset(node.clades.keys())
                for resolution in expand_func(node.label):
                    if hash((resolution, clades)) in nodedict:
                        newnode = nodedict[hash((resolution, clades))]
                    else:
                        newnode = node.node_self()
                        newnode.label = resolution
                        nodedict[hash(newnode)] = newnode
                        new_nodes.add(newnode)
                    # Add all edges into and out of node to newnode
                    for target in node.children():
                        newnode.add_edge(target)
                    for parent in node.parents:
                        parent.add_edge(newnode)
                # Delete old node
                node.remove_node(nodedict=nodedict)
        return new_nodes

    def summary(self):
        print(f"Nodes:\t{sum(1 for _ in self.postorder())}")
        print(f"Trees:\t{self.count_trees()}")
        utils.hist(self.get_weight_counts_with_ambiguities()[self.dagroot.label])

    def get_weight_counts_with_ambiguities(self, distance_func=utils.wrapped_hamming_distance):
        r"""like get_weight_counts, but creates dictionaries of Counter objects at each node,
        keyed by possible sequences at that node.
        The total number of trees will be greater than count_trees(), as these are possible
        disambiguations of trees. These disambiguations may not be unique (?), but if two are
        the same, they come from different subtrees of the DAG.
        TODO: replace with call to abstract method"""

        for node in self.postorder():
            node.weight_counters = {}
            for sequence in utils.sequence_resolutions(node.label):
                if node.is_leaf():
                    node.weight_counters[sequence] = Counter({0: 1})
                else:
                    cladelists = [
                        [
                            addweight(
                                target_wc,
                                distance_func(target_seq, sequence),
                            )
                            for target in node.children(clade=clade)
                            for target_seq, target_wc in target.weight_counters.items()
                        ]
                        for clade in node.clades
                    ]
                    cladecounters = [counter_sum(cladelist) for cladelist in cladelists]
                    node.weight_counters[sequence] = counter_prod(cladecounters, sum)
        return self.dagroot.weight_counters

    ######## Abstract dp method and derivatives: ########

    def postorder_cladetree_accum(
        self,
        leaf_func: Callable[["HistoryDagNode"], Weight],
        edge_func: Callable[["HistoryDagNode", "HistoryDagNode"], Weight],
        accum_within_clade: Callable[[List[Counter]], Counter],
        accum_between_clade: Callable[[List[Counter]], Counter],
    ):
        """TODO Docstring"""
        for node in self.postorder():
            if node.is_leaf():
                node._dp_data = leaf_func(node)
            else:
                node._dp_data = accum_between_clade(
                    [
                        accum_within_clade(
                            [
                                accum_between_clade(
                                    [target._dp_data, edge_func(node, target)],
                                )
                                for target in node.children(clade=clade)
                            ]
                        )
                        for clade in node.clades
                    ]
                )
        return self.dagroot._dp_data

    def optimal_weight_annotate(
        self,
        start_func: Callable[["HistoryDagNode"], Weight] = lambda n: 0,
        edge_weight_func: Callable[
            ["HistoryDagNode", "HistoryDagNode"], Weight
        ] = lambda n1, n2: utils.wrapped_hamming_distance(n1.label, n2.label),
        accum_func: Callable[[List[Weight]], Weight] = sum,
        optimal_func: Callable[[List[Weight]], Weight] = min,
    ):
        r"""
        Dynamically annotates each node in the DAG with the optimal weight of a clade sub-tree beneath it, so that the DAG root node is annotated with the optimal weight of a clade tree in the DAG.

        Args:
            start_func: A function which assigns starting weights to leaves.
            edge_weight_func: A function which assigns weights to DAG edges based on the parent node and the child node, in that order.
            accum_func: A function which takes a list of weights and returns a weight, like sum.
            optimal_func: A function which takes a list of weights and returns the optimal one, like min.

        At each node, if :math:`U` is the set of clades under the node, and for each clade :math:`C\in U`, :math:`V_C` is the set of target nodes beneath that clade, and if for each node :math:`v\in V_C`, :math:`OW(v)` is the optimal weight of a clade sub-tree below :math:`v`, then:

        .. math::

            OW(node) = node_weight_func(node) + \sum\limits_{C\in U} \text{optimal\_func}\left (\left \{ OW(v) + \text{edge\_weight\_func}(node.label, v.label) \mid v\in V_C\right \} \right )

        where :math:`\sum` and :math:`+` are used for clarity, while accum_func is meant.
        """
        return self.postorder_cladetree_accum(
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
        ] = lambda n1, n2: utils.wrapped_hamming_distance(n1.label, n2.label),
        accum_func: Callable[[List[Weight]], Weight] = sum,
    ):
        return self.postorder_cladetree_accum(
            lambda n: Counter([start_func(n)]),
            lambda n1, n2: Counter([edge_weight_func(n1, n2)]),
            counter_sum,
            lambda x: counter_prod(x, accum_func),
        )

    def hamming_parsimony_count(self):
        return self.weight_count(**utils.hamming_distance_countfuncs)

    def to_newicks(
        self, **kwargs
    ):
        """Returns a list of extended newick strings formed with label fields."""

        newicks = self.weight_count(**utils.make_newickcountfuncs(**kwargs)).elements()
        return [newick[1:-1] + ";" for newick in newicks]

    def count_trees(
        self,
        expand_func=lambda l: [l]
    ):
        r"""Annotates each node in the DAG with the number of complete trees underneath (extending to leaves,
        and containing exactly one edge for each node-clade pair). Returns the total number of unique
        complete trees below the root node.
        TODO: Replace with abstract dp method call"""
        return self.postorder_cladetree_accum(
            lambda n: 1,
            lambda parent, child: len(list(expand_func(child.label))),
            sum,
            prod
        )

    def trim_optimal_weight(
        self,
        edge_weight_func: Callable[[HistoryDagNode, HistoryDagNode], Weight] = lambda n1, n2: utils.wrapped_hamming_distance(n1.label, n2.label),
        accum_func: Callable[[List[Weight]], Weight] = sum,
        optimal_func: Callable[[List[Weight]], Weight] = min,
        **kwargs,
    ):
        """Trims the DAG to only express trees with optimal weight.
        This is guaranteed to be possible when edge_weight_func depends only on the labels of
        an edge's parent and child node.

        Requires that weights are of a type that supports reliable equality testing. In particular,
        floats are not recommended. Instead, consider defining weights to be a precursor type, and
        define `optimal_func` to choose the one whose corresponding float is maximized/minimized.

        If floats must be used, a Numpy type may help.

        For argument details, see :meth:`HistoryDag.optimal_weight_annotate`."""
        self.optimal_weight_annotate(
            edge_weight_func=edge_weight_func,
            accum_func=accum_func,
            optimal_func=optimal_func,
            **kwargs,
        )
        # It may not be okay to use preorder here. May need reverse postorder
        # instead?
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
                newtargets = []
                newweights = []
                for weight, target, index in weightlist:
                    if weight == optimalweight:
                        newtargets.append(target)
                        newweights.append(eset.weights[index])
                eset.targets = newtargets
                eset.weights = newweights
                n = len(eset.targets)
                eset.probs = [1.0 / n] * n

    def serialize(self):
        r"""Serializes a HistoryDag object as a bytestring.
        Since a HistoryDag is a recursive data structure, and contains types defined in
        function scope, modifications must be made for pickling.

        Returns:
            A bytestring expressing a dictionary containing:
                label_fields: The names of label fields.
                label_list: labels used in nodes, without duplicates. Indices are mapped to nodes in node_list
                node_list: node tuples containing
                (node label index in label_list, frozenset of frozensets of leaf label indices, node.attr).
                edge_list: a tuple for each edge:
                (origin node index, target node index, edge weight, edge probability)"""
        label_fields = list(self.dagroot.children())[0].label._fields
        label_list = []
        node_list = []
        attr_list = []
        edge_list = []
        label_indices = {}
        node_indices = {id(node): idx for idx, node in enumerate(self.postorder())}

        def cladesets(node):
            clades = {
                frozenset({label_indices[label] for label in clade})
                for clade in node.clades
            }
            return frozenset(clades)

        for node in self.postorder():
            if node.label not in label_indices:
                label_indices[node.label] = len(label_list)
                label_list.append(utils.ignore_ualabel(None)(tuple)(node.label))
                assert label_list[label_indices[node.label]] == node.label or isinstance(node.label, UALabel)
            node_list.append((label_indices[node.label], cladesets(node), node.attr))
            node_idx = len(node_list) - 1
            for eset in node.clades.values():
                for idx, target in enumerate(eset.targets):
                    edge_list.append(
                        (
                            node_idx,
                            node_indices[id(target)],
                            eset.weights[idx],
                            eset.probs[idx],
                        )
                    )
        serial_dict = {
            "label_fields": label_fields,
            "label_list": label_list,
            "node_list": node_list,
            "edge_list": edge_list,
        }
        return pickle.dumps(serial_dict)

    def recompute_parents(self):
        for node in self.postorder():
            node.parents = set()
        for node in self.postorder():
            for child in node.children():
                child.parents.add(node)

    def convert_to_collapsed(self):
        r"""Rebuilds the DAG so that no edge connects two nodes with the same label,
        unless one of them is a leaf node.
        This should be the same as building a new DAG by collapsing edges in all
        trees expressed by the old DAG."""

        self.recompute_parents()
        nodes = list(self.postorder())
        nodedict = {hash(node): node for node in nodes}
        edgequeue = [
            [parent, target] for parent in nodes for target in parent.children()
        ]

        while edgequeue:
            parent, child = edgequeue.pop()
            clade = child.under_clade()
            if (
                parent.label == child.label
                and hash(parent) in nodedict
                and hash(child) in nodedict
                and not child.is_leaf()
            ):
                parent_clade_edges = len(parent.clades[clade].targets)
                new_parent_clades = (
                    frozenset(parent.clades.keys()) - {clade}
                ) | frozenset(child.clades.keys())
                if hash((parent.label, new_parent_clades)) in nodedict:
                    newparent = nodedict[hash((parent.label, new_parent_clades))]
                else:
                    newparent = empty_node(parent.label, new_parent_clades)
                    nodedict[hash(newparent)] = newparent
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
                parent.clades[clade].remove_from_edgeset_byid(child)
                # Clean up the DAG:
                # Delete old parent if it is no longer a valid node
                if parent_clade_edges == 1:
                    # Remove old parent as child of all of its parents
                    # no need for recursion here, all of its parents had
                    # edges added to new parent from the same clade.
                    upclade = parent.under_clade()
                    for grandparent in parent.parents:
                        grandparent.clades[upclade].remove_from_edgeset_byid(parent)
                    for child2 in parent.children():
                        child2.parents.remove(parent)
                        if not child2.parents:
                            child2.remove_node(nodedict=nodedict)
                    nodedict.pop(hash(parent))

                # Remove child, if child no longer has parents
                if parent in child.parents:
                    child.parents.remove(parent)
                if not child.parents:
                    # This recursively removes children of child too, if necessary
                    child.remove_node(nodedict=nodedict)
        self.recompute_parents()

    # ######## DAG Traversal Methods ########

    def postorder(self):
        visited = set()

        def traverse(node: HistoryDagNode):
            visited.add(id(node))
            if not node.is_leaf():
                for child in node.children():
                    if not id(child) in visited:
                        yield from traverse(child)
            yield node

        yield from traverse(self.dagroot)


    def preorder(self):
        """Careful! remember this is not guaranteed to visit a parent node before any of its children.
        for that, need reverse postorder traversal."""
        visited = set()

        def traverse(node: HistoryDagNode):
            visited.add(id(node))
            yield node
            if not node.is_leaf():
                for child in node.children():
                    if not id(child) in visited:
                        yield from traverse(child)

        yield from traverse(self.dagroot)



class EdgeSet:
    r"""Goal: associate targets (edges) with arbitrary parameters, but support
    set-like operations like lookup and enforce that elements are unique."""

    def __init__(self, *args, weights: list = None, probs: list = None):
        r"""Takes no arguments, or an ordered iterable containing target nodes"""
        if len(args) > 1:
            raise TypeError(f"Expected at most one argument, got {len(args)}")
        elif args:
            self.targets = list(args[0])
            n = len(self.targets)
            if weights is not None:
                self.weights = weights
            else:
                self.weights = [0] * n

            if probs is not None:
                self.probs = probs
            else:
                self.probs = [float(1) / n] * n
        else:
            self.targets = []
            self.weights = []
            self.probs = []
            self._hashes = set()

        self._hashes = {hash(self.targets[i]) for i in range(len(self.targets))}
        if not len(self._hashes) == len(self.targets):
            raise TypeError("First argument may not contain duplicate target nodes")
        # Should probably also check to see that all passed lists have same length

    def __iter__(self):
        return (
            (self.targets[i], self.weights[i], self.probs[i])
            for i in range(len(self.targets))
        )

    # def copy(self):
    #     return EdgeSet(
    #         [node.copy() for node in self.targets],
    #         weights=self.weights.copy(),
    #         probs=self.probs.copy(),
    #     )

    def shallowcopy(self):
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
            self._hashes = {hash(node) for node in self.targets}

    def sample(self):
        """Returns a randomly sampled child edge, and the corresponding entry from the
        weight vector."""
        index = random.choices(list(range(len(self.targets))), weights=self.probs, k=1)[
            0
        ]
        return (self.targets[index], self.weights[index])

    def add_to_edgeset(
        self, target, weight=0, prob=None, prob_norm=True
    ):
        """currently does nothing if edge is already present. Also does nothing
        if the target node has one child clade, and parent node is not the DAG root.
        Returns whether an edge was added"""
        if target.is_root():
            raise ValueError("Edges that target UA nodes are not allowed. "
                             f"Target node has label {target.label} and therefore "
                             "is assumed to be the DAG UA root node.")
        elif hash(target) in self._hashes:
            return False
        else:
            self._hashes.add(hash(target))
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


######## DAG Creation Functions ########


def empty_node(label, clades):
    return HistoryDagNode(label, {clade: EdgeSet() for clade in clades})


def from_tree(
    tree: ete3.TreeNode,
    label_features: List[str],
    label_functions: Mapping[str, Callable[[ete3.TreeNode], Any]] = {},
    attr_func: Callable[[ete3.TreeNode], Any] = lambda n: None,
):
    """
    Build a full tree from an ete3 TreeNode.

    Args:
        tree: ete3 tree to be converted to HistoryDag clade tree
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
        TODO remove root unifurcation if present.
    """
    feature_maps = {name: (lambda n: getattr(n, name)) for name in label_features}
    feature_maps.update(label_functions)
    Label = namedtuple(
        "Label", list(feature_maps.keys()), defaults=[None] * len(feature_maps)
    )

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
        )
        return dag

    # Check for unique leaf labels:
    if len(list(tree.get_leaves())) != len(leaf_names(tree)):
        raise ValueError(
            "This tree's leaves are not labeled uniquely. Check your tree, "
            "or modify the label fields so that leaves are unique.\n"
            + str(leaf_names(tree))
        )

    # Checking for unifurcation is handled in HistoryDagNode.__init__.

    dag = _unrooted_from_tree(tree)
    dagroot = HistoryDagNode(
        UALabel(),
        {
            frozenset({taxon for s in dag.clades for taxon in s}): EdgeSet(
                [dag], weights=[tree.dist]
            )
        },
    )
    dagroot.add_edge(dag, weight=0)
    return HistoryDag(dagroot)


def from_newick(
    tree: str,
    label_features: List[str],
    label_functions: Mapping[str, Callable[[ete3.TreeNode], Any]] = {},
    newick_format=8,
    attr_func: Callable[[ete3.TreeNode], Any] = lambda n: None,
):
    """
    Make a history DAG using a newick string.
    Internally, utilizes newick parsing features provided by ete3, then calls :meth:`from_tree`
    on the resulting ete3.Tree object.

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
    newicklist,
    label_features: List[str],
    label_functions: Mapping[str, Callable[[ete3.TreeNode], Any]] = {},
    attr_func: Callable[[ete3.TreeNode], Any] = lambda n: None,
    newick_format=1,
):
    return history_dag_from_clade_trees(
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
    treelist,
    label_features: List[str],
    label_functions: Mapping[str, Callable[[ete3.TreeNode], Any]] = {},
    attr_func: Callable[[ete3.TreeNode], Any] = lambda n: None,
):
    return history_dag_from_clade_trees(
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


def history_dag_from_clade_trees(treelist):
    # merge checks that all clade trees have the same leaf label set.
    # Is copying the first enough to avoid mutating treelist?
    dag = treelist[0].copy()
    for tree in treelist[1:]:
        dag.merge(tree)
    return dag


######## Miscellaneous Functions ########


def deserialize(bstring):
    """reloads an HistoryDagNode serialized object, as ouput by HistoryDagNode.serialize"""
    serial_dict = pickle.loads(bstring)
    label_list = serial_dict["label_list"]
    node_list = serial_dict["node_list"]
    edge_list = serial_dict["edge_list"]
    label_fields = serial_dict["label_fields"]
    Label = namedtuple("Label", label_fields, defaults=[None] * len(label_fields))

    def unpack_labels(labelset):
        res = frozenset({Label(*label_list[idx]) for idx in labelset})
        return res

    node_postorder = [
        HistoryDagNode(
            (utils.UALabel() if label_list[labelidx] is None
             else Label(*label_list[labelidx])),
            {unpack_labels(clade): EdgeSet() for clade in clades},
            attr=attr,
        )
        for labelidx, clades, attr in node_list
    ]
    # Last node in list is root
    for origin_idx, target_idx, weight, prob in edge_list:
        node_postorder[origin_idx].add_edge(
            node_postorder[target_idx], weight=weight, prob=prob, prob_norm=False
        )
    return HistoryDag(node_postorder[-1])
