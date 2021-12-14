import pickle
import operator
from functools import reduce
import math
import graphviz as gv
from historydag import utils
import ete3
import random
from typing import List, Callable, Any
from collections import Counter
from gctree import CollapsedTree
from multiset import FrozenMultiset
from historydag.counterops import *


class HistoryDag:
    r"""A recursive representation of a history DAG object
    - a dictionary keyed by clades (frozensets) containing EdgeSet objects
    - a label
    """

    def __init__(self, label, clades: dict = {}):
        self.clades = clades
        # If passed a nonempty dictionary, need to add self to children's parents
        self.label = label
        self.parents = set()
        if self.clades:
            for _, edgeset in self.clades.items():
                edgeset.parent = self
            for child in self.children():
                child.parents.add(self)

    def __repr__(self):
        return str((self.label, set(self.clades.keys())))

    def __hash__(self):
        return hash((self.label, frozenset(self.clades.keys())))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def node_self(self):
        return HistoryDag(self.label, {clade: EdgeSet() for clade in self.clades})

    def under_clade(self):
        r"""Returns the union of all child clades"""
        if self.is_leaf():
            return frozenset([self.label])
        else:
            return frozenset().union(*self.clades.keys())

    def copy(self, use_recursion=False):
        r"""Add each child node's copy, below this node, by merging.
        The non-recursive version uses bytestring serialization.
        non-recursive is an order of magnitude faster in one test"""
        if use_recursion:
            newnode = self.node_self()
            from_root = newnode.label == "DAG_root"
            for clade in self.clades:
                for index, target in enumerate(self.clades[clade].targets):
                    othernode = self.node_self()
                    # In this line lies the recursive call
                    othernode.clades[clade].add_to_edgeset(
                        target.copy(), from_root=from_root
                    )
                    newnode.merge(othernode)
                    newnode.clades[clade].weights[index] = self.clades[clade].weights[
                        index
                    ]
                    newnode.clades[clade].probs[index] = self.clades[clade].probs[index]
            return newnode
        else:
            return deserialize(self.serialize())

    def merge(self, node):
        r"""performs post order traversal to add node and all of its children,
        without introducing duplicate nodes in self. Requires given node is a root"""
        if not hash(self) == hash(node):
            raise ValueError(
                f"The given node must be a root node on identical taxa.\n{self}\nvs\n{node}"
            )
        selforder = postorder(self)
        nodeorder = postorder(node)
        hashdict = {hash(n): n for n in selforder}
        for n in nodeorder:
            if hash(n) in hashdict:
                pnode = hashdict[hash(n)]
            else:
                pnode = n.node_self()
                hashdict[hash(n)] = pnode

            for _, edgeset in n.clades.items():
                for child, weight, _ in edgeset:
                    pnode.add_edge(hashdict[hash(child)], weight=weight)

    def add_edge(self, target, weight=0, prob=None, prob_norm=True):
        r"""Adds edge, if not already present and allowed. Returns if edge was added."""
        # target clades must union to a clade of self
        key = target.under_clade()
        if key not in self.clades:
            raise KeyError("Target clades' union is not a clade of this parent node")
        else:
            from_root = self.label == "DAG_root"
            target.parents.add(self)
            return self.clades[key].add_to_edgeset(
                target,
                weight=weight,
                prob=prob,
                prob_norm=prob_norm,
                from_root=from_root,
            )

    def is_leaf(self):
        return not bool(self.clades)

    def partitions(self):
        return self.clades.keys()

    def children(self, clade=None):
        r"""If clade is provided, returns generator object of edge targets from that
        clade. If no clade is provided, generator includes all children of self.
        """
        if clade is None:
            return (
                target for clade in self.clades for target, _, _ in self.clades[clade]
            )
        else:
            return (child for child, _, _ in self.clades[clade])

    def add_all_allowed_edges(
        self, new_from_root=True, adjacent_labels=True, preserve_parent_labels=False
    ):
        r"""Add all allowed edges to the DAG, returning the number that were added.
        if new_from_root is False, no edges are added that start at DAG root.
        This is useful to enforce a single ancestral sequence.
        If adjacent_labels is False, no edges will be added between nodes with the same labels.
        preserve_parent_labels was to show something was true....?"""
        n_added = 0
        clade_dict = {node.under_clade(): [] for node in postorder(self)}
        if preserve_parent_labels is True:
            self.recompute_parents()
            uplabels = {
                node: {parent.label for parent in node.parents}
                for node in postorder(self)
            }
        for node in postorder(self):
            clade_dict[node.under_clade()].append(node)
        for node in postorder(self):
            if new_from_root is False and node.label == "DAG_root":
                continue
            else:
                for clade in node.clades:
                    for target in clade_dict[clade]:
                        if adjacent_labels is False and target.label == node.label:
                            continue
                        if (
                            preserve_parent_labels is True
                            and node.label not in uplabels[target]
                        ):
                            continue
                        n_added += node.add_edge(target)
        return n_added

    def to_newick(self, namedict={}, fixed_order=True, include_root=False):
        r"""Converts to extended newick format with arbitrary node names and a
        sequence feature. For use on tree-shaped DAG.
        If fixed_order is True, with same namedict, two trees' newick representations are
        equal iff the two trees have the same topology and node sequences.
        Should be usable for equality in this sense on general DAG too, with include_root
        True"""

        def newick(node):
            if node.label in namedict:
                name = namedict[node.label]
            else:
                name = "unnamed_seq"
            if node.is_leaf():
                return f"{name}[&&NHX:sequence={node.label}]"
            else:
                if fixed_order:
                    # Sort child nodes by their hash, which is a function of
                    # label and child clades, and up to hash collisions is
                    # unique on nodes in the DAG.
                    children = sorted(node.children(), key=hash)
                else:
                    children = node.children()
                return (
                    "("
                    + ",".join([newick(node2) for node2 in children])
                    + ")"
                    + f"{name}[&&NHX:sequence={node.label}]"
                )

        if self.label == "DAG_root" and include_root is False:
            return newick(next(self.children())) + ";"
        else:
            return newick(self) + ";"

    def to_ete(self, namedict={}):
        return ete3.TreeNode(newick=self.to_newick(namedict=namedict), format=1)

    def to_graphviz(self, labelfunc=None, namedict={}, show_partitions=True):
        r"""Converts to graphviz Digraph object. Namedict must associate sequences
        of all leaf nodes to a name
        """

        def taxa(clade):
            l = [labeller(taxon) for taxon in clade]
            l.sort()
            return ",".join(l)

        def labeller(sequence):
            if sequence in namedict:
                return str(namedict[sequence])
            elif len(sequence) < 11:
                return sequence
            else:
                return hash(sequence)

        if labelfunc is None:

            def labelfunc(node):
                return labeller(node.label)

        G = gv.Digraph("labeled partition DAG", node_attr={"shape": "record"})
        for node in postorder(self):
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
        nonleaf_parents = (len(n.parents) for n in postorder(self) if not n.is_leaf())
        n = 0
        cumsum = 0
        for sum in nonleaf_parents:
            n += 1
            cumsum += sum
        # Exclude root:
        return cumsum / float(n - 1)

    def min_weight_annotate(self, distance_func=utils.hamming_distance):
        self.optimal_weight_annotate(edge_weight_func=(lambda x, y: distance_func(x.label, y.label)))

    def sample(self, min_weight=False, distance_func=utils.hamming_distance):
        r"""Samples a sub-history-DAG that is also a tree containing the root and
        all leaf nodes. Returns a new HistoryDag object."""
        from_root = self.label == "DAG_root"
        if from_root:
            self.min_weight_annotate(distance_func=distance_func)
        sample = self.node_self()
        for clade, eset in self.clades.items():
            sampled_target, target_weight = eset.sample(
                min_weight=min_weight, distance_func=distance_func
            )
            sample.clades[clade].add_to_edgeset(
                sampled_target.sample(min_weight=min_weight),
                weight=target_weight,
                from_root=from_root,
            )
        return sample

    def count_trees(
        self,
        min_weight=False,
        count_resolutions=False,
        distance_func=utils.hamming_distance,
    ):
        r"""Annotates each node in the DAG with the number of complete trees underneath (extending to leaves,
        and containing exactly one edge for each node-clade pair). Returns the total number of unique
        complete trees below the root node."""
        if min_weight:
            if not count_resolutions:
                c = self.get_weight_counts(distance_func=distance_func)
                return c[min(c)]
            else:
                c = self.get_weight_counts_with_ambiguities(distance_func=distance_func)
                min_weights = [min(valdict) for valdict in c.values()]
                min_weight = min(min_weights)
                return sum([valdict[min_weight] for valdict in c.values()])
        else:
            # Replace prod function later:
            prod = lambda l: l[0] * prod(l[1:]) if l else 1
            # logic below requires prod([]) == 1!!
            for node in postorder(self):
                node.trees_under = prod(
                    [
                        sum(
                            [
                                target.trees_under
                                for target in node.clades[clade].targets
                            ]
                        )
                        for clade in node.clades
                    ]
                )
                if count_resolutions:
                    node.trees_under *= len(
                        list(utils.sequence_resolutions(node.label))
                    )
            return self.trees_under

    def get_trees(self, min_weight=False, distance_func=utils.hamming_distance):
        r"""Return a generator to iterate through all trees expressed by the DAG."""

        if min_weight:
            dag = self.copy().trim_min_weight(distance_func=distance_func)
        else:
            dag = self

        def genexp_func(clade):
            # Return generator expression of all possible choices of tree
            # structure from dag below clade
            def f():
                eset = dag.clades[clade]
                return (
                    (clade, targettree, i)
                    for i, target in enumerate(eset.targets)
                    for targettree in target.get_trees(
                        min_weight=False, distance_func=distance_func
                    )
                )

            return f

        optionlist = [genexp_func(clade) for clade in self.clades]

        for option in utils.cartesian_product(optionlist):
            tree = dag.node_self()
            from_root = tree.label == "DAG_root"
            for clade, targettree, index in option:
                tree.clades[clade].add_to_edgeset(
                    targettree,
                    weight=dag.clades[clade].weights[index],
                    from_root=from_root,
                )
            yield tree

    def disambiguate_sitewise(self):
        r"""
        Finds all sites at which DAG nodes have ambiguous bases, then expands ambiguities at each of those sites individually, pruning the expanded nodes that do not have optimal below-node tree weight. This method may be guaranteed to find all min-weight disambiguations, but if some of the original trees have higher minimum disambiguation weight, then some or all of their disambiguations may be missing from the resulting DAG structure.
        """
        # Does not work yet!
        ambigset = set()
        for node in postorder(self):
            if node.label != "DAG_root":
                ambigset.update(
                    {
                        site
                        for site, base in enumerate(node.label)
                        if base not in utils.bases
                    }
                )
        print(ambigset)
        for site in ambigset:
            dist_func = utils.compare_site_func(site)
            new_nodes = self.expand_ambiguities(focus_site=site)
            print(sum(1 for _ in postorder(self)))
            self.min_weight_annotate(distance_func=dist_func)
            rporder = reversed(list(postorder(self)))
            for node in rporder:
                for clade, eset in node.clades.items():
                    min_weights = [
                        target.min_weight_under + dist_func(node.label, target.label)
                        for target in eset.targets
                    ]
                    min_weight = min(min_weights)
                    to_delete = [
                        index
                        for index, weight in enumerate(min_weights)
                        if weight != min_weight
                    ]
                    # Guaranteed to preserve at least one eset target, but
                    # could result in orphaned targets
                    for index in reversed(to_delete):
                        # only allow removal of newly created nodes
                        if eset.targets[index] in new_nodes:
                            oldtarget = eset.targets.pop(index)
                            if node in oldtarget.parents:
                                oldtarget.parents.remove(node)
                            # Remove orphaned targets
                            if not oldtarget.parents:
                                oldtarget.remove_node()
                            eset.weights.pop(index)
                            eset.probs.pop(index)
            # Remove orphaned nodes?
            self.recompute_parents()

    def two_pass_sankoff(self, distance_func=utils.hamming_distance):
        r"""Disambiguate using a two-pass Sankoff algorithm. The first pass is provided by the
        get_weight_counts_with_ambiguities method. The second (downward) pass involves choosing a minimum weight
        resolution of each node, then updating sequence weights of all child nodes."""

        def addweight(c, w):
            return Counter({key + w: val for key, val in c.items()})

        self.get_weight_counts_with_ambiguities(distance_func=distance_func)
        rporder = list(postorder(self))
        rporder.reverse()
        for node in rporder:
            min_weight = float("inf")
            best_sequence = None
            for sequence in node.weight_counters:
                this_min = min(node.weight_counters[sequence])
                if this_min < min_weight:
                    min_weight = this_min
                    best_sequence = sequence
            node.label = best_sequence
            for child in node.children():
                for sequence in child.weight_counters:
                    before = child.weight_counters[sequence]
                    after = addweight(before, distance_func(sequence, node.label))
                    child.weight_counters[sequence] = after

    def disambiguate_dag(self, distance_func=utils.hamming_distance):
        r"""like get_weight_counts_with_ambiguities, but creates dictionaries of Counter objects at each node,
        keyed by possible sequences at that node.
        chooses a sequence that minimizes the below-node tree weight.
        This is a one-pass Sankoff algorithm??"""

        for node in postorder(self):
            node.weight_counters = {}
            min_weight = float("inf")
            best_sequence = None
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
                            for target in node.clades[clade].targets
                            for target_seq, target_wc in target.weight_counters.items()
                        ]
                        for clade in node.clades
                    ]
                    cladecounters = [counter_sum(cladelist) for cladelist in cladelists]
                    node.weight_counters[sequence] = counter_prod(cladecounters, sum)
                print(node.label)
                print(node.weight_counters)
                this_sequence_min_weight = min(node.weight_counters[sequence])
                if this_sequence_min_weight < min_weight:
                    min_weight = this_sequence_min_weight
                    best_sequence = sequence
            n_min_weight = node.weight_counters[sequence][min_weight]
            node.weight_counters = {sequence: Counter({min_weight: n_min_weight})}
            node.label = best_sequence

        return self.weight_counters

    def expand_ambiguities(self, focus_site=None):
        r"""Each node with an ambiguous sequence is replaced by nodes with all possible disambiguations of the original sequence, and the same clades and parent and child edges."""

        def sequence_resolutions(sequence):
            if focus_site is not None:
                for base in utils.sequence_resolutions(sequence[focus_site]):
                    yield sequence[:focus_site] + base + sequence[focus_site + 1 :]
            else:
                yield from utils.sequence_resolutions(sequence)

        def is_ambiguous(sequence):
            if focus_site is not None:
                return utils.is_ambiguous(sequence[focus_site])
            else:
                return utils.is_ambiguous(sequence)

        self.recompute_parents()
        nodedict = {hash(node): node for node in postorder(self)}
        nodeorder = list(postorder(self))
        new_nodes = set()
        for node in nodeorder:
            if node.label != "DAG_root" and is_ambiguous(node.label):
                clades = frozenset(node.clades.keys())
                for resolution in sequence_resolutions(node.label):
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
        print(f"Nodes:\t{sum(1 for _ in postorder(self))}")
        print(f"Trees:\t{self.count_trees()}")
        utils.hist(self.get_weight_counts_with_ambiguities()["DAG_root"])

    def get_weight_counts_with_ambiguities(self, distance_func=utils.hamming_distance):
        r"""like get_weight_counts, but creates dictionaries of Counter objects at each node,
        keyed by possible sequences at that node.
        The total number of trees will be greater than count_trees(), as these are possible
        disambiguations of trees. These disambiguations may not be unique (?), but if two are
        the same, they come from different subtrees of the DAG."""

        for node in postorder(self):
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
                            for target in node.clades[clade].targets
                            for target_seq, target_wc in target.weight_counters.items()
                        ]
                        for clade in node.clades
                    ]
                    cladecounters = [counter_sum(cladelist) for cladelist in cladelists]
                    node.weight_counters[sequence] = counter_prod(cladecounters, sum)
        return self.weight_counters

    def get_weight_counts(self, distance_func=utils.hamming_distance):
        r"""Annotate each node in the DAG, in postorder traversal, with a Counter object
        keyed by weight, with values the number of possible unique trees below the node
        with that weight."""
        for node in postorder(self):
            if node.is_leaf():
                node.weight_counter = Counter({0: 1})
            else:
                cladelists = [
                    [
                        addweight(
                            target.weight_counter,
                            distance_func(target.label, node.label),
                        )
                        for target in node.clades[clade].targets
                    ]
                    for clade in node.clades
                ]
                cladecounters = [counter_sum(cladelist) for cladelist in cladelists]
                node.weight_counter = counter_prod(cladecounters, sum)
        return self.weight_counter

    def optimal_weight_annotate(
        self,
        edge_weight_func: Callable[[Any, Any], Any] = (
            lambda x, y: utils.hamming_distance(x.label, y.label)
        ),
        node_weight_func: Callable[[Any], Any] = (lambda node: 0),
        accum_func: Callable[[Any, Any], Any] = operator.add,
        accum_identity: Any = 0,
        optimal_func: Callable[[List[Any]], Any] = min,
    ):
        r"""
        Dynamically annotates each node in the DAG with the optimal weight of a clade sub-tree beneath it, so that the DAG root node is annotated with the optimal weight of a clade tree in the DAG.

        Args:
            edge_weight_func: A function which assigns weights to DAG edges based on the label of the parent node and the label of the child node, in that order.
            node_weight_func: A function which assigns weights to DAG nodes.
            accum_func: A function which takes two weights and returns a weight, like sum.
            accum_identity: A left identity value for accum_func, so that accum_func(accum_identity, x) = x. "If we want to accum_func a list of 0 items, what should the result be? accum_identity."
            optimal_func: A function which takes a list of weights and returns the optimal one, like min.

        At each node, if :math:`U` is the set of clades under the node, and for each clade :math:`C\in U`, :math:`V_C` is the set of target nodes beneath that clade, and if for each node :math:`v\in V_C`, :math:`OW(v)` is the optimal weight of a clade sub-tree below :math:`v`, then:

        .. math::

            OW(node) = node_weight_func(node) + \sum\limits_{C\in U} \text{optimal\_func}\left (\left \{ OW(v) + \text{edge\_weight\_func}(node.label, v.label) \mid v\in V_C\right \} \right )

        where :math:`\sum` and :math:`+` are used for clarity, while accum_func is meant.
        """
        for node in postorder(self):
            if node.is_leaf():
                node.min_weight_under = node_weight_func(node)
            else:
                min_sum = reduce(
                    accum_func,
                    [
                        optimal_func(
                            [
                                accum_func(
                                    child.min_weight_under,
                                    edge_weight_func(node, child),
                                )
                                for child in node.clades[clade].targets
                            ]
                        )
                        for clade in node.clades
                    ],
                    accum_identity,
                )
                node.min_weight_under = accum_func(min_sum, node_weight_func(node))
        return self.min_weight_under


    def trim_optimal_weight(
        self,
        edge_weight_func: Callable[[Any, Any], Any] = (
            lambda x, y: utils.hamming_distance(x.label, y.label)
        ),
        accum_func: Callable[[Any, Any], Any] = operator.add,
        optimal_func: Callable[[List[Any]], Any] = min,
        rel_tol=1e-09,
        abs_tol=0.0,
        **kwargs,
    ):
        self.optimal_weight_annotate(
            edge_weight_func=edge_weight_func,
            accum_func=accum_func,
            optimal_func=optimal_func,
            **kwargs,
        )
        # It may not be okay to use preorder here. May need reverse postorder
        # instead?
        for node in preorder(self):
            for clade, eset in node.clades.items():
                weightlist = [
                    (
                        accum_func(target.min_weight_under, edge_weight_func(node, target)),
                        target,
                        index,
                    )
                    for index, target in enumerate(eset.targets)
                ]
                optimalweight = optimal_func([weight for weight, _, _ in weightlist])
                newtargets = []
                newweights = []
                for weight, target, index in weightlist:
                    if math.isclose(
                        weight, optimalweight, rel_tol=rel_tol, abs_tol=abs_tol
                    ):
                        newtargets.append(target)
                        newweights.append(eset.weights[index])
                eset.targets = newtargets
                eset.weights = newweights
                n = len(eset.targets)
                eset.probs = [1.0 / n] * n

    def trim_min_weight(self, distance_func=utils.hamming_distance, focus_site=None):
        if focus_site is None:
            def getlabel(node):
                return node.label
        else:
            def getlabel(node):
                if node.label == 'DAG_root':
                    return 'DAG_root'
                else:
                    return node.label[focus_site]
        self.trim_optimal_weight(edge_weight_func=(lambda x, y: distance_func(x.label, y.label)))

    def serialize(self):
        r"""Represents HistoryDag object as a list of sequences, a list of node tuples containing
        (node sequence index, frozenset of frozenset of leaf sequence indices)
        and an edge list containing a tuple for each edge:
        (origin node index, target node index, edge weight, edge probability)"""
        sequence_list = []
        node_list = []
        edge_list = []
        sequence_indices = {}
        node_indices = {id(node): idx for idx, node in enumerate(postorder(self))}

        def cladesets(node):
            clades = {
                frozenset({sequence_indices[sequence] for sequence in clade})
                for clade in node.clades
            }
            return frozenset(clades)

        for node in postorder(self):
            if node.label not in sequence_indices:
                sequence_indices[node.label] = len(sequence_list)
                sequence_list.append(node.label)
                assert sequence_list[sequence_indices[node.label]] == node.label
            node_list.append((sequence_indices[node.label], cladesets(node)))
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
            "sequence_list": sequence_list,
            "node_list": node_list,
            "edge_list": edge_list,
        }
        return pickle.dumps(serial_dict)

    def weight(self):
        "Sums weights of all edges in the DAG"
        nodes = postorder(self)
        edgesetsums = (
            sum(edgeset.weights) for node in nodes for edgeset in node.clades.values()
        )
        return sum(edgesetsums)

    def recompute_parents(self):
        for node in postorder(self):
            node.parents = set()
        for node in postorder(self):
            for child in node.children():
                child.parents.add(node)

    def remove_node(self, nodedict={}):
        r"""Recursively removes node self and any orphaned children from dag.
        May not work on root?
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

    def convert_to_collapsed(self):
        r"""Rebuilds the DAG so that no edge connects two nodes with the same label,
        unless one of them is a leaf node.
        This should be the same as building a new DAG by collapsing edges in all
        trees expressed by the old DAG."""

        self.recompute_parents()
        nodes = list(postorder(self))
        nodedict = {hash(node): node for node in nodes}
        edgequeue = [
            [parent, target] for parent in nodes for target in parent.children()
        ]

        # Replace with remove_node method above (TODO)
        def remove_node(node):
            if hash(node) in nodedict:
                nodedict.pop(hash(node))
            for child in node.children():
                if node in child.parents:
                    child.parents.remove(node)
                if not child.parents:
                    remove_node(child)

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
                    # Why not use remove_node on parent?
                    upclade = parent.under_clade()
                    for grandparent in parent.parents:
                        grandparent.clades[upclade].remove_from_edgeset_byid(parent)
                    for child2 in parent.children():
                        child2.parents.remove(parent)
                        if not child2.parents:
                            remove_node(child2)
                    nodedict.pop(hash(parent))

                # Remove child, if child no longer has parents
                if parent in child.parents:
                    child.parents.remove(parent)
                if not child.parents:
                    # This recursively removes children of child too, if necessary
                    remove_node(child)
        self.recompute_parents()

    def add_abundances(self, abundances: dict):
        r"""Expects DAG to be collapsed so that only edges targeting leaf nodes may have same label on parent and child nodes. Abundances should be a dictionary containing integers, keyed by node labels (sequences?)"""
        for node in postorder(self):
            # Although it's guaranteed to happen in max parsimony setting, only want internal nodes that are adjacent to a leaf with the same label to carry a nonzero abundance
            if node.label in abundances:
                if node.is_leaf() or frozenset({node.label}) in node.clades:
                    node.abundance = int(abundances[node.label])
                else:
                    node.abundance = 0
            else:
                node.abundance = 0

        # This corresponds to https://github.com/matsengrp/gctree/blob/375f61e60ec87aabfb0e5a9d0575cf56f46f06a5/gctree/branching_processes.py#L319
        nondagroot = list(self.children())[0]
        if len(nondagroot.clades) == 1:
            nondagroot.abundance = 1

    def bp_loglikelihoods(self, p, q):
        r"""Expects DAG to be collapsed so that only edges targeting leaf nodes may have the same label on parent and child nodes. p and q are branching process likelihoods."""

        # Need to make this traversal ignore edges between nodes with the
        # same label
        for node in postorder(self):
            if node.is_leaf():
                node.below_loglikelihoods = Counter(
                    [CollapsedTree._ll_genotype(node.abundance, 0, p, q)[0]]
                )
            else:
                cladelists = [
                    [
                        target.below_loglikelihoods
                        for target in node.clades[clade].targets
                    ]
                    for clade in node.clades
                    if not (len(clade) == 1 and list(clade)[0] == node.label)
                ]
                # Because construction of cladelists ignores child clades whose
                # sole target is a leaf with the same sequence:
                m = len(cladelists)
                cladecounters = [counter_sum(cladelist) for cladelist in cladelists]
                if node.label == "DAG_root":
                    node.below_loglikelihoods = counter_prod(cladecounters, sum)
                    return node.below_loglikelihoods
                else:
                    node.below_loglikelihoods = addweight(
                        counter_prod(cladecounters, sum),
                        CollapsedTree._ll_genotype(node.abundance, m, p, q)[0],
                    )

    def cmcounters(self):
        r"""Expects DAG to be collapsed so that only edges targeting leaf nodes may have the same label on parent and child nodes. p and q are branching process likelihoods."""
        self.cmlist = self._construct_cmlists()
        return self.below_cm

    def _construct_cmlists(self):
        r"""Constructs a Counter object, containing one FrozenMultiset object for each
        (sub) tree below each node. The FrozenMultiset contains (c, m) tuples (c is abundance and
        m is mutant descendants) for the corresponding (sub) tree."""
        # Need to make this traversal ignore edges between nodes with the
        # same label
        def accumfunc(counterlist):
            return counter_sum(counterlist, counter_type=FrozenMultiset)

        for node in postorder(self):
            if node.is_leaf():
                node.below_cm = Counter([FrozenMultiset([(node.abundance, 0)])])
            else:
                cladelists = [
                    [target.below_cm for target in node.clades[clade].targets]
                    for clade in node.clades
                    if not (len(clade) == 1 and list(clade)[0] == node.label)
                ]
                # Because construction of cladelists ignores child clades whose
                # sole target is a leaf with the same sequence:
                m = len(cladelists)
                cladecounters = [counter_sum(cladelist) for cladelist in cladelists]
                if node.label == "DAG_root":
                    node.below_cm = counter_prod(cladecounters, accumfunc)
                    return node.below_cm
                else:
                    node.below_cm = addweight(
                        counter_prod(cladecounters, accumfunc), {(node.abundance, m)}
                    )


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

    def copy(self):
        return EdgeSet(
            [node.copy() for node in self.targets],
            weights=self.weights.copy(),
            probs=self.probs.copy(),
        )

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

    def sample(self, min_weight=False, distance_func=utils.hamming_distance):
        """Returns a randomly sampled child edge, and the corresponding entry from the
        weight vector. If min_weight is True, samples only target nodes with lowest
        min_weight_under attribute, ignoring edge probabilities."""
        if min_weight:
            mw = min(
                node.min_weight_under + distance_func(self.parent.label, node.label)
                for node in self.targets
            )
            options = [
                i
                for i, node in enumerate(self.targets)
                if (
                    node.min_weight_under + distance_func(self.parent.label, node.label)
                )
                == mw
            ]
            index = random.choices(options, k=1)[0]
            return (self.targets[index], self.weights[index])
        else:
            index = random.choices(
                list(range(len(self.targets))), weights=self.probs, k=1
            )[0]
            return (self.targets[index], self.weights[index])

    def add_to_edgeset(
        self, target, weight=0, prob=None, prob_norm=True, from_root=False
    ):
        """currently does nothing if edge is already present. Also does nothing
        if the target node has one child clade, and parent node is not the DAG root.
        Returns whether an edge was added"""
        if target.label == "DAG_root":
            return False
        elif hash(target) in self._hashes:
            return False
        elif not from_root and len(target.clades) == 1:
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


def from_tree(tree: ete3.TreeNode):
    def leaf_names(r: ete3.TreeNode):
        return frozenset((node.sequence for node in r.get_leaves()))

    def _unrooted_from_tree(tree):
        dag = HistoryDag(
            tree.sequence,
            {
                leaf_names(child): EdgeSet(
                    [_unrooted_from_tree(child)], weights=[child.dist]
                )
                for child in tree.get_children()
            },
        )
        return dag

    dag = _unrooted_from_tree(tree)
    dagroot = HistoryDag(
        "DAG_root",
        {
            frozenset({taxon for s in dag.clades for taxon in s}): EdgeSet(
                [dag], weights=[tree.dist]
            )
        },
    )
    dagroot.add_edge(dag, weight=0)
    return dagroot


def from_newick(tree: str):
    etetree = ete3.Tree(tree, format=8)
    return from_tree(etetree)


def postorder(dag: HistoryDag):
    visited = set()

    def traverse(node: HistoryDag):
        visited.add(id(node))
        if not node.is_leaf():
            for child in node.children():
                if not id(child) in visited:
                    yield from traverse(child)
        yield node

    yield from traverse(dag)


def preorder(dag: HistoryDag):
    """Careful! remember this is not guaranteed to visit a parent node before any of its children.
    for that, need reverse postorder traversal."""
    visited = set()

    def traverse(node: HistoryDag):
        visited.add(id(node))
        yield node
        if not node.is_leaf():
            for child in node.children():
                if not id(child) in visited:
                    yield from traverse(child)

    yield from traverse(dag)


def deserialize(bstring):
    """reloads an HistoryDag serialized object, as ouput by HistoryDag.serialize"""
    serial_dict = pickle.loads(bstring)
    sequence_list = serial_dict["sequence_list"]
    node_list = serial_dict["node_list"]
    edge_list = serial_dict["edge_list"]

    def unpack_seqs(seqset):
        return frozenset({sequence_list[idx] for idx in seqset})

    node_postorder = [
        HistoryDag(
            sequence_list[idx], {unpack_seqs(clade): EdgeSet() for clade in clades}
        )
        for idx, clades in node_list
    ]
    # Last node in list is root
    for origin_idx, target_idx, weight, prob in edge_list:
        node_postorder[origin_idx].add_edge(
            node_postorder[target_idx], weight=weight, prob=prob, prob_norm=False
        )
    return node_postorder[-1]


def history_dag_from_newicks(newicklist):
    treelist = list(map(lambda x: ete3.Tree(x, format=8), newicklist))
    for tree in treelist:
        for node in tree.traverse():
            node.sequence = node.name
    return history_dag_from_etes(treelist)


def history_dag_from_etes(treelist):
    dag = from_tree(treelist[0])
    for tree in treelist[1:]:
        dag.merge(from_tree(tree))
    return dag


def recalculate_parsimony(tree: HistoryDag, distance_func=utils.hamming_distance):
    for node in postorder(tree):
        for clade, eset in node.clades.items():
            for i in range(len(eset.targets)):
                eset.weights[i] = distance_func(eset.targets[i].label, node.label)
    return tree.weight()


def empty_node(label, clades):
    return HistoryDag(label, {clade: EdgeSet() for clade in clades})

def add_abundances(self, abundances: dict):
    """Expects DAG to be collapsed so that only edges targeting leaf nodes may have same label on parent and child nodes. Abundances should be a dictionary containing integers, keyed by node labels (sequences?)"""
    for node in postorder(self):
        # Although it's guaranteed to happen in max parsimony setting, only want internal nodes that are adjacent to a leaf with the same label to carry a nonzero abundance
        if node.label in abundances:
            if node.is_leaf() or frozenset({node.label}) in node.clades:
                node.abundance = int(abundances[node.label])
            else:
                node.abundance = 0
        else:
            node.abundance = 0

    # This corresponds to https://github.com/matsengrp/gctree/blob/375f61e60ec87aabfb0e5a9d0575cf56f46f06a5/gctree/branching_processes.py#L319
    nondagroot = list(self.children())[0]
    if len(nondagroot.clades) == 1:
        nondagroot.abundance = 1
