import ete3
import pickle
import graphviz as gv
import random
from Bio.Data.IUPACData import ambiguous_dna_values
from collections import Counter

bases = "AGCT-"
ambiguous_dna_values.update({"?": "GATC-", "-": "-"})


def weight_function(func):
    """A wrapper to allow distance to label 'DAG_root' to be zero"""

    def wrapper(s1, s2):
        if s1 == "DAG_root" or s2 == "DAG_root":
            return 0
        else:
            return func(s1, s2)

    return wrapper


@weight_function
def hamming_distance(s1: str, s2: str) -> int:
    if len(s1) != len(s2):
        raise ValueError("Sequences must have the same length!")
    return sum(x != y for x, y in zip(s1, s2))


class SdagNode:
    """A recursive representation of an sDAG
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
        return SdagNode(self.label, {clade: EdgeSet() for clade in self.clades})

    def under_clade(self):
        """Returns the union of all child clades"""
        if self.is_leaf():
            return frozenset([self.label])
        else:
            return frozenset().union(*self.clades.keys())

    def copy(self, use_recursion=False):
        """Add each child node's copy, below this node, by merging.
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
        """performs post order traversal to add node and all of its children,
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
        """Adds edge, if not already present and allowed. Returns if edge was added."""
        # target clades must union to a clade of self
        key = target.under_clade()
        if key not in self.clades:
            print(key)
            print(self.clades)
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
        """If clade is provided, returns generator object of edge targets from that
        clade. If no clade is provided, generator includes all children of self.
        """
        if clade is None:
            return (
                target for clade in self.clades for target, _, _ in self.clades[clade]
            )
        else:
            return (child for child, _, _ in self.clades[clade])

    def add_all_allowed_edges(self, new_from_root=True, adjacent_labels=True):
        """Add all allowed edges to the DAG, returning the number that were added.
        if new_from_root is False, no edges are added that start at DAG root.
        This is useful to enforce a single ancestral sequence.
        If adjacent_labels is False, no edges will be added between nodes with the same labels"""
        n_added = 0
        clade_dict = {node.under_clade(): [] for node in postorder(self)}
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
                        n_added += node.add_edge(target)
        return n_added

    def to_newick(self, namedict={}, fixed_order=True, include_root=False):
        """Converts to extended newick format with arbitrary node names and a
        sequence feature. For use on tree-shaped DAG.
        If fixed_order is True, with same namedict, two trees' newick representations are
        equal iff the two trees have the same topology and node sequences.
        Should be usable for equality in this sense on general DAG too, with include_root
        True"""

        def newick(node):
            if node.label in namedict:
                name = namedict[node.label]
            else:
                name = "1"
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

    def to_graphviz(self, namedict={}, show_partitions=True):
        """Converts to graphviz Digraph object. Namedict must associate sequences
        of all leaf nodes to a name
        """

        def taxa(clade):
            l = [labeller(taxon) for taxon in clade]
            l.sort()
            return ",".join(l)

        def min_weight_under(node):
            try:
                return node.min_weight_under
            except:
                return ""

        def labeller(sequence):
            if len(sequence) < 11:
                return sequence
            elif sequence in namedict:
                return str(namedict[sequence])
            else:
                return hash(sequence)

        G = gv.Digraph("labeled partition DAG", node_attr={"shape": "record"})
        for node in postorder(self):
            if node.is_leaf() or show_partitions is False:
                G.node(str(id(node)), f"<label> {labeller(node.label)}")
            else:
                splits = "|".join(
                    [f"<{taxa(clade)}> {taxa(clade)}" for clade in node.clades]
                )
                G.node(
                    str(id(node)), f"{{ <label> {labeller(node.label)} |{{{splits}}} }}"
                )
            for clade in node.clades:
                for target, weight, prob in node.clades[clade]:
                    label = ""
                    if prob < 1.0:
                        label += f"p:{prob:.2f}"
                    if weight > 0.0:
                        label += f"w:{weight}"
                    G.edge(
                        f"{id(node)}:{taxa(clade) if show_partitions else 'label'}",
                        f"{id(target)}:label",
                        label=label,
                    )
        return G

    def weight(self):
        "Sums weights of all edges in the DAG"
        nodes = postorder(self)
        edgesetsums = (
            sum(edgeset.weights) for node in nodes for edgeset in node.clades.values()
        )
        return sum(edgesetsums)

    def internal_avg_parents(self):
        """Returns the average number of parents among internal nodes
        A simple measure of similarity of trees that the DAG expresses."""
        nonleaf_parents = (len(n.parents) for n in postorder(self) if not n.is_leaf())
        n = 0
        cumsum = 0
        for sum in nonleaf_parents:
            n += 1
            cumsum += sum
        # Exclude root:
        return cumsum / float(n - 1)

    def sample(self, min_weight=False, distance_func=hamming_distance):
        """Samples a sub-history-DAG that is also a tree containing the root and
        all leaf nodes. Returns a new SdagNode object"""
        sample = self.node_self()
        from_root = sample.label == "DAG_root"
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
        self, min_weight=False, count_resolutions=False, distance_func=hamming_distance
    ):
        """Annotates each node in the DAG with the number of complete trees underneath (extending to leaves,
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
                    node.trees_under *= len(list(sequence_resolutions(node.label)))
            return self.trees_under

    def get_trees(self, min_weight=False, distance_func=hamming_distance):
        """Return a generator to iterate through all trees expressed by the DAG."""

        if min_weight:
            dag = self.prune_min_weight(distance_func=distance_func)
        else:
            dag = self

        def genexp_func(clade):
            # Return generator expression of all possible choices of tree
            # structure from dag below clade
            def f():
                eset = self.clades[clade]
                return (
                    (clade, targettree, i)
                    for i, target in enumerate(eset.targets)
                    for targettree in target.get_trees(
                        min_weight=min_weight, distance_func=distance_func
                    )
                )

            return f

        optionlist = [genexp_func(clade) for clade in self.clades]

        for option in product(optionlist):
            tree = dag.node_self()
            from_root = tree.label == "DAG_root"
            for clade, targettree, index in option:
                tree.clades[clade].add_to_edgeset(
                    targettree,
                    weight=dag.clades[clade].weights[index],
                    from_root=from_root,
                )
            yield tree

    def min_weight_annotate(self, distance_func=hamming_distance):
        for node in postorder(self):
            if node.is_leaf():
                node.min_weight_under = 0
            else:
                min_sum = sum(
                    [
                        min(
                            [
                                child.min_weight_under
                                + distance_func(child.label, node.label)
                                for child in node.clades[clade].targets
                            ]
                        )
                        for clade in node.clades
                    ]
                )
                node.min_weight_under = min_sum
        return self.min_weight_under

    def disambiguate_dag(self, distance_func=hamming_distance):
        """like get_weight_counts_with_ambiguities, but creates dictionaries of Counter objects at each node,
        keyed by possible sequences at that node.
        chooses a sequence that minimizes the below-node tree weight.
        This is a one-pass Sankoff algorithm??"""
        # Replace prod function later

        prod = lambda l: l[0] * prod(l[1:]) if l else 1

        def counter_prod(counterlist):
            newc = Counter()
            for combi in product([c.items for c in counterlist]):
                weights, counts = [[t[i] for t in combi] for i in range(len(combi[0]))]
                newc.update({sum(weights): prod(counts)})
            return newc

        def counter_sum(counterlist):
            newc = Counter()
            for c in counterlist:
                newc += c
            return newc

        def addweight(c, w):
            return Counter({key + w: val for key, val in c.items()})

        for node in postorder(self):
            node.weight_counters = {}
            min_weight = float('inf')
            best_sequence = None
            for sequence in sequence_resolutions(node.label):
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
                    node.weight_counters[sequence] = counter_prod(cladecounters)
                    this_sequence_min_weight = min(node.weight_counters[sequence])
                    if this_sequence_min_weight < min_weight:
                        min_weight = this_sequence_min_weight
                        best_sequence = sequence
            n_min_weight = node.weight_counters[sequence][min_weight]
            node.weight_counters = {sequence: Counter({min_weight: n_min_weight})}

        return self.weight_counters

    def get_weight_counts_with_ambiguities(self, distance_func=hamming_distance):
        """like get_weight_counts, but creates dictionaries of Counter objects at each node,
        keyed by possible sequences at that node.
        The total number of trees will be greater than count_trees(), as these are possible
        disambiguations of trees. These disambiguations may not be unique (?), but if two are
        the same, they come from different subtrees of the DAG."""
        # Replace prod function later

        prod = lambda l: l[0] * prod(l[1:]) if l else 1

        def counter_prod(counterlist):
            newc = Counter()
            for combi in product([c.items for c in counterlist]):
                weights, counts = [[t[i] for t in combi] for i in range(len(combi[0]))]
                newc.update({sum(weights): prod(counts)})
            return newc

        def counter_sum(counterlist):
            newc = Counter()
            for c in counterlist:
                newc += c
            return newc

        def addweight(c, w):
            return Counter({key + w: val for key, val in c.items()})

        for node in postorder(self):
            node.weight_counters = {}
            for sequence in sequence_resolutions(node.label):
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
                    node.weight_counters[sequence] = counter_prod(cladecounters)
        return self.weight_counters

    def get_weight_counts(self, distance_func=hamming_distance):
        """Annotate each node in the DAG, in postorder traversal, with a Counter object
        keyed by weight, with values the number of possible unique trees below the node
        with that weight."""
        # Replace prod function later

        prod = lambda l: l[0] * prod(l[1:]) if l else 1

        def counter_prod(counterlist):
            newc = Counter()
            for combi in product([c.items for c in counterlist]):
                weights, counts = [[t[i] for t in combi] for i in range(len(combi[0]))]
                newc.update({sum(weights): prod(counts)})
            return newc

        def counter_sum(counterlist):
            newc = Counter()
            for c in counterlist:
                newc += c
            return newc

        def addweight(c, w):
            return Counter({key + w: val for key, val in c.items()})

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
                node.weight_counter = counter_prod(cladecounters)
        return self.weight_counter

    def prune_min_weight(self, distance_func=hamming_distance):
        newdag = self.copy()
        newdag.min_weight_annotate(distance_func=distance_func)
        # It may not be okay to use preorder here. May need reverse postorder
        # instead?
        for node in preorder(newdag):
            for clade, eset in node.clades.items():
                weightlist = [
                    (
                        target.min_weight_under
                        + distance_func(target.label, node.label),
                        target,
                        index,
                    )
                    for index, target in enumerate(eset.targets)
                ]
                minweight = min([weight for weight, _, _ in weightlist])
                newtargets = []
                newweights = []
                for weight, target, index in weightlist:
                    if weight == minweight:
                        newtargets.append(target)
                        newweights.append(eset.weights[index])
                eset.targets = newtargets
                eset.weights = newweights
                n = len(eset.targets)
                eset.probs = [1.0 / n] * n
        return newdag

    def serialize(self):
        """Represents SdagNode object as a list of sequences, a list of node tuples containing
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


class EdgeSet:
    """Goal: associate targets (edges) with arbitrary parameters, but support
    set-like operations like lookup and enforce that elements are unique."""

    def __init__(self, *args, weights: list = None, probs: list = None):
        """Takes no arguments, or an ordered iterable containing target nodes"""
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

    def sample(self, min_weight=False, distance_func=hamming_distance):
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
        dag = SdagNode(
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
    dagroot = SdagNode(
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


def postorder(dag: SdagNode):
    visited = set()

    def traverse(node: SdagNode):
        visited.add(id(node))
        if not node.is_leaf():
            for child in node.children():
                if not id(child) in visited:
                    yield from traverse(child)
        yield node

    yield from traverse(dag)


def preorder(dag: SdagNode):
    """Careful! remember this is not guaranteed to visit a parent node before any of its children.
    for that, need reverse postorder traversal."""
    visited = set()

    def traverse(node: SdagNode):
        visited.add(id(node))
        yield node
        if not node.is_leaf():
            for child in node.children():
                if not id(child) in visited:
                    yield from traverse(child)

    yield from traverse(dag)


def sdag_from_newicks(newicklist):
    treelist = list(map(lambda x: ete3.Tree(x, format=8), newicklist))
    for tree in treelist:
        for node in tree.traverse():
            node.sequence = node.name
    return sdag_from_etes(treelist)


def sdag_from_etes(treelist):
    dag = from_tree(treelist[0])
    for tree in treelist[1:]:
        dag.merge(from_tree(tree))
    return dag


def disambiguate(tree: ete3.TreeNode, random_state=None) -> ete3.TreeNode:
    """Resolve tree and return list of all possible resolutions"""
    code_vectors = {
        code: [
            0 if base in ambiguous_dna_values[code] else float("inf") for base in bases
        ]
        for code in ambiguous_dna_values
    }
    cost_adjust = {
        base: [int(not i == j) for j in range(5)] for i, base in enumerate(bases)
    }
    if random_state is None:
        random.seed(tree.write(format=1))
    else:
        random.setstate(random_state)

    for node in tree.traverse(strategy="postorder"):

        def cvup(node, site):
            cv = code_vectors[node.sequence[site]].copy()
            if not node.is_leaf():
                for i in range(5):
                    for child in node.children:
                        cv[i] += min(
                            [
                                sum(v)
                                for v in zip(child.cvd[site], cost_adjust[bases[i]])
                            ]
                        )
            return cv

        # Make dictionary of cost vectors for each site
        node.cvd = {site: cvup(node, site) for site in range(len(node.sequence))}

    disambiguated = [tree.copy()]
    ambiguous = True
    while ambiguous:
        ambiguous = False
        treesindex = 0
        while treesindex < len(disambiguated):
            tree2 = disambiguated[treesindex]
            treesindex += 1
            for node in tree2.traverse(strategy="preorder"):
                ambiguous_sites = [
                    site for site, code in enumerate(node.sequence) if code not in bases
                ]
                if not ambiguous_sites:
                    continue
                else:
                    ambiguous = True
                    # Adjust cost vectors for ambiguous sites base on above
                    if not node.is_root():
                        for site in ambiguous_sites:
                            base_above = node.up.sequence[site]
                            node.cvd[site] = [
                                sum(v)
                                for v in zip(node.cvd[site], cost_adjust[base_above])
                            ]
                    option_dict = {site: "" for site in ambiguous_sites}
                    # Enumerate min-cost choices
                    for site in ambiguous_sites:
                        min_cost = min(node.cvd[site])
                        min_cost_sites = [
                            bases[i]
                            for i, val in enumerate(node.cvd[site])
                            if val == min_cost
                        ]
                        option_dict[site] = "".join(min_cost_sites)

                    sequences = list(_options(option_dict, node.sequence))
                    # Give this tree the first sequence, append copies with all
                    # others to disambiguated.
                    numseqs = len(sequences)
                    for idx, sequence in enumerate(sequences):
                        node.sequence = sequence
                        if idx < numseqs - 1:
                            disambiguated.append(tree2.copy())
                    break
    for tree in disambiguated:
        for node in tree.traverse():
            try:
                node.del_feature("cvd")
            except KeyError:
                pass
    return disambiguated


def product(optionlist, accum=tuple()):
    """Takes a list of functions which each return a fresh generator
    on options at that site"""
    if optionlist:
        for term in optionlist[0]():
            yield from product(optionlist[1:], accum=(accum + (term,)))
    else:
        yield accum


def _options(option_dict, sequence):
    """option_dict is keyed by site index, with iterables containing
    allowed bases as values"""
    if option_dict:
        site, choices = option_dict.popitem()
        for choice in choices:
            sequence = sequence[:site] + choice + sequence[site + 1 :]
            yield from _options(option_dict.copy(), sequence)
    else:
        yield sequence


def sequence_resolutions(sequence):
    """Returns iterator on possible resolutions of sequence, replacing ambiguity codes with bases."""
    if sequence == "DAG_root":
        yield sequence
    else:
        ambiguous_sites = [
            site for site, code in enumerate(sequence) if code not in bases
        ]
        if not ambiguous_sites:
            yield sequence
        else:
            option_dict = {
                site: ambiguous_dna_values[sequence[site]] for site in ambiguous_sites
            }
            yield from _options(option_dict, sequence)


def disambiguate_all(treelist):
    resolvedsamples = []
    for sample in treelist:
        resolvedsamples.extend(disambiguate(sample))
    return resolvedsamples


def recalculate_ete_parsimony(
    tree: ete3.TreeNode, distance_func=hamming_distance
) -> float:
    tree.dist = 0
    for node in tree.iter_descendants():
        node.dist = distance_func(node.sequence, node.up.sequence)
    return total_weight(tree)


def recalculate_parsimony(tree: SdagNode, distance_func=hamming_distance):
    for node in postorder(tree):
        for clade, eset in node.clades.items():
            for i in range(len(eset.targets)):
                eset.weights[i] = distance_func(eset.targets[i].label, node.label)
    return tree.weight()


def hist(c: Counter, samples=1):
    l = list(c.items())
    l.sort()
    print("Weight\t| Frequency\n------------------")
    for weight, freq in l:
        print(f"{weight}  \t| {freq if samples==1 else freq/samples}")


def total_weight(tree: ete3.TreeNode) -> float:
    return sum(node.dist for node in tree.traverse())


def deserialize(bstring):
    """reloads an SdagNode serialized object, as ouput by SdagNode.serialize"""
    serial_dict = pickle.loads(bstring)
    sequence_list = serial_dict["sequence_list"]
    node_list = serial_dict["node_list"]
    edge_list = serial_dict["edge_list"]

    def unpack_seqs(seqset):
        return frozenset({sequence_list[idx] for idx in seqset})

    node_postorder = [
        SdagNode(
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

def collapse_adjacent_sequences(tree: ete3.TreeNode) -> ete3.TreeNode:
    """Collapse nonleaf nodes that have the same sequence"""
    # Need to keep doing this until the tree fully collapsed. See gctree for this!
    to_delete = []
    for node in tree.get_descendants():
        if not node.is_leaf() and node.sequence == node.up.sequence:
            to_delete.append(node)
    for node in to_delete:
        node.delete()
    return(tree)

