import ete3
import graphviz as gv
import random
from Bio.Data.IUPACData import ambiguous_dna_values
from collections import Counter
from gctree.utils import hamming_distance


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

    def add_edge(self, target, weight=0):
        # target clades must union to a clade of self
        if target.is_leaf():
            key = frozenset([target.label])
        else:
            key = frozenset().union(*target.clades.keys())
        if key not in self.clades:
            raise KeyError("Target clades' union is not a clade of this parent node")
        else:
            self.clades[key].add(target, weight=weight)
            target.parents.add(self)

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

    def to_newick(self):
        """Converts to extended newick format with arbitrary node names and a
        sequence feature. For use on tree-shaped DAG"""

        def newick(node):
            if node.is_leaf():
                return f"1[&&NHX:sequence={node.label}]"
            else:
                return (
                    "("
                    + ",".join([newick(node2) for node2 in node.children()])
                    + ")"
                    + f"1[&&NHX:sequence={node.label}]"
                )

        if self.label == "root":
            return newick(next(self.children())) + ";"
        else:
            return newick(self) + ";"

    def to_ete(self):
        return ete3.TreeNode(newick=self.to_newick(), format=1)

    def to_graphviz(self, namedict, use_sequences=False):
        """Converts to graphviz Digraph object. Namedict must associate sequences
        of all leaf nodes to a name
        """

        def taxa(clade):
            if use_sequences:
                l = [taxon for taxon in clade]
            else:
                l = [str(namedict[taxon]) for taxon in clade]
            l.sort()
            return ",".join(l)

        def min_weight_under(node):
            try:
                return(node.min_weight_under)
            except:
                return(None)

        def labeller(node):
            if use_sequences:
                return node.label
            else:
                return hash(node.label)

        def leaf_labeller(node):
            if use_sequences:
                return node.label
            else:
                return namedict[node.label]

        G = gv.Digraph("labeled partition DAG", node_attr={"shape": "record"})
        for node in postorder(self):
            if node.is_leaf():
                G.node(str(id(node)), f"<label> {leaf_labeller(node)}")
            else:
                splits = "|".join(
                    [f"<{taxa(clade)}> {taxa(clade)}" for clade in node.clades]
                )
                G.node(str(id(node)), f"{{ <label> {labeller(node)} {min_weight_under(node)} |{{{splits}}} }}")
                for clade in node.clades:
                    for target, weight, prob in node.clades[clade]:
                        label = ""
                        if prob < 1.0:
                            label += f"p:{prob:.2f}"
                        if weight > 0.0:
                            label += f"w:{weight}"
                        G.edge(
                            f"{id(node)}:{taxa(clade)}",
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

    def sample(self, min_weight=False):
        """Samples a sub-history-DAG that is also a tree containing the root and
        all leaf nodes. Returns a new SdagNode object"""
        sample = self.node_self()
        for clade, eset in self.clades.items():
            sampled_target, target_weight = eset.sample(min_weight=min_weight)
            sample.clades[clade].add(sampled_target.sample(min_weight=min_weight), weight=target_weight)
        return sample

    def get_trees(self, min_weight=False):
        """Return a generator to iterate through all trees expressed by the DAG."""
        def product(terms, accum=tuple()):
            if terms:
                for term in terms[0]:
                    yield from product(terms[1:], accum=(accum + (term, )))
            else:
                yield accum

        if min_weight:
            dag = self
        else:
            dag = self.prune_min_weight()

        optionlist = [((clade, targettree, i) for i, target in enumerate(eset.targets)
                       for targettree in target.get_trees(min_weight=min_weight))
                      for clade, eset in self.clades.items()]

        for option in product(optionlist):
            tree = dag.node_self()
            for clade, targettree, index in option:
                tree.clades[clade].add(targettree, weight=eset.weights[index])
            yield tree    



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

        self._hashes = {hash(self.targets[i]): i for i in range(len(self.targets))}
        if not len(self._hashes) == len(self.targets):
            raise TypeError("First argument may not contain duplicate target nodes")
        # Should probably also check to see that all passed lists have same length

    def __iter__(self):
        return (
            (self.targets[i], self.weights[i], self.probs[i])
            for i in range(len(self.targets))
        )

    def sample(self, min_weight=False):
        '''Returns a randomly sampled child edge, and the corresponding entry from the
        weight vector. If min_weight is True, samples only target nodes with lowest
        min_weight_under attribute, ignoring edge probabilities.'''
        if min_weight:
            mw = min(node.min_weight_under + dag_hamming_distance(self.parent.label, node.label) for node in self.targets)
            options = [i for i, node in enumerate(self.targets) if (node.min_weight_under + dag_hamming_distance(self.parent.label, node.label)) == mw]
            index = random.choices(options, k=1)[0]
            return (self.targets[index], self.weights[index])
        else:
            choice = random.choices(self.targets, weights=self.probs, k=1)
            return (choice[0], self.weights[self._hashes[hash(choice[0])]])

    def add(self, target, weight=0, prob=None):
        """currently does nothing if edge is already present"""
        if not hash(target) in self._hashes:
            self._hashes[hash(target)] = len(self.targets)
            self.targets.append(target)
            self.weights.append(weight)

            if prob is None:
                prob = float(1) / len(self.targets)
            self.probs = list(
                map(lambda x: x * (1 - prob) / sum(self.probs), self.probs)
            )
            self.probs.append(prob)
        else:
            pass
            # index = self._hashes[hash(target)]
            # self.weight[index] = weight # Could do something here!!


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
        "root",
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
    bases = "AGCT-"
    ambiguous_dna_values.update({"?": "GATC-", "-": "-"})
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

                    def _options(option_dict, sequence):
                        if option_dict:
                            site, choices = option_dict.popitem()
                            for choice in choices:
                                sequence = (
                                    sequence[:site] + choice + sequence[site + 1 :]
                                )
                                yield from _options(option_dict.copy(), sequence)
                        else:
                            yield sequence

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


def dag_analysis(in_trees, n_samples=100):
    in_tree_weights = [recalculate_ete_parsimony(tree) for tree in in_trees]
    print(f"Input trees have the following weight distribution:")
    hist(Counter(in_tree_weights), samples=len(in_tree_weights))
    resolvedset = {from_tree(tree).to_newick() for tree in in_trees}
    print(len(resolvedset), " unique input trees")
    dag = sdag_from_etes(in_trees)

    dagsamples = []
    for _ in range(n_samples):
        dagsamples.append(dag.sample())
    dagsampleweights = [sample.weight() for sample in dagsamples]
    sampleset = {tree.to_newick() for tree in dagsamples}
    print(f"\nSampled trees have the following weight distribution:")
    hist(Counter(dagsampleweights), samples=n_samples)
    print(len(sampleset), " unique sampled trees")
    print(len(sampleset - resolvedset), " sampled trees were not DAG inputs")


def disambiguate_all(treelist):
    resolvedsamples = []
    for sample in treelist:
        resolvedsamples.extend(disambiguate(sample))
    return resolvedsamples

def dag_hamming_distance(s1, s2):
    if s1 == "root" or s2 == "root":
        return 0
    else:
        return hamming_distance(s1, s2)


def recalculate_ete_parsimony(tree: ete3.TreeNode) -> float:
    tree.dist = 0
    for node in tree.iter_descendants():
        node.dist = hamming_distance(node.sequence, node.up.sequence)
    return total_weight(tree)


def recalculate_parsimony(tree: SdagNode):
    for node in postorder(tree):
        for clade, eset in node.clades.items():
            for i in range(len(eset.targets)):
                eset.weights[i] = dag_hamming_distance(eset.targets[i].label, node.label)
    return tree.weight()


def hist(c: Counter, samples=1):
    l = list(c.items())
    l.sort()
    print(f"Weight | Frequency", "\n------------------")
    for weight, freq in l:
        print(f"{weight}   | {freq/samples}")

def total_weight(tree: ete3.TreeNode) -> float:
    return(sum(node.dist for node in tree.traverse()))
