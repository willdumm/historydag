"""Utility functions and classes for working with HistoryDag objects."""

import ete3
from Bio.Data.IUPACData import ambiguous_dna_values
from collections import Counter
from functools import wraps
from collections import UserDict
from decimal import Decimal
from typing import (
    List,
    Any,
    TypeVar,
    Callable,
    Union,
    Iterable,
    Generator,
    Tuple,
    NamedTuple,
    Optional,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from historydag.dag import HistoryDagNode, HistoryDag

Weight = Any
Label = Union[NamedTuple, "UALabel"]
F = TypeVar("F", bound=Callable[..., Any])


class UALabel(str):
    _fields: Tuple = tuple()

    def __new__(cls):
        return super(UALabel, cls).__new__(cls, "UA_Node")

    def __eq__(self, other):
        return isinstance(other, UALabel)

    def __hash__(self):
        return hash("UA_Node")

    def __iter__(self):
        raise RuntimeError("Attempted to iterate from dag root UALabel")

    def _asdict(self):
        raise RuntimeError("Attempted to iterate from dag root UALabel")


bases = "AGCT-"
ambiguous_dna_values.update({"?": bases, "-": "-"})


# ######## Decorators ########
def access_nodefield_default(fieldname: str, default: Any) -> Any:
    """A decorator for accessing label fields on a HistoryDagNode. Converts a
    function taking some label field's values as positional arguments, to a
    function taking HistoryDagNodes as positional arguments.

    Args:
        fieldname: The name of the label field whose value the function takes as arguments
        default: A value that should be returned if one of the arguments is the DAG UA node.

    For example, instead of
    `lambda n1, n2: default if n1.is_ua_node() or n2.is_ua_node() else func(n1.label.fieldname, n2.label.fieldname)`,
    this wrapper allows one to write `access_nodefield_default(fieldname, default)(func)`.
    """

    def decorator(func):
        @ignore_uanode(default)
        @access_field("label")
        @access_field(fieldname)
        @wraps(func)
        def wrapper(*args: Label, **kwargs: Any) -> Weight:
            return func(*args, **kwargs)

        return wrapper

    return decorator


def access_field(fieldname: str) -> Callable[[F], F]:
    """A decorator for conveniently accessing a field in a label.

    To be used instead of something like `lambda l1, l2:
    func(l1.fieldname, l2.fieldname)`. Instead just write
    `access_field(fieldname)(func)`. Supports arbitrarily many
    positional arguments, which are all expected to be labels
    (namedtuples) with field `fieldname`.
    """

    def decorator(func: F):
        @wraps(func)
        def wrapper(*args: Label, **kwargs: Any) -> Any:
            newargs = [getattr(label, fieldname) for label in args]
            return func(*newargs, **kwargs)

        return wrapper

    return decorator


def ignore_uanode(default: Any) -> Callable[[F], F]:
    """A decorator to return a default value if any argument is a UANode.

    For instance, to allow distance between two nodes to be zero if one
    is UANode
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args: "HistoryDagNode", **kwargs: Any):
            for node in args:
                if node.is_ua_node():
                    return default
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def explode_label(labelfield: str):
    """A decorator to make it easier to expand a Label by a certain field.

    Args:
        labelfield: the name of the field whose contents the wrapped function is expected to
            explode

    Returns:
        A decorator which converts a function which explodes a field value, into a function
        which explodes the whole label at that field.
    """

    def decorator(
        func: Callable[[Any], Iterable[Any]]
    ) -> Callable[[Label], Iterable[Label]]:
        @wraps(func)
        def wrapfunc(label, *args, **kwargs):
            Label = type(label)
            d = label._asdict()
            for newval in func(d[labelfield], *args, **kwargs):
                d[labelfield] = newval
                yield Label(**d)

        return wrapfunc

    return decorator


# ######## Distances and comparisons... ########


def hamming_distance(s1: str, s2: str) -> int:
    """The sitewise sum of base differences between s1 and s2."""
    if len(s1) != len(s2):
        raise ValueError("Sequences must have the same length!")
    return sum(x != y for x, y in zip(s1, s2))


@access_nodefield_default("sequence", 0)
def wrapped_hamming_distance(s1, s2) -> int:
    """The sitewise sum of base differences between sequence field contents of
    two nodes.

    Takes two HistoryDagNodes as arguments.

    If l1 or l2 is a UANode, returns 0.
    """
    return hamming_distance(s1, s2)


def hamming_distance_leaf_ambiguous(n1, n2):
    """Same as wrapped_hamming_distance, but correctly calculates parsimony
    scores if leaf nodes have ambiguous sequences."""
    if n2.is_leaf():
        # Then its sequence may be ambiguous
        s1 = n1.label.sequence
        s2 = n2.label.sequence
        if len(s1) != len(s2):
            raise ValueError("Sequences must have the same length!")
        return sum(
            pbase not in ambiguous_dna_values[cbase] for pbase, cbase in zip(s1, s2)
        )
    else:
        return wrapped_hamming_distance(n1, n2)


def is_ambiguous(sequence: str) -> bool:
    """Returns whether the provided sequence contains IUPAC nucleotide
    ambiguity codes."""
    return any(code not in bases for code in sequence)


def cartesian_product(
    optionlist: List[Callable[[], Iterable]], accum=tuple()
) -> Generator[Tuple, None, None]:
    """The cartesian product of iterables in a list.

    Takes a list of functions which each return a fresh generator on
    options at that site, and returns a generator yielding tuples, which
    are elements of the cartesian product of the passed generators'
    contents.
    """
    if optionlist:
        for term in optionlist[0]():
            yield from cartesian_product(optionlist[1:], accum=(accum + (term,)))
    else:
        yield accum


@explode_label("sequence")
def sequence_resolutions(sequence: str) -> Generator[str, None, None]:
    """Iterates through possible disambiguations of sequence, recursively.

    Recursion-depth-limited by number of ambiguity codes in sequence,
    not sequence length.
    """

    def _sequence_resolutions(sequence, _accum=""):
        if sequence:
            for index, base in enumerate(sequence):
                if base in bases:
                    _accum += base
                else:
                    for newbase in ambiguous_dna_values[base]:
                        yield from _sequence_resolutions(
                            sequence[index + 1 :], _accum=(_accum + newbase)
                        )
                    return
        yield _accum

    return _sequence_resolutions(sequence)


@access_field("sequence")
def sequence_resolutions_count(sequence: str) -> int:
    """Count the number of possible sequence resolutions Equivalent to the
    length of the list returned by :meth:`sequence_resolutions`."""
    base_options = [
        len(ambiguous_dna_values[base])
        for base in sequence
        if base in ambiguous_dna_values
    ]
    return prod(base_options)


def hist(c: Counter, samples: int = 1):
    """Pretty prints a counter Normalizing counts using the number of samples,
    passed as the argument `samples`."""
    ls = list(c.items())
    ls.sort()
    print("Weight\t| Frequency\n------------------")
    for weight, freq in ls:
        print(f"{weight}  \t| {freq if samples==1 else freq/samples}")


def is_collapsed(tree: ete3.TreeNode) -> bool:
    """Return whether the provided tree is collapsed.

    Collapsed means that any edge whose target is not a leaf node
    connects nodes with different sequences.
    """
    return not any(
        node.sequence == node.up.sequence and not node.is_leaf()
        for node in tree.iter_descendants()
    )


def collapse_adjacent_sequences(tree: ete3.TreeNode) -> ete3.TreeNode:
    """Collapse nonleaf nodes that have the same sequence."""
    # Need to keep doing this until the tree fully collapsed. See gctree for this!
    tree = tree.copy()
    to_delete = []
    for node in tree.get_descendants():
        # This must stay invariably hamming distance, since it's measuring equality of strings
        if (
            not node.is_leaf()
            and hamming_distance(node.up.sequence, node.sequence) == 0
        ):
            to_delete.append(node)
    for node in to_delete:
        node.delete()
    return tree


class AddFuncDict(UserDict):
    """Container for function keyword arguments to
    :meth:`historydag.HistoryDag.weight_count`. This is primarily useful
    because it allows instances to be added. Passing the result to
    `weight_count` as keyword arguments counts the weights jointly. A
    :class:`historydag.utils.AddFuncDict` which may be passed as keyword
    arguments to :meth:`historydag.HistoryDag.weight_count`,
    :meth:`historydag.HistoryDag.trim_optimal_weight`, or
    :meth:`historydag.HistoryDag.optimal_weight_annotate` methods to trim or
    annotate a :meth:`historydag.HistoryDag` according to the weight that the
    contained functions implement.

    For example, `dag.weight_count(**(utils.hamming_distance_countfuncs + make_newickcountfuncs()))`
    would return a Counter object in which the weights are tuples containing hamming parsimony and newickstrings.

    Args:
        initialdata: A dictionary containing functions keyed by "start_func", "edge_weight_func", and
            "accum_func". "start_func" specifies weight assigned to leaf HistoryDagNodes.
            "edge_weight_func" specifies weight assigned to an edge between two HistoryDagNodes, with the
            first argument the parent node, and the second argument the child node.
            "accum_func" specifies how to 'add' a list of weights. See :meth:`historydag.HistoryDag.weight_count`
            for more details.
        name: A string containing a name for the weight to be counted. If a tuple of weights will be returned,
            use ``names`` instead.
        names: A tuple of strings containing names for the weights to be counted, if a tuple of weights will
            be returned by passed functions. If only a single weight will be returned, use ``name`` instead.
    """

    requiredkeys = {"start_func", "edge_weight_func", "accum_func"}

    def __init__(self, initialdata, name: str = None, names: Tuple[str] = None):
        self.name: Optional[str]
        self.names: Tuple[str]
        if name is not None and names is not None:
            raise ValueError(
                "Pass a value to either keyword argument 'name' or 'names'."
            )
        elif name is None and names is None:
            self.name = "unknown weight"
            self.names = (self.name,)
        elif name is not None:
            self.name = name
            self.names = (self.name,)
        elif names is not None:
            self.names = names
            self.name = None
        if not set(initialdata.keys()) == self.requiredkeys:
            raise ValueError(
                "Must provide functions named " + ", ".join(self.requiredkeys)
            )
        super().__init__(initialdata)

    def __add__(self, other) -> "AddFuncDict":
        fdict1 = self._convert_to_tupleargs()
        fdict2 = other._convert_to_tupleargs()
        n = len(fdict1.names)

        def newaccumfunc(weightlist):
            return fdict1["accum_func"](
                [weight[0:n] for weight in weightlist]
            ) + fdict2["accum_func"]([weight[n:] for weight in weightlist])

        def addfuncs(func1, func2):
            def newfunc(*args):
                return func1(*args) + func2(*args)

            return newfunc

        return AddFuncDict(
            {
                "start_func": addfuncs(fdict1["start_func"], fdict2["start_func"]),
                "edge_weight_func": addfuncs(
                    fdict1["edge_weight_func"], fdict2["edge_weight_func"]
                ),
                "accum_func": newaccumfunc,
            },
            names=fdict1.names + fdict2.names,
        )

    def _convert_to_tupleargs(self):
        if self.name is not None:

            def node_to_weight_decorator(func):
                @wraps(func)
                def wrapper(*args):
                    return (func(*args),)

                return wrapper

            def list_of_weights_to_weight_decorator(func):
                @wraps(func)
                def wrapper(weighttuplelist: List[Weight]):
                    return (func([wt[0] for wt in weighttuplelist]),)

                return wrapper

            return AddFuncDict(
                {
                    "start_func": node_to_weight_decorator(self["start_func"]),
                    "edge_weight_func": node_to_weight_decorator(
                        self["edge_weight_func"]
                    ),
                    "accum_func": list_of_weights_to_weight_decorator(
                        self["accum_func"]
                    ),
                },
                names=(self.name,),
            )
        else:
            return self


hamming_distance_countfuncs = AddFuncDict(
    {
        "start_func": lambda n: 0,
        "edge_weight_func": wrapped_hamming_distance,
        "accum_func": sum,
    },
    name="HammingParsimony",
)
"""Provides functions to count hamming distance parsimony.
For use with :meth:`historydag.HistoryDag.weight_count`."""

node_countfuncs = AddFuncDict(
    {
        "start_func": lambda n: 0,
        "edge_weight_func": lambda n1, n2: 1,
        "accum_func": sum,
    },
    name="NodeCount",
)
"""Provides functions to count the number of nodes in trees.
For use with :meth:`historydag.HistoryDag.weight_count`."""


def sum_rfdistance_funcs(reference_dag: "HistoryDag"):
    """Provides functions to compute the sum over all histories in the provided
    reference DAG, of rooted RF distances to those histories.

    Args:
        reference_dag: The reference DAG. The sum will be computed over all RF
            distances to histories in this DAG

    The reference DAG must have the same taxa as all the trees in the DAG on which these count
    functions are used.

    The edge weight is computed using the expression 2 * N[c_e] - |T| where c_e is the clade under
    the relevant edge, and |T| is the number of trees in the reference dag. This provide rooted RF
    distances, meaning that the clade below each edge is used for RF distance computation.

    The weights are represented by an IntState object and are shifted by a constant K,
    which is the sum of number of clades in each tree in the DAG.
    """
    n_histories = reference_dag.count_histories()
    N = reference_dag.count_nodes(collapse=True)

    # adjust clade union counts appearing on tree root nodes
    for child in reference_dag.dagroot.children():
        N[child.clade_union()] -= n_histories

    # Remove the UA node clade union from N
    try:
        N.pop(frozenset())
    except KeyError:
        pass

    # K is the constant that the weights are shifted by
    K = sum(N.values())

    num_trees = reference_dag.count_histories()

    def make_intstate(n):
        return IntState(n + K, state=n)

    def edge_func(n1, n2):
        if n1.is_ua_node():
            return make_intstate(0)
        else:
            clade = n2.clade_union()
            if clade in N:
                weight = num_trees - (2 * N[n2.clade_union()])
            else:
                # This clade's count should then just be 0:
                weight = num_trees
            return make_intstate(weight)

    kwargs = AddFuncDict(
        {
            "start_func": lambda n: make_intstate(0),
            "edge_weight_func": edge_func,
            "accum_func": lambda wlist: make_intstate(
                sum(w.state for w in wlist)
            ),  # summation over edge weights
        },
        name="RF_rooted_sum",
    )
    return kwargs


def make_rfdistance_countfuncs(ref_tree: "HistoryDag", rooted: bool = False):
    """Provides functions to compute Robinson-Foulds (RF) distances of trees in
    a DAG, relative to a fixed reference tree.

    We use :meth:`ete3.TreeNode.robinson_foulds` as the reference implementation for both
    rooted and unrooted RF distance.

    Args:
        ref_tree: A tree with respect to which Robinson-Foulds distance will be computed.
        rooted: If False, use edges' splits for RF distance computation. Otherwise, use
            the clade below each edge.

    The reference tree must have the same taxa as all the trees in the DAG.

    This calculation relies on the observation that the symmetric distance between
    the splits A in a tree in the DAG, and the splits B in the reference tree, can
    be computed as:
    ``|A ^ B| = |A U B| - |A n B| = |A - B| + |B| - |A n B|``

    As long as tree edges are in bijection with splits, this can be computed without
    constructing the set A by considering each edge's split independently.

    In order to accommodate multiple edges with the same split in a tree with root
    bifurcation, we keep track of the contribution of such edges separately.

    The weight type is a tuple wrapped in an IntState object. The first tuple value `a` is the
    contribution of edges which are not part of a root bifurcation, where edges whose splits are in B
    contribute `-1`, and edges whose splits are not in B contribute `-1`, and the second tuple
    value `b` is the contribution of the edges which are part of a root bifurcation. The value
    of the IntState is computed as `a + sign(b) + |B|`, which on the UA node of the hDAG gives RF distance.
    """
    taxa = frozenset(n.label for n in ref_tree.get_leaves())

    if not rooted:

        def split(node):
            cu = node.clade_union()
            return frozenset({cu, taxa - cu})

        ref_splits = frozenset(split(node) for node in ref_tree.preorder())
        # Remove above-root split, which doesn't map to any tree edge:
        ref_splits = ref_splits - {
            frozenset({taxa, frozenset()}),
        }
        shift = len(ref_splits)

        n_taxa = len(taxa)

        def is_history_root(n):
            return len(list(n.clade_union())) == n_taxa

        def sign(n):
            return (-1) * (n < 0) + (n > 0)

        def summer(tupseq):
            a, b = 0, 0
            for ia, ib in tupseq:
                a += ia
                b += ib
            return (a, b)

        def make_intstate(tup):
            return IntState(tup[0] + shift + sign(tup[1]), state=tup)

        def edge_func(n1, n2):
            spl = split(n2)
            if n1.is_ua_node():
                return make_intstate((0, 0))
            if len(n1.clades) == 2 and is_history_root(n1):
                if spl in ref_splits:
                    return make_intstate((0, -1))
                else:
                    return make_intstate((0, 1))
            else:
                if spl in ref_splits:
                    return make_intstate((-1, 0))
                else:
                    return make_intstate((1, 0))

        kwargs = AddFuncDict(
            {
                "start_func": lambda n: make_intstate((0, 0)),
                "edge_weight_func": edge_func,
                "accum_func": lambda wlist: make_intstate(
                    summer(w.state for w in wlist)
                ),
            },
            name="RF_unrooted_distance",
        )
    else:
        ref_cus = frozenset(
            node.clade_union() for node in ref_tree.preorder(skip_ua_node=True)
        )
        ref_cus = ref_cus - {
            taxa,
        }
        shift = len(ref_cus)

        def make_intstate(n):
            return IntState(n + shift, state=n)

        def edge_func(n1, n2):
            if n1.is_ua_node():
                return make_intstate(0)
            if n2.clade_union() in ref_cus:
                return make_intstate(-1)
            else:
                return make_intstate(1)

        kwargs = AddFuncDict(
            {
                "start_func": lambda n: make_intstate(0),
                "edge_weight_func": edge_func,
                "accum_func": lambda wlist: make_intstate(sum(w.state for w in wlist)),
            },
            name="RF_rooted_distance",
        )

    return kwargs


def make_newickcountfuncs(
    name_func=lambda n: "unnamed",
    features=None,
    feature_funcs={},
    internal_labels=True,
    collapse_leaves=False,
):
    """Provides functions to count newick strings. For use with
    :meth:`historydag.HistoryDag.weight_count`.

    Arguments are the same as for
    :meth:`historydag.HistoryDag.to_newick`.
    """

    def _newicksum(newicks):
        # Filter out collapsed/deleted edges
        snewicks = sorted(newicks)
        if len(snewicks) == 2 and ";" in [newick[-1] for newick in snewicks if newick]:
            # Then we are adding an edge above a complete tree
            return "".join(
                sorted(snewicks, key=lambda n: ";" == n[-1] if n else False)
            )[:-1]
        else:
            # Then we're just accumulating options between clades
            return "(" + ",".join(snewicks) + ")"

    def _newickedgeweight(n1, n2):
        if collapse_leaves and n2.is_leaf() and n1.label == n2.label:
            return "COLLAPSED_LEAF;"
        elif (
            internal_labels
            or n2.is_leaf()
            or (collapse_leaves and frozenset({n2.label}) in n2.clades)
        ):
            return (
                n2._newick_label(
                    name_func=name_func, features=features, feature_funcs=feature_funcs
                )
                + ";"
            )
        else:
            return ";"

    return AddFuncDict(
        {
            "start_func": lambda n: "",
            "edge_weight_func": _newickedgeweight,
            "accum_func": _newicksum,
        },
        name="NewickString",
    )


def _history_method(method):
    """HistoryDagNode method decorator to ensure that the method is only run on
    history DAGs which are histories."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self.is_history():
            raise ValueError(
                "to_newick requires the history DAG to be a history. "
                "To extract newicks from a general DAG, see to_newicks"
            )
        else:
            return method(self, *args, **kwargs)

    return wrapper


def prod(ls: list):
    """Return product of elements of the input list.

    if passed list is empty, returns 1.
    """
    n = len(ls)
    if n > 0:
        accum = ls[0]
        if n > 1:
            for item in ls[1:]:
                accum *= item
    else:
        accum = 1
    return accum


# Unfortunately these can't be made with a class factory (just a bit too meta for Python)
# short of doing something awful like https://hg.python.org/cpython/file/b14308524cff/Lib/collections/__init__.py#l232
def _remstate(kwargs):
    if "state" not in kwargs:
        kwargs["state"] = None
    intkwargs = kwargs.copy()
    intkwargs.pop("state")
    return intkwargs


class IntState(int):
    """A subclass of int, with arbitrary, mutable state.

    State is provided to the constructor as the keyword argument
    ``state``. All other arguments will be passed to ``int``
    constructor. Instances should be functionally indistinguishable from
    ``int``.
    """

    def __new__(cls, *args, **kwargs):
        intkwargs = _remstate(kwargs)
        return super(IntState, cls).__new__(cls, *args, **intkwargs)

    def __init__(self, *args, **kwargs):
        self.state = kwargs["state"]

    def __copy__(self):
        return IntState(int(self), state=self.state)

    def __getstate__(self):
        return {"val": int(self), "state": self.state}

    def __setstate__(self, statedict):
        self.state = statedict["state"]


class FloatState(float):
    """A subclass of float, with arbitrary, mutable state.

    State is provided to the constructor as the keyword argument
    ``state``. All other arguments will be passed to ``float``
    constructor. Instances should be functionally indistinguishable from
    ``float``.
    """

    def __new__(cls, *args, **kwargs):
        intkwargs = _remstate(kwargs)
        return super(FloatState, cls).__new__(cls, *args, **intkwargs)

    def __init__(self, *args, **kwargs):
        self.state = kwargs["state"]

    def __copy__(self):
        return FloatState(float(self), state=self.state)

    def __getstate__(self):
        return {"val": float(self), "state": self.state}

    def __setstate__(self, statedict):
        self.state = statedict["state"]


class DecimalState(Decimal):
    """A subclass of ``decimal.Decimal``, with arbitrary, mutable state.

    State is provided to the constructor as the keyword argument
    ``state``. All other arguments will be passed to ``Decimal``
    constructor. Instances should be functionally indistinguishable from
    ``Decimal``.
    """

    def __new__(cls, *args, **kwargs):
        intkwargs = _remstate(kwargs)
        return super(DecimalState, cls).__new__(cls, *args, **intkwargs)

    def __init__(self, *args, **kwargs):
        self.state = kwargs["state"]

    def __copy__(self):
        return DecimalState(Decimal(self), state=self.state)

    def __getstate__(self):
        return {"val": Decimal(self), "state": self.state}

    def __setstate__(self, statedict):
        self.state = statedict["state"]


class StrState(str):
    """A subclass of string, with arbitrary, mutable state.

    State is provided to the constructor as the keyword argument
    ``state``. All other arguments will be passed to ``str``
    constructor. Instances should be functionally indistinguishable from
    ``str``.
    """

    def __new__(cls, *args, **kwargs):
        intkwargs = _remstate(kwargs)
        return super(StrState, cls).__new__(cls, *args, **intkwargs)

    def __init__(self, *args, **kwargs):
        self.state = kwargs["state"]

    def __copy__(self):
        return StrState(str(self), state=self.state)

    def __getstate__(self):
        return {"val": str(self), "state": self.state}

    def __setstate__(self, statedict):
        self.state = statedict["state"]


def load_fasta(fastapath):
    fasta_records = []
    with open(fastapath, "r") as fh:
        for line in fh:
            if line[0] == ">":
                fasta_records.append([line[1:].strip(), ""])
            else:
                fasta_records[-1][-1] += line.strip()
    return dict(fasta_records)
