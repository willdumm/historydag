"""Utility functions and classes for working with HistoryDag objects."""

import ete3
from math import log
from collections import Counter
from functools import wraps
import operator
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
        if not node.is_leaf() and node.up.sequence == node.sequence:
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

    For example, `dag.weight_count(**(parsimony_utils.hamming_distance_countfuncs + make_newickcountfuncs()))`
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
            if not isinstance(names, tuple):
                raise ValueError("``names`` keyword argument expects a tuple.")
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

    def __str__(self) -> str:
        return f"AddFuncDict[{', '.join(str(it) for it in self.names)}]"

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

    def linear_combination(self, coeffs, significant_digits=8):
        """Convert an AddFuncDict implementing a tuple of weights to a linear
        combination of those weights.

        This only works when the weights computed by the AddFuncDict use plain
        `sum` as their accum_func.
        Otherwise, although the resulting AddFuncDict may be usable without errors,
        its behavior is undefined.

        Args:
            coeffs: The coefficients to be multiplied with each weight before summing.
            significant_digits: To combat floating point errors, only this many digits
                after the decimal will be significant in comparisons between weights.

        Returns:
            A new AddFuncDict object which computes the specified linear combination
            of weights.
        """
        n = len(self.names)
        if len(coeffs) != n:
            raise ValueError(
                f"Expected {n} ranking coefficients but received {len(coeffs)}."
            )
        if n == 1:
            raise ValueError(
                "linear_combination should only be called on AddFuncDict"
                " objects which compute more than one weight, e.g."
                " resulting from summing one or more AddFuncDicts."
            )

        def make_floatstate(val):
            return FloatState(round(val, significant_digits), state=val)

        def _lc(weight_tuple):
            return make_floatstate(sum(c * w for c, w in zip(coeffs, weight_tuple)))

        def accum_func(weights):
            return make_floatstate(sum(w.state for w in weights))

        start_func = self["start_func"]
        edge_func = self["edge_weight_func"]

        def new_start_func(n):
            return _lc(start_func(n))

        def new_edge_func(n1, n2):
            return _lc(edge_func(n1, n2))

        return AddFuncDict(
            {
                "start_func": new_start_func,
                "edge_weight_func": new_edge_func,
                "accum_func": accum_func,
            },
            name="("
            + " + ".join(
                str(c) + "(" + name + ")" for c, name in zip(coeffs, self.names)
            )
            + ")",
        )


class HistoryDagFilter:
    def __init__(
        self,
        weight_funcs: AddFuncDict,
        optimal_func,
        ordering_name=None,
        eq_func=operator.eq,
    ):
        self.weight_funcs = weight_funcs
        self.optimal_func = optimal_func
        self.eq_func = eq_func
        end_idx = len(self.weight_funcs.names)
        if ordering_name is None:
            if optimal_func == min:
                self.ordering_names = (("minimum", end_idx),)
            elif optimal_func == max:
                self.ordering_names = (("maximum", end_idx),)
            else:
                self.ordering_names = (("optimal", end_idx),)
        else:
            self.ordering_names = ((ordering_name, end_idx),)

    def __str__(self) -> str:
        start_idx = 0
        descriptions = []
        for ordering_name, end_idx in self.ordering_names:
            these_names = self.weight_funcs.names[start_idx:end_idx]
            if len(these_names) > 1:
                descriptions.append(
                    f"{ordering_name} ({', '.join(str(it) for it in these_names)})"
                )
            else:
                descriptions.append(ordering_name + " " + these_names[0])
            start_idx = end_idx
        return "HistoryDagFilter[" + " then ".join(descriptions) + "]"

    def __getitem__(self, item):
        if item == "optimal_func":
            return self.optimal_func
        elif item == "eq_func":
            return self.eq_func
        else:
            return self.weight_funcs[item]

    # Or should it be &?
    def __add__(self, other):
        if not isinstance(other, HistoryDagFilter):
            raise TypeError(
                f"Can only add HistoryDagFilter to HistoryDagFilter, not f{type(other)}"
            )
        split_idx = len(self.weight_funcs.names)

        def new_optimal_func(weight_tuple_seq):
            weight_tuple_seq = tuple(weight_tuple_seq)
            first_optimal_val = self.optimal_func(
                t[:split_idx] for t in weight_tuple_seq
            )
            second_optimal_val = other.optimal_func(
                t[split_idx:]
                for t in weight_tuple_seq
                if self.eq_func(t[:split_idx], first_optimal_val)
            )
            return first_optimal_val + second_optimal_val

        if self.eq_func == operator.eq and other.eq_func == operator.eq:
            new_eq_func = operator.eq
        else:

            def new_eq_func(a, b):
                return self.eq_func(a[:split_idx], b[:split_idx]) and other.eq_func(
                    a[split_idx:], b[split_idx:]
                )

        ret = HistoryDagFilter(
            self.weight_funcs + other.weight_funcs,
            new_optimal_func,
            eq_func=new_eq_func,
        )
        ret.ordering_names = self.ordering_names + tuple(
            (name, idx + split_idx) for name, idx in other.ordering_names
        )
        return ret

    def keys(self):
        yield from self.weight_funcs.keys()
        yield from ("optimal_func", "eq_func")

    # def with_linear_combination_ordering(self, ranking_coeffs, eq_func=operator.eq):
    #     ranking_coeffs = tuple(ranking_coeffs)
    #     n = len(self.weight_funcs.names)
    #     if len(ranking_coeffs) != n:
    #         raise ValueError(f"Expected {n} ranking coefficients but received {len(ranking_coeffs)}.")

    #     def _lc(weight_tuple):
    #         return sum(c * w for c, w in zip(ranking_coeffs, weight_tuple))

    #     def new_optimal_func(weight_tuple_sequence):
    #         return min(weight_tuple_sequence, key=_lc)

    #     def new_eq_func(weight_tup1, weight_tup2):
    #         return eq_func(_lc(weight_tup1), _lc(weight_tup2))

    #     ret = HistoryDagFilter(self.weight_funcs, new_optimal_func, eq_func=new_eq_func)
    #     new_optimal_func_name = ("minimum ("
    #                              + '+'.join(str(c) + chr(97 + i) for i, c in enumerate(ranking_coeffs))
    #                              + ") for ("
    #                              + ','.join(chr(97 + i) for i in range(n))
    #                              + ") =")
    #     ret.ordering_names = ((new_optimal_func_name, n),)
    #     return ret


node_countfuncs = AddFuncDict(
    {
        "start_func": lambda n: 0,
        "edge_weight_func": lambda n1, n2: 1,
        "accum_func": sum,
    },
    name="NodeCount",
)
"""Provides functions to count the number of nodes in trees.

For use with :meth:`historydag.HistoryDag.weight_count`.
"""


def natural_edge_probability(parent, child):
    """Return the downward-conditional edge probability of the edge from parent
    to child.

    This is defined as 1/n, where n is the number of edges descending
    from the same child clade of ``parent`` as this edge.
    """
    if parent.is_ua_node():
        return 1 / len(list(parent.children()))
    else:
        eset = parent.clades[child.clade_union()]
        return 1 / len(eset.targets)


log_natural_probability_funcs = AddFuncDict(
    {
        "start_func": lambda n: 0,
        "edge_weight_func": lambda n1, n2: log(natural_edge_probability(n1, n2)),
        "accum_func": sum,
    },
    name="LogNaturalProbability",
)
"""Provides functions to count the probabilities of histories in a DAG,
according to the natural distribution induced by the DAG topology."""


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
    N = reference_dag.count_nodes(collapse=True)

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

    We use :meth:`ete3.TreeNode.robinson_foulds` as the reference implementation for
    unrooted RF distance.

    Rooted Robinson-Foulds is simply the cardinality of the symmetric difference of
    the clade sets of two trees, including the root clade.
    Since we include the root clade in this calculation, our rooted RF distance does
    not match the rooted :meth:`ete3.TreeNode.robinson_foulds` implementation.

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

        shift = len(ref_cus)

        def make_intstate(n):
            return IntState(n + shift, state=n)

        def edge_func(n1, n2):
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


def edge_difference_funcs(reference_dag: "HistoryDag", key=lambda n: n):
    """Provides functions to compute the number of edges in a history which do
    not appear in a reference HistoryDag.

    This is useful for taking history-wise intersections of DAGs, or counting
    the number of histories which would appear in such an intersection.

    Args:
        reference_dag: The reference DAG. These functions will count the
            number of edges in a history which do not appear in this DAG.

    Returns:
        :class:`utils.AddFuncDict` object for use with HistoryDag methods for
        trimming and weight counting/annotation.
    """
    edge_set = set(
        (key(n), key(c)) for n in reference_dag.preorder() for c in n.children()
    )

    def edge_weight_func(n1, n2):
        return int((key(n1), key(n2)) not in edge_set)

    return AddFuncDict(
        {
            "start_func": lambda n: 0,
            "edge_weight_func": edge_weight_func,
            "accum_func": sum,
        },
        name="EdgeDifference",
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
