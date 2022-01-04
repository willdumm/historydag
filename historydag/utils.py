import ete3
from Bio.Data.IUPACData import ambiguous_dna_values
from collections import Counter
from functools import wraps
from collections import UserDict
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
)

Weight = Any
NTLabel = NamedTuple
Label = Union["UALabel", NTLabel]
F = TypeVar("F", bound=Callable[..., Any])

bases = "AGCT-"
ambiguous_dna_values.update({"?": bases, "-": "-"})


class UALabel:
    """A history DAG universal ancestor (UA) node label"""

    _fields: Any = tuple()

    def __init__(self):
        pass

    def __repr__(self):
        return "UA_node"

    def __hash__(self):
        return 0

    def __eq__(self, other):
        if isinstance(other, UALabel):
            return True
        else:
            return False

    # For typing:
    def __iter__(self):
        raise RuntimeError("Attempted to iterate from dag root UALabel")

    def _asdict(self):
        raise RuntimeError("Attempted to iterate from dag root UALabel")


# ######## Decorators ########
def access_nodefield_default(fieldname: str, default: Any) -> Any:
    """A wrapper to convert a function taking some label field's values as positional
    arguments, to a function taking HistoryDagNodes as positional arguments.

    Args:
        fieldname: The name of the label field whose value the function takes as arguments
        default: A value that should be returned if one of the arguments is the DAG UA node.

    For example, instead of
    `lambda n1, n2: default if n1.is_root() or n2.is_root() else func(n1.label.fieldname, n2.label.fieldname)`,
    this wrapper allows one to write `access_nodefield_default(fieldname, default)(func)`.
    """

    def decorator(func):
        @access_field("label")
        @ignore_ualabel(default)
        @access_field(fieldname)
        @wraps(func)
        def wrapper(*args: Label, **kwargs: Any) -> Weight:
            return func(*args, **kwargs)

        return wrapper

    return decorator


def access_field(fieldname: str) -> Callable[[F], F]:
    """A decorator for conveniently accessing a field in a label.

    To be used instead of something like `lambda l1, l2: func(l1.fieldname, l2.fieldname)`.
    Instead just write `access_field(fieldname)(func)`. Supports arbitrarily many positional
    arguments, which are all expected to be labels (namedtuples) with field `fieldname`."""

    def decorator(func: F):
        @wraps(func)
        def wrapper(*args: Label, **kwargs: Any) -> Any:
            newargs = [getattr(label, fieldname) for label in args]
            return func(*newargs, **kwargs)

        return wrapper

    return decorator


def ignore_ualabel(default: Any) -> Callable[[F], F]:
    """A decorator to return a default value if any argument is a UALabel.
    For instance, to allow distance between two labels to be zero if one is UALabel"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args: Union[Label, UALabel], **kwargs: Any):
            for label in args:
                if isinstance(label, UALabel):
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
            which explodes the whole label at that field."""

    def decorator(
        func: Callable[[Any], Iterable[Any]]
    ) -> Callable[[Label], Iterable[Label]]:
        @wraps(func)
        def wrapfunc(label, *args, **kwargs):
            if isinstance(label, UALabel):
                yield label
            else:
                Label = type(label)
                d = label._asdict()
                for newval in func(d[labelfield], *args, **kwargs):
                    d[labelfield] = newval
                    yield Label(**d)

        return wrapfunc

    return decorator


# ######## Distances and comparisons... ########


def hamming_distance(s1: str, s2: str) -> int:
    """The sitewise sum of base differences between s1 and s2"""
    if len(s1) != len(s2):
        raise ValueError("Sequences must have the same length!")
    return sum(x != y for x, y in zip(s1, s2))


@ignore_ualabel(0)
@access_field("sequence")
def wrapped_hamming_distance(s1, s2) -> int:
    """The sitewise sum of base differences between sequence field contents of two labels.

    Takes two Labels as arguments.

    If l1 or l2 is a UALabel, returns 0."""
    return hamming_distance(s1, s2)


def is_ambiguous(sequence: str) -> bool:
    """Returns whether the provided sequence contains IUPAC nucleotide ambiguity codes"""
    return any(code not in bases for code in sequence)


def cartesian_product(
    optionlist: List[Callable[[], Iterable]], accum=tuple()
) -> Generator[Tuple, None, None]:
    """Takes a list of functions which each return a fresh generator
    on options at that site, and returns a generator yielding tuples, which are
    elements of the cartesian product of the passed generators' contents."""
    if optionlist:
        for term in optionlist[0]():
            yield from cartesian_product(optionlist[1:], accum=(accum + (term,)))
    else:
        yield accum


@explode_label("sequence")
def sequence_resolutions(sequence: str) -> Generator[str, None, None]:
    """Iterates through possible disambiguations of sequence, recursively.
    Recursion-depth-limited by number of ambiguity codes in
    sequence, not sequence length.
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


def hist(c: Counter, samples: int = 1):
    """Pretty prints a counter, normalizing counts using the number of samples,
    passed as the argument `samples`."""
    ls = list(c.items())
    ls.sort()
    print("Weight\t| Frequency\n------------------")
    for weight, freq in ls:
        print(f"{weight}  \t| {freq if samples==1 else freq/samples}")


def is_collapsed(tree: ete3.TreeNode) -> bool:
    """Return whether the provided tree is collapsed, meaning that any edge whose target
    is not a leaf node connects nodes with different sequences."""
    return not any(
        node.sequence == node.up.sequence and not node.is_leaf()
        for node in tree.iter_descendants()
    )


def collapse_adjacent_sequences(tree: ete3.TreeNode) -> ete3.TreeNode:
    """Collapse nonleaf nodes that have the same sequence"""
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
    """Provides a container for function keyword arguments to :meth:`dag.HistoryDag.weight_count`.
    This is primarily useful because it allows instances to be added. Passing the result to `weight_count`
    as keyword arguments counts the weights jointly.

    For example, `dag.weight_count(**(utils.hamming_distance_countfuncs + make_newickcountfuncs()))`
    would return a Counter object in which the weights are tuples containing hamming parsimony and newickstrings."""

    requiredkeys = {"start_func", "edge_weight_func", "accum_func"}

    def __init__(self, initialdata, names="unnamed_weight"):
        self.names = names
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
        if isinstance(self.names, str):

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
                names=(self.names,),
            )
        else:
            return self


hamming_distance_countfuncs = AddFuncDict(
    {
        "start_func": lambda n: 0,
        "edge_weight_func": lambda n1, n2: wrapped_hamming_distance(n1.label, n2.label),
        "accum_func": sum,
    },
    names="HammingParsimony",
)
"""Provides functions necessary to count hamming distance parsimony of trees in a history DAG,
using :meth:`dag.HistoryDag.weight_count`."""


def make_newickcountfuncs(
    name_func=lambda n: "unnamed", features=None, feature_funcs={}, internal_labels=True
):
    """Provides functions necessary to count newick strings of trees in a history DAG,
    using :meth:`dag.HistoryDag.weight_count`.

    Arguments are the same as for :meth:`dag.HistoryDag.to_newick`."""

    def _newicksum(newicks):
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
        if internal_labels or n2.is_leaf():
            return (
                n2._newick_label(
                    name_func=name_func, features=features, feature_funcs=feature_funcs
                )
                + ";"
            )
        else:
            # Right now required to have resulting string well-formed.
            return ";"

    return AddFuncDict(
        {
            "start_func": lambda n: "",
            "edge_weight_func": _newickedgeweight,
            "accum_func": _newicksum,
        },
        names="NewickString",
    )
