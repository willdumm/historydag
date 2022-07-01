import ete3
import functools
import random
import dag_pb2 as dpb
from collections import Counter
import bte as mat
from pathlib import Path
import historydag as hdag
from historydag.dag import HistoryDagNode, HistoryDag, EdgeSet
import click
import pickle
from sequence import mutate, distance
from frozendict import frozendict
import json
from typing import NamedTuple
nuc_lookup = {0: "A", 1: "C", 2: "G", 3: "T"}
nuc_codes = {nuc: code for code, nuc in nuc_lookup.items()}

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    """
    A collection of tools for feeding MATs to history DAG.
    """
    pass

def merge(first, others):
    r"""Graph union first history DAG with a generator of others."""
    selforder = first.postorder()
    # hash and __eq__ are implemented for nodes, but we need to retrieve
    # the actual instance that's the same as a proposed node-to-add:
    nodedict = {n: n for n in selforder}
    for other in others:
        if first.refseq != other.refseq:
            raise NotImplementedError("Cannot yet merge history DAGs with different UA node reference sequences.")
        for n in other.postorder():
            if n in nodedict:
                pnode = nodedict[n]
            else:
                pnode = n.node_self()
                nodedict[n] = pnode

            for _, edgeset in n.clades.items():
                for child, weight, _ in edgeset:
                    pnode.add_edge(nodedict[child], weight=weight)

def apply_muts(sequence, muts, reverse=False):
    for mut in muts:
        sequence = mutate(sequence, mut, reverse=reverse)
    return sequence

def build_tree_from_lists(node_info, edges):
    # make ete tree:
    nd = {}
    for name, muts in node_info:
        t = ete3.Tree(name=name)
        t.add_feature("mutations", muts)
        nd[name] = t
    for pname, cname in edges:
        nd[pname].add_child(child=nd[cname])
    return nd[node_info[0][0]]

def build_tree_from_mat(infile):
    mattree = mat.MATree(infile)
    nl = mattree.depth_first_expansion()
    node_info = [(n.id, n.mutations) for n in nl]
    edges = [(n.id, child.id) for n in nl for child in n.children]
    return build_tree_from_lists(node_info, edges)

def process_from_mat(file, refseqid):
    tree = build_tree_from_mat(file)
    # reconstruct root sequence:
    try:
        known_node = tree & refseqid
    except:
        warnings.warn(f"{refseqid} not found in loaded MAT, assuming this sequence is for the root node")
        known_node = tree
        
    known_node.add_feature("mutseq", frozendict())
    while not known_node.is_root():
        known_node.up.add_feature("mutseq", apply_muts(known_node.mutseq, known_node.mutations, reverse=True))
        known_node = known_node.up
    # reconstruct all sequences from root sequence:
    with open('testree.p', 'wb') as fh:
        fh.write(pickle.dumps(tree))
    for node in tree.iter_descendants(strategy='preorder'):
        node.add_feature("mutseq", apply_muts(node.up.mutseq, node.mutations))
    # remove root unifurcations
    while len(tree.children) == 1:
            tree.children[0].delete(prevent_nondicotomic=False)
    # remove unifurcations
    while True:
        to_delete = [node for node in tree.traverse() if len(node.children) == 1]
        if len(to_delete) == 0:
            break
        for node in to_delete:
            node.delete(prevent_nondicotomic=False)
    return tree

def load_MAD_pbdata(filename):
    with open(filename, 'rb') as fh:
        pb_data = dpb.data()
        pb_data.ParseFromString(fh.read())
    return pb_data

def load_dag(dagname):
    last_suffix = dagname.split('.')[-1]
    if last_suffix == 'p':
        with open(dagname, 'rb') as fh:
            dag, refseqtuple = pickle.load(fh)
            dag.refseq = refseqtuple
            return dag
    elif last_suffix == 'json':
        with open(dagname, 'r') as fh:
            json_dict = json.load(fh)
        return unflatten(json_dict)
    elif last_suffix == 'pb':
        return pb_to_dag(load_MAD_pbdata(dagname))
    else:
        raise ValueError("Unrecognized file format. Provide either pickled dag (*.p), or json serialized dags (*.json), or protobuf (*.pb).")

class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, frozendict):
            return dict(obj)
        elif isinstance(obj, frozenset):
            return list(obj)
        return json.JSONEncoder.default(self, obj)

def write_dag(dag, dagpath, sort=False, **kwargs):
    """Write to pickle file, json, or MAD protobuf (`pb`)"""
    extension = str(dagpath).split('.')[-1].lower()
    if extension == 'p':
        with open(dagpath, 'wb') as fh:
            fh.write(pickle.dumps((dag, dag.refseq)))
    elif extension == 'json':
        with open(dagpath, 'w') as fh:
            fh.write(json.dumps(flatten(dag, sort_compact_genomes=sort), cls=Encoder))
    elif extension == 'pb':
        with open(dagpath, 'wb') as fh:
            fh.write(dag_to_mad_pb(dag, **kwargs).SerializeToString())
    else:
        raise ValueError("unrecognized output file extension. Supported extensions are .p and .json.")

def sequence_to_cg(sequence, ref_seq):
    cg = {zero_idx + 1: (old_base, new_base)
          for zero_idx, (old_base, new_base) in enumerate(zip(ref_seq, sequence))
          if old_base != new_base}
    return frozendict(cg)

def cg_to_sequence(cg, ref_seq):
    newseq = []
    newseq = list(ref_seq)
    for idx, (ref_base, newbase) in cg.items():
        if ref_base != newseq[idx - 1]:
            print("cg_to_sequence warning: reference base doesn't match cg reference base")
        newseq[idx - 1] = newbase
    return ''.join(newseq)

def _test_sequence_cg_convert():
    seqs = [
        'AAAA',
        'TAAT',
        'CTGA',
        'TGCA',
    ]
    for refseq in seqs:
        for seq in seqs:
            cg = sequence_to_cg(seq, refseq)
            reseq = cg_to_sequence(cg, refseq)
            if reseq != seq:
                print("\nUnmatched reconstructed sequence:")
                print("ref sequence:", refseq)
                print("sequence:", seq)
                print("cg:", cg)
                print("reconstructed sequence:", reseq)
                assert False

def cg_diff(parent_cg, child_cg):
    """Yields mutations in the format (parent_nuc, child_nuc, sequence_index)
    distinguishing two compact genomes, such that applying the
    resulting mutations to `parent_cg` would yield `child_cg`"""
    keys = set(parent_cg.keys()) | set(child_cg.keys())
    for key in keys:
        if key in parent_cg:
            parent_base = parent_cg[key][1]
        else:
            parent_base = child_cg[key][0]
        if key in child_cg:
            new_base = child_cg[key][1]
        else:
            new_base = parent_cg[key][0]
        if parent_base != new_base:
            yield (parent_base, new_base, key)

def str_mut_from_tups(tup_muts):
    for tup_mut in tup_muts:
        par_nuc, child_nuc, idx = tup_mut
        yield par_nuc + str(idx) + child_nuc

def _test_cg_diff():
    cgs = [
        frozendict({287: ('C', 'G')}),
        frozendict({287: ('C', 'G'), 318: ('C', 'A'), 495: ('C', 'T')}),
        frozendict({287: ('C', 'G'), 80: ('A', 'C'), 257: ('C', 'G'), 591: ('G', 'A')}),
        frozendict({287: ('C', 'G'), 191: ('A', 'G'), 492: ('C', 'G'), 612: ('C', 'G'), 654: ('A', 'G')}),
        frozendict({287: ('C', 'G'), 318: ('C', 'A'), 495: ('C', 'T')}),
    ]
    for parent_cg in cgs:
        for child_cg in cgs:
            assert apply_muts(parent_cg, str_mut_from_tups(cg_diff(parent_cg, child_cg))) == child_cg
            assert all(par_nuc != child_nuc for par_nuc, child_nuc, idx in cg_diff(parent_cg, child_cg))

def string_seq_diff(parent_seq, child_seq):
    return ((par_nuc, child_nuc, zero_idx + 1)
            for zero_idx, (par_nuc, child_nuc) in enumerate(zip(parent_seq, child_seq))
            if par_nuc != child_nuc)



def dag_to_mad_pb(dag, leaf_data_func=None, from_mutseqs=True):
    """convert a DAG with compact genome data on each node, to a MAD protobuf with mutation
    information on edges.

    Args:
        dag: the history DAG to be converted
        leaf_data_func: a function taking a DAG node and returning a string to store
            in the protobuf node_name field `condensed_leaves` of leaf nodes
        from_mutseqs: if True, passed DAG must contain compact genomes in mutseq label
            attributes. Otherwise, DAG must contain string sequences in sequence label attributes.
    """
    if from_mutseqs:
        refseqid, refseq = dag.refseq
        def mut_func(pnode, cnode):
            if pnode.is_root():
                parent_seq = frozendict()
            else:
                parent_seq = pnode.label.mutseq
            return cg_diff(parent_seq, child.label.mutseq)
    else:
        refseq = next(dag.preorder(skip_root=True)).label.sequence
        refseqid = 'unknown_seq_id'
        def mut_func(pnode, cnode):
            if pnode.is_root():
                parent_seq = refseq
            else:
                parent_seq = pnode.label.sequence
            return string_seq_diff(parent_seq, cnode.label.sequence)

    node_dict = {}
    data = dpb.data()
    for idx, node in enumerate(dag.postorder()):
        node_dict[node] = idx
        node_name = data.node_names.add()
        node_name.node_id = idx
        if leaf_data_func is not None:
            if node.is_leaf():
                node_name.condensed_leaves.append(leaf_data_func(node))

    for node in dag.postorder():
        for cladeidx, (clade, edgeset) in enumerate(node.clades.items()):
            for child in edgeset.targets:
                edge = data.edges.add()
                edge.parent_node = node_dict[node]
                edge.parent_clade = cladeidx
                edge.child_node = node_dict[child]
                for par_nuc, child_nuc, idx in mut_func(node, child):
                    mut = edge.edge_mutations.add()
                    mut.position = idx
                    mut.par_nuc = nuc_codes[par_nuc.upper()]
                    mut.mut_nuc.append(nuc_codes[child_nuc.upper()])
    data.reference_seq = refseq
    data.reference_id = refseqid
    return data

def pb_mut_to_str(mut):
    return nuc_lookup[mut.par_nuc] + str(mut.position) + nuc_lookup[mut.mut_nuc[0]]

def pb_to_dag(pbdata):
    """Convert a MAD protobuf to a history DAG with compact genomes in the `mutseq` label attribute"""
    # use HistoryDag.__setstate__ to make this happen
    # all of a node's parent edges
    parent_edges = {node.node_id: [] for node in pbdata.node_names}
    # a list of list of a node's child edges
    child_edges = {node.node_id: [] for node in pbdata.node_names}
    for edge in pbdata.edges:
        parent_edges[edge.child_node].append(edge)
        child_edges[edge.parent_node].append(edge)


    # now each node id is in parent_edges and child_edges as a key,
    # fix the UA node's compact genome (could be done in function but this
    # asserts only one node has no parent edges)
    (ua_node_id, ) = [node_id for node_id, eset in parent_edges.items() if len(eset) == 0]
    @functools.cache
    def get_node_compact_genome(node_id):
        if node_id == ua_node_id:
            return frozendict()
        else:
            edge = parent_edges[node_id][0]
            parent_seq = get_node_compact_genome(edge.parent_node)
            str_mutations = tuple(pb_mut_to_str(mut) for mut in edge.edge_mutations)
            return apply_muts(parent_seq, str_mutations)


    label_list = []
    label_dict = {}

    for node_record in pbdata.node_names:
        cg = get_node_compact_genome(node_record.node_id)
        if cg in label_dict:
            cg_idx = label_dict[cg]
        else:
            cg_idx = len(label_list)
            label_dict[cg] = cg_idx
            label_list.append(cg)

    # now build clade unions by dynamic programming:
    @functools.cache
    def get_clade_union(node_id):
        if len(child_edges[node_id]) == 0:
            # it's a leaf node
            return frozenset({label_dict[get_node_compact_genome(node_id)]})
        else:
            return frozenset({label
                              for child_edge in child_edges[node_id]
                              for label in get_clade_union(child_edge.child_node)})

    def get_child_clades(node_id):
        return frozenset({get_clade_union(child_edge.child_node) for child_edge in child_edges[node_id]})

    # order node_ids in postordering
    visited = set()

    def traverse(node_id):
        visited.add(node_id)
        child_ids = [edge.child_node for edge in child_edges[node_id]]
        if len(child_ids) > 0:
            for child_id in child_ids:
                if not child_id in visited:
                    yield from traverse(child_id)
        yield node_id

    id_postorder = list(traverse(ua_node_id))
    # Start building DAG data
    node_index_d = {node_id: idx for idx, node_id in enumerate(id_postorder)}
    node_list = [(label_dict[get_node_compact_genome(node_id)],
                  get_child_clades(node_id),
                  {"node_id": node_id})
                 for node_id in id_postorder]
    
    edge_list = [(node_index_d[edge.parent_node], node_index_d[edge.child_node], 0, 1)
                 for edge in pbdata.edges]
    #fix label list
    label_list = [(item, ) for item in label_list]
    label_list.append(None)
    ua_node = list(node_list[-1])
    ua_node[0] = len(label_list) - 1
    node_list[-1] = tuple(ua_node)
    dag = hdag.HistoryDag(None)
    dag.__setstate__({"label_fields": ("mutseq",),
                      "label_list": label_list,
                      "node_list": node_list,
                      "edge_list": edge_list,
                      "attr": None})
    dag.refseq = (pbdata.reference_id, pbdata.reference_seq)
    return dag

@cli.command('summarize')
@click.argument('dagpath')
@click.option('-t', '--treedir', help='include parsimony counts for .pb trees in the given directory')
@click.option('-c', '--csv_data', nargs=1, help='print information as csv row, with passed identifier')
@click.option('-p', '--print_header', is_flag=True, help='also print csv header row')
def summarize(dagpath, treedir, csv_data, print_header):
    """output summary information about the provided input file(s)"""
    @hdag.utils.access_nodefield_default("mutseq", default=0)
    def dist(seq1, seq2):
        return distance(seq1, seq2)
    dag = load_dag(dagpath)
    data = []
    data.append(("before_collapse_n_trees", dag.count_trees()))
    data.append(("before_collapse_n_nodes", len(list(dag.preorder()))))
    wc = dag.weight_count(edge_weight_func=dist)
    data.append(("before_collapse_max_pars", max(wc.keys())))
    data.append(("before_collapse_min_pars", min(wc.keys())))
    dag.add_all_allowed_edges()
    dag.trim_optimal_weight(edge_weight_func=dist)
    dag.convert_to_collapsed()
    data.append(("n_trees", dag.count_trees()))
    wc = dag.weight_count(edge_weight_func=dist)
    data.append(("parsimony_score", min(wc.keys())))
    data.append(("avg_node_parents", dag.internal_avg_parents()))
    data.append(("n_nodes", len(list(dag.preorder()))))
    data.append(("n_edges", sum(len(list(n.children())) for n in dag.preorder())))
    data.append(("n_leaves", len([n for n in dag.preorder() if n.is_leaf()])))
    node_counts = dag.weight_count(edge_weight_func=lambda n1, n2: 1)
    data.append(("tree_min_n_nodes", min(node_counts.keys())))
    data.append(("tree_max_n_nodes", max(node_counts.keys())))
    data.append(("n_tree_roots", len(list(dag.dagroot.children()))))
    if treedir:
        treepath = Path(treedir)
        treefiles = list(treepath.glob('*.pb'))
        wc = count_parsimony(treefiles)
        data.append(("n_input_trees", len(treefiles)))
        data.append(("input_min_pars", min(wc.keys())))
        data.append(("input_max_pars", max(wc.keys())))
    if csv_data:
        if print_header:
            print(','.join(['Identifier'] + [str(stat[0]) for stat in data]))
        print(','.join([csv_data] + [str(stat[1]) for stat in data]))
    else:
        for stat in data:
            print(stat[0], stat[1])


@cli.command('merge')
@click.argument('input_dags', nargs=-1, type=click.Path(exists=True))
@click.option('-o', '--outdagpath', default='dag.p', nargs=1, help='output history DAG file')
def merge_dags(input_dags, outdagpath):
    """merge a list of provided history DAGs"""
    if len(input_dags) < 2:
        print('At least two DAGs must be passed for merging')
        return
    dag_gen = (load_dag(dagname) for dagname in input_dags)
    start_dag = next(dag_gen)
    merge(start_dag, dag_gen)
    write_dag(start_dag, outdagpath)

def parsimony(etetree):
    return sum(distance(n.up.mutseq, n.mutseq) for n in etetree.iter_descendants())

def count_parsimony(trees):
    return(Counter(parsimony(process_from_mat(str(file), 'node_1')) for file in trees))

def load_fasta(fastapath):
    fasta_records = []
    current_seq = ''
    with open(fastapath, 'r') as fh:
        for line in fh:
            if line[0] == '>':
                fasta_records.append([line[1:].strip(), ''])
            else:
                fasta_records[-1][-1] += line.strip()
    return dict(fasta_records)


@cli.command('count_parsimony')
@click.argument('trees', nargs=-1, type=click.Path(exists=True))
def count_parsimony_command(trees):
    """Count the parsimony scores of the passed trees"""
    print(count_parsimony(trees))

@cli.command('aggregate')
@click.argument('trees', nargs=-1, type=click.Path(exists=True))
@click.option('-i', '--dagpath', default=None, help='input history DAG')
@click.option('-o', '--outdagpath', default='dag.p', help='output history DAG file')
@click.option('-d', '--outtreedir', default=None, help='directory to move input trees to once added to DAG')
@click.option('--refseq', help='fasta file containing a reference sequence id found in all trees, and that reference sequence')
# @click.option('--refseqid', help='fasta file containing a reference sequence id found in all trees, and that reference sequence')
def aggregate_trees(trees, dagpath, outdagpath, outtreedir, refseq):
    """Aggregate the passed trees (MAT protobufs) into a history DAG"""
    
    ((refseqid, refsequence), ) = load_fasta(refseq).items()

    parsimony_counter = Counter()
    treecounter = []

    def singledag(etetree):
        parsimony_counter.update({parsimony(etetree): 1})
        treecounter.append(1)
        print(len(treecounter))
        dag = hdag.history_dag_from_etes(
            [etetree],
            ["mutseq"],
            attr_func=lambda n: {
                "name": n.name,
            }
        )
        dag.refseq = (refseqid, refsequence)
        return dag

    print(f"loading {len(trees)} trees lazily...")
    ushertrees = (process_from_mat(str(file), refseqid) for file in trees)
    if dagpath is not None:
        print("opening old DAG...")
        olddag = load_dag(dagpath)
    else:
        etetree = next(ushertrees)
        print("Creating DAG from first tree...")
        olddag = singledag(etetree)
    print("Adding trees to history DAG...")
    merge(olddag, (singledag(etetree) for etetree in ushertrees))
    # for idx, etetree in enumerate(ushertrees):
    #     print(idx)
    #     parsimony_counter.update({parsimony(etetree): 1})
    #     newdag = singledag(etetree)
    #     olddag.merge(newdag)
    print("\nParsimony scores of added trees:")
    print(parsimony_counter)
    print("writing new DAG...")
    olddag.refseq = (refseqid, refsequence)
    write_dag(olddag, outdagpath)

    if outtreedir is not None:
        print("moving added treefiles...")
        treepath = Path(outtreedir)
        treepath.mkdir(parents=True, exist_ok=True)
        for path in pathlist:
            path.replace(treepath / path.name)

def collapse_by_mutseq(tree):
    to_collapse = []
    for node in tree.iter_descendants():
        if node.mutseq == node.up.mutseq:
            to_collapse.append(node)
    for node in to_collapse:
        node.delete(prevent_nondicotomic=False)
    return tree

class TreeComparer:
    def __init__(self, tree):
        tree = collapse_by_mutseq(tree.copy())
        for node in tree.traverse(strategy='postorder'):
            seqlist = [val[0] + str(key) + val[1] for key, val in node.mutseq.items()]
            node.name = ':'.join(sorted(seqlist))
            node.children.sort(key=lambda n: n.name)
        self.tree = tree.write(format=8, format_root_node=True)

    def __eq__(self, other):
        return self.tree == other.tree

    def __hash__(self):
        return hash(self.tree)

@cli.command('count-unique')
@click.argument('trees', nargs=-1, type=click.Path(exists=True))
def count_unique(trees):
    """Count the number of unique trees represented by MAT protobufs passed to this function"""
    ushertrees = {TreeComparer(process_from_mat(str(file), 'node_1')) for file in trees}
    print(len(ushertrees))

@cli.command('find-duplicates')
@click.option('-t', '--tree', help='tree in which to search for duplicate samples')
@click.option('-o', '--duplicatefile', default='duplicates.txt', help='output file containing sample names to keep')
@click.option('--refseqid')
def aggregate_trees(tree, duplicatefile, refseqid):
    """Search for duplicate samples in the provided MAT protobuf, and output their names.
    samples not named in the output represent an exhaustive list of unique samples in the provided tree."""
    ushertree = process_from_mat(tree, refseqid)
    try:
        refmutseq = (ushertree & refseqid).mutseq
    except:
        raise RuntimeError(f"{refseqid} not found in loaded tree")
    leaf_d = {n.mutseq: n.name for n in ushertree.iter_leaves()}
    leaf_d.update({refmutseq: refseqid})
    with open(duplicatefile, 'w') as fh:
        for _, name in leaf_d.items():
            print(name, file=fh)
    # rerooted_trees = [process_tree(tree) for tree in ushertrees]
# #### fitting stuff:
# forest = bp.CollapsedForest(rerooted_trees, sequence_counts)



@cli.command('convert')
@click.argument('dag_path')
@click.argument('out_path')
@click.option('-s', '--sort', is_flag=True)
def convert(dag_path, out_path, sort):
    """convert the provided history DAG to the format specified by the extension on `out_path`"""
    write_dag(load_dag(dag_path), out_path, sort=sort)

@cli.command('find-leaf')
@click.argument('infile')
@click.option('-o', '--outfile', default=None)
def find_closest_leaf(infile, outfile):
    """Find a leaf id in the passed MAT protobuf file, and write to outfile, if provided"""
    mattree = mat.MATree(infile)
    nl = mattree.depth_first_expansion()
    ll = [n for n in nl if n.is_leaf()]
    if outfile is not None:
        with open(outfile, 'w') as fh:
            fh.write(ll[0].id)
    click.echo(ll[0].id)

@cli.command('find-leaf-seq')
@click.argument('infile')
@click.argument('reference_seq_fasta')
@click.option('-o', '--outfile', default=None)
@click.option('-i', '--leaf-id', default=None)
@click.option('-f', '--leaf-id-file', default=None)
@click.option('-u', '--filter-unique', is_flag=True)
def find_leaf_sequence(infile, reference_seq_fasta, outfile, leaf_id, leaf_id_file, filter_unique):
    """given a MAT protobuf, its reference sequence, and a sequence ID of interest (or a file containing many sequence IDs)
    output a fasta file containing all of the sequences for the given sequence IDs.

    if filter-unique is True, some sequence IDs may be omitted so that the fasta will not contain duplicate sequences"""
    if leaf_id is not None:
        leaf_ids = {leaf_id}
    else:
        assert leaf_id_file is not None
        with open(leaf_id_file, 'r') as fh:
            leaf_ids = set(line.strip() for line in fh)
    def apply_muts_to_string(sequence, muts, reverse=False):
        for mut in muts:
            oldbase = mut[0]
            newbase = mut[-1]
            # muts seem to be 1-indexed!
            zero_idx = int(mut[1:-1]) - 1
            if reverse:
                newbase, oldbase = oldbase, newbase
            if zero_idx > len(sequence):
                print(zero_idx, len(sequence))
            if sequence[zero_idx] != oldbase:
                print("warning: sequence does not have expected (old) base at site")
            sequence = sequence[: zero_idx] + newbase + sequence[zero_idx + 1 :]
        return sequence
    fasta_data = load_fasta(reference_seq_fasta)
    ((_, refseq_constant), ) = fasta_data.items()

    mattree = mat.MATree(infile)
    nl = mattree.depth_first_expansion()
    seqdict = {nl[0].id: refseq_constant}
    focus_leaves = [node for node in nl if node.id in leaf_ids]

    def compute_node_sequence(treenode):
        if treenode.id in seqdict:
            return seqdict[treenode.id]
        else:
            refseq = compute_node_sequence(treenode.parent)
            this_seq = apply_muts_to_string(refseq, treenode.mutations)
            seqdict[treenode.id] = this_seq
            return this_seq

    outfasta = {}
    visited_set = set()
    for current_node in focus_leaves:
        leaf_id = current_node.id
        node_seq = compute_node_sequence(current_node)
        if filter_unique and node_seq in visited_set:
            continue
        else:
            visited_set.add(node_seq)
            outfasta[leaf_id] = node_seq
    with open(outfile, 'w') as fh:
        for seqid, seq in outfasta.items():
            print('>' + seqid + '\n' + seq, file=fh)


@cli.command('lookup-in-fasta')
@click.argument('fasta')
@click.argument('seqid')
def lookup_in_fasta(fasta, seqid):
    fasta_data = load_fasta(fasta)
    print(fasta_data[seqid])
    return fasta_data[seqid]


@cli.command('test-equal')
@click.argument('dagpath1')
@click.argument('dagpath2')
def test_equal(dagpath1, dagpath2):
    """Test whether the two provided history DAGs are equal, by comparing their JSON serializations"""
    paths = [dagpath1, dagpath2]
    def is_sorted(ls):
        return all(ls[i] <= ls[i+1] for i in range(len(ls) - 1))
    def load_json(path):
        if path.split('.')[-1] == 'p':
            dag = load_dag(path)
            flatdag = flatten(dag, sort_compact_genomes=True)
            if is_sorted(flatdag['compact_genomes']):
                return flatdag
            else:
                raise ValueError("Set sort_compact_genomes flag to True when flattening dags for comparison")
        elif path.split('.')[-1] == 'json':
            with open(path, 'r') as fh:
                return json.load(fh)
        else:
            raise ValueError("Provide either the filenames of pickled dags (*.p), or sorted json serialized dags (*.json).")
    jsons = [load_json(path) for path in paths]
    print(equal_flattened(*jsons))

def equal_flattened(flatdag1, flatdag2):
    """Test whether two flattened history DAGs are equal.
    Provided history DAGs must have been flattened with sort_compact_genomes=True."""
    
    cg_list1 = flatdag1["compact_genomes"]
    cg_list2 = flatdag2["compact_genomes"]

    def get_edge_set(flatdag):
        edgelist = flatdag['edges']
        nodelist = flatdag['nodes']
        
        def convert_flatnode(flatnode):
            label_idx, clade_list = flatnode
            clades = frozenset(frozenset(label_idx_list) for label_idx_list in clade_list)
            return (label_idx, clades)

        nodelist = [convert_flatnode(node) for node in nodelist]
        return frozenset((nodelist[p_idx], nodelist[c_idx]) for p_idx, c_idx, _ in edgelist)

    return cg_list1 == cg_list2 and get_edge_set(flatdag1) == get_edge_set(flatdag2)

def flatten(dag, sort_compact_genomes=False):
    """return a dictionary containing four keys:

    * `refseq` is a list containing the reference sequence id, and the reference sequence (the implied sequence on the UA node)
    * `compact_genome_list` is a list of compact genomes, where each compact genome is a list of nested lists `[seq_idx, [old_base, new_base]]` where `seq_idx` is (1-indexed) nucleotide sequence site. If sort_compact_genomes is True, compact genomes and `compact_genome_list` are sorted.
    * `node_list` is a list of `[label_idx, clade_list]` pairs, where
        * `label_idx` is the index of the node's compact genome in `compact_genome_list`, and
        * `clade_list` is a list of lists of `compact_genome_list` indices, encoding sets of child clades.

    * `edge_list` is a list of triples `[parent_idx, child_idx, clade_idx]`, where
        * `parent_idx` is the index of the edge's parent node in `node_list`,
        * `child_idx` is the index of the edge's child node in `node_list`, and
        * `clade_idx` is the index of the clade in the parent node's `clade_list` from which this edge descends."""
    compact_genome_list = []
    node_list = []
    edge_list = []
    node_indices = {}
    cg_indices = {}

    def get_child_clades(node):
        return [frozenset(cg_indices[label] for label in clade) for clade in node.clades]

    def get_compact_genome(node):
        if node.is_root():
            return []
        else:
            ret = [[idx, list(bases)] for idx, bases in node.label.mutseq.items()]

        if sort_compact_genomes:
            ret.sort()
        return ret

    for node in dag.postorder():
        node_cg = get_compact_genome(node)
        if node.label not in cg_indices:
            cg_indices[node.label] = len(compact_genome_list)
            compact_genome_list.append(node_cg)

    if sort_compact_genomes:
        cgindexlist = sorted(enumerate(compact_genome_list), key=lambda t: t[1])
        compact_genome_list = [cg for _, cg in cgindexlist]
        # the rearrangement is a bijection of indices
        indexmap = {oldidx: newidx for newidx, (oldidx, _) in enumerate(cgindexlist)}
        for key in cg_indices:
            cg_indices[key] = indexmap[cg_indices[key]]

    for node_idx, node in enumerate(dag.postorder()):
        node_indices[id(node)] = node_idx
        node_list.append((cg_indices[node.label], get_child_clades(node)))
        for clade_idx, (clade, eset) in enumerate(node.clades.items()):
            for child in eset.targets:
                edge_list.append((node_idx, node_indices[id(child)], clade_idx))

    return {
        "refseq": dag.refseq,
        "compact_genomes": compact_genome_list,
        "nodes": node_list,
        "edges": edge_list
    }

def unflatten(flat_dag):
    """Takes a dictionary like that returned by flatten, and returns a HistoryDag"""
    compact_genome_list = [frozendict({idx: tuple(bases) for idx, bases in flat_cg}) for flat_cg in flat_dag["compact_genomes"]]
    node_list = flat_dag["nodes"]
    edge_list = flat_dag["edges"]
    Label = NamedTuple("Label", [("mutseq", frozendict)])
    def unpack_cladelabellists(cladelabelsetlist):
        return [frozenset(Label(compact_genome_list[cg_idx]) for cg_idx in idx_clade)
                for idx_clade in cladelabelsetlist]

    node_postorder = []
    # a list of (node, [(clade, eset), ...]) tuples
    for cg_idx, cladelabellists in node_list:
        clade_eset_list = [(clade, EdgeSet()) for clade in unpack_cladelabellists(cladelabellists)]
        if len(clade_eset_list) == 1:
            # This must be the UA node
            label = hdag.utils.UALabel()
        else:
            label = Label(compact_genome_list[cg_idx])
        node = HistoryDagNode(label, dict(clade_eset_list), attr=None)
        node_postorder.append((node, clade_eset_list))

    # adjust UA node label
    node_postorder[-1][0].label = hdag.utils.UALabel()

    # Add edges
    for parent_idx, child_idx, clade_idx in edge_list:
        node_postorder[parent_idx][1][clade_idx][1].add_to_edgeset(node_postorder[child_idx][0])

    # UA node is last in postorder
    dag = HistoryDag(node_postorder[-1][0])
    dag.refseq = tuple(flat_dag["refseq"])
    return dag


@cli.command("make-testcase")
@click.argument("pickled_forest")
@click.argument("outdir")
@click.option("-n", "--num_trees")
@click.option("-s", "--random_seed", default=1)
def make_testcase(pickled_forest, outdir, num_trees, random_seed):
    random.seed(random_seed)
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    with open(pickled_forest, 'rb') as fh:
        forest = pickle.load(fh)
    dag = forest._forest
    trees = [dag.sample() for _ in range(int(num_trees))]
    refseqs = [next(tree.preorder(skip_root=True)).label.sequence for tree in trees]
    assert len(set(refseqs)) == 1
    # These dags have full sequence strings instead of compact genomes, which
    # is why we pass from_mutseqs to write_dag (which passes it on to
    # dag_to_mad_pb)
    for idx, tree in enumerate(trees):
        write_dag(tree, outdir / f"tree_{str(idx)}.pb", from_mutseqs=False)
    newdag = trees[0]
    for tree in trees[1:]:
        newdag.merge(tree)
    write_dag(newdag, outdir / "full_dag.pb", from_mutseqs=False)
    print(f"Test case dag contains {newdag.count_trees()} trees")

@cli.command("change-ref")
@click.argument("in_pb")
@click.argument("out_dag")
@click.argument("new_ref_fasta")
def change_ref(in_pb, out_dag, new_ref_fasta):
    """Change the reference sequence on the provided protobuf DAG, and output
    to a new protobuf file"""""
    pbdata = load_MAD_pbdata(in_pb)
    oldref = pbdata.reference_seq
    ((newrefid, newref),) = load_fasta(new_ref_fasta).items()
    if len(newref) != len(oldref):
        raise ValueError("New reference length does not match old reference length")
    # use new ref as reference sequence
    newstart_cg = sequence_to_cg(oldref, newref)
    # find DAG UA node:
    parent_edges = {}
    child_edges = {}
    for edge in pbdata.edges:
        if edge.parent_node in child_edges:
            child_edges[edge.parent_node].append(edge)
        else:
            child_edges[edge.parent_node] = [edge]
        if edge.child_node in parent_edges:
            parent_edges[edge.child_node].append(edge)
        else:
            parent_edges[edge.child_node] = [edge]
    ua_node_set = set(child_edges.keys()) - set(parent_edges.keys())
    (ua_node_id, ) = set(child_edges.keys()) - set(parent_edges.keys())
    # add newmuts to all edges descending from DAG UA node:
    for edge in child_edges[ua_node_id]:
        child_cg = newstart_cg.copy()
        for mut in edge.edge_mutations:
            mutstring = nuc_lookup[mut.par_nuc] + str(mut.position) + nuc_lookup[mut.mut_nuc[0]]
            sequence.mutate(child_cg, mutstring)
        # clear edge mutations
        while len(edge.edge_mutations) > 0:
            edge.edge_mutations.pop()
        for (old_base, new_base, idx) in cg_diff(frozendict(), child_cg):
            mut = edge.edge_mutations.add()
            mut.position = idx
            mut.par_nuc = nuc_codes[old_base]
            mut.mut_nuc.append(nuc_codes[new_base])
    pbdata.reference_seq = newref
    pbdata.reference_id = newrefid
    # load and export modified pbdata to fix internal mutations' parent bases
    dag = pb_to_dag(pbdata)
    write_dag(dag, out_dag)
    
@cli.command('test')
def _cli_test():
    """Run all functions beginning with _test_ in this script
    (They must take no arguments)"""
    namespace = globals()
    print("Running all test functions:")
    for key, val in namespace.items():
        if '_test_' in key and key[:6] == '_test_':
            try:
                print(key + '\t', end='')
                val()
                print("Passed")
            except:
                print("FAILED")
                raise

@cli.command('get-leaf-ids')
@click.option('-t', '--treepath')
@click.option('-o', '--outfile')
def get_leaf_seqs(treepath, outfile):
    """Write list of leaf sequences to a file"""
    mattree = mat.MATree(treepath)
    with open(outfile, 'w') as fh:
        for node in mattree.depth_first_expansion():
            if node.is_leaf():
                print(node.id, file=fh)


# This duplicates find-leaf-seq I think?
@cli.command('extract-fasta')
@click.option('-t', '--treepath')
@click.option('--refseqfasta')
@click.option('--selected-leaves', default=None)
@click.option('-o', '--fasta-path', default='out.fasta')
@click.option('-u', '--filter-unique', is_flag=True)
def extract_fasta(treepath, refseqfasta, selected_leaves, fasta_path, filter_unique):
    """Extract a fasta alignment for the leaves of a given MAT protobuf"""
    if selected_leaves is not None:
        with open(selected_leaves, 'r') as fh:
            leaves_to_write = {line.strip() for line in fh}
    else:
        leaves_to_write = set()
    ((refseqid, refseq), ) = load_fasta(refseqfasta).items()
    tree = process_from_mat(treepath, refseqid)
    towrite = []
    visited_seqs = {}
    for node in tree.iter_leaves():
        if node.mutseq:
            if node.mutseq in visited_seqs:
                continue
            visited_seqs.add(node.mutseq)
        if selected_leaves is None or node.name in leaves_to_write:
            towrite.append('>' + node.name + '\n' + cg_to_sequence(n.mutseq, refseq))

    with open(fasta_path, 'w') as fh:
        for line in towrite:
            print(line, file=fh)

if __name__ == '__main__':
    cli()
