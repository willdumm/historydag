from math import floor
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
import warnings
import time
import os
import matplotlib.pyplot as plt
nuc_lookup = {0: "A", 1: "C", 2: "G", 3: "T"}
nuc_codes = {nuc: code for code, nuc in nuc_lookup.items()}

@hdag.utils.access_nodefield_default("mutseq", default=0)
def dist(seq1, seq2):
    return distance(seq1, seq2)

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    """
    A collection of tools for feeding MATs to history DAG.
    """
    pass

def merge(first, others, accumulation_data=None, resolution=10):
    r"""Graph union first history DAG with a generator of others."""
    selforder = first.postorder()
    # hash and __eq__ are implemented for nodes, but we need to retrieve
    # the actual instance that's the same as a proposed node-to-add:
    nodedict = {n: n for n in selforder}
    accum_data = []

    def compute_accum_data(dag):
        tdag = dag.copy()
        tdag.add_all_allowed_edges()
        pscore = tdag.trim_optimal_weight(edge_weight_func=dist)
        tdag.convert_to_collapsed()
        ntrees = tdag.count_trees()
        return ntrees, pscore

    if accumulation_data is not None:
        accum_data.append((0, *compute_accum_data(first)))
    for oidx, other in enumerate(others):
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
        if accumulation_data is not None and oidx % resolution == 0:
            accum_data.append((oidx, *compute_accum_data(first)))
    if accumulation_data is not None:
        accum_data.append((oidx, *compute_accum_data(first)))
    if accumulation_data is not None:
        with open(accumulation_data, 'w') as fh:
            print("iteration,ntrees,parsimonyscore", file=fh)
            for line in accum_data:
                print(','.join([str(it) for it in line]), file=fh)



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

def process_from_mat(file, refseqid, known_node_cg=frozendict({})):
    """
    Given a protobuf file containing a subseted tree, the id of a known node, and that nodes compact genome,
    Return an ete tree annotated with compact genomes.
    """
    tree = build_tree_from_mat(file)
    # reconstruct root sequence
    try:
        known_node = tree & refseqid
    except:
        warnings.warn(f"{refseqid} not found in loaded MAT, assuming this sequence is for the root node using {tree.name}")
        known_node = tree

    known_node.add_feature("mutseq", known_node_cg)
    while not known_node.is_root():
        known_node.up.add_feature("mutseq", apply_muts(known_node.mutseq, known_node.mutations, reverse=True))
        known_node = known_node.up
    
    for node in tree.iter_descendants(strategy='preorder'):
        node.add_feature("mutseq", apply_muts(node.up.mutseq, node.mutations))
    # # remove root unifurcations
    # while len(tree.children) == 1:
    #         tree.children[0].delete(prevent_nondicotomic=False)
    # remove unifurcations
    while True:
        to_delete = [node for node in tree.traverse() if len(node.children) == 1 and not node.is_root()]
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

        def key_func(cladeitem):
            clade, _ = cladeitem
            return sorted(sorted(idx for idx in label.mutseq) for label in clade)
    else:
        refseq = next(dag.preorder(skip_ua_node=True)).label.sequence
        refseqid = 'unknown_seq_id'
        def mut_func(pnode, cnode):
            if pnode.is_root():
                parent_seq = refseq
            else:
                parent_seq = pnode.label.sequence
            return string_seq_diff(parent_seq, cnode.label.sequence)

        def key_func(cladeitem):
            clade, _ = cladeitem
            return sorted(sorted(idx for idx in sequence_to_cg(label.sequence, refseq)) for label in clade)

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
        for cladeidx, (clade, edgeset) in enumerate(sorted(node.clades.items(), key=key_func)):
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
        return tuple(get_clade_union(child_edge.child_node) for child_edge in child_edges[node_id])
        # maybe we need this??
        # return frozenset({get_clade_union(child_edge.child_node) for child_edge in child_edges[node_id]})

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
    dag = hdag.HistoryDag(hdag.dag.UANode(hdag.dag.EdgeSet()))
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
@click.option('-o', '--outfile', help='print output to the provided file')
@click.option('-c', '--csv_data', nargs=1, help='print information as csv row, with passed identifier')
@click.option('-p', '--print_header', is_flag=True, help='also print csv header row')
def summarize(dagpath, treedir, outfile, csv_data, print_header):
    """output summary information about the provided input file(s)"""
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
    data.append(("n_nodes_unlabeled", len(list(dag.unlabel().preorder()))))
    data.append(("n_tree_roots", len(list(dag.dagroot.children()))))
    if treedir:
        treepath = Path(treedir)
        treefiles = list(treepath.glob('*.pb'))
        wc = count_parsimony(treefiles)
        data.append(("n_input_trees", len(treefiles)))
        data.append(("n_unique_inputs", count_unique(treefiles)))
        data.append(("input_min_pars", min(wc.keys())))
        data.append(("input_max_pars", max(wc.keys())))
    outstring = ''
    if csv_data:
        if print_header:
            outstring += ','.join(['Identifier'] + [str(stat[0]) for stat in data]) + '\n'
        outstring += ','.join([csv_data] + [str(stat[1]) for stat in data]) + '\n'
    else:
        for stat in data:
            outstring += str(stat[0]) + str(stat[1]) + '\n'
    if outfile is not None:
        with open(outfile, 'w') as fh:
            print(outstring, file=fh)
    else:
        print(outstring)


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
@click.option('--accumulation-data', default=None, help='A file to save accumulation data')
# @click.option('--refseqid', help='fasta file containing a reference sequence id found in all trees, and that reference sequence')
def aggregate_trees(trees, dagpath, outdagpath, outtreedir, refseq, accumulation_data):
    """Aggregate the passed trees (MAT protobufs) into a history DAG"""
    
    ((refseqid, refsequence), ) = load_fasta(refseq).items()

    parsimony_counter = Counter()
    treecounter = []

    def singledag(etetree):
        #################################################################################     
        # print(etetree.get_ascii(show_internal=True))

        # Re-root ete tree
        node = etetree.search_nodes(name=refseqid)[0]
        etetree = reroot(node)

        ancestral = ete3.TreeNode(name=f"{node.name}_leaf")
        etetree.add_child(ancestral)
        ancestral.add_feature("mutseq", etetree.mutseq)

        # print("New version")
        # print(etetree.get_ascii(show_internal=True))
        #################################################################################

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
    if accumulation_data is not None:
        ushertrees = iter({TreeComparer(tree): tree for tree in ushertrees}.values())
    if dagpath is not None:
        print("opening old DAG...")
        olddag = load_dag(dagpath)
    else:
        etetree = next(ushertrees)        
        print("Creating DAG from first tree...")
        olddag = singledag(etetree)

    print(f"Adding {len(trees)} trees to history DAG...")
    merge(olddag, (singledag(etetree) for etetree in ushertrees), accumulation_data=accumulation_data)
    
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

def reroot(new_root):
    """ Edits the tree that the given node, new_root, is a part of so that it becomes the root.
    Returns pointer to the new root. Also, removes any unifurcations caused by edits.
    """
    node_path = [new_root]
    curr = new_root
    while not curr.is_root():
        node_path.append(curr.up)
        curr = curr.up

    root = node_path[-1]
    delete_root = len(root.children) <= 2
    
    while len(node_path) >= 2:
        curr_node = node_path[-1]
        curr_child = node_path[-2]
        curr_child.detach()
        curr_child.add_child(curr_node)
        node_path = node_path[:-1]
    
    if delete_root:
        root.delete()

    return curr_child



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

def count_unique(trees):
    """Count the number of unique trees represented by MAT protobufs passed to this function"""
    ushertrees = {TreeComparer(process_from_mat(str(file), 'node_1')) for file in trees}
    return len(ushertrees)

@cli.command('count-unique')
@click.argument('trees', nargs=-1, type=click.Path(exists=True))
def cli_count_unique(trees):
    """Count the number of unique trees represented by MAT protobufs passed to this function"""
    print(count_unique(trees))

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

    # Find ancestral node
    mattree = mat.MATree(infile)
    nl = mattree.depth_first_expansion()
    seqdict = {nl[0].id: refseq_constant}   # Assign root it's seq from reference file
    focus_leaves = [node for node in nl if node.id in leaf_ids]

    node2path = {}
    min_len = 99999999
    for node in focus_leaves:
        curr = node
        path = []
        while not curr.id == mattree.root.id:
            path.append(curr)
            curr = curr.parent
        path.append(mattree.root)
        min_len = min(min_len, len(path))
        path.reverse()
        node2path[node] = path

    finished = False
    for i in range(min_len):
        first = None
        for node, path in node2path.items():
            if first is None:
                first = path[i]
            else:
                if first.id != path[i].id:
                    finished = True
                    break
        
        if finished:
            mrca = path[i-1]
            break

    # Testing that this indeed is the MRCA
    assert mrca is not None
    assert set([node.id for node in mattree.get_leaves(nid=mrca.id)]) == set([node.id for node in focus_leaves])
    assert mrca.parent is not None # NOTE: Fails if MRCA is the root node of giant usher tree.

    # Condition our node support on the parent of MRCA: the ancestral node
    ancestral_node = mrca.parent

    def compute_node_sequence(treenode):
        if treenode.id in seqdict:
            return seqdict[treenode.id]
        else:
            refseq = compute_node_sequence(treenode.parent)
            this_seq = apply_muts_to_string(refseq, treenode.mutations)
            seqdict[treenode.id] = this_seq
            return this_seq

    # Write ancestral node to outfile
    with open(outfile, 'w') as fh:
        seqid = ancestral_node.id
        seq = compute_node_sequence(ancestral_node)
        print('>' + seqid + '\n' + seq, file=fh)

    # TODO: You could check here to make sure that there actually are duplicate sequences

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
    with open(outfile, 'a') as fh:
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
        try:
            node = HistoryDagNode(label, dict(clade_eset_list), attr=None)
        except ValueError:
            node = hdag.dag.UANode(clade_eset_list[0][1])
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
    try:
        dag = forest._forest
    except AttributeError:
        dag = forest
    trees = [dag.sample() for _ in range(int(num_trees))]
    refseqs = [next(tree.preorder(skip_ua_node=True)).label.sequence for tree in trees]
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

@cli.command('check-parsimony')
@click.argument('cladedir')
@click.argument('refseqid')
@click.argument('outnewick')
def check_parsimony(cladedir, refseqid, outnewick):
    """check parsimony of a tree sampled from the DAG, and output a newick string"""
    cladedir = Path(cladedir)
    with open(cladedir / 'trimmed_dag.pkl', 'rb') as fh:
        dag = pickle.load(fh)
    tree = dag.sample()
    print("trimmed DAG parsimony score from sampled tree, from Python")
    @hdag.utils.access_nodefield_default("mutseq", default=0)
    def dist(seq1, seq2):
        return distance(seq1, seq2)
    print(tree.optimal_weight_annotate(edge_weight_func=dist))
    with open(outnewick, 'w') as fh:
        print(tree.to_newick(name_func=lambda n: n.attr['name'] if n.is_leaf() else '', features=[]), file=fh)
    with open(cladedir / 'annotated_modified_toi.pk', 'rb') as fh:
        toi = pickle.load(fh)
    print("modified TOI parsimony score, from Python")
    print(parsimony(toi))
    with open(cladedir / 'tmpdir/modified_toi.nk', 'w') as fh:
        print(toi.write(format_root_node=True, format=9), file=fh)

    original_toi = process_from_mat(str(cladedir / 'subset_mat.pb'), 'node_1')
    print("original TOI parsimony score, from Python")
    print(parsimony(original_toi))


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

def distance_between_nodes(n1, n2):
    """ Returns the hamming distance between two hdag nodes that represent their sequences with
    collapsed genomes
    """
    if n1.is_root() or n2.is_root():
        return 0
    else:
        return distance(n1.label.mutseq, n2.label.mutseq)
 

@cli.command("annotate-support")
@click.argument("subset_mat_file")
@click.argument("reference_file")
@click.argument("clade_dir")
@click.argument("unique_seqs_file")
def annotate_support(subset_mat_file, reference_file, clade_dir, unique_seqs_file):
    """ Given a protobuf file containing the TOI, a reference file containing the ancestral node
    id and sequence, the directory to output files to, and a file containing all unique leaf node
    sequences and ids, write the tree of interest with annotated nodes in newick format.
    """
    import time
    begin = time.time()
    with open(reference_file, "r") as f:
        ancestral_seq_id = f.readline()[1:-1]   # First line of fasta file
        ancestral_seq = f.readline()[:-1]       # Remove newline character
    
    with open(unique_seqs_file) as f:
        found_leaf = False
        for line in f.readlines():
            if line.startswith(">") and not line.startswith(">node"):   # First leaf node id
                leaf_id = line[1:-1]
                found_leaf = True
            elif found_leaf:
                leaf_seq = line[:-1]
                break
    
    print("Processing MAT into ete tree...")

    cg = {}
    for i, (anc_char, leaf_char) in enumerate(zip(ancestral_seq, leaf_seq)):
        if anc_char != leaf_char:
            cg[i+1] = (anc_char, leaf_char)
    tree = process_from_mat(subset_mat_file, leaf_id, known_node_cg = frozendict(cg))

    # Edit TOI to conform to our USHER generated trees (i.e., ancestral seq for root and on leaf)
    ancestral_node = ete3.TreeNode(name=ancestral_seq_id)
    ancestral_node.add_feature("mutseq", frozendict({}))
    ancestral_leaf = ete3.TreeNode(name=f"{ancestral_seq_id}_leaf")
    ancestral_leaf.add_feature("mutseq", frozendict({}))
    ancestral_node.add_child(tree)
    ancestral_node.add_child(ancestral_leaf)
    tree = ancestral_node   # Reroot tree on the ancestral node

    print("Summary of ToI (ete) info before edits:")
    print("\tparsimony score", parsimony(tree))
    print(f"\tcontains {sum([1 for _ in tree.traverse()])} nodes")
    print()

    # TODO: Delete this method soon... Other one works fine
    # Find all unique leaf nodes
    with open(unique_seqs_file) as f:
        unique_seqs = set([ancestral_leaf.name])
        for line in f.readlines():
            if line.startswith(">"):
                unique_seqs.add(line[1:-1]) # Remove '>' and '\n'
        print(f"Using {len(unique_seqs)} unique seqs")
    mutseqs = set()
    unique_seqs_new = set()
    for node in tree.traverse():
        if node.is_leaf() and node.mutseq not in mutseqs:
            unique_seqs_new.add(node.name)
            mutseqs.add(node.mutseq)
    print(f"There are {len(unique_seqs_new)} unique seqs for leaves")
    unique_seqs = unique_seqs_new # TODO: Shouldn't have to do this...

    # NOTE: DEBUG =========================================
    #   ==> Discovered that the set of mutseqs is the same, but
    #       the unique_seqs set differs by the ancestral node
    #
    # mutseqs_old = set()
    # for node in tree.traverse():
    #     if node.name in unique_seqs:
    #         mutseqs_old.add(node.mutseq)

    # print("unique_seqs are equal:", unique_seqs == unique_seqs_new)
    # print(len(unique_seqs), len(unique_seqs_new))
    # print(len(unique_seqs.intersection(unique_seqs_new)))
    # for seq in unique_seqs:
    #     if seq not in unique_seqs_new:
    #         print("Not in new way of computing", seq)
    # print("mut_seqs are equal:", mutseqs_old == mutseqs)
    # print(len(mutseqs_old), len(mutseqs))
    # print(len(mutseqs_old.intersection(mutseqs)))
    # =====================================================
    
    # Delete non-unique leaf nodes
    to_delete = []
    for node in tree.traverse():
        if node.is_leaf() and node.name not in unique_seqs:
            to_delete.append(node)
    print(f"\tDeleting {len(to_delete)} duplicate leaves")
    for node in to_delete:
        node.delete(prevent_nondicotomic=False)

    # Remove unifurcations
    to_delete = []
    for node in tree.traverse():
        if len(node.children) == 1:
            to_delete.append(node)
    print(f"\tDeleting {len(to_delete)} unifurcacious nodes")
    for node in to_delete:
        node.delete(prevent_nondicotomic=False)        
    print()

    toidag = hdag.history_dag_from_etes(
        [tree],
        ["mutseq"],
        attr_func=lambda n: {
            "name": n.name,
        }
    )
    hist = toidag.weight_count(edge_weight_func=distance_between_nodes)
    print("=> TOI (hdag) has parsimony score", hist)

    dag_path = f"{clade_dir}/full_dag.p"
    with open(dag_path, 'rb') as fh:
        dag_stuff = pickle.load(fh)
        refseqid, refsequence = dag_stuff[1]
        dag = dag_stuff[0]

    hist = dag.weight_count(edge_weight_func=distance_between_nodes)
    print("=> Parsimony scores of trees in big hdag")
    print(hist)

    # Ensure all trees are on the same leaves
    toidag_leaves = set()
    for node in toidag.preorder():
        if node.is_leaf():
            toidag_leaves.add(node.label.mutseq)
    dag_leaves = set()
    for node in dag.preorder():
        if node.is_leaf():
            dag_leaves.add(node.label.mutseq)

    assert toidag_leaves == dag_leaves

    verbose = False # NOTE: Very slow to count all the trees!

    print("Merging tree dag into hdag...")
    print(f"\t{dag.count_trees()} trees before merge")
    if verbose:
        hist = dag.weight_count(edge_weight_func=distance_between_nodes)
        for k, v in hist.items():
            print(k, "\t", v)
        print()

        histogram_path = clade_dir + "/parsimony_hists"
        if not os.path.isdir(histogram_path):
            os.makedirs(histogram_path)
        with open(clade_dir + "/parsimony_hists/before_merge.pkl", "wb") as f:
            pickle.dump(hist, f)

    dag.merge([toidag])
    if verbose:
        print(f"\t{dag.count_trees()} trees after merge")
        hist = dag.weight_count(edge_weight_func=distance_between_nodes)
        for k, v in hist.items():
            print(k, "\t", v)
        print()
        with open(clade_dir + "/parsimony_hists/after_merge.pkl", "wb") as f:
            pickle.dump(hist, f)
    dag.add_all_allowed_edges()
    if False:   # NOTE: For many clades this is infeasible to compute in a reasonable amount of time
        print(f"\t{dag.count_trees()} after adding edges")
        hist = dag.weight_count(edge_weight_func=distance_between_nodes)
        for k, v in hist.items():
            print(k, "\t", v)
        print()
        with open(clade_dir + "/parsimony_hists/after_all_edges.pkl", "wb") as f:
            pickle.dump(hist, f)
    
    # NOTE: Trimming to optimal weight often removes ToI nodes
    trim = True
    if trim:
        dag.trim_optimal_weight(edge_weight_func=distance_between_nodes)
        dag.recompute_parents()
        with open(clade_dir + "/trimmed_dag.pkl", "wb") as f:
            pickle.dump(dag, f)
        hist = dag.weight_count(edge_weight_func=distance_between_nodes)
        print(f"\t{dag.count_trees()} trees after trim")
        if verbose:
            with open(clade_dir + "/parsimony_hists/after_trim.pkl", "wb") as f:
                pickle.dump(hist, f)


    # Annotate the original version of the tree:
    raw_tree = process_from_mat(subset_mat_file, leaf_id, known_node_cg = frozendict(cg))
    print("Annotating raw tree...")
    raw_tree = annotate_ete(dag, raw_tree)
    raw_tree.write(outfile=f"{clade_dir}/annotated_toi.nh")
    with open(f"{clade_dir}/annotated_toi.pk", "wb") as f:
        pickle.dump(raw_tree, f)

    print("Annotating modified tree...")
    tree = annotate_ete(dag, tree)
    tree.write(outfile=f"{clade_dir}/annotated_modified_toi.nh")
    with open(f"{clade_dir}/annotated_modified_toi.pk", "wb") as f:
        pickle.dump(tree, f)
    
    for node in tree.traverse():
        if not node.is_leaf():
            print(node.support, "\troot:", node.is_root())

    print(f"\t--- Annotation took {time.time() - begin} seconds ---")
    num_uncertain = 0
    total_leaves = 0
    support_vals = {}
    for node in tree.traverse():
        if not node.is_leaf():
            total_leaves += 1
            if node.support < 1:
                num_uncertain += 1
                if node.support not in support_vals:
                    support_vals[node.support] = 0
                support_vals[node.support] += 1
    
    print("Uncertainty:", num_uncertain, "/", total_leaves)
    for sup, count in support_vals.items():
        print(f"{sup:4g}\t{count}")

    return tree

def annotate_ete(dag, tree):
    start_time = time.time()
    print("Counting nodes!...")
    node2count = dag.count_nodes()
    print(f"\t--- Counting took {time.time() - start_time} seconds ---")

    total_trees = dag.count_trees()
    
    print(f"hDAG contains {total_trees} trees")

    # Set of mutation dicts -> support
    clade2support = {}

    # Collapse hdag node counts by clade union
    for node, count in node2count.items():
        if node.is_leaf():
            clade_union = frozenset([node.label.mutseq])
        else:
            clade_union = frozenset([label.mutseq for label in node.clade_union()])

        if clade_union not in clade2support:
            clade2support[clade_union] = 0
        clade2support[clade_union] += count / total_trees

    # Annotate ete tree nodes with their support values
    node2clade = {}
    for ete_node in tree.traverse("postorder"):    # Visit children first
        if ete_node.is_leaf():
            clade_union = frozenset([ete_node.mutseq])
            node2clade[ete_node] = clade_union
        else:
            clade_union = frozenset().union(*[node2clade[child] for child in ete_node.children])
            node2clade[ete_node] = clade_union

        if clade_union not in clade2support:
            ete_node.support = 1 / total_trees
        else:
            ete_node.support = clade2support[clade_union]
    
    return tree


def load_toi(subset_mat_file, reference_file, unique_seqs_file, annotated_ete_file=None):
    """
    Loads the TOI (modified with ancestral leaf) from the given subset_mat_file
    """
    with open(reference_file, "r") as f:
        ancestral_seq_id = f.readline()[1:-1]   # First line of fasta file
        ancestral_seq = f.readline()[:-1]       # Remove newline character

    if annotated_ete_file is None:   # Create the modified TOI from scratch        
        with open(unique_seqs_file) as f:
            found_leaf = False
            for line in f.readlines():
                if line.startswith(">") and not line.startswith(">node"):   # First leaf node id
                    leaf_id = line[1:-1]
                    found_leaf = True
                elif found_leaf:
                    leaf_seq = line[:-1]
                    break
        cg = {}
        for i, (anc_char, leaf_char) in enumerate(zip(ancestral_seq, leaf_seq)):
            if anc_char != leaf_char:
                cg[i+1] = (anc_char, leaf_char)
        
        tree = process_from_mat(subset_mat_file, leaf_id, known_node_cg = frozendict(cg))

    else:   # Load TOI from newick
        tree = ete3.Tree(annotated_ete_file)


    # Edit TOI to conform to our USHER generated trees (i.e., ancestral seq for root and on leaf)
    ancestral_node = ete3.TreeNode(name=ancestral_seq_id)
    ancestral_node.add_feature("mutseq", frozendict({}))
    ancestral_leaf = ete3.TreeNode(name=f"{ancestral_seq_id}_leaf")
    ancestral_leaf.add_feature("mutseq", frozendict({}))
    ancestral_node.add_child(tree)
    ancestral_node.add_child(ancestral_leaf)
    tree = ancestral_node   # Reroot tree on the ancestral node

    return tree

### Explore annotated support values ##############################################################
# TODO: This should be in a separate file

@cli.command("explore-annotation")
@click.argument("subset_mat_file")
@click.argument("reference_file")
@click.argument("clade_dir")
@click.argument("unique_seqs_file")
def explore_annotation(subset_mat_file, reference_file, clade_dir, unique_seqs_file):
    """ Plots intra-clade statistics
    """
    BIGGER_SIZE = 12
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plots_path = clade_dir + "/plots"
    if not os.path.isdir(plots_path):
        os.makedirs(plots_path)

    annotated_ete_file = clade_dir + "/annotated_toi.nh"
    print("Loading annotated TOI as ete...")
    tree_nw = load_toi(subset_mat_file, reference_file, unique_seqs_file, annotated_ete_file=annotated_ete_file)
    with open(f"{clade_dir}/annotated_modified_toi.pk", "rb") as f:
        tree = pickle.load(f)

    # TODO: Scatter plot of node size vs support for uncertain nodes
    num_uncertain = 0
    total_non_leaves = 0
    support_vals = {}
    for node in tree.traverse():
        if not node.is_leaf():
            total_non_leaves += 1
            if node.support < 1:
                num_uncertain += 1
                if node.support not in support_vals:
                    support_vals[node.support] = 0
                support_vals[node.support] += 1
    
    print("Uncertainty:", num_uncertain, "/", total_non_leaves)
    for sup, count in support_vals.items():
        print(f"{sup:4g}\t{count}")

    name2size = {}
    name2dist = {}
    for node in tree.traverse('postorder'):
        if node.is_leaf():
            name2size[node.name] = 1
            name2dist[node.name] = 0
        else:
            size = 0
            dist = -1 #10**18 # max int
            for child in node.children:
                size += name2size[child.name]
                dist = max(dist, name2dist[child.name])
            name2size[node.name] = size
            name2dist[node.name] = dist+1


    sups = []
    sizes = []
    dists = []
    num_children_uncertain = []
    num_children = []
    for node in tree.traverse():
        if not node.is_leaf():
            num_children.append(len(node.children))
            if node.support < 1:
                sups.append(node.support)
                sizes.append(name2size[node.name])
                dists.append(name2dist[node.name])
                num_children_uncertain.append(len(node.children))


    plt.scatter(sups, sizes, alpha=0.4)
    plt.ylabel("Support")
    plt.xlabel("Size")
    plt.title(f"Support vs Node Size for {num_uncertain} / {total_non_leaves} Uncertain Nodes")
    plt.savefig(plots_path + "/supp_vs_size_scatter.png")
    plt.clf()

    plt.scatter(sups, dists, alpha=0.4)
    plt.ylabel("Support")
    plt.xlabel("Shortest Distance to Leaf")
    plt.title(f"Support vs Node Height for {num_uncertain} / {total_non_leaves} Uncertain Nodes")
    plt.savefig(plots_path + "/supp_vs_height_scatter.png")
    plt.clf()
    
    plt.hist(sups)
    plt.ylabel("Count")
    plt.xlabel("Support")
    plt.title(f"Support Values for {num_uncertain} / {total_non_leaves} Uncertain Nodes")
    plt.savefig(plots_path + "/support_hist.png")
    plt.clf()

    dag_path = f"{clade_dir}/full_dag.p"
    with open(dag_path, 'rb') as fh:
        dag_stuff = pickle.load(fh)
        dag = dag_stuff[0]

    print("Sampling from DAG...")
    dag_samples = []
    for i in range(100):
        if i % 10 == 0:
            print(i)
        sample_history = dag.sample()
        sample_tree = sample_history.to_ete()
        dag_samples.append(sample_tree)
    print()

    def parsimony(etetree):
        return sum(distance(n.up.mutseq, n.mutseq) for n in etetree.iter_descendants())

    toi_parsiomny = parsimony(tree)
    toi_leaves = set()
    for node in tree.traverse():
        if not node.is_leaf():
            if node.support < 1:
                sups.append(node.support)
    
    plt.hist(sups)
    plt.yscale("log")
    plt.ylabel("count")
    plt.xlabel("support")
    plt.title(f"Support Values for {num_uncertain} / {total_non_leaves} Uncertain Nodes")
    plt.savefig(plots_path + "/support_hist.png")
    plt.clf()

    data = [num_children_uncertain, num_children]
    labels = ["Uncertain", "All Nodes"]
    plt.hist(data, label=labels)
    plt.ylabel("Number of Nodes")
    plt.yscale("log")
    plt.xlabel("Number of Children")
    plt.title(f"Multifurcation Distribution")
    plt.legend(loc='upper right')
    plt.savefig(plots_path + "/multifurcation_hist.png")
    plt.clf()

    # We'd like to be able to aggregate this sort of information across all clades.
    # How can we normalize it so that each clade is on the same playing field?
    #   --> Normalize by tree height.
    #   --> Color by clade
    #   --> Bucket by support value

    fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True)
    fig.set_size_inches(8, 10)
    ax1.scatter(sups, sizes, alpha=0.4)
    ax1.set_ylabel("Size")
    ax1.set_yscale("log")
    ax2.scatter(sups, dists, alpha=0.4)
    ax2.set_ylabel("Distance")
    ax3.hist(sups)
    ax3.set_ylabel("Node Count")
    ax3.set_xlabel("Support")
    plt.tight_layout()
    plt.savefig(plots_path + "/location_vs_support.png")
    plt.clf()


    # NOTE: Ideally we'd have the same size figures for each of these plots
    plt.rc('axes', titlesize= 15)     # fontsize of the axes title
    plt.rc('xtick', labelsize= 15)    # fontsize of the tick labels
    plt.rc('ytick', labelsize= 23)    # fontsize of the tick labels

    # Plot the parsimony histograms
    hist_dir = clade_dir + "/parsimony_hists"
    if len(os.listdir(hist_dir)) < 4:
        return
    # Reorder the file paths
    histograms = ['before_merge.pkl', 'after_merge.pkl', 'after_all_edges.pkl', 'after_trim.pkl']

    fig, axes = plt.subplots(1,4, sharey=True, sharex=True)
    fig.set_size_inches(14, 8)
    largest_ymax = 0
    for ax, file_name in zip(axes, histograms):
        hist_path = hist_dir + "/" + file_name
        print("Reading", hist_path)
        with open(hist_path, "rb") as f:
            pars2count = pickle.load(f)
        
        max_key = -1
        min_key = 10**18
        for key, val in pars2count.items():
            max_key = max(max_key, key)
            min_key = min(min_key, key)

            largest_ymax = max(largest_ymax, val) 
        
        for i in range(min_key-2, max_key+2):
            if i not in pars2count:
                pars2count[i] = 0.1

        ax.bar(list(pars2count.keys()), pars2count.values())
        title = ' '.join(hist_path.split("/")[-1][:-4].split('_'))
        ax.set_title(title)
        ax.set_ylim(ymin = 0.1, ymax=float(largest_ymax*10))
        ax.set_yscale("log")

    plt.savefig(hist_dir + "/clade_reconstruction.png")
    plt.clf()



@cli.command("plot-hists")
def plot_hists():
    """ Aggregates support statistics across clades using a couple histograms
    """

    plot_path = "clades/plots"
    if not os.path.isdir(plot_path):
        os.makedirs(plot_path)
    
    with open("focus_clades.txt", "r") as f:
        clade_list = f.readlines()

    percent_uncertain = []
    sample_uncertain = {}
    best_pars_diffs = [[], [], []]
    heights = [[], [], [], []] # Buckets: |0-0.33| _ |0.33-0.66| _ |0.66-0.99| _ | 1 | 

    for clade in clade_list:
        clade_dir = "clades/" + clade[:-1] # Remove newline char

        print(clade_dir)
        print("Loading annotated TOI as ete...")
        with open(f"{clade_dir}/annotated_modified_toi.pk", "rb") as f:
            tree = pickle.load(f)
       
        dag_path = f"{clade_dir}/trimmed_dag.pkl"
        with open(dag_path, 'rb') as f:
            dag = pickle.load(f)
        
        num_samples = 0
        for i in range(num_samples):
            ete_tree = dag.sample().to_ete()
            ete_tree = annotate_ete(dag, ete_tree)

            num_certain = 0
            total_non_leaves = 0
            for node in ete_tree.traverse():
                if not node.is_leaf():
                    total_non_leaves += 1
                    if node.support == 1:
                        num_certain += 1

            if i not in sample_uncertain:
                sample_uncertain[i] = []
            sample_uncertain[i].append(num_certain / total_non_leaves)
    
        num_uncertain = 0
        total_non_leaves = 0
        support_vals = {}
        for node in tree.traverse():
            if not node.is_leaf():
                total_non_leaves += 1
                if node.support < 1:
                    num_uncertain += 1
                    if node.support not in support_vals:
                        support_vals[node.support] = 0
                    support_vals[node.support] += 1

        percent_uncertain.append((total_non_leaves - num_uncertain) / total_non_leaves)
        
        print("Uncertainty:", num_uncertain, "/", total_non_leaves)
        for sup, count in support_vals.items():
            print(f"{sup:4g}\t{count}")

        hist_dir = clade_dir + "/parsimony_hists"
        histograms = ['before_merge.pkl', 'after_merge.pkl', 'after_trim.pkl']
        stage = 2 # Looking at parsimony compared to FINAL tree set
        
        toi_parsimony = parsimony(tree)

        for diffs_list, file_name in zip(best_pars_diffs, histograms):
            hist_path = hist_dir + "/" + file_name
            print("\t=> Reading", hist_path)
            with open(hist_path, "rb") as f:
                pars2count = pickle.load(f)

            best_pars = 10 ** 18
            for pars_val in pars2count.keys():
                best_pars = min(best_pars, pars_val)

            diffs_list.append(toi_parsimony / best_pars)

        print("Pars diff is", best_pars_diffs[stage][-1])

        name2size = {}
        name2dist = {}
        for node in tree.traverse('postorder'):
            if node.is_leaf():
                name2size[node.name] = 1
                name2dist[node.name] = 0
            else:
                size = 0
                dist = -1
                for child in node.children:
                    size += name2size[child.name]
                    dist = max(dist, name2dist[child.name])
                name2size[node.name] = size
                name2dist[node.name] = dist+1
        
        tree_height = name2dist[tree.name] # Height at root
        for node in tree.traverse():
            # Only looking at distribution of uncertain nodes
            if node.support <= 1:
                idx = floor(node.support * 3)
                heights[idx].append(name2dist[node.name] / tree_height)
            else:
                # TODO: Look into why this is happen
                print("\n")
                print("\t FOUND NODE WITH SUPPORT:\t", node.support)
                print("\n")
        
        for height in heights:
            print(f"\tlen {len(height)} nums: {height[-3:]}")

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
    data = heights[0]
    ax1.hist(data)
    ax1.set_ylabel("Support: 0.00-0.33")
    data = heights[1]
    ax2.hist(data)
    ax2.set_ylabel("Support: 0.33-0.66")
    data = heights[2]
    ax3.hist(data)
    ax3.set_ylabel("Support: 0.66-0.99")
    data = heights[3]
    ax4.hist(data)
    ax4.set_ylabel("Support = 1.0")
    fig.set_size_inches(8, 14)
    plt.savefig(plot_path + "/height_for_support_buckets.png")
    plt.tight_layout()
    plt.clf()

    plt.figure(figsize=(8, 6)) 
    data = heights
    labels = ["0.00-0.33", "0.33-0.66", "0.66-0.99", "1"]
    plt.hist(data, label=labels, density=True)
    plt.ylabel("Normalized # Nodes")
    # plt.yscale('log')
    plt.xlabel("Height (% total)")
    plt.title(f"Height distribution for different support buckets")
    plt.legend(loc='upper right')
    plt.savefig(plot_path + "/heights.png")
    plt.clf()
    
    for clade, pars_diff in zip(clade_list, best_pars_diffs[stage]):
        print(clade, "\t", pars_diff)
       
    # Put percent uncertain into a histogram and title it nicely and stuff...
    plt.hist(percent_uncertain)
    plt.xlabel("Percentage")
    plt.ylabel("Number of Clades")
    plt.title("Percentage of TOI (non-leaf) Nodes that are Certain")

    plt.savefig(plot_path + "/certainty_hisogram.png")
    plt.clf()

    for i, percent_uncertain in sample_uncertain.items():
        plt.hist(percent_uncertain)
        plt.xlabel("Percentage")
        plt.ylabel("Number of Clades")
        plt.title("Percentage of Sampled Tree (non-leaf) Nodes that are Certain")

        plt.savefig(plot_path + f"/certainty_hisogram_{i}.png")
        plt.clf()


    fig, ax = plt.subplots(1)
    titles = ["before merge"]#, "after merge", "after trim"]
    
    # ax.hist([np.log(val) for val in best_pars_diffs[0]], bins=50)#bins=1+10.0**np.arange(-5, 1))
    data = best_pars_diffs[stage]
    ax.hist(data) #, bins=range(min(data), max(data) + binwidth, binwidth))
    ax.set_title("Parsimony Difference Between TOI and Best Pars Tree")
    ax.set_ylabel("Number of Clades")
    ax.set_xlabel("Difference in Parsimony: pars(TOI) / pars(Other)")
    # ax.set_xscale('log')

    plt.savefig(plot_path + "/toi_difference.png")
    plt.clf()


### TODO: ERemove these and add to DAG
from math import log
def most_supported_trees(dag):
    """ Trims the DAG to only express the trees that have the highest support.
    """
    node2count = dag.count_nodes()        
    total_trees = dag.count_trees()
    clade2support = {}
    for node, count in node2count.items():
        if node.clade_union() not in clade2support:
            clade2support[node.clade_union()] = 0
        clade2support[node.clade_union()] += count / total_trees

    dag.trim_optimal_weight(
        start_func= lambda n: 0,
        edge_weight_func= lambda n1, n2: log(clade2support[n2.clade_union()]),
        accum_func= lambda weights: sum([w for w in weights]),
        optimal_func=max,
        #TODO: Add equality_func that checks first 5 decimals
    )
    dag.recompute_parents()
    
    return dag.dagroot._dp_data

def support_count(dag, clade2support=None):
    if clade2support is None:
        node2count = dag.count_nodes()        
        total_trees = dag.count_trees()
        clade2support = {}
        for node, count in node2count.items():
            if node.clade_union() not in clade2support:
                clade2support[node.clade_union()] = 0
            clade2support[node.clade_union()] += count / total_trees
    
    support_hist = Counter()
    for node in dag.postorder():
        support_hist[clade2support[node.clade_union()]] += 1
    return support_hist


@cli.command("explore-most-supported-trees")
def explore_most_supported_trees():
    clade_dir = "/fh/fast/matsen_e/whowards/usher-clade-reconstructions/clades"
    with open("focus_clades.txt", "r") as f:
        clade_list = f.readlines()

    num_nodes_before = []
    num_nodes_after = []
    toi_sups = []
    best_sups = []

    for clade in clade_list:
        print(clade)
        dag_path = f"{clade_dir}/{clade[:-1]}/trimmed_dag.pkl"
        with open(dag_path, 'rb') as fh:
            dag = pickle.load(fh)
        
        toi_path = f"{clade_dir}/{clade[:-1]}/annotated_modified_toi.pk"
        with open(toi_path, 'rb') as fh:
            toi = pickle.load(fh)

        # Compute the support for the toi
        toi_support = 0
        for node in toi.traverse():
            if not node.is_leaf():
                toi_support += log(node.support)
        toi_sups.append(toi_support)

        node2count = dag.count_nodes()        
        total_trees = dag.count_trees()
        clade2support = {}
        for node, count in node2count.items():
            if node.clade_union() not in clade2support:
                clade2support[node.clade_union()] = 0
            clade2support[node.clade_union()] += count / total_trees
        
        num_trees_before = dag.count_trees()
        support_hist = support_count(dag, clade2support)
        best_sup = most_supported_trees(dag)
        best_sups.append(best_sup)
        num_trees_after = dag.count_trees()

        print(f"\t{num_trees_before} -> {num_trees_after}\tsup: {best_sup}\ttoi_sup: {toi_support}")
        # print("\tsupport hist:", support_hist)
        bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        plt.hist(support_hist, bins=bins)
        plt.hist(support_count(dag, clade2support), bins=bins)
        plt.legend(["Before Trim", "After Trim"])
        plt.xlabel("Support Values")
        plt.ylabel("Number of Nodes in DAG")
        plt.title(f"Support distribution before/after trimming to best trees")
        plt.savefig(f"{clade_dir}/{clade[:-1]}/support_for_full_dag.png")
        plt.clf()

        num_nodes_before.append(num_trees_before)
        num_nodes_after.append(num_trees_after)

    # TODO: Scatter plot of number of trees in hDAG before and after
    plt.scatter(num_nodes_before, num_nodes_after)
    plt.xlabel("Number trees before trim")
    plt.ylabel("Number trees after trim")
    plt.xscale('log')
    plt.savefig(f"{clade_dir}/plots/num_trees_before_after_scatter_.png")
    plt.clf()

    plt.scatter(num_nodes_before, num_nodes_after)
    plt.xlabel("Number trees before trim")
    plt.ylabel("Number trees after trim")
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f"{clade_dir}/plots/num_trees_before_after_scatter.png")
    plt.clf()

    plt.scatter(toi_sups, best_sups)
    plt.plot(plt.ylim(), plt.ylim(), ls="--", c=".3")
    plt.xlabel("TOI ln-support")
    plt.ylabel("Best ln-support")
    plt.savefig(f"{clade_dir}/plots/TOI_vs_best_support.png")
    plt.clf()

    diffs = [log(toi_sup / best_sup) for toi_sup, best_sup in zip(toi_sups, best_sups)]
    plt.hist(diffs)#, bins=[1+i for i in range(0, 350, 25)])
    # plt.xticks([1+i for i in range(0, 350, 50)])
    plt.xlabel("TOI ln-support / Best Tree ln-support")
    plt.ylabel("Number of Clades")
    plt.savefig(f"{clade_dir}/plots/TOI_vs_best_diffs.png")
    plt.clf()


@cli.command("get-clade-stats")
def select_clades():
    """ Given a path to a MAT, save the subtree, the leaves, and the ancestral sequences for each 
    """

    working_dir = "/fh/fast/matsen_e/whowards/usher-clade-reconstructions/clade_selection"

    # Load big MAT tree
    bigmat_file = "clade_selection/data/mat_unique_leaves.pb" #"public-latest.all.masked.pb.gz"
    refseq_file = "public-latest-reference.fasta"

    print("Loading big MAT...", end="\n\t")
    mattree = mat.MATree(bigmat_file)
    node_list = mattree.breadth_first_expansion(reverse=True)

    print(f"Gathering stats from big MAT with {len(node_list)} nodes...")
    # Gather stats for nodes at each level
    id2height = {}
    id2clade = {}
    id2num_muts = {}
    leaves = []

    for i, node in enumerate(node_list):
        height = -1
        if node.is_leaf():
            clade = 1
        else:
            clade = 0
            for child in node.children:
                height = max(height, id2height[child.id])
                clade += id2clade[child.id]
        
        id2height[node.id] = height+1
        id2clade[node.id] = clade
        id2num_muts[node.id] = len(node.mutations)

        if i % 5000 == 0:
            print(i, "height:", height+1, "\tclade len:", clade, "\tnum muts:", len(node.mutations), "\tnum leaves:", len(leaves))

        if node.is_leaf():
            leaves.append(node.id)

    with open(working_dir + "/usher_stats.pkl", "wb") as f:
        stats = {
            "id2height": id2height,
            "id2clade": id2clade,
            "id2num_muts": id2num_muts,
            "root_leaves": leaves
        }
        pickle.dump(stats, f)

@cli.command("analyze-clades")
def analyze_clades():
    # TODO:
    # - Figure out why the maximum clade size at a level decreases when you merge clade sets.
    # --> Because you're joining and summing over height. NOT actually summing over nodes in your partition

    working_dir = "/fh/fast/matsen_e/whowards/usher-clade-reconstructions/clade_selection"

    with open(working_dir + "/usher_stats.pkl", "rb") as f:
        stats = pickle.load(f)
    
    id2height = stats["id2height"]
    id2clade = stats["id2clade"]
    id2num_muts = stats["id2num_muts"]

    height2ids = {}
    height2range = {}
    for id, height in id2height.items():
        if height not in height2ids:
            height2ids[height] = []
        height2ids[height].append(id)

        if height not in height2range:
            height2range[height] = (id2clade[id], id2clade[id])
        else:
            height2range[height] = (min(height2range[height][0], id2clade[id]), max(height2range[height][1], id2clade[id]))
            

    for height, ids in height2ids.items():
        print(height, "\t", len(ids), f"\t\tclades: [{height2range[height][0]}, {height2range[height][1]}]")

    height2cladesizes = {}
    for height, ids in height2ids.items():
        height2cladesizes[height] = []
        for id in ids:
            height2cladesizes[height].append(id2clade[id])
    
    # Put the axes in a flattened list
    fig, axes_tup = plt.subplots(2,3)
    axes = []
    for row in axes_tup:
        for col in row:
            axes.append(col)

    for ax, height in zip(axes, range(5, 11)):
        ax.hist(height2cladesizes[height])
        ax.set_yscale('log')
        ax.set_title(f"height {height}")

    plt.tight_layout()
    plt.savefig(working_dir + "/output/clade_distrib.png")
    plt.clf()

    muts_dist = Counter()
    for id, muts in id2num_muts.items():
        muts_dist[muts] += 1
    print("Distribution of num mutations on each edge:")
    print(muts_dist)

    heights = []
    num_muts = []
    clade_sizes = []
    for id, height in id2height.items():
        heights.append(height)
        num_muts.append(id2num_muts[id])
        clade_sizes.append(id2clade[id])
    
    plt.scatter(heights, num_muts, alpha=0.1)
    plt.xlabel("height")
    plt.ylabel("num mutations")
    plt.savefig(working_dir + "/output/height_vs_num_muts.png")
    plt.clf()

    plt.scatter(clade_sizes, num_muts, alpha=0.1)
    plt.xlabel("clade sizes")
    plt.xscale('log')
    plt.ylabel("num mutations")
    plt.savefig(working_dir + "/output/clade_size_vs_num_muts.png")
    plt.clf()




    


    

    



if __name__ == '__main__':
    cli()
