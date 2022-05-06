import ete3
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
        raise RuntimeError(f"{refseqid} not found in loaded MAT")
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

def load_dag(dagname):
    if dagname.split('.')[-1] == 'p':
        with open(dagname, 'rb') as fh:
            dag, refseqtuple = pickle.load(fh)
            dag.refseq = refseqtuple
            return dag
    elif dagname.split('.')[-1] == 'json':
        with open(dagname, 'r') as fh:
            json_dict = json.load(fh)
        return unflatten(json_dict)
    else:
        raise ValueError("Unrecognized file format. Provide either pickled dag (*.p), or json serialized dags (*.json).")

class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, frozendict):
            return dict(obj)
        elif isinstance(obj, frozenset):
            return list(obj)
        return json.JSONEncoder.default(self, obj)

def write_dag(dag, dagpath, sort=False):
    extension = dagpath.split('.')[-1].lower()
    if extension == 'p':
        with open(dagpath, 'wb') as fh:
            fh.write(pickle.dumps((dag, dag.refseq)))
    elif extension == 'json':
        with open(dagpath, 'w') as fh:
            fh.write(json.dumps(flatten(dag, sort_compact_genomes=sort), cls=Encoder))
    else:
        raise ValueError("unrecognized output file extension. Supported extensions are .p and .json.")

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


## TODO this needs to be updated to apply recorded mutations at each node
## to the given reference sequence
# @cli.command('extract-fasta')
# @click.option('-t', '--treepath')
# @click.option('--refseqid')
# @click.option('--refseqpath')
# @click.option('-o', '--fasta_path', default='out.fasta')
# def extract_fasta(treepath, refseqid, refseqpath, fasta_path):
#     refseq = open_refseqpath(refseqpath)
#     tree = process_from_mat(treepath, refseqid, refseq)
#     towrite = ['>' + n.name + '\n' + n.sequence for n in tree.iter_leaves()]
#     with open(fasta_path, 'w') as fh:
#         for line in towrite:
#             print(line, file=fh)

if __name__ == '__main__':
    cli()
