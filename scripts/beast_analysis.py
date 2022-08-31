
import click
import time
from bs4 import BeautifulSoup
import pickle
import os

from collections import Counter

import parsimony as pars
import historydag.dag as hdag

import ete3
import Bio.Data.IUPACData
ambig_codes = Bio.Data.IUPACData.ambiguous_dna_values

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    """
    A collection of tools for feeding MATs to history DAG.
    """
    pass


def parsimony(etetree):
    """ hamming parsimony on trees with sequence labels
    """
    return sum(str_distance(n.up.sequence, n.sequence) for n in etetree.iter_descendants())

def str_distance(str1, str2):
    return sum([c1 != c2 for c1, c2 in zip(str1, str2)])

def node_distance(hdag_node1, hdag_node2):
    if hdag_node1.is_root() or hdag_node2.is_root():
        return 0
    return str_distance(hdag_node1.label.sequence, hdag_node2.label.sequence)

@cli.command("extract-fasta")
def extract_fasta():
    """ Given a XML Beast file, extract a fasta file of all the sequences and their ids
    """

    print("=> Extracting fasta file...")
    with open("/fh/fast/matsen_e/whowards/beast_analysis/mccrone-beast-parsimony/clade_13.GTR.xml", "r") as f:
        file = f.read()

    soup = BeautifulSoup(file, 'xml')
    taxa = soup.find_all('sequence')
    id2seq = {}
    for taxon in taxa:
        id2seq[taxon.contents[1]["idref"]] = taxon.contents[2][4:-3]

    with open("raw_seqs.fasta", "w") as f:
        for id, seq in id2seq.items():
            f.write(f">{id}\n{seq}\n\n")
    
    print(f"\tFinding ambiguous columns from {len(taxon.contents[2][4:-3])} bases...")
    invariant_idxs = []
    for idx in range(len(taxon.contents[2][4:-3])):
        col_bases = None
        for id, seq in id2seq.items():
            # Convert unkown chars to `N`
            if seq[idx] not in ambig_codes:
                id2seq[id] = seq[:idx] + "N" + seq[idx+1:]
                assert len(seq) == len(id2seq[id])
                assert id2seq[id][idx] == "N"
                seq = id2seq[id]

            if col_bases is None:
                col_bases = set([char for char in ambig_codes[seq[idx]]])
            else:
                col_bases = col_bases.intersection([char for char in ambig_codes[seq[idx]]])
        
        if len(col_bases) != 0:
            invariant_idxs.append(idx)
    
    print(f"\tRemoving {len(invariant_idxs)} columns...")
    for id, seq in id2seq.items():
        new_seq = ""
        for i in range(1, len(invariant_idxs)-1):
            new_seq += seq[invariant_idxs[i-1]+1:invariant_idxs[i]]

        id2seq[id] = new_seq + seq[invariant_idxs[-1]+1:]

    # Check that all sequences are the same length
    first_seq_len = len(id2seq[id])
    for id, seq in id2seq.items():
        assert len(seq) == first_seq_len

    with open("seqs_new.fasta", "w") as f:
        for id, seq in id2seq.items():
            f.write(f">{id}\n{seq}\n\n")


@cli.command("generate-tree")
@click.argument("start", type=int, default=None)
@click.argument("end", type=int, default=None)
@click.argument("topology_file", default="mccrone-beast-parsimony/reduced.uniq.trees")
@click.argument("fasta_file", default="seqs_new.fasta")
def generate_trees(start, end, topology_file, fasta_file):
    """ Given a newick file that contains a tree on leaf ids in newick format, and a fasta file
    with leaf ids and their sequences, infer internal nodes with sankhoff algorithm. Interval is
    the start and end line of the topology file that you want to process 
    """

    id2seq = pars.load_fasta(fasta_file)

    with open(topology_file, "r") as f:
        topologies = f.readlines()  # NOTE: you could save compute by passing in the portion of this list you're using

    # print("first end:", end, "start:", start)
    if start is None:
        start = 0
    if end is None or end > len(topologies):
        end = len(topologies)
    assert end > start
    # print("new end:", end, "start:", start)

    if not os.path.isdir("disambiguated"):
        os.makedirs("disambiguated")

    start_time = time.time()
    tree_list = []
    for i in range(start, end):
        topology = topologies[i]
        tree = pars.build_tree(topology, id2seq)
        tree = pars.disambiguate(tree)#, min_ambiguities=True) # TODO: Add ambiguities later
        # TODO: Add option to dismbiguate that returns a list of ete trees corresponding to 
        #       all the possible MP trees on the ambiguous tree.
        # Idea: Put ambig tree in DAG, explode, and trim!
        #       --> Do this one tree at a time. However, we might end up with more trees in
        #           the DAG than were actually being expressed by BEAST

        tree_list.append(tree)

        if i % 100 == 1:
            secs = time.time() - start_time
            print(i, "\t", secs, secs/i, "per tree")

    with open(f"disambiguated/trees_{start}-{end}.pkl", "wb") as f:
        pickle.dump(tree_list, f)

    print(f"Disambiguated {end-start} trees!")



@cli.command('aggregate')
@click.argument('trees_file')
@click.option('-o', '--outdagpath', default='dag.p', help='output history DAG file')
@click.option('-a', '--accumulation-data', default=None, help='A file to save accumulation data')
def aggregate_trees(trees_file, outdagpath, accumulation_data):
    """Aggregate the passed trees (lists of ete pickles) into a history DAG"""

    big_dag = pickle.load(open("dag.p", "rb"))
    big_dag.convert_to_collapsed()
    print(big_dag.weight_count())

    
    parsimony_counter = Counter()
    treecounter = []
    def singledag(etetree_list):
        etetree_list = list({TreeComparer(tree): tree for tree in etetree_list}.values())
        for tree in etetree_list:
            # TODO: This is wrong because we're double counting trees
            # ----> Need to keep track of what unique trees we've seen
            parsimony_counter.update({parsimony(tree): 1})
            treecounter.append(1)
            print(len(treecounter))
        
        dag = hdag.history_dag_from_etes(
            etetree_list,
            ["sequence"],
            attr_func=lambda n: {
                "name": n.name,
            }
        )
        dag.convert_to_collapsed() # NOTE: here for efficiency reasons?
        return dag

    print(f"loading {len(os.listdir(trees_file))} tree lists lazily...")
    tree_lists = (pickle.load(open(trees_file + file, "rb")) for file in os.listdir(trees_file))

    # if accumulation_data is not None:
    #     tree_lists = iter({TreeComparer(tree): tree for tree in tree_lists}.values())
    
    tree_list = next(tree_lists)        
    print("Creating DAG from first tree...")
    olddag = singledag(tree_list)

    print(f"Adding {len(os.listdir(trees_file))-1} tree lists with {len(tree_list)} elements to history DAG...")
    merge(olddag, (singledag(tree_list) for tree_list in tree_lists), accumulation_data=accumulation_data, edge_weight_func=node_distance)
    olddag.convert_to_collapsed()
    
    print("\nParsimony scores of added trees:")
    print(olddag.weight_count())
    print("writing new DAG...")
    with open(outdagpath, 'wb') as fh:
        pickle.dump(olddag, fh)

def merge(first, others, accumulation_data=None, resolution=10, edge_weight_func=node_distance):
    r"""Graph union first history DAG with a generator of others."""
    selforder = first.postorder()
    # hash and __eq__ are implemented for nodes, but we need to retrieve
    # the actual instance that's the same as a proposed node-to-add:
    nodedict = {n: n for n in selforder}
    accum_data = []

    def compute_accum_data(dag, edge_weight_func=edge_weight_func):
        tdag = dag.copy()
        tdag.add_all_allowed_edges()
        pscore = tdag.trim_optimal_weight(edge_weight_func=edge_weight_func)
        tdag.convert_to_collapsed()
        ntrees = tdag.count_trees()
        return ntrees, pscore

    if accumulation_data is not None:
        accum_data.append((0, *compute_accum_data(first)))
    for oidx, other in enumerate(others):
        # NOTE: This check interferes with using sequences for hdag labels
        # if first.refseq != other.refseq:
        #     raise NotImplementedError("Cannot yet merge history DAGs with different UA node reference sequences.")
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

@cli.command("explore-trees")
def explore_trees():
    tree_path = "disambiguated/"
    total_unique_trees = set()
    total_pars_hist = Counter()
    print()
    full_dag = pickle.load(open("dag.p", "rb"))
    print("Full dag weight count:")
    print(full_dag.weight_count())
    for tree_list_path in os.listdir(tree_path):
        with open(tree_path + tree_list_path, "rb") as f:
            trees = pickle.load(f)

        tree2tree = {}
        for tree in trees:
            comparer = TreeComparer(tree)
            if comparer not in tree2tree:
                tree2tree[comparer] = []
            tree2tree[comparer].append(tree)

        # for tree, tree_list in tree2tree:
        #     if tree not in total_unique_trees:
        #         total_unique_trees[tree] = tree_list[0]
        
        print("-->", len(total_unique_trees))
        parshist = Counter()
        for tree, tree_list in tree2tree.items():
            parshist[parsimony(tree_list[0])] += 1
            if tree not in total_unique_trees:
                total_pars_hist[parsimony(tree_list[0])] += 1

        print("There are", len(tree2tree), "unique trees in this list")
        total_unique_trees = total_unique_trees.union(*[tree2tree.keys()])

        print("parshist for unique ete trees:")
        print(parshist)

        dag = hdag.history_dag_from_etes(
                trees,
                ["sequence"],
                attr_func=lambda n: {
                    "name": n.name,
                }
            )
        dag.add_all_allowed_edges()
        dag.trim_optimal_weight()
        dag.convert_to_collapsed()
        total_trees = dag.weight_count()
        print("There are", total_trees, "unique trees in the dag")
        print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
    
    print()
    print("Total number of unique trees:", len(total_unique_trees))
    print("Total parsimony histogram:")  # NOTE: not doing anything with ambiguity codes...
    print(total_pars_hist)
    # pars = Counter()
    # for tree in total_unique_trees:
    #     pars[parsimony(tree_list[0])] += 1
    # print(pars)

# Re-written for sequence data
class TreeComparer:
    def __init__(self, tree):
        tree = self.collapse_by_seq(tree.copy())
        for node in tree.traverse(strategy='postorder'):
            node.name = node.sequence
            node.children.sort(key=lambda n: n.name)
        self.tree = tree.write(format=8, format_root_node=True)
    def __eq__(self, other):
        return self.tree == other.tree
    def __hash__(self):
        return hash(self.tree)
    def collapse_by_seq(self, tree):
        to_collapse = []
        for node in tree.iter_descendants():
            if node.sequence == node.up.sequence:
                to_collapse.append(node)
        for node in to_collapse:
            node.delete(prevent_nondicotomic=False)
        return tree

# import subprocess
# @cli.command("driver")
# def driver():
#     """ Shell scripts for parallel aggregation
#     """
#     subprocess.run(["ls", "-la"])

#     process = subprocess.Popen(["ls", "-la"])
#     print("Completed!")

#     process = subprocess.Popen(["ls", "-la"])
#     process.wait()

#     print("Completed!")





if __name__ == '__main__':
    cli()

