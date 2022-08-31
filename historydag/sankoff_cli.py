import parsimony as pars
import click
import pickle


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def _cli():
    """A collection of tools for calculating parsimony scores of newick trees,
    and using them to create a history DAG."""
    pass


@_cli.command("remove-invariants")
@click.argument("in_fasta")
@click.argument("out_fasta")
@click.argument("out_variant_sites")
def _cli_remove_invariant_sites(in_fasta, out_fasta, out_variant_sites):
    fasta_map = pars.load_fasta(in_fasta)
    newfasta, variant_sites = pars.remove_invariant_sites(fasta_map)
    with open(out_fasta, "w") as fh:
        for seqid, seq in newfasta.items():
            print(">" + seqid, file=fh)
            print(seq, file=fh)
    with open(out_variant_sites, "w") as fh:
        print(",".join([str(site) for site in variant_sites]), file=fh)


@_cli.command("build-trees")
@click.argument("treefiles", nargs=-1)
@click.option(
    "-f",
    "--fasta-file",
    required=True,
    help="Filename of a fasta file containing sequences appearing on nodes of newick tree",
)
@click.option(
    "-r",
    "--root-id",
    default=None,
    help="The fasta identifier of the fixed root of provided trees. May be omitted if there is no fixed root sequence.",
)
@click.option(
    "-F",
    "--newick-format",
    default=1,
    help=("Newick format of the provided newick file. See "
          "http://etetoolkit.org/docs/latest/reference/reference_tree.html#ete3.TreeNode"),
)
@click.option(
    "-i",
    "--include-internal-sequences",
    is_flag=True,
    help="include non-leaf node labels, and associated sequences in the fasta file.",
)
@click.option(
    "-g",
    "--gap-as-char",
    is_flag=True,
    help="Treat gap character `-` as a fifth character. Otherwise treated as ambiguous `N`.",
)
@click.option(
    "-a",
    "--preserve-ambiguities",
    is_flag=True,
    help=("Do not disambiguate fully, but rather preserve ambiguities to express "
          "all maximally parsimonious assignments at each site."),
)
@click.option(
    "-o", "--outdir", default=".", help="Directory in which to write pickled trees."
)
@click.option(
    "-c",
    "--clean-trees",
    is_flag=True,
    help="remove cost vectors from tree, resulting in smaller pickled tree files",
)
def _cli_build_trees(
    treefiles,
    fasta_file,
    root_id,
    newick_format,
    include_internal_sequences,
    gap_as_char,
    preserve_ambiguities,
    outdir,
    clean_trees,
):
    trees = pars.build_trees_from_files(
        treefiles,
        fasta_file,
        reference_id=root_id,
        ignore_internal_sequences=(not include_internal_sequences),
    )
    trees = (
        pars.disambiguate(
            tree,
            gap_as_char=gap_as_char,
            min_ambiguities=preserve_ambiguities,
            remove_cvs=clean_trees,
        )
        for tree in trees
    )
    for treefile, tree in zip(treefiles, trees):
        print("saving tree")
        with open(outdir + f"tree_{treefile.split('/')[-1]}.p", "wb") as fh:
            fh.write(pickle.dumps(tree))
        print("saved tree")


@_cli.command("parsimony_scores")
@click.argument("treefiles", nargs=-1)
@click.option(
    "-f",
    "--fasta-file",
    required=True,
    help="Filename of a fasta file containing sequences appearing on nodes of newick tree",
)
@click.option(
    "-r",
    "--root-id",
    default=None,
    help="The fasta identifier of the fixed root of provided trees. May be omitted if there is no fixed root sequence.",
)
@click.option(
    "-F",
    "--newick-format",
    default=1,
    help=("Newick format of the provided newick file. See "
          "http://etetoolkit.org/docs/latest/reference/reference_tree.html#ete3.TreeNode"),
)
@click.option(
    "-i",
    "--include-internal-sequences",
    is_flag=True,
    help="include non-leaf node labels, and associated sequences in the fasta file.",
)
@click.option(
    "-d",
    "--save-to-dag",
    default=None,
    help="Combine loaded and disambiguated trees into a history DAG, and save pickled DAG to provided path.",
)
@click.option(
    "-g",
    "--gap-as-char",
    is_flag=True,
    help="Treat gap character `-` as a fifth character. Otherwise treated as ambiguous `N`.",
)
@click.option(
    "-a",
    "--preserve-ambiguities",
    is_flag=True,
    help=("Do not disambiguate fully, but rather preserve ambiguities to express "
          "all maximally parsimonious assignments at each site."),
)
def _cli_parsimony_score_from_files(
    treefiles,
    fasta_file,
    root_id,
    newick_format,
    include_internal_sequences,
    save_to_dag,
    gap_as_char,
    preserve_ambiguities,
):
    """Print the parsimony score of one or more newick files."""
    pscore_gen = pars.parsimony_scores_from_files(
        treefiles,
        fasta_file,
        reference_id=root_id,
        gap_as_char=gap_as_char,
        ignore_internal_sequences=(not include_internal_sequences),
        remove_invariants=True,
    )

    for score, treepath in zip(pscore_gen, treefiles):
        print(treepath)
        print(score)

    if save_to_dag is not None:
        trees = pars.build_trees_from_files(
            treefiles,
            fasta_file,
            reference_id=root_id,
            ignore_internal_sequences=(not include_internal_sequences),
        )
        trees = (
            pars.disambiguate(
                tree, gap_as_char=gap_as_char, min_ambiguities=preserve_ambiguities
            )
            for tree in trees
        )
        dag = pars.build_dag_from_trees(trees)
        with open(save_to_dag, "wb") as fh:
            fh.write(pickle.dumps(dag))


@_cli.command("summarize-dag")
@click.argument("dagpath")
def _cli_summarize_dag(dagpath):
    """print summary information about the provided history DAG."""
    with open(dagpath, "rb") as fh:
        dag = pickle.load(fh)
    pars.summarize_dag(dag)


if __name__ == "__main__":
    _cli()
