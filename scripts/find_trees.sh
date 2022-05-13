#!/bin/bash
set -eu

print_help()
{
    echo "DESCRIPTION:"
    echo "This script takes a fasta file as input, and uses Usher and matOptimize to \
search for maximally parsimonious trees on those sequences. The output is \
a directory containing many MAT protobufs, each a tree on the same set of \
sequences. The first line of the fasta is expected to contain the name of the \
sequence that will be used as a reference (the sequence of the root node of \
the trees output). \

The total number of trees found will be, at maximum, the value passed to '-M' times the value passed to '-d' \
times the value passed to -n."
    echo
    echo "Script requires Usher, matOptimize, and faToVcf."
    echo
    echo "SYNTAX:    find_trees.sh -f INPUT_FASTA [-h|o|M|d]"
    echo
    echo "OPTIONS:"
    echo "-f    Provide an input fasta file (required)"
    echo "-n    Specify the number of times to start from scratch rebuilding the tree (default 1)"
    echo "-o    Specify an output directory for created trees"
    echo "          (default a directory called 'output_trees' in the current directory)"
    echo "-M    Specify the maximum number of alternative placements"
    echo "          to be kept when building initial trees. (default 200)"
    echo "-d    Specify the number of tree moves to apply when drifting (default 4)"
    echo "-h    Print this help message and exit"
    echo
}

OUTDIR=output_trees
FASTA=""
MAX_ALTERNATE_PLACEMENTS=200
DRIFTING_MOVES=4
NRUNS=1

while getopts "n:f:ho:M:d:" option; do
    case $option in
        n)
            NRUNS=$OPTARG;;
        f)
            FASTA=$OPTARG;;
        h)
            print_help
            exit;;
        o)
            OUTDIR=$OPTARG;;
        M)
            MAX_ALTERNATE_PLACEMENTS=$OPTARG;;
        d)
            DRIFTING_MOVES=$OPTARG;;
    esac
done

[ ! -n "${FASTA}" ] && { echo "You must provide an input fasta with '-f'"; exit 0; }
[ -e $OUTDIR ] && { echo "$OUTDIR already exists! Exiting."; exit 0; }
REFID=$(head -1 $FASTA)
REFID=$(echo "${REFID:1}" | xargs)

mkdir $OUTDIR
TMPDIR=$OUTDIR/tmp
mkdir $TMPDIR
VCF=$OUTDIR/out.vcf

faToVcf $FASTA $VCF -ref=$REFID
echo "($REFID)1;" > $TMPDIR/starttree.nh
for ((run=1;i<=NRUNS;i++)); do
    echo optimize $run
    # place samples in the tree in up to MAX_ALTERNATE_PLACEMENTS different
    # ways
    usher -t $TMPDIR/starttree.nh -v $VCF -o $TMPDIR/mat.pb -d $TMPDIR/ushertree/ -M $MAX_ALTERNATE_PLACEMENTS
    for intree in $TMPDIR/ushertree/*.nh; do
        # Optimize each resulting tree DRIFTING_MOVES times
        usher -t $intree -v $VCF -o $TMPDIR/mat.pb
        for optrun in {1..4}; do
            echo optimize $run
            matOptimize -i $TMPDIR/mat.pb -o $TMPDIR/opt_mat.pb -d $DRIFTING_MOVES
            mv $TMPDIR/opt_mat.pb $TMPDIR/mat.pb
            cp $TMPDIR/mat.pb $OUTDIR/${run}$(basename $intree)${optrun}.pb
        done
        rm $TMPDIR/mat.pb
        rm -f *intermediate*
    done
    rm -f *intermediate*
    rm -f $TMPDIR/ushertree/*.nh
    rm -f $TMPDIR/ushertree/*.txt
    rm -f $TMPDIR/ushertree/*.tsv
done
rm -rf $TMPDIR
