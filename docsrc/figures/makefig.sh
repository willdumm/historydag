#!/bin/zsh
set -eu

for dotfile in *.dot; do
    dot -Tsvg -Gmargin=0 $dotfile -o ${dotfile:r}.svg
done
