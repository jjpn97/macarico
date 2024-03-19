R255 L2S experiments 

1) Plots, results (in /experiments/synthetic/) for synthetic sequence reversal task: ./run_experiments.sh
2) Plots, results (in /experiments/real/) for POS-tagging task: ./run_experiments_real.sh
    - Data used is Universal Dependencies 2.13/ud-treebanks-v2.13/UD_English-ParTUT, standard train/val/test splits
    - https://universaldependencies.org/treebanks/en_partut/index.html

Logic for experiments can be found in seq_label.py

3) Plots, results (in /experiments/lols/) for LOLS variants: ./run_experiments_lols.sh

Logic for experiments can be found in lols_variants.py