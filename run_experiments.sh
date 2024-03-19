#!/bin/bash

# Define the learners
learners=("lols" "aggrevate" "dagger" "searn")

# Define the oracle thresholds
oracle_thresholds=(0.1 0.5 1.0)

# Loop over learners
for learner in "${learners[@]}"
do
    # Loop over oracle thresholds
    for threshold in "${oracle_thresholds[@]}"
    do
        # Construct the command
        cmd="python seq_label.py --learner $learner --task synthetic --n_epochs 8 --n_types 5 --n_labels 5 --n_hidden 8 --oracle_threshold $threshold"
        
        # Print the command
        echo "Running: $cmd"
        
        # Execute the command
        $cmd
        
        # Print a separator
        echo "---------------------------------------------------"
    done
done