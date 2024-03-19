#!/bin/bash

# Define the oracle thresholds
oracle_thresholds=(0.1 0.5 1.0)

# Define the beta list
policy_list=('learned' 'ref' 'mix')

# Loop over oracle thresholds
for threshold in "${oracle_thresholds[@]}"
do
    # Loop over oracle thresholds
    for rollin in "${policy_list[@]}"
    do
        # Loop over oracle thresholds
        for rollout in "${policy_list[@]}"
        do
            # Construct the command
            cmd="python lols_variants.py --rollin_type $rollin --rollout_type $rollout --task synthetic --n_epochs 8 --n_types 5 --n_labels 5 --n_hidden 8 --oracle_threshold $threshold"
            
            # Print the command
            echo "Running: $cmd"
            
            # Execute the command
            $cmd
            
            # Print a separator
            echo "---------------------------------------------------"
        done
    done
done