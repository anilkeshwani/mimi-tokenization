#!/usr/bin/env bash

first_block=11
last_block=599
split='train'

# Launcher
for i in $(seq -f "%0${#last_block}g" $first_block $last_block); do
    echo "Tokenizing split ${split}, block ${i} of ${last_block} (zero indexing)"
    tmux new-window -d -t "mimi-${split}" -n "mimi_$i" -- bash -c "
        srun --partition a6000 --time=01:00:00 --qos=cpu \
            conda run --live-stream -n mimi \
                /mnt/scratch-artemis/anilkeshwani/mimi/mls.py --split ${split} ${i} && 
                bash" # stay open (tmux window)
    sleep 0.5
done
