#!/bin/bash

# Define the argument spaces
numElements=(10000 1000000 10000000 50000000) 
numBlocks=(1 8 30 60)
selectivities=(0.1 0.5 0.8)
numThreadsInBlock=(128 256 512 1024)
gallatin=("ON" "OFF")

SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Prepare the CSV file
output_file="$SCRIPT_DIR/Data/GrowingBufferFresh.csv"
first_iteration=true
rm -r $output_file

BUILDDIR=../build
pushd "$BUILDDIR"
for use_gallatin in "${gallatin[@]}"; do
    make clean 
    cmake .. -DINITIAL_CAPACITY="32768" -DUSE_GALLATIN=$use_gallatin
    make GrowingBufferTest
    pushd "$BUILDDIR/lingodbDSA"
    for selectivity in "${selectivities[@]}"; do
        for numRows in "${numElements[@]}"; do
            for blocks in "${numBlocks[@]}"; do
                for threads in "${numThreadsInBlock[@]}"; do
                    if [ "$use_gallatin" == "ON" ] && [ "$threads" -gt 256 ]; then
                        continue
                    fi
                    if $first_iteration; then
                        printHeader=1
                        first_iteration=false
                    else
                        printHeader=0
                    fi
                    command="./GrowingBufferTest $numRows $blocks $threads 5 $selectivity $printHeader >> $output_file"
                    echo "Running: $command"
                    eval "$command"
                done
            done
        done
    done

    popd
done
popd

echo "Running python3 plot_malloc.py --csv-path=$output_file"
python3 plot_malloc.py --csv-path=$output_file