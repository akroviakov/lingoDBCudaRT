#!/bin/bash

# Define the argument spaces
numElements=(1000 10000 1000000 10000000 100000000)
numBlocks=(1 8 32 128)
numColumns=(1 2 5)
numThreadsInBlock=(64 128 256) # 512 1024)
initSizes=(1024) # 4096 16384) #(256 512 1024 2048)
gallatin=("ON" "OFF")

SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Prepare the CSV file
output_file="$SCRIPT_DIR/Data/GrowingBuffer.csv"
first_iteration=true
rm -r $output_file

BUILDDIR=../build
pushd "$BUILDDIR"
for use_gallatin in "${gallatin[@]}"; do
    for init_size in "${initSizes[@]}"; do
        make clean 
        cmake .. -DINITIAL_CAPACITY="$init_size" -DUSE_GALLATIN=$use_gallatin
        make GrowingBufferTest
        pushd "$BUILDDIR/lingodbDSA"
        for columns in "${numColumns[@]}"; do
            for elements in "${numElements[@]}"; do
                for blocks in "${numBlocks[@]}"; do
                    for threads in "${numThreadsInBlock[@]}"; do
                        if $first_iteration; then
                            printHeader=1
                            first_iteration=false
                        else
                            printHeader=0
                        fi
                        command="./GrowingBufferTest $elements $blocks $threads $columns $printHeader >> $output_file"
                        echo "Running: $command"
                        eval "$command"
                    done
                done
            done
        done
        popd
    done
done
popd