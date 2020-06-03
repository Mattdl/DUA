#!/bin/bash

# Add root path
root_path=./../
echo "root_path="$root_path
export PYTHONPATH=$PYTHONPATH:$root_path
echo "Project root added to Python path"


# Run python file
START_TIME=$(date +%s.%N)

echo "running script with ARGS=$*"
python $1 "${@:2}" || exit
echo 'Python file executed.'

END_TIME=$(date +%s.%N)
TOTAL_TIME=$(echo "$END_TIME - $START_TIME" | bc)
echo "TOTAL Execution TIME = $TOTAL_TIME"
