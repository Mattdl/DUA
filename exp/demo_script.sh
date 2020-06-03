#!/bin/bash

grid="demo"
model="vgg11_pretrain"

printf "\n******************************************************\n"
echo "Saving paths defined in config.init:"
cat ../config.init
printf "\n******************************************************\n"

######################
# Prepare dataset
./run_wrapper.sh ../data/MITscenes_prep.py

######################
# TRAIN SERVER

# Train server-side task-specific models
train_script="../train/main_train.py"
args="--method IMM --lmbL2trans 0.001 --bs 20 --model_name $model --gridsearch_name $grid"
./run_wrapper.sh "$train_script $args"

######################
# ADAPT/TEST USERS
BN_mode="None" # adaBNTUNE (AdaBN-S), adaBN (AdaBN), None (No additional Adaptive BN)

# Test IMM server models for users
test_script="../test/main_test.py"
args="--method IMM --IMM_mode mode --BN_mode $BN_mode --bs 20 --model_name $model --gridsearch_name $grid"
./run_wrapper.sh "$test_script $args"

# Test RACL user-specific models
test_script="../test/main_test.py"
args="--method LA --LA_mode plain --BN_mode $BN_mode --bs 20 --model_name $model --gridsearch_name $grid"
./run_wrapper.sh "$test_script $args"

######################
# POSTPROCESS RESULTS
plot_script="../plot/plot_configs/plot_demo.py"
./run_wrapper.sh "$plot_script $args"


