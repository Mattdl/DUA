#!/bin/bash

grid="demo"

# Numbers SETUP
model="MLP_cl_100_100"
ds="numbers_nb" # Numbers dataset -> use MLP

############################################
# TRAIN SERVER
train_script="../train/main_train.py"
epochs="10"
bs="20"

# FIM/MAS-IMM/LACL + Task Experts (Common training of server models)
args="--method IMM --lr 0.001 --lmbL2trans 0.001 --bs $bs --epochs $epochs --model_name $model --gridsearch_name $grid --ds_name $ds"

# MAS
args="--method MAS --lr 0.001 --lmb 1 --bs $bs --epochs $epochs --model_name $model --gridsearch_name $grid --ds_name $ds"

# EWC
args="--method EWC --lr 0.001 --lmb 400 --bs $bs --epochs $epochs --model_name $model --gridsearch_name $grid --ds_name $ds"

# LWF
args="--method LWF --lr 0.001 --lmb 1 --bs $bs --epochs $epochs --model_name $model --gridsearch_name $grid --ds_name $ds"

# JOINT
args="--method JOINT --lr 0.001 --bs $bs --epochs $epochs --model_name $model --gridsearch_name $grid --ds_name $ds"

./run_wrapper.sh "$train_script $args"


############################################
# ADAPT/TEST USERS
test_script="../test/main_test.py"

# FIM-IMM
args="--method IMM --IMM_mode mode --bs 20 --model_name $model --gridsearch_name $grid --ds_name $ds"

# MAS-IMM
args="--method IMM --IMM_mode mode_MAS --bs 20 --model_name $model --gridsearch_name $grid --ds_name $ds"

# MAS-LACL
args="--method LA --LA_mode plain --bs 20 --model_name $model --gridsearch_name $grid --ds_name $ds"

# FIM-LACL
args="--method LA --LA_mode FIM --bs 20 --model_name $model --gridsearch_name $grid --ds_name $ds"

# FT (No AdaBN applicable)
args="--method FT --bs 20 --model_name $model --gridsearch_name $grid --ds_name $ds"

# Same for other methods
method="JOINT"
#method="EWC"
#method="LWF"
#method="MAS"
args="--method $method --bs 20 --model_name $model --gridsearch_name $grid --ds_name $ds"

./run_wrapper.sh "$test_script $args"



