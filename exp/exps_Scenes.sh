#!/bin/bash

grid="demo"

# CatPrior/TransPrior SETUP
ds="MITindoorscenes"
ds="MITusertransform"

model="vgg11_pretrain"
#model="alexnet_pretrain" # Same hyperparams

############################################
# TRAIN SERVER
train_script="../train/main_train.py"

# FIM/MAS-IMM/LACL + Task Experts (Common training of server models)
args="--method IMM --lr 0.001 --lmbL2trans 0.001 --bs 30 --model_name $model --gridsearch_name $grid --ds_name $ds"

# MAS
args="--method MAS --lr 0.001 --lmb 1 --bs 30 --model_name $model --gridsearch_name $grid --ds_name $ds"

# EWC
args="--method EWC --lr 0.001 --lmb 400 --bs 30 --model_name $model --gridsearch_name $grid --ds_name $ds"

# LWF
args="--method LWF --lr 0.001 --lmb 1 --bs 30 --model_name $model --gridsearch_name $grid --ds_name $ds"

# JOINT
args="--method JOINT --lr 0.001 --bs 30 --model_name $model --gridsearch_name $grid --ds_name $ds"

./run_wrapper.sh "$train_script $args"


############################################
# ADAPT/TEST USERS
test_script="../test/main_test.py"
BN_mode="None" # adaBNTUNE (AdaBN-S), adaBN (AdaBN), None (No additional Adaptive BN)

# FIM-IMM
args="--method IMM --IMM_mode mode --BN_mode $BN_mode --bs 20 --model_name $model --gridsearch_name $grid --ds_name $ds"

# MAS-IMM
args="--method IMM --IMM_mode mode_MAS --BN_mode $BN_mode --bs 20 --model_name $model --gridsearch_name $grid --ds_name $ds"

# MAS-LACL
args="--method LA --LA_mode plain --BN_mode $BN_mode --bs 20 --model_name $model --gridsearch_name $grid --ds_name $ds"

# FIM-LACL
args="--method LA --LA_mode FIM --BN_mode $BN_mode --bs 20 --model_name $model --gridsearch_name $grid --ds_name $ds"

# FT (No AdaBN applicable)
args="--method FT --BN_mode None --bs 20 --model_name $model --gridsearch_name $grid --ds_name $ds"

# Same for other methods
method="JOINT"
#method="EWC"
#method="LWF"
#method="MAS"
args="--method $method --BN_mode $BN_mode --bs 20 --model_name $model --gridsearch_name $grid --ds_name $ds"

./run_wrapper.sh "$test_script $args"



