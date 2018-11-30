#!/bin/bash

function bool_to_int()
{
	if $1 ; then
		echo 1
	else
		echo 0
	fi
}

TRAIN=true

##########################################

RECOMPILE=false
DATASET="test_dataset.h"
N_SAMPLES=50000
N_DIM_IN=12
N_DIM_OUT=1

##########################################
N_LAG=0
STDNORM=1
TESTING_METHOD=true
NETWORK_FILENAME="kann_net.bin"
PREDICTIONS_FILENAME="predictions.csv"
FEATURE_SCALING_MIN=0.00
FEATURE_SCALING_MAX=1.00
NETWORK_TYPE=0
N_LAYERS=3
N_NEURONS=120
N_EPOCHS=3000
MINIBATCH_SIZE=1
N_TIMESTEPS=1
LEARNING_RATE=0.001
DROPOUT=0.00
TRAINING_IDX=70
VALIDATION_IDX=10
LAYER_NORMALIZATION=true
N_THREADS=8
RANDOM_SEED=$RANDOM
##########################################


if $RECOMPILE ; then
	gcc -O3 -march=native -DDATASET=\"$DATASET\" -DN_SAMPLES=$N_SAMPLES -DN_FEATURES=$N_DIM_IN -DN_DIM_OUT=$N_DIM_OUT annpfe.c kann/kann.c kann/kautodiff.c -lm -o annpfe
fi

if $TRAIN ; then
	./annpfe $N_LAG $STDNORM $(bool_to_int $TESTING_METHOD) $NETWORK_FILENAME $PREDICTIONS_FILENAME $FEATURE_SCALING_MIN $FEATURE_SCALING_MAX $NETWORK_TYPE $N_LAYERS $N_NEURONS $N_EPOCHS $MINIBATCH_SIZE $TIMESTEPS $LEARNING_RATE $DROPOUT $TRAINING_IDX $VALIDATION_IDX $(bool_to_int $LAYER_NORMALIZATION) $N_THREADS $RANDOM_SEED
else
	./annpfe $N_LAG $STDNORM $(bool_to_int $TESTING_METHOD) $NETWORK_FILENAME $PREDICTIONS_FILENAME $FEATURE_SCALING_MIN $FEATURE_SCALING_MAX $NETWORK_TYPE
fi
