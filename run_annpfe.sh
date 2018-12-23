#!/bin/bash
#################################################################
#   ANNPFE - Artificial Neural Network Prototyping Front-End    #
#             Final Built, v0.1 - 23/12/2018                    #
#                    Authors/Developer:                         #
#             Marco Chiarelli        @ UNISALENTO & CMCC        #
#               Gabriele Accarino    @ UNISALENTO & CMCC        #
#                      marco_chiarelli@yahoo.it                 #
#                      marco.chiarelli@cmcc.it                  #
#                     gabriele.accarino@cmcc.it                 #
#                     gabriele.accarino@unisalento.it           #
#################################################################


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
RECOMPILE=true
PROGRAM_PRECISION=0
STATIC_DATASET="test_dataset.h"
N_SAMPLES=50000
N_DIM_IN=12
N_DIM_OUT=1
##########################################
RUN_NEURAL=true
DATASET="none.csv"
NEW_DATASET="new_dataset.csv"
TRAINING_LOSS_FILE="training_loss.csv"
VALIDATION_LOSS_FILE="validation_loss.csv"
ERROR_SCORE_FILE="error_score.csv"
DELIMITER=','
TRANSFORMATION=0
TRAP_FP_EXCEPTIONS=true
N_LAG=0
MINIBATCH_SIZE=1
STDNORM=1
TESTING_METHOD=true
NETWORK_FILENAME="kann_net.bin"
PREDICTIONS_FILENAME="predictions.csv"
FEATURE_SCALING_MIN=0.00
FEATURE_SCALING_MAX=1.00
NETWORK_TYPE=0
VERBOSE=true
METRICS=true
N_LAYERS=3
N_NEURONS=120
N_EPOCHS=3000
N_TIMESTEPS=1
LEARNING_RATE=0.001
DROPOUT=0.00
ACTIVATION_FUNCTION=2
BREAK_TRAIN_SCORE=0.00
BREAK_VAL_SCORE=0.00
TRAINING_IDX=70
VALIDATION_IDX=10
LAYER_NORMALIZATION=false
N_THREADS=8
RANDOM_SEED=$RANDOM
##########################################


if $RECOMPILE ; then
	gcc -O3 -march=native -DPROGRAM_PRECISION=$PROGRAM_PRECISION -DDATASET=\"$STATIC_DATASET\" -DN_SAMPLES=$N_SAMPLES -DN_FEATURES=$N_DIM_IN -DN_DIM_OUT=$N_DIM_OUT annpfe.c kann_atyp/kann.c kann_atyp/kautodiff.c -lm -o annpfe
fi


if $TRAIN ; then
	./annpfe $(bool_to_int $RUN_NEURAL) $DATASET $DELIMITER $NEW_DATASET $TRANSFORMATION $(bool_to_int $TRAP_FP_EXCEPTIONS) $TRAINING_LOSS_FILE $VALIDATION_LOSS_FILE $ERROR_SCORE_FILE $N_LAG $MINIBATCH_SIZE $STDNORM $(bool_to_int $TESTING_METHOD) $NETWORK_FILENAME $PREDICTIONS_FILENAME $FEATURE_SCALING_MIN $FEATURE_SCALING_MAX $NETWORK_TYPE $(bool_to_int $VERBOSE) $(bool_to_int $METRICS) $N_LAYERS $N_NEURONS $N_EPOCHS $N_TIMESTEPS $LEARNING_RATE $DROPOUT $ACTIVATION_FUNCTION $BREAK_TRAIN_SCORE $BREAK_VAL_SCORE $TRAINING_IDX $VALIDATION_IDX $(bool_to_int $LAYER_NORMALIZATION) $N_THREADS $RANDOM_SEED
else
	./annpfe $(bool_to_int $RUN_NEURAL) $DATASET $DELIMITER $NEW_DATASET $TRANSFORMATION $(bool_to_int $TRAP_FP_EXCEPTIONS) $TRAINING_LOSS_FILE $VALIDATION_LOSS_FILE $ERROR_SCORE_FILE $N_LAG $MINIBATCH_SIZE $STDNORM $(bool_to_int $TESTING_METHOD) $NETWORK_FILENAME $PREDICTIONS_FILENAME $FEATURE_SCALING_MIN $FEATURE_SCALING_MAX $NETWORK_TYPE $(bool_to_int $VERBOSE) $(bool_to_int $METRICS)
fi
