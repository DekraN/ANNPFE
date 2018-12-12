# ANNPFE
<b>Artificial Neural Network Prototyping Front-End</b>

COMPILE:
<code>
  gcc -O3 -march=native annpfe.c kann/kann.c kann/kautodiff.c -lm -o annpfe
</code>

Simply compile a dataset like a C array as in the example (first columns are the outputs, other are inputs), or format it in a file named <code>input_file</code> whose lines are <code>delimiter</code>-separated, then open run_annpfe.sh and manually edit bash input variables or directly launch <b><i>annpfe</i></b> from command line:

# TRAINING
<code>
  ./annpfe [input_file] [delimiter] [n_lag] [minibatch_size] [normal-standard-ization_method] [testing_method] [network_filename] [predictions_filename] [feature_scaling_min] [feature_scaling_max] [net_type] [verbose] [metrics] [n_h_layers] [n_h_neurons] [max_epoch] [timesteps] [learning_rate] [dropout] [activation_function] [break_train_score] [break_val_score] [training_idx] [validation_idx] [want_layer_normalization] [n_threads] [random_seed]
</code>

For example,

<code>
./annpfe none , 0 2 0 1 kann_net.bin predictions.csv 0.00 1.00 0 1 1 120 1000 16 0.001 0.8 1 0.0 0.0 60 20 0 8 1542718307
</code>

# TEST
<code>
./annpfe [input_file] [delimiter] [n_lag] [minibatch_size] [normal-standard-ization_method] [testing_method] [network_filename] [predictions_filename] [feature_scaling_min] [feature_scaling_max] [net_type] [metrics]
</code>

For example,
<code>
  ./annpfe none , 0 0 0 kann_net.bin predictions.csv 0.00 1.00 0 1 1
</code>
