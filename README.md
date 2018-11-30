# ANNPFE
<b>Artificial Neural Network Prototyping Front-End</b>

COMPILE: gcc -O3 -march=native annpfe.c kann/kann.c kann/kautodiff.c -lm -o annpfe

Simply compile a dataset like a C array as in the example (first columns are the outputs, other are inputs), then, open run_annpfe.sh and manually edit bash input variables or directly launch annpfe from command line:

# TRAINING
<code>
  ./annpfe [n_lag] [normal-standard-ization_method] [testing_method] [network_filename] [predictions_filename] [feature_scaling_min] [feature_scaling_max] [net_type] [n_h_layers] [n_h_neurons] [max_epoch] [minibatch_size] [timesteps] [learning_rate] [dropout] [training_idx] [validation_idx] [want_layer_normalization] [n_threads] [random_seed]
</code>

<example-string>
./annpfe 0 0 1 kann_net.bin predictions.csv 0 1 0 1 10 1000 2 1 0.001 0.2 60 20 0 8 1542718307 </example-string>

/*EXAMPLE*/ ./annpfe 0 0 1 kann_net.bin predictions.csv 0 1 0 1 10 1000 1 1 0.001 0.2 60 20 0 8 1542718307

/*MODIFIED*/ ./annpfe 0 0 1 kann_net.bin predictions.csv 0 1 0 1 120 1000 2 16 0.001 0.8 60 20 0 8 1542718307

# TEST
<code>
./annpfe [n_lag] [normal-standard-ization_method] [testing_method] [network_filename] [predictions_filename] [feature_scaling_min] [feature_scaling_max] [net_type]
</code>

<example-string> ./annpfe 0 0 0 kann_net.bin predictions.csv 0 1 0 </example-string>
