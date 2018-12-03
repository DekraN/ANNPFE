/*
#################################################################
#   ANNPFE - Artificial Neural Network Prototyping Front-End    #
#             Final Built, v0.1 - 01/12/2018                    #
#                    Authors/Developer:                         #
#             Marco Chiarelli        @ UNISALENTO & CMCC        #
#               Gabriele Accarino    @ UNISALENTO & CMCC        #
#                      marco_chiarelli@yahoo.it                 #
#                      marco.chiarelli@cmcc.it                  #
#                     gabriele.accarino@cmcc.it                 #
#                     gabriele.accarino@unisalento.it           #
#################################################################
*/

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>

#include "inter_h.h"
#include "kann_atyp/kann.h"

#ifndef DATASET
	#define DATASET "test_dataset.h"
#endif

#ifndef N_SAMPLES
	#define N_SAMPLES 50000
#endif

#ifndef N_FEATURES
	#define N_FEATURES 12
#endif

#ifndef N_DIM_OUT
	#define N_DIM_OUT 1
#endif

#ifndef TRAINING_LOSS_FILE
	#define TRAINING_LOSS_FILE "training_loss.csv"
#endif

#ifndef VALIDATION_LOSS_FILE
	#define VALIDATION_LOSS_FILE "validation_loss.csv"
#endif

#ifndef ERROR_SCORE_FILE
	#define ERROR_SCORE_FILE "error_score.csv"
#endif

#define NET_TYPE 0
#define N_H_LAYERS 3
#define N_NEURONS 120
#define N_EPOCHS 3000
#define LEARNING_RATE 0.001f
#define DROPOUT 0.00f // 0.20f
#define BREAK_SCORE 0.00f
#define N_DIM_IN N_FEATURES
#define N_TIMESTEPS 1
#define N_MINIBATCH 1 // ONLINE LEARNING
#define STDNORM 1
#define L_NORM 1
#define N_THREADS 8
#define RANDOM_SEED 11 // (time(NULL))
#define TRAINING_IDX 0.7f
#define VALIDATION_IDX 0.1f
#define TESTING_METHOD 1
#define N_LAG 0
#define METRICS 1
#define TOT_FEATURES (N_DIM_OUT+N_DIM_IN)
#define DATASET_SIZE TOT_FEATURES*N_SAMPLES

#define FEATURE_SCALING_MIN 0.00f
#define FEATURE_SCALING_MAX 1.00f

#define TO_APPLY 1
#define TRAINING_DESC stderr
#define ERROR_DESC stderr
#define NET_BINARY_NAME "kann_net.bin"
#define PREDICTIONS_NAME "predictions.csv"

#include DATASET

enum
{
	NOERROR,
	ERROR_FILE,
	ERROR_MEMORY,
	ERROR_SYNTAX,
	ERROR_UNKNOWN
};

enum
{
	ANN_FF,
	ANN_RNN,
	ANN_GRU,
	ANN_LSTM
};

static unsigned char train_exec = 1;

static atyp MIN(atyp a[], int n, int pitch, int max_pitch)
{
	int c, index;
	atyp min = a[pitch];

	index = pitch;

	for (c = pitch+max_pitch; c+max_pitch < n; c+=max_pitch)
		if (a[c] < min)
		{
			index = c;
			min = a[c];
		}

	return min;
}

static atyp MAX(atyp a[], int n, int pitch, int max_pitch)
{
	int c, index;
	atyp max = a[pitch];

	index = pitch;

	for (c = pitch+max_pitch; c+max_pitch < n; c+=max_pitch)
		if (a[c] > max)
		{
			index = c;
			max = a[c];
		}

	return max;
}

static atyp MEAN(atyp a[], int n, int pitch, int max_pitch)
{
	int c;
	atyp mean = 0.00; 

	for (c = pitch; c < n; c+=max_pitch)	
		mean += a[c];


	return mean / n;
}

static atyp VARIANCE(atyp a[], int n, int pitch, int max_pitch)
{
	int c;
	atyp variance = 0.00;
	atyp mean = MEAN(a, n, pitch, max_pitch); 

	for (c = pitch; c < n; c+=max_pitch)	
		variance += (a[c]-mean)*(a[c]-mean);


	return variance / n;
}

static inline atyp STD(atyp a[], int n, int pitch, int max_pitch)
{
	return sqrtf(VARIANCE(a, n, pitch, max_pitch));
}

static void normalize_minmax(atyp *vec, int size, int pitch, int max_pitch, atyp min_x, atyp max_x, atyp a, atyp b)
{
	int i;

	for (i = pitch; i < size; i+=max_pitch)
		vec[i] = (b - a)*((vec[i] - min_x)/(max_x - min_x)) + a;

}

static void normalize_std(atyp *vec, int size, int pitch, int max_pitch, atyp mean, atyp var, atyp unused1, atyp unused2)
{
	#pragma unused(unused1)
	#pragma unused(unused2)
	int i;

	for (i = pitch+max_pitch; i+max_pitch < size; i+=max_pitch)
		vec[i] = vec[i]*var + mean;

}

static inline atyp identity_denormalize(register atyp y, atyp min_x, atyp max_x, atyp a, atyp b)
{
	#pragma unused(min_x)
	#pragma unused(max_x)
	#pragma unused(a)
	#pragma unused(b)
	return y;
}

static inline atyp minmax_denormalize(register atyp y, atyp min_x, atyp max_x, atyp a, atyp b)
{
	return ((y-a)/(b-a))*(max_x-min_x) + min_x;
}

static inline atyp z_unscoring(register atyp y, atyp mean, atyp var, atyp a, atyp b)
{
	#pragma unused(a)
	#pragma unused(b)
	return y*var + mean;
}

static int train(kann_t *net, atyp *train_data, int n_samples, float lr, int ulen, int mbs, int max_epoch, double break_score, float train_idx, float val_idx, int n_threads, unsigned char metrics)
{
	int k;
	kann_t *ua;
	atyp *r;
	atyp **x, **y;
	float best_cost = 1e30f;
	struct timeval tp;
	int n_var = kann_size_var(net); 
	int n_dim_in = kann_dim_in(net);
	int n_dim_out = kann_dim_out(net);

	int n_train_ex = (int)(train_idx*n_samples);	
	int n_val_ex = (int)(val_idx*n_samples);

	FILE *train_fd, *val_fd;

	if((x = (atyp**)calloc(ulen, sizeof(atyp*))) == NULL) // an unrolled has _ulen_ input nodes
	{
		fprintf(ERROR_DESC, "Memory error on input vector allocation.\n");
		return ERROR_MEMORY; 
	}

	if((y = (atyp**)calloc(ulen, sizeof(atyp*))) == NULL) // ... and _ulen_ truth nodes
	{
		free(x);
		fprintf(ERROR_DESC, "Memory error on output vector allocation.\n");
		return ERROR_MEMORY;
	}

	for (k = 0; k < ulen; ++k)
	{
		if((x[k] = (atyp*)calloc(n_dim_in * mbs, sizeof(atyp))) == NULL) // each input node takes a (1,n_dim_in) 2D array
		{
			while(--k)
				free(x[k]), free(y[k]);
			free(x), free(y);
			fprintf(ERROR_DESC, "Memory error on input vector elements allocation.\n");
			return ERROR_MEMORY; 
		}
			
		if((y[k] = (atyp*)calloc(n_dim_out * mbs, sizeof(atyp))) == NULL) // ... where 1 is the mini-batch size
		{
			free(x[k]);
			while(--k)
				free(x[k]), free(y[k]);
			free(x), free(y);
			fprintf(ERROR_DESC, "Memory error on output vector elements allocation.\n");
			return ERROR_MEMORY;
		}
	}

	if((r = (atyp*)calloc(n_var, sizeof(atyp))) == NULL) // temporary array for RMSprop
	{

		for (k = 0; k < ulen; ++k)
		{
			free(x[k]);
			free(y[k]);
		}

		free(x), free(y);
		fprintf(ERROR_DESC, "Memory error on RMSprop's temporary vector.\n");
		return ERROR_MEMORY;
	}

	if(metrics)
	{
		if(!(train_fd = fopen(TRAINING_LOSS_FILE, "w+")))
		{
			for (k = 0; k < ulen; ++k)
			{
				free(x[k]);
				free(y[k]);
			}
	
			free(x), free(y), free(r);		
			fprintf(ERROR_DESC, "Unable to write on "TRAINING_LOSS_FILE".\n");
			return ERROR_FILE;
		}			

		if(val_idx && !(val_fd = fopen(VALIDATION_LOSS_FILE, "w+")))
		{
			for (k = 0; k < ulen; ++k)
			{
				free(x[k]);
				free(y[k]);
			}
	
			fclose(train_fd);
			free(x), free(y), free(r);	
			fprintf(ERROR_DESC, "Unable to write on "VALIDATION_LOSS_FILE".\n");
			return ERROR_FILE;
		}
	}

	ua = ulen > 1 ? kann_unroll(net, ulen) : net;            // unroll; the mini batch size is 1
	kann_feed_bind(ua, KANN_F_IN,    0, x); // bind _x_ to input nodes
	kann_feed_bind(ua, KANN_F_TRUTH, 0, y); // bind _y_ to truth nodes
	kann_set_batch_size(ua, mbs);
	kann_switch(ua, 1);

	if(n_threads > 1)
		kann_mt(ua, n_threads, n_threads);

	int i, j, b, l;
	int train_tot, val_tot;
	double train_cost, val_cost;
	double mean_train_cost, mean_val_cost;
	gettimeofday(&tp, NULL);
	double elaps = -(double)(tp.tv_sec + tp.tv_usec/1000000.0);

	for (i = 0; i < max_epoch && train_exec; ++i)
	{		
		train_tot = val_tot = 0;
		train_cost = val_cost = 0.00f;
		kann_switch(ua, 1);

		for (j = 0; j < n_train_ex; j += ulen * mbs)
		{

			for (b = 0; b < mbs; ++b)
			{ // loop through a mini-batch
				for (k = 0; k < ulen; ++k)
				{
					memset(x[k], 0, n_dim_in * mbs * sizeof(atyp));
					memset(y[k], 0, n_dim_out * mbs * sizeof(atyp));
		
					memcpy(&x[k][b * n_dim_in], &train_data[(j + b*ulen + k)*(n_dim_in+n_dim_out) + n_dim_out], n_dim_in * sizeof(atyp));
					memcpy(&y[k][b * n_dim_out], &train_data[(j + b*ulen + k)*(n_dim_in+n_dim_out)], n_dim_out * sizeof(atyp));
	
				}
				
				
				train_cost += kann_cost(ua, 0, 1) * ulen *mbs;
				train_tot += ulen * mbs;

			}
			
			for (k = 0; k < n_var; ++k)
				ua->g[k] /= (atyp) mbs; // gradients are the average of this mini batch
			
			kann_RMSprop(n_var, lr, 0, 0.9f, ua->g, ua->x, r); // update all variables

		}

		kann_switch(ua, 0);

		for (j = 0; j < n_val_ex; j += ulen * mbs)
		{

			for (b = 0; b < mbs; ++b)
			{ // loop through a mini-batch
				for (k = 0; k < ulen; ++k)
				{
					
					memset(x[k], 0, n_dim_in * mbs * sizeof(atyp));
					memset(y[k], 0, n_dim_out * mbs * sizeof(atyp));
	
					memcpy(&x[k][b * n_dim_in], &train_data[(j + n_train_ex + b*ulen + k)*(n_dim_in+n_dim_out) + n_dim_out], n_dim_in * sizeof(atyp));
					memcpy(&y[k][b * n_dim_out], &train_data[(j + n_train_ex + b*ulen + k)*(n_dim_in+n_dim_out)], n_dim_out * sizeof(atyp));		
	
				}	
				
				val_cost += kann_cost(ua, 0, 0) * ulen *mbs;
				val_tot += ulen * mbs;

			}

		}

		mean_train_cost = train_cost / train_tot;
		fprintf(TRAINING_DESC, "epoch: %d; Training cost: %g", i+1, mean_train_cost);

		if(metrics)
			fprintf(train_fd, "%d,%g\n", i+1, mean_train_cost);

		if(val_idx)
		{
			mean_val_cost = val_cost / val_tot;
			fprintf(TRAINING_DESC, "; Validation cost: %g", mean_val_cost);

			if(metrics)
				fprintf(val_fd, "%d,%g\n", i+1, mean_val_cost);
		}

		fprintf(TRAINING_DESC, ";\n"); 
	
		if(break_score && mean_train_cost <= break_score)
		{
			fprintf(stderr, "break for scoring\n");
			break;
		}

	}

	gettimeofday(&tp, NULL);
	elaps += ((double)(tp.tv_sec + tp.tv_usec/1000000.0));
	printf("\nAverage test time: %lf\n", elaps);
	
	for (k = 0; k < ulen; ++k)
	{
		free(x[k]);
		free(y[k]);
	}

 	free(x);
	free(y); 

	if(ulen > 1)
		kann_delete_unrolled(ua); // for an unrolled network, don't use kann_delete()!

	if(metrics)
	{
		fclose(train_fd);
		if(val_idx)
			fclose(val_fd);
	}

	free(r);
	return NOERROR;
}

static int test(kann_t *net, atyp *test_data, int n_test_ex, double *tot_cost, atyp *min_x, atyp *max_x, atyp *mean, atyp *std, char * p_name, unsigned char net_type, unsigned char stdnorm, unsigned char metrics, atyp a, atyp b)
{
	int i, j, k, l;
	struct timeval tp;
	FILE * fp, * err_fd;
	atyp y1_denorm;
	double elaps = 0.00;
	double cur_cost = 0.00;
	int out_idx;
	int n_dim_in = kann_dim_in(net);
	int n_dim_out = kann_dim_out(net);
	atyp *x1;
	atyp *expected;
	const atyp *y1;


	static const atyp (* const denorm_functions[3])(register atyp, atyp, atyp, atyp, atyp) =
	{
		identity_denormalize,
		minmax_denormalize,
		z_unscoring
	};

	atyp (* denorm_function)(register atyp, atyp, atyp, atyp, atyp) = denorm_functions[stdnorm]; 

	if((x1 = (atyp*)calloc(n_dim_in, sizeof(atyp))) == NULL)
	{
		fprintf(ERROR_DESC, "Memory error on input vector allocation.\n");
		return ERROR_MEMORY; 
	}

	if((expected = (atyp*)calloc(n_dim_out, sizeof(atyp))) == NULL)
	{
		free(x1);
		fprintf(ERROR_DESC, "Memory error on expected vector allocation.\n");
		return ERROR_MEMORY;
	}

	kann_feed_bind(net, KANN_F_IN, 0, &x1);
	kann_set_batch_size(net, 1);
	kann_switch(net, 0);
	out_idx = kann_find(net, KANN_F_OUT, 0);

	if(net_type)
		kann_rnn_start(net);
	
	printf("Test Begin\n");
	printf("Number of ex: %d\n", n_test_ex);

	if(!(fp=fopen(p_name, "w+")))
	{
		free(x1), free(expected);
		fprintf(ERROR_DESC, "Unable to write on %s.\n", p_name);
		return ERROR_FILE;
	}

	if(metrics && !(err_fd = fopen(ERROR_SCORE_FILE, "w+")))
	{
		fclose(fp);
		free(x1), free(expected);
		fprintf(ERROR_DESC, "Unable to write on "ERROR_SCORE_FILE".\n");
		return ERROR_FILE;
	}

	// fprintf(fp, ",0\n");
	
	for (i = 0; i < n_test_ex; ++i)
	{
		memcpy(x1, &test_data[i*(n_dim_in+n_dim_out) + n_dim_out], n_dim_in * sizeof(atyp));
		memcpy(expected, &test_data[i*(n_dim_in+n_dim_out)], n_dim_out * sizeof(atyp));
		gettimeofday(&tp, NULL);
		elaps += -((double)(tp.tv_sec + tp.tv_usec/1000000.0));
		kann_eval(net, KANN_F_OUT, 0);
		y1 = net->v[out_idx]->x;
		gettimeofday(&tp, NULL);
		elaps += ((double)(tp.tv_sec + tp.tv_usec/1000000.0));
		fprintf(fp, "%d", i+1);

		if(metrics)
			fprintf(err_fd, "%d", i+1);

		cur_cost = 0.00f;

		for (l = 0; l < n_dim_out; ++l)
		{
			y1_denorm = stdnorm == 3 ? z_unscoring(minmax_denormalize(y1[l], min_x[l], max_x[l], a, b), mean[l], std[l], a, b) : denorm_function(y1[l], min_x[l], max_x[l], a, b);

			fprintf(fp, ",%g", y1_denorm);

			if(metrics)
				fprintf(err_fd, ",%g", y1_denorm - expected[l]);

			cur_cost += (y1_denorm - expected[l])*(y1_denorm - expected[l]);
		}
	
		fprintf(fp, "\n");

		if(metrics)
			fprintf(err_fd, "\n");

		cur_cost /= n_dim_out;
		*tot_cost += cur_cost;
	}

	fclose(fp);
	*tot_cost = sqrtf(*tot_cost/n_test_ex);
	printf("Test Ended.\n");
	elaps /= n_test_ex;
	printf("\nAverage test time: %lf.\n", elaps);

	if(net_type)
		kann_rnn_end(net);

	free(x1), free(expected);
	return NOERROR;
}

static void sigexit(int sign)
{
	train_exec = 0;
	return;
}

int main(int argc, char *argv[])
{
	
	int i, j;
	kann_t *ann = NULL;
	double break_score;
	char *p_name = PREDICTIONS_NAME;
	char *fn_in = NET_BINARY_NAME, *fn_out = 0;
	float lr, dropout, t_idx, val_idx;
	atyp feature_scaling_min, feature_scaling_max;
	const unsigned char to_apply = argc > 10;
	int net_type, metrics, n_h_layers, n_h_neurons, mini_size, timesteps, max_epoch, t_method, n_lag, stdnorm, l_norm, n_threads, seed;

	printf("\n\n#################################################################\n");
	printf("#   ANNPFE - Artificial Neural Network Prototyping Front-End    #\n");
	printf("#             Final Built, v0.1 - 01/12/2018                    #\n");
	printf("#                    Authors/Developer:                         #\n");
	printf("#             Marco Chiarelli        @ UNISALENTO & CMCC        #\n");
	printf("#               Gabriele Accarino    @ UNISALENTO & CMCC        #\n");
	printf("#                      marco_chiarelli@yahoo.it                 #\n");
	printf("#                      marco.chiarelli@cmcc.it                  #\n");
	printf("#                     gabriele.accarino@cmcc.it                 #\n");
	printf("#                     gabriele.accarino@unisalento.it          	#\n");
	printf("#################################################################\n\n");

	printf("Type ./annpfe help for help\n\n");

	if(!strcmp(argv[1], "help"))
	{
		printf("USAGE: ./annpfe [n_lag] [normal-standard-ization_method] [testing_method] [network_filename] [predictions_filename] [feature_scaling_min] [feature_scaling_max] [net_type] [metrics] [n_h_layers] [n_h_neurons] [max_epoch] [minibatch_size] [timesteps] [learning_rate] [dropout] [break_score] [training_idx[\%%]] [validation_idx[\%%]] [want_layer_normalization] [n_threads] [random_seed]\n");
		printf("Enter executable name without params for testing.\n");	
		return 2;
	}

	n_lag = argc > 1 ? atoi(argv[1]) : N_LAG;

	if(n_lag < 0 || n_lag > N_SAMPLES)
	{
		fprintf(ERROR_DESC, "Number of lag must be a positive integer.\n");
		return ERROR_SYNTAX;	
	}

	stdnorm = argc > 2 ? atoi(argv[2]) : STDNORM;

	if(stdnorm < 0 || stdnorm > 3)
	{
		fprintf(ERROR_DESC, "Normal/Standard-ization method must be an integer >= 0 and <= 3.\n");
		return ERROR_SYNTAX;	
	}

	// test-only parameters

	t_method = argc > 3 ? atoi(argv[3]): TESTING_METHOD;

	if(t_method != 0 && t_method != 1)
	{
		fprintf(ERROR_DESC, "Testing method must be a boolean number.\n");
		return ERROR_SYNTAX;
	}

	if(argc > 4)
		fn_in = argv[4];

	if(argc > 5)
		p_name = argv[5];


	feature_scaling_min = argc > 6 ? atof(argv[6]) : FEATURE_SCALING_MIN;
	feature_scaling_max = argc > 7 ? atof(argv[7]) : FEATURE_SCALING_MAX;

	if(feature_scaling_max < feature_scaling_min)
	{
		fprintf(ERROR_DESC, "Feature-scaling min must be lesser than feature-scaling max.");	
		return ERROR_SYNTAX;
	}

	net_type = argc > 8 ? atoi(argv[8]) : NET_TYPE;	

	if(net_type < 0 || net_type > 3)
	{
		fprintf(ERROR_DESC, "Network type must be an integer >= 0 and <= 3.\n");
		return ERROR_SYNTAX;	
	}

	metrics = argc > 9 ? atoi(argv[9]) : METRICS;

	if(metrics != 0 && metrics != 1)
	{
		fprintf(ERROR_DESC, "Error metrics must be a boolean number.\n");
		return ERROR_SYNTAX;
	}

	// train only parameters

	n_h_layers = argc > 10 ? atoi(argv[10]) : N_H_LAYERS;	

	if(n_h_layers <= 0)
	{
		fprintf(ERROR_DESC, "Number of layers must be a non-zero positive integer.\n");
		return ERROR_SYNTAX;	
	}

	n_h_neurons = argc > 11 ? atoi(argv[11]) : N_NEURONS;

	if(n_h_neurons <= 0)
	{
		fprintf(ERROR_DESC, "Number of neurons must be a non-zero positive integer.\n");
		return ERROR_SYNTAX;	
	}

	max_epoch = argc > 12 ? atoi(argv[12]) : N_EPOCHS;

	if(max_epoch <= 0)
	{
		fprintf(ERROR_DESC, "Max epoch must be a non-zero positive integer.\n");
		return ERROR_SYNTAX;	
	}


	mini_size = argc > 13 ? atoi(argv[13]) : N_MINIBATCH;

	if(mini_size < 1)
	{
		fprintf(ERROR_DESC, "Minibatch size must be an integer >= 1.\n");
		return ERROR_SYNTAX;	
	}
	
	timesteps = argc > 14 ? atoi(argv[14]) : N_TIMESTEPS;

	if(timesteps <= 0)
	{
		fprintf(ERROR_DESC, "Timesteps must be a non-zero positive integer.\n");
		return ERROR_SYNTAX;	
	}

	lr = argc > 15 ? ((float) atof(argv[15])) : LEARNING_RATE;

	if(lr <= 0 || lr >= 1.00f)
	{
		fprintf(ERROR_DESC, "Learning rate must be a float > 0 and <= 1.0.\n");
		return ERROR_SYNTAX;	
	}

	dropout = argc > 16 ? ((float) atof(argv[16])) : DROPOUT;

	if(dropout < 0 || dropout >= 1.00f)
	{
		fprintf(ERROR_DESC, "Dropout must be a float >= 0 and <= 1.0.\n");
		return ERROR_SYNTAX;	
	}

	break_score = argc > 17 ? ((float) atof(argv[17])) : BREAK_SCORE;

	if(break_score < 0)
	{
		fprintf(ERROR_DESC, "Dropout must be a float >= 0.\n");
		return ERROR_SYNTAX;	
	}

	t_idx = argc > 18 ? ((float)atof(argv[18])*0.01f) : TRAINING_IDX;

	if(t_idx <= 0 || t_idx > 1.00f)
	{
		fprintf(ERROR_DESC, "Training index must be a float > 0%% and <= 100%%.\n");
		return ERROR_SYNTAX;	
	}

	val_idx = argc > 19 ? ((float)atof(argv[19])*0.01f) : VALIDATION_IDX;

	if(val_idx < 0 || val_idx >= 1.00f)
	{
		fprintf(ERROR_DESC, "Validation index must be a float >= 0\%% and <= 100%%.\n");
		return ERROR_SYNTAX;	
	}

	if(val_idx > t_idx)
	{
		fprintf(ERROR_DESC, "Training index must be greater than Validation index.\n");
		return ERROR_SYNTAX;
	}

	l_norm = argc > 20 ? atoi(argv[20]) : L_NORM;

	if(l_norm != 0 && l_norm != 1)
	{
		fprintf(ERROR_DESC, "Layer normalization must be a boolean number.\n");
		return ERROR_SYNTAX;
	}

	n_threads = argc > 21 ? atoi(argv[21]) : N_THREADS;

	if(n_threads <= 0)
	{
		fprintf(ERROR_DESC, "Number of threads must be a non-zero positive integer.\n");
		return ERROR_SYNTAX;	
	}

	seed = argc > 22 ? atoi(argv[22]) : RANDOM_SEED;

	(void) signal(SIGINT, sigexit);

	kad_trap_fe();
	kann_srand(seed);

	atyp * output_feature_c = NULL;
	atyp * output_feature_d = NULL;
	atyp * train_data = NULL;

	const int tot_features_lag = TOT_FEATURES+n_lag;
	const int dataset_size = DATASET_SIZE+n_lag*N_SAMPLES-tot_features_lag*n_lag;

	atyp output_feature_a[tot_features_lag];
	atyp output_feature_b[tot_features_lag];

	if(n_lag)
	{
		
		train_data = calloc(dataset_size, sizeof(atyp));	

		for(i=0; i<N_SAMPLES-n_lag; ++i)
		{
			for(j=0; j<n_lag+1; ++j)
				train_data[i*tot_features_lag + j] = train_data_base[(i+n_lag-j)*TOT_FEATURES];
			for( ; j<tot_features_lag; ++j)
				train_data[i*tot_features_lag + j] = train_data_base[(i+n_lag)*TOT_FEATURES+j-n_lag];
		}
	}
	else
		train_data = train_data_base;
	
	if(stdnorm)
	{
		if(stdnorm != 3)
		{
			atyp (* const norm_functions[2][2])(atyp [], int, int, int) =
			{	
				{
					MIN,	
					MAX
				},
				{
					MEAN,
					STD
				}
			};

			void (* const norm_routine[2])(atyp *, int, int, int, atyp, atyp, atyp, atyp) =
			{
				normalize_minmax,
				normalize_std
			};

			const char * const norm_names[2][2] =
			{
				{
					"min",	
					"max"
				},
				{
					"mean",
					"std"
				}
			};
			

			for(i=tot_features_lag-1; i>=0; --i)
			{
				output_feature_a[i] = norm_functions[stdnorm-1][0](train_data, dataset_size, i, tot_features_lag);
				output_feature_b[i] = norm_functions[stdnorm-1][1](train_data, dataset_size, i, tot_features_lag); 

				printf("i is %d, out-%s: %g, out-%s: %g\n", i, norm_names[stdnorm-1][0], output_feature_a[i], norm_names[stdnorm-1][1], output_feature_b[i]);
				norm_routine[stdnorm-1](train_data, dataset_size, i, tot_features_lag, output_feature_a[i], output_feature_b[i], feature_scaling_min, feature_scaling_max);
			}
		}
		else
		{
			output_feature_c = calloc(tot_features_lag, sizeof(atyp));
			output_feature_d = calloc(tot_features_lag, sizeof(atyp));

			for(i=tot_features_lag-1; i>=0; --i)
			{
				output_feature_c[i] = MEAN(train_data, dataset_size, i, tot_features_lag);
				output_feature_d[i] = STD(train_data, dataset_size, i, tot_features_lag); 
				normalize_std(train_data, dataset_size, i, tot_features_lag, output_feature_c[i], output_feature_d[i], feature_scaling_min, feature_scaling_max);

				output_feature_a[i] = MIN(train_data, dataset_size, i, tot_features_lag);
				output_feature_b[i] = MAX(train_data, dataset_size, i, tot_features_lag); 
				
				normalize_minmax(train_data, dataset_size, i, tot_features_lag, output_feature_c[i], output_feature_d[i], feature_scaling_min, feature_scaling_max);

				printf("i is %d, out-min: %g, out-max: %g\n", i, output_feature_a[i], output_feature_b[i]);
				printf("i is %d, out-mean: %g, out-std: %g\n", i, output_feature_c[i], output_feature_d[i]);
			}
		}
	}
	
	if (to_apply)
	{
		// model generation
		kad_node_t *t;
		int rnn_flag = KANN_RNN_VAR_H0;
		if (l_norm) rnn_flag |= KANN_RNN_NORM;
		t = kann_layer_input(N_DIM_IN+n_lag); // t = kann_layer_input(d->n_in);

		for (i = 0; i < n_h_layers; ++i)
		{
			switch(net_type)
			{
				case ANN_FF:	
					t = kad_sigm(kann_layer_dense(t, n_h_neurons));
					break;
				case ANN_RNN:
					t = kann_layer_rnn(t, n_h_neurons, rnn_flag);
					break;
				case ANN_GRU:
					t = kann_layer_gru(t, n_h_neurons, rnn_flag);
					break;
				case ANN_LSTM:
					t = kann_layer_lstm(t, n_h_neurons, rnn_flag);
					break;
				default:
					break;
			}

			if(dropout)
				t = kann_layer_dropout(t, dropout);
		}

		ann = kann_new(kann_layer_cost(t, N_DIM_OUT, KANN_C_MSE), 0);
		printf("\nTRAINING...\n");
		if(!train(ann, train_data, N_SAMPLES-n_lag, lr, net_type ? timesteps : 1, mini_size, max_epoch, break_score, t_idx, val_idx, n_threads, metrics)) 
		{
			kann_save(fn_in, ann);
			printf("\nTraining succeeded!\n");
		}
		
	}
	else
	{
		double tot_cost = 0.00;
		ann = kann_load(fn_in);
		printf("\nTEST...\n");

		const int exit_code = t_method ? test(ann, &train_data[(int)(tot_features_lag*(N_SAMPLES-n_lag)*(t_idx+val_idx))], N_SAMPLES - (int)(N_SAMPLES*(t_idx+val_idx)), &tot_cost, output_feature_a, output_feature_b, output_feature_c, output_feature_d, p_name, net_type, stdnorm, metrics, feature_scaling_min, feature_scaling_max) : test(ann, train_data, N_SAMPLES-n_lag, &tot_cost, output_feature_a, output_feature_b, output_feature_c, output_feature_d, p_name, net_type, stdnorm, metrics, feature_scaling_min, feature_scaling_max);	

		if(!exit_code) 		
			printf("\nTest total cost: %g\n", tot_cost);
	}

	kann_delete(ann);
	printf("\nDeleted kann network.\n");
	return NOERROR;
}
