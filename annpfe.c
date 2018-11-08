/*
#################################################################
#   ANNPFE - Artificial Neural Network Prototyping Front-End    #
#             Final Built, v0.1 - 08/11/2018                    #
#                    Authors/Developer:                         #
#             Marco Chiarelli        @ UNISALENTO & CMCC        #
#               Gabriele Accarino    @ UNISALENTO & CMCC        #
#                      marco_chiarelli@yahoo.it                 #
#                      marco.chiarelli@cmcc.it                  #
#                     gabriele.accarino@cmcc.it                	#
#################################################################
*/

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "kann/kann.h"

#include <sys/time.h>

#define N_H_LAYERS 3
#define N_NEURONS 120
#define N_EPOCHS 3000
#define LEARNING_RATE 0.001f
#define DROPOUT 0.00f // 0.20f
#define N_FEATURES 12
#define N_DIM_IN N_FEATURES
#define N_DIM_OUT 1
#define N_TIMESTEPS 1
#define N_MINIBATCH 1 // ONLINE LEARNING
#define STDNORM 1
#define L_NORM 1
#define N_THREADS 8
#define RANDOM_SEED 11
#define TRAINING_IDX 0.7f
#define VALIDATION_IDX 0.1f
#define TESTING_METHOD 1

#define FEATURE_SCALING_MIN 0.00f
#define FEATURE_SCALING_MAX 1.00f

#define TO_APPLY 1
#define TRAINING_DESC stderr
#define ERROR_DESC stderr
#define NET_BINARY_NAME "kann_net.bin"
#define POINT_HEADER "test_dataset.h"
#define PREDICTIONS_NAME "predictions.csv"

#define N_SAMPLES_PER_POINT 50000
#include POINT_HEADER

static unsigned char train_exec = 1;

static float MIN(float a[], int n, int pitch, int max_pitch)
{
	int c, index;
	float min = a[pitch];

	index = pitch;

	for (c = pitch+max_pitch; c+max_pitch < n; c+=max_pitch)
		if (a[c] < min)
		{
			index = c;
			min = a[c];
		}

	return min;
}

static float MAX(float a[], int n, int pitch, int max_pitch)
{
	int c, index;
	float max = a[pitch];

	index = pitch;

	for (c = pitch+max_pitch; c+max_pitch < n; c+=max_pitch)
		if (a[c] > max)
		{
			index = c;
			max = a[c];
		}

	return max;
}

static float MEAN(float a[], int n, int pitch, int max_pitch)
{
	int c;
	float mean = 0.00; 

	for (c = pitch+max_pitch; c+max_pitch < n; c+=max_pitch)	
		mean += a[c];


	return mean / n;
}

static float VARIANCE(float a[], int n, int pitch, int max_pitch)
{
	int c;
	float variance = 0.00;
	float mean = MEAN(a, n, pitch, max_pitch); 

	for (c = pitch+max_pitch; c+max_pitch < n; c+=max_pitch)	
		variance += (a[c]-mean)*(a[c]-mean);


	return variance / n;
}

static inline float STD(float a[], int n, int pitch, int max_pitch)
{
	return sqrtf(VARIANCE(a, n, pitch, max_pitch));
}

static void normalize_minmax(float *vec, int size, int pitch, int max_pitch, float min_x, float max_x, float a, float b)
{
	int i;

	for (i = pitch+max_pitch; i+max_pitch < size; i+=max_pitch)
		vec[i] = (b - a)*((vec[i] - min_x)/(max_x - min_x)) + a;

}

static void normalize_std(float *vec, int size, int pitch, int max_pitch, float mean, float var, float unused1, float unused2)
{
	#pragma unused unused1
	#pragma unused unused2
	int i;

	for (i = pitch+max_pitch; i+max_pitch < size; i+=max_pitch)
		vec[i] = vec[i]*var + mean;

}

static inline float identity_denormalize(register float y, float min_x, float max_x, float a, float b)
{
	#pragma unused min_x
	#pragma unused max_x
	#pragma unused a
	#pragma unused b
	return y;
}

static inline float minmax_denormalize(register float y, float min_x, float max_x, float a, float b)
{
	return ((y-a)/(b-a))*(max_x-min_x) + min_x;
}

static inline float z_unscoring(register float y, float mean, float var, float a, float b)
{
	#pragma unused a
	#pragma unused b
	return y*var + mean;
}

static int train(kann_t *net, float *train_data, int n_samples, float lr, int ulen, int mbs, int max_epoch, float train_idx, float val_idx, int n_threads)
{
	int k;
	kann_t *ua;
	float *r;
	float **x, **y;
	float best_cost = 1e30f;
	int n_var = kann_size_var(net); 
	int n_dim_in = kann_dim_in(net);
	int n_dim_out = kann_dim_out(net);

	int n_train_ex = (int)(train_idx*n_samples);	
	int n_val_ex = (int)(val_idx*n_samples);

	if((x = (float**)calloc(ulen, sizeof(float*))) == NULL) // an unrolled has _ulen_ input nodes
		return 1;

	if((y = (float**)calloc(ulen, sizeof(float*))) == NULL) // ... and _ulen_ truth nodes
		return 1;

	for (k = 0; k < ulen; ++k)
	{
		if((x[k] = (float*)calloc(n_dim_in * mbs, sizeof(float))) == NULL) // each input node takes a (1,n_dim_in) 2D array
			return 1;
		if((y[k] = (float*)calloc(n_dim_out * mbs, sizeof(float))) == NULL) // ... where 1 is the mini-batch size
			return 1;
	}

	if((r = (float*)calloc(n_var, sizeof(float))) == NULL) // temporary array for RMSprop
		return 1;

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
					memset(x[k], 0, n_dim_in * mbs * sizeof(float));
					memset(y[k], 0, n_dim_out * mbs * sizeof(float));

					for (l = n_dim_out; l < n_dim_in+n_dim_out; ++l)
						x[k][b * n_dim_in + l-n_dim_out] = train_data[(j + b*ulen + k)*(n_dim_in+n_dim_out) + l] ;
		
					for (l = 0; l < n_dim_out; ++l)
						y[k][b * n_dim_out + l] = train_data[(j + b*ulen + k)*(n_dim_in+n_dim_out) + l];
	
				}
				
				
				train_cost += kann_cost(ua, 0, 1) * ulen *mbs;
				train_tot += ulen * mbs;

			}
			
			for (k = 0; k < n_var; ++k)
				ua->g[k] /= (double) mbs; // gradients are the average of this mini batch
			
			kann_RMSprop(n_var, lr, 0, 0.9f, ua->g, ua->x, r); // update all variables

		}

		kann_switch(ua, 0);

		for (j = 0; j < n_val_ex; j += ulen * mbs)
		{

			for (b = 0; b < mbs; ++b)
			{ // loop through a mini-batch
				for (k = 0; k < ulen; ++k)
				{
					memset(x[k], 0, n_dim_in * mbs * sizeof(float));
					memset(y[k], 0, n_dim_out * mbs * sizeof(float));

					for (l = n_dim_out; l < n_dim_in+n_dim_out; ++l)
						x[k][b * n_dim_in + l-n_dim_out] = train_data[(j + n_train_ex + b*ulen + k)*(n_dim_in+n_dim_out) + l] ;
		
					for (l = 0; l < n_dim_out; ++l)
						y[k][b * n_dim_out + l] = train_data[(j + n_train_ex + b*ulen + k)*(n_dim_in+n_dim_out) + l];
	
				}	
				
				val_cost += kann_cost(ua, 0, 0) * ulen *mbs;
				val_tot += ulen * mbs;

			}

		}

		fprintf(TRAINING_DESC, "epoch: %d; Training cost: %g", i+1, train_cost / train_tot);

		if(val_idx)
			fprintf(TRAINING_DESC, "; Validation cost: %g", val_cost / val_tot);

		fprintf(TRAINING_DESC, ";\n"); 

	}
	

 	free(y); free(x);
	if(ulen > 1)
		kann_delete_unrolled(ua); // for an unrolled network, don't use kann_delete()!
	free(r);
	return 0;
}

static int test(kann_t *net, float *test_data, int n_test_ex, double *tot_cost, float *min_x, float *max_x, float *mean, float *std, char * p_name, unsigned char stdnorm, float a, float b)
{
	FILE * fp;
	int i, j, k, l;
	struct timeval tp;
	float y1_denorm;
	double cur_cost = 0.00;
	double cpu_time = 0.00;
	int n_dim_in = kann_dim_in(net);
	int n_dim_out = kann_dim_out(net);
	float *x1;
	float *expected;
	const float *y1;

	static const float (* const denorm_functions[3])(register float, float, float, float, float) =
	{
		identity_denormalize,
		minmax_denormalize,
		z_unscoring
	};

	float (* denorm_function)(register float, float, float, float, float) = denorm_functions[stdnorm]; 

	kann_switch(net, 0);

	if((x1 = (float*)calloc(n_dim_in, sizeof(float))) == NULL)
		return 1;

	if((expected = (float*)calloc(n_dim_out, sizeof(float))) == NULL)
		return 1;

	// kann_rnn_start(net);
	printf("Test Begin\n");
	printf("Number of ex: %d\n", n_test_ex);
	fp=fopen(p_name, "w+");
	// fprintf(fp, ",0\n");

	for (i = 0; i < n_test_ex; ++i)
	{
		for (j = n_dim_out; j < n_dim_in+n_dim_out; ++j)
			x1[j-n_dim_out] = test_data[i*(n_dim_in+n_dim_out) + j];
		
		for (k = 0; k < n_dim_out; ++k)
			expected[k] = test_data[i*(n_dim_in+n_dim_out) + k];

		gettimeofday(&tp, NULL);
		double elaps = -(double)(tp.tv_sec + tp.tv_usec/1000000.0);
		y1 = kann_apply1(net, x1);
		gettimeofday(&tp, NULL);
		cpu_time += elaps+((double)(tp.tv_sec + tp.tv_usec/1000000.0));
		fprintf(fp, "%d", i);
		cur_cost = 0.00f;

		for (l = 0; l < n_dim_out; ++l)
		{
			y1_denorm = stdnorm == 3 ? z_unscoring(minmax_denormalize(y1[l], min_x[l], max_x[l], a, b), mean[l], std[l], a, b) : denorm_function(y1[l], min_x[l], max_x[l], a, b);
			fprintf(fp, ",%g", y1_denorm);
			cur_cost += (y1_denorm - expected[l])*(y1_denorm - expected[l]);
		}
	
		fprintf(fp, "\n");
		cur_cost /= n_dim_out;
		*tot_cost += cur_cost;
	}

	fclose(fp);
	*tot_cost = *tot_cost/n_test_ex;
	printf("Test Ended.\n");
	cpu_time /= n_test_ex;
	printf("\nAverage test time: %lf.\n", cpu_time);
	// kann_rnn_end(net);
	return 0;
}

static void sigexit(int sign)
{
	train_exec = 0;
	return;
}

int main(int argc, char *argv[])
{
	int i;
	kann_t *ann = NULL;
	char *p_name = PREDICTIONS_NAME;
	char *fn_in = NET_BINARY_NAME, *fn_out = 0;
	float lr, dropout, t_idx, val_idx;
	float feature_scaling_min, feature_scaling_max;
	const unsigned char to_apply = argc > 7;
	int n_h_layers, n_h_neurons, mini_size, max_epoch, t_method, stdnorm, l_norm, n_threads, seed;

	printf("\n\n#################################################################\n");
	printf("#   ANNPFE - Artificial Neural Network Prototyping Front-End    #\n");
	printf("#             Final Built, v0.1 - 08/11/2018                    #\n");
	printf("#                    Authors/Developer:                         #\n");
	printf("#             Marco Chiarelli        @ UNISALENTO & CMCC        #\n");
	printf("#               Gabriele Accarino    @ UNISALENTO & CMCC        #\n");
	printf("#                      marco_chiarelli@yahoo.it                 #\n");
	printf("#                      marco.chiarelli@cmcc.it                  #\n");
	printf("#                     gabriele.accarino@cmcc.it                 #\n");
	printf("#################################################################\n\n");

	printf("Type ./annpfe help for help\n\n");

	if(!strcmp(argv[1], "help"))
	{
		printf("USAGE: ./annpfe [normal-standard-ization_method] [testing_method] [network_filename] [predictions_filename] [feature_scaling_min] [feature_scaling_max] [n_h_layers] [n_h_neurons] [minibatch_size] [max_epoch] [learning_rate] [dropout] [training_idx] [validation_idx] [want_layer_normalization] [n_threads] [random_seed]\n");
		printf("Enter executable name without params for testing.\n");	
		return 2;
	}

	stdnorm = argc > 1 ? atoi(argv[1]) : STDNORM;

	if(stdnorm < 0 || stdnorm > 3)
	{
		fprintf(ERROR_DESC, "Normal/Standard-ization method must be an integer >= 0 and <= 3.\n");
		return 1;	
	}

	// test-only parameters

	t_method = argc > 2 ? atoi(argv[2]): TESTING_METHOD;

	if(t_method != 0 && t_method != 1)
	{
		fprintf(ERROR_DESC, "Testing method must be a boolean number.\n");
		return 1;
	}

	if(argc > 3)
		fn_in = argv[3];

	if(argc > 4)
		p_name = argv[4];


	feature_scaling_min = argc > 5 ? atof(argv[5]) : FEATURE_SCALING_MIN;
	feature_scaling_max = argc > 6 ? atof(argv[6]) : FEATURE_SCALING_MAX;

	if(feature_scaling_max < feature_scaling_min)
	{
		fprintf(ERROR_DESC, "Feature-scaling min must be lesser than feature-scaling max.");	
		return 1;
	}

	// train only parameters

	n_h_layers = argc > 7 ? atoi(argv[7]) : N_H_LAYERS;	

	if(n_h_layers <= 0)
	{
		fprintf(ERROR_DESC, "Number of layers must be a non-zero positive integer.\n");
		return 1;	
	}

	n_h_neurons = argc > 8 ? atoi(argv[8]) : N_NEURONS;

	if(n_h_neurons <= 0)
	{
		fprintf(ERROR_DESC, "Number of neurons must be a non-zero positive integer.\n");
		return 1;	
	}

	mini_size = argc > 9 ? atoi(argv[9]) : N_MINIBATCH;

	if(mini_size < 1)
	{
		fprintf(ERROR_DESC, "Minibatch size must be an integer >= 1.\n");
		return 1;	
	}

	max_epoch = argc > 10 ? atoi(argv[10]) : N_EPOCHS;

	if(max_epoch <= 0)
	{
		fprintf(ERROR_DESC, "Max epoch must be a non-zero positive integer.\n");
		return 1;	
	}

	lr = argc > 11 ? atof(argv[11]) : LEARNING_RATE;

	if(lr <= 0 || lr >= 1.00f)
	{
		fprintf(ERROR_DESC, "Learning rate must be a float > 0 and <= 1.0.\n");
		return 1;	
	}

	dropout = argc > 12 ? atof(argv[12]) : DROPOUT;

	if(dropout < 0 || dropout >= 1.00f)
	{
		fprintf(ERROR_DESC, "Dropout must be a float >= 0 and <= 1.0.\n");
		return 1;	
	}

	t_idx = argc > 13 ? (atof(argv[13])*0.01f) : TRAINING_IDX;

	if(t_idx <= 0 || t_idx >= 1.00f)
	{
		fprintf(ERROR_DESC, "Training index must be a float > 0%% and < 100%%.\n");
		return 1;	
	}

	val_idx = argc > 14 ? (atof(argv[14])*0.01f) : VALIDATION_IDX;

	if(val_idx < 0 || val_idx >= 1.00f)
	{
		fprintf(ERROR_DESC, "Validation index must be a float >= 0\%% and <= 100%%.\n");
		return 1;	
	}

	if(val_idx > t_idx)
	{
		fprintf(ERROR_DESC, "Training index must be greater than Validation index.\n");
		return 1;
	}

	l_norm = argc > 15 ? atoi(argv[15]) : L_NORM;

	if(l_norm != 0 && l_norm != 1)
	{
		fprintf(ERROR_DESC, "Layer normalization must be a boolean number.\n");
		return 1;
	}

	n_threads = argc > 16 ? atoi(argv[16]) : N_THREADS;

	if(n_threads <= 0)
	{
		fprintf(ERROR_DESC, "Number of threads must be a non-zero positive integer.\n");
		return 1;	
	}

	seed = argc > 17 ? atoi(argv[17]) : RANDOM_SEED;

	(void) signal(SIGINT, sigexit);

	kad_trap_fe();
	kann_srand(seed);

	float output_feature_a[N_DIM_OUT+N_DIM_IN];
	float output_feature_b[N_DIM_OUT+N_DIM_IN];
	
	float * output_feature_c = NULL;
	float * output_feature_d = NULL;
	
	if(stdnorm)
		if(stdnorm != 3)
		{
			float (* const norm_functions[2][2])(float [], int, int, int) =
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

			void (* const norm_routine[2])(float *, int, int, int, float, float, float, float) =
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
			

			for(i=N_FEATURES+N_DIM_OUT-1; i>=0; --i)
			{
				output_feature_a[i] = norm_functions[stdnorm-1][0](train_data, N_SAMPLES_PER_POINT* (N_FEATURES+N_DIM_OUT), i, (N_FEATURES+N_DIM_OUT));
				output_feature_b[i] = norm_functions[stdnorm-1][1](train_data, N_SAMPLES_PER_POINT* (N_FEATURES+N_DIM_OUT), i, (N_FEATURES+N_DIM_OUT)); 

				printf("i is %d, out-%s: %g, out-%s: %g\n", i, norm_names[stdnorm-1][0], output_feature_a[i], norm_names[stdnorm-1][1], output_feature_b[i]);
				norm_routine[stdnorm-1](train_data, N_SAMPLES_PER_POINT*(N_FEATURES+N_DIM_OUT), i, (N_FEATURES+N_DIM_OUT), output_feature_a[i], output_feature_b[i], feature_scaling_min, feature_scaling_max);
			}
		}
		else
		{
			output_feature_c = calloc(N_DIM_IN+N_DIM_OUT, sizeof(float));
			output_feature_d = calloc(N_DIM_IN+N_DIM_OUT, sizeof(float));

			for(i=N_FEATURES+N_DIM_OUT-1; i>=0; --i)
			{
				output_feature_c[i] = MEAN(train_data, N_SAMPLES_PER_POINT* (N_FEATURES+N_DIM_OUT), i, (N_FEATURES+N_DIM_OUT));
				output_feature_d[i] = STD(train_data, N_SAMPLES_PER_POINT* (N_FEATURES+N_DIM_OUT), i, (N_FEATURES+N_DIM_OUT)); 
				normalize_std(train_data, N_SAMPLES_PER_POINT*(N_FEATURES+N_DIM_OUT), i, (N_FEATURES+N_DIM_OUT), output_feature_c[i], output_feature_d[i], feature_scaling_min, feature_scaling_max);

				output_feature_a[i] = MIN(train_data, N_SAMPLES_PER_POINT* (N_FEATURES+N_DIM_OUT), i, (N_FEATURES+N_DIM_OUT));
				output_feature_b[i] = MAX(train_data, N_SAMPLES_PER_POINT* (N_FEATURES+N_DIM_OUT), i, (N_FEATURES+N_DIM_OUT)); 
				
				normalize_minmax(train_data, N_SAMPLES_PER_POINT*(N_FEATURES+N_DIM_OUT), i, (N_FEATURES+N_DIM_OUT), output_feature_c[i], output_feature_d[i], feature_scaling_min, feature_scaling_max);

				printf("i is %d, out-min: %g, out-max: %g\n", i, output_feature_a[i], output_feature_b[i]);
				printf("i is %d, out-mean: %g, out-std: %g\n", i, output_feature_c[i], output_feature_d[i]);
			}
		}
	
	(void) getchar();
	
	if (to_apply)
	{
		// model generation
		kad_node_t *t;
		int rnn_flag = KANN_RNN_VAR_H0;
		if (l_norm) rnn_flag |= KANN_RNN_NORM;
		t = kann_layer_input(N_DIM_IN); // t = kann_layer_input(d->n_in);

		for (i = 0; i < n_h_layers; ++i)
		{
			t = kad_sigm(kann_layer_dense(t, n_h_neurons));
			// t = kann_layer_rnn(t, n_h_neurons, rnn_flag);
			// t = kad_sigm(t);
			if(dropout)
				t = kann_layer_dropout(t, dropout);
		}

		ann = kann_new(kann_layer_cost(t, N_DIM_OUT, KANN_C_MSE), 0);
		printf("\nTRAINING...\n");
		train(ann, train_data, N_SAMPLES_PER_POINT, lr, N_TIMESTEPS, mini_size, max_epoch, t_idx, val_idx, n_threads); // max_epoch);
		kann_save(fn_in, ann);
		printf("\nTraining succeeded!\n");
		
	}
	else
	{
		double tot_cost = 0.00;

		ann = kann_load(fn_in);
		printf("\nTEST...\n");

		if(t_method)
			test(ann, &train_data[(int)((N_DIM_IN+N_DIM_OUT)*N_SAMPLES_PER_POINT*(t_idx+val_idx))], N_SAMPLES_PER_POINT - (int)(N_SAMPLES_PER_POINT*(t_idx+val_idx)), &tot_cost, output_feature_a, output_feature_b, output_feature_c, output_feature_d, p_name, stdnorm, feature_scaling_min, feature_scaling_max);
		else
			test(ann, train_data, N_SAMPLES_PER_POINT, &tot_cost, output_feature_a, output_feature_b, output_feature_c, output_feature_d, p_name, stdnorm, feature_scaling_min, feature_scaling_max);	
		
		printf("\nTest total cost: %g\n", tot_cost);
	}

	kann_delete(ann);
	printf("\nDeleted kann network.\n");
	return 0;
}
