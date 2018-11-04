/*
#################################################################
# ANNPFE - Artificial Neural Network Prototyping Front-End      #
# 			Final Built, v0.1 - 05/11/2018			            #
#					Authors/Developer: 						    #
# 	      		Marco Chiarelli @ UNISALENTO & CMCC			    #
# 			Gabriele Accarino   @ UNISALENTO & CMCC				#
#  					marco_chiarelli@yahoo.it  					#
#  					marco.chiarelli@cmcc.it   					#
#  				   gabriele.accarino@cmcc.it					#
#################################################################
*/

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>
#include <string.h>

#include "kann/kann.h"

#include <sys/time.h>

#define N_H_LAYERS 3
#define N_NEURONS 120
#define N_EPOCHS 3000
#define LEARNING_RATE 0.001f
#define DROPOUT 0.2f
#define N_FEATURES 12 // 19 // 6
#define N_DIM_IN N_FEATURES
#define N_DIM_OUT 1
#define N_TIMESTEPS 1
#define N_MINIBATCH 1 // ONLINE LEARNING
#define L_NORM 1
#define N_THREADS 8
#define RANDOM_SEED 11
#define TRAINING_IDX 0.8f


#define TO_APPLY 1
#define NET_BINARY_NAME "kann_net.bin"
#define POINT_HEADER "test_dataset.h" // _masks.h"

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

static void normalize_minmax(float *vec, int size, int pitch, int max_pitch, float min_x, float max_x, float a, float b)
{
	int i;

	for (i = pitch+max_pitch; i+max_pitch < size; i+=max_pitch)
		vec[i] = (b - a)*((vec[i] - min_x)/(max_x - min_x)) + a;

}

static inline float atomic_denormalize(register float y, float min_x, float max_x, float a, float b)
{
	return ((y-a)/(b-a))*(max_x-min_x) + min_x;
}

static int train(kann_t *net, float *train_data, int n_train_ex, float lr, int ulen, int mbs, int max_epoch, int n_threads)
{
	int k;
	kann_t *ua;
	float *r;
	float **x, **y;
	float best_cost = 1e30f;
	int n_var = kann_size_var(net); 
	int n_dim_in = kann_dim_in(net);
	int n_dim_out = kann_dim_out(net);

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

	ua = kann_unroll(net, ulen);            // unroll; the mini batch size is 1
	kann_feed_bind(ua, KANN_F_IN,    0, x); // bind _x_ to input nodes
	kann_feed_bind(ua, KANN_F_TRUTH, 0, y); // bind _y_ to truth nodes
	kann_set_batch_size(ua, mbs);
	kann_switch(ua, 1);

	kann_mt(ua, n_threads, n_threads);

	int i, j, b, l;

	for (i = 0; i < max_epoch && train_exec; ++i)
	{
		int iter = 0;
		int nelem = 0;
		double train_cost = 0.0;
		int train_tot = 0, val_tot =0, n_cerr = 0;
		for (j = 0; j + ulen * mbs < n_train_ex; j += ulen * mbs)
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
		fprintf(stderr, "epoch: %d; Training cost: %g;\n", i+1, train_cost / train_tot);

	}
	

 	free(y); free(x);
	kann_delete_unrolled(ua); // for an unrolled network, don't use kann_delete()!
	free(r);
	return 0;
}

static int test(kann_t *net, float *test_data, int n_test_ex, double *tot_cost, float *min_x, float *max_x, float a, float b)
{
	struct timeval tp;
	float y1_denorm;
	double cur_cost = 0.00;
	double cpu_time = 0.00;
	int n_dim_in = kann_dim_in(net);
	int n_dim_out = kann_dim_out(net);
	
	// kann_rnn_start(net);
	float *x1;
	float *expected;
	const float *y1;

	kann_switch(net, 0);


	if((x1 = (float*)calloc(n_dim_in, sizeof(float))) == NULL)
		return 1;

	if((expected = (float*)calloc(n_dim_out, sizeof(float))) == NULL)
		return 1;

	int i, j, k, l;
	printf("Test Begin\n");
	printf("Number of ex: %d\n", n_test_ex);

	FILE * fp;
	
	fp=fopen("predictions.csv", "w+");

	// fprintf(fp, ",0\n");

	for (i = 0; i < n_test_ex; ++i){
		for (j = n_dim_out; j < n_dim_in+n_dim_out; ++j)
			x1[j-n_dim_out] = test_data[i*(n_dim_in+n_dim_out) + j];
		
		for (k = 0; k < n_dim_out; ++k)
			expected[k] = test_data[i*(n_dim_in+n_dim_out) + k];

		gettimeofday(&tp, NULL);
		double elaps = -(double)(tp.tv_sec + tp.tv_usec/1000000.0);
		y1 = kann_apply1(net, x1);
		gettimeofday(&tp, NULL);
		cpu_time += elaps+((double)(tp.tv_sec + tp.tv_usec/1000000.0));
		double cur_cost = 0;
	
		fprintf(fp, "%d", i);

		for (l = 0; l < n_dim_out; ++l)
		{
			y1_denorm = atomic_denormalize(y1[l], min_x[l], max_x[l], a, b);
			fprintf(fp, ",%g", y1_denorm);
			cur_cost += (y1_denorm - expected[l])*(y1_denorm - expected[l]);
		}
	
		fprintf(fp, "\n");

		cur_cost /= n_dim_out;
		*tot_cost += cur_cost;
	}

	fclose(fp);
	*tot_cost = *tot_cost/n_test_ex;
	printf("Test Ended\n");
	cpu_time /= n_test_ex;
	printf("\nAverage test time: %lf\n", cpu_time);
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
	char *fn_in, *fn_out = 0;
	float lr, dropout, t_idx;
	const unsigned char to_apply = argc > 1;
	int n_h_layers, n_h_neurons, mini_size, max_epoch, norm, n_threads, seed;
	

	if(to_apply)	
	{
		n_h_layers = argc > 1 ? atoi(argv[1]) : N_H_LAYERS;	
		n_h_neurons = argc > 2 ? atoi(argv[2]) : N_NEURONS;
		mini_size = argc > 3 ? atoi(argv[3]) : N_MINIBATCH;
		max_epoch = argc > 4 ? atoi(argv[4]) : N_EPOCHS;
		lr = argc > 5 ? atof(argv[5]) : LEARNING_RATE;
		dropout = argc > 6 ? atof(argv[6]) : DROPOUT;
		t_idx = argc > 7 ? atof(argv[7]) : TRAINING_IDX;
		norm = argc > 8 ? atoi(argv[8]) : L_NORM;
		n_threads = argc > 9 ? atoi(argv[9]) : N_THREADS;
		seed = argc > 10 ? atoi(argv[10]) : RANDOM_SEED;
		fn_in = argc > 11 ? argv[11] : NET_BINARY_NAME;
	}

	(void) signal(SIGINT, sigexit);

	kad_trap_fe();
	kann_srand(seed);

	float output_feature_min[N_DIM_OUT+N_DIM_IN];
	float output_feature_max[N_DIM_OUT+N_DIM_IN];

	for(i=N_FEATURES+N_DIM_OUT-1; i>=0; --i)
	{
		output_feature_min[i] = MIN(train_data, N_SAMPLES_PER_POINT* (N_FEATURES+N_DIM_OUT), i, (N_FEATURES+N_DIM_OUT));
		output_feature_max[i] = MAX(train_data, N_SAMPLES_PER_POINT* (N_FEATURES+N_DIM_OUT), i, (N_FEATURES+N_DIM_OUT)); 
		if(output_feature_min == output_feature_max && (output_feature_min[i] == 1.00f || !output_feature_min[i]))
			continue;
		 printf("i is %d, outmin: %g, outmax: %g\n", i, output_feature_min[i], output_feature_max[i]);
		normalize_minmax(train_data, N_SAMPLES_PER_POINT*(N_FEATURES+N_DIM_OUT), i, (N_FEATURES+N_DIM_OUT), output_feature_min[i], output_feature_max[i], 0.00f, 1.00f);
	}
	
	(void) getchar();
	
	if (to_apply)
	{
		// model generation
		kad_node_t *t;
		int rnn_flag = KANN_RNN_VAR_H0;
		if (norm) rnn_flag |= KANN_RNN_NORM;
		t = kann_layer_input(N_DIM_IN); // t = kann_layer_input(d->n_in);

		for (i = 0; i < n_h_layers; ++i)
			t = kad_sigm(kann_layer_dense(t, n_h_neurons));
			// t = kann_layer_rnn(t, n_h_neurons, rnn_flag);
			// t = kad_sigm(t);
			if(dropout)
				t = kann_layer_dropout(t, dropout);

		ann = kann_new(kann_layer_cost(t, N_DIM_OUT, KANN_C_MSE), 0);
		train(ann, train_data, (int)(TRAINING_IDX*N_SAMPLES_PER_POINT), lr, N_TIMESTEPS, mini_size, max_epoch, n_threads); // max_epoch);
		kann_save(NET_BINARY_NAME, ann);
		printf("\ntraining succeeded\n");
		
	}
	else
	{
		double tot_cost = 0.00;
		ann = kann_load(NET_BINARY_NAME);

		printf("\nTEST...\n");
		//test(ann, train_data, N_SAMPLES_PER_POINT, &tot_cost, output_feature_min, output_feature_max, 0.00f, 1.00f);
		test(ann, &train_data[(int)((N_DIM_IN+N_DIM_OUT)*N_SAMPLES_PER_POINT*TRAINING_IDX)], N_SAMPLES_PER_POINT - (int)(N_SAMPLES_PER_POINT*TRAINING_IDX), &tot_cost, output_feature_min, output_feature_max, 0.00f, 1.00f);	
		printf("\ntest tot cost: %g\n", tot_cost);
	}

	kann_delete(ann);
	printf("\ndeleted kann network\n");
	return 0;
}
