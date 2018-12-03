from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib import colors
import pandas as pd
import numpy as np
import gc
import os
from math import sqrt
from sklearn.metrics import mean_squared_error
#from point_tags import point_tags
#from manneR_conf import *
from mpi4py import MPI

# MPI Settings
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

plot_ts = False
yyhat_dir = "yyhat"

np.set_printoptions(precision = 32)
var = '8_8_8'
#test_dim = 998
#point_tags=[var]
#point_tags=['5_7_10']
#point_tags=['14_2_2']
#point_tags=['29_17_14']
#point_tags=['29_17_14_25000']
#point_tags=['23_14_29_boundary']
#point_tags=['8_2_20_boundary']

# @ Plot results of simulation
def plot_results(inv_yhat, inv_y, var):
    # --> TO DO (2) parametrico
    time_steps = len(inv_yhat)
    fig = plt.figure()
    DPI = fig.get_dpi()
    fig.set_size_inches(1960/float(DPI), 1080/float(DPI))
    plt.xticks(np.arange(1, time_steps, int(time_steps / 10)))
    print(len(inv_y))
    plt.plot(inv_y[:50000], label="Real Values")
    plt.plot(inv_yhat[:50000], label='Estimated Values')
    #plt.plot(inv_y[2264:], label="Real Values")
    #plt.plot(inv_yhat[2264:-10], label='Estimated Values')
    #plt.plot(inv_y[35000:], label="Real Values")
    #plt.plot(inv_yhat[10:-10], label='Estimated Values')
    #plt.plot(inv_yhat.iloc[ time_steps - test_dim : ][:], label='Test Values', color = 'red', markersize = 2)
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=1, borderaxespad=0.)
    plt.legend()
    plt.xlabel('Time Steps')
    plt.ylabel(str(var))
    plt.title('Advection')
    plt.grid(True)
    #fig.savefig(var+'.png')
    #plt.close()
    plt.show()

def loss_plot(train_loss, validation_loss):
    plt.plot(train_loss, label='training_loss')
    plt.plot(validation_loss, label='validation_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.legend()
    plt.show()

def error_plot(error_score):
    plt.plot(error_score, label='error_score')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Error Plot')
    plt.legend()
    plt.show()

train_loss = pd.read_csv('training_loss.csv', delimiter = ',', engine = 'c', index_col = 0, dtype = np.float64).values
validation_loss = pd.read_csv('validation_loss.csv', delimiter = ',', engine = 'c', index_col = 0, dtype = np.float64).values
error_score = pd.read_csv('error_score.csv', delimiter = ',', engine = 'c', index_col = 0, dtype = np.float64).values
y = pd.read_csv('y_df_' + var + '.csv', delimiter = ',', engine = 'c', index_col = 0, dtype = np.float64).values
yhat = pd.read_csv('predictions.csv', delimiter = ',', engine = 'c', index_col = 0, dtype = np.float64).values

'''
for i in range(len(train_loss)):
    print(train_loss[i], validation_loss[i])
    if i == 0:
        break
'''

loss_plot(train_loss, validation_loss)
error_plot(error_score)
plot_results(yhat, y, var)

#yhat = pd.read_csv(yyhat_dir + '/y_df_' + pt + '.csv', delimiter = ',', engine = 'c', index_col = 0, dtype = np.float64).values
#yhat = pd.read_csv(yyhat_dir + '/yhat_df_' + pt + '.csv', delimiter = ',', engine = 'c', index_col = 0, dtype = np.float64).values
#yhat = pd.read_csv(yyhat_dir + '/yhat_df_' + pt + '_onlyptab.csv', delimiter = ',', engine = 'c', index_col = 0, dtype = np.float64).values
#yhat = pd.read_csv(yyhat_dir + '/yhat_df_8_8_8_ptx_neighbours_nolag.csv', delimiter = ',', engine = 'c', index_col = 0, dtype = np.float64).values
#yhat = pd.read_csv(yyhat_dir + '/yhat_df_8_8_8_ptx_neighbours.csv', delimiter = ',', engine = 'c', index_col = 0, dtype = np.float64).values
#yhat = pd.read_csv(yyhat_dir + '/yhat_df_8_8_8_ptx.csv', delimiter = ',', engine = 'c', index_col = 0, dtype = np.float64).values

# rmse = sqrt(mean_squared_error(y, yhat))

#plot_results(yhat, y, 'pta_' + pt)

    #gc.collect()

#history = pd.read_csv('history_df.csv', delimiter = ',', engine = 'c', index_col = 0, dtype = np.float64)
#plot_loss(history)
