import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import time
import matplotlib.pylab as pylab
from matplotlib.ticker import MaxNLocator

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def print_runtime(start, p_flag=True):
    end = time.time()
    if p_flag:
        print('Runtime: %d min %d sec' % ((end-start)//60, (end-start)%60))
        return None
    else:
        return 'Runtime: %d min %d sec' % ((end-start)//60, (end-start)%60)
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def plotter(xlabel=None, ylabel=None, title=None, xlim=None, ylim=None):
    fig = plt.figure()
    ax = plt.gca()
    fig.set_size_inches((15,5))
    plt.grid('on')
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    if xlim: plt.xlim((0, xlim));
    if ylim: plt.ylim((0, ylim));
    if title: plt.title(title)
    return ax, fig

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def split_data(Y_indices, Y):
    N = Y.shape[0]
    i0 = int(N*.64)
    i1 = int(N*.80)
    
    train_Y_indices = Y_indices[:i0]
    train_Y = Y[:i0]
    
    cv_Y_indices = Y_indices[i0:i1]
    cv_Y = Y[i0:i1]
    
    test_Y_indices = Y_indices[i1:]
    test_Y = Y[i1:]
    
    return train_Y_indices, train_Y, cv_Y_indices, cv_Y, test_Y_indices, test_Y

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def shuffler(a,b):
    i = np.arange(a.shape[0])
    np.random.shuffle(i)
    a = a[i]
    b = b[i]
    return a,b
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

class Batch(object):
    def __init__(self, Y_indices, Y_values, BATCH_SIZE):
        self.Y_indices = Y_indices
        self.Y_values = Y_values
        self.BATCH_SIZE = BATCH_SIZE
        self.i0 = np.inf
        self.i1 = np.inf
        self.epoch = 0
        self.broken = False
        self.last_batch = False

    def next(self):
        if self.i1 >= len(self.Y_values):
            # new epoch.
            self.epoch += 1
            self.broken = False
            self.last_batch = False
            # reset the counter. 
            self.i0 = 0
            self.i1 = self.i0 + self.BATCH_SIZE
            print('\nNew epoch: %d %s' % (self.epoch, '*'*30))
            return self.Y_indices[self.i0:self.i1], self.Y_values[self.i0:self.i1]

        self.i0 = self.i0 + self.BATCH_SIZE
        self.i1 = min(self.i0 + self.BATCH_SIZE, len(self.Y_values))
        if self.i1 - self.i0 < self.BATCH_SIZE:
            # broken batch.
            self.broken = True
        if self.i1 == len(self.Y_values):
            self.last_batch = True
        return self.Y_indices[self.i0:self.i1], self.Y_values[self.i0:self.i1]
        

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def get_sliced_UV(Y_indices, Y, U, V, k, BATCH_SIZE):
    u_idx = Y_indices[:,0]
    v_idx = Y_indices[:,1]
    rows_U = tf.transpose(np.ones((k,1), dtype=np.int32)*u_idx)
    rows_V = tf.transpose(np.ones((k,1), dtype=np.int32)*v_idx)
    cols = np.arange(k, dtype=np.int32).reshape((1,-1))
    cols = tf.tile(cols, [BATCH_SIZE,1])

    indices_U = tf.stack([rows_U, cols], -1)
    indices_V = tf.stack([rows_V, cols], -1)
    sliced_U = tf.gather_nd(U, indices_U)
    sliced_V = tf.gather_nd(V, indices_V)
    
    return sliced_U, sliced_V

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def evaluate(cv_Y, cv_Y_indices, _U, _V, BATCH_SIZE):
    i0 = (-1) * BATCH_SIZE
    i1 = 0
    preds = np.zeros((len(cv_Y) // BATCH_SIZE) * BATCH_SIZE)
    for step in range(len(cv_Y) // BATCH_SIZE):
        i0 = i0 + BATCH_SIZE
        i1 = i1 + BATCH_SIZE
        u_idx = cv_Y_indices[i0:i1, 0]
        v_idx = cv_Y_indices[i0:i1, 1]
        sliced_U = _U[u_idx] 
        sliced_V = _V[v_idx] 
        preds[i0:i1] = np.sum(np.multiply(sliced_U, sliced_V), axis=1)
        
    mae = np.sum(np.abs(cv_Y[:i1] - preds)) / preds.shape[0]
    return preds, mae

        
        
        
        