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

def plotter(mae_train_arr, mae_cv_arr, mae_test_arr, loss_arr):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    n_epoch = len(mae_train_arr)
    ax1.plot(range(1,n_epoch+1), mae_train_arr, 'kd-', alpha=.6);
    ax1.plot(range(1,n_epoch+1), mae_cv_arr, 'rd-', alpha=.6);
    ax1.plot(range(1,n_epoch+1), mae_test_arr, 'bd-', alpha=.6);
    ax1.legend(['train', 'cv', 'test'])
    ax2.plot(range(1,n_epoch+1), loss_arr, 'kd-', alpha=.6);
    
    ax1.set_title('MAE')
    ax2.set_title('Training Loss (RMS error)')
    ax1.set_xlabel('epochs')
    ax2.set_xlabel('epochs')

    i_min = np.argmin(mae_cv_arr)
    ax1.plot(i_min+1,mae_cv_arr[i_min], 'ro', markersize=13,
                markeredgewidth=2, markerfacecolor='None')

    return ax1, ax2

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def split_data(Y_indices, Y):
    # training data takes up 64% of shuffled data
    # cv data 16% and
    # test data 20%
    
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

def evaluate(sess, cv_Y, cv_Y_indices, Y_pred, Y_indices, Y, BATCH_SIZE):
    i0 = (-1) * BATCH_SIZE
    i1 = 0
    preds = np.zeros((len(cv_Y) // BATCH_SIZE) * BATCH_SIZE)
    for step in range(len(cv_Y) // BATCH_SIZE):
        i0 = i0 + BATCH_SIZE
        i1 = i1 + BATCH_SIZE
        preds[i0:i1] = sess.run(Y_pred, feed_dict={Y_indices: cv_Y_indices[i0:i1], Y: cv_Y[i0:i1]})

    mae = np.sum(np.abs(cv_Y[:i1] - preds)) / preds.shape[0]
    return preds, mae
    
    
    