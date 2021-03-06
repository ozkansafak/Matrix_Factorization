import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import time
import os
import matplotlib.pylab as pylab
from matplotlib.ticker import MaxNLocator

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def get_train_u_mean(train_R_indices, u_mean_dict, v_mean_dict):
    N = train_R_indices.shape[0]
    train_u_mean = np.zeros((N, 1))
    train_v_mean = np.zeros((N, 1))
    for i in range(N):
        u_idx = train_R_indices[i,0]
        v_idx = train_R_indices[i,1]
        train_u_mean[i] = u_mean_dict[u_idx]
        train_v_mean[i] = v_mean_dict[v_idx]

    return train_u_mean, train_v_mean

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

def plotter(mae_train_arr, mae_cv_arr, mae_test_arr, loss_arr, BATCH_SIZE):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    n_epoch = len(mae_train_arr)
    ax1.plot(range(1, n_epoch+1), mae_test_arr, 'bd-', alpha=.6);
    ax1.plot(range(1, n_epoch+1), mae_cv_arr, 'rd-', alpha=.6);
    ax1.plot(range(1, n_epoch+1), mae_train_arr, 'kd-', alpha=.6);
    ax1.set_xlim(left=1);
    ax1.set_ylim(0.5, 0.65)
    i_min = np.argmin(mae_cv_arr)
    ax1.set_title('MAE_cv: {:6.4f}, BATCH_SIZE: {}*1024'.format(mae_cv_arr[i_min], BATCH_SIZE//1024))
    ax1.legend(['test', 'cv', 'train'])
    ax2.plot(range(1, n_epoch+1), loss_arr, 'kd-', alpha=.6);
    
    ax2.set_title('Training Loss (RMS)')
    ax1.set_xlabel('epochs')
    ax2.set_xlabel('epochs')
    ax2.set_ylim(0.7, 1)
    i_min = np.argmin(mae_cv_arr)
    ax1.plot(i_min+1,mae_cv_arr[i_min], 'ro', markersize=13,
                markeredgewidth=2, markerfacecolor='None')
    
    return ax1, ax2

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def calculate_u_mean(train_R_indices, train_R, cv_R_indices, cv_R):
    start = time.time()
    users = np.concatenate((train_R_indices, cv_R_indices))[:,0]
    users = set(users)
    all_data_indices = np.concatenate((train_R_indices, cv_R_indices))
    all_data_R = np.concatenate((train_R, cv_R))
    u_mean = dict() 
    print('Calculating u_mean.... (it takes ~45 mins)')
    
    for u in users:
        u_mean[u] = np.mean(all_data_R[all_data_indices[:,0] == u])
    
    print('Writing to u_mean.pkl....')
    with open('data/u_mean.pkl', 'wb') as f:
        pickle.dump(u_mean, f, pickle.HIGHEST_PROTOCOL)

    return

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def calculate_v_mean(train_R_indices, train_R, cv_R_indices, cv_R):
    start = time.time()
    movies = np.concatenate((train_R_indices, cv_R_indices))[:,1]
    movies = set(movies)
    all_data_indices = np.concatenate((train_R_indices, cv_R_indices))
    all_data_R = np.concatenate((train_R, cv_R))
    v_mean = dict() 
    print('Calculating v_mean.... (it takes ~45 mins)')
    
    i = 0
    for v in movies:
        i += 1
        print('i:%d,  v:%d' % (i,v), end='\r')
        v_mean[v] = np.mean(all_data_R[all_data_indices[:,1] == v])
    
    print('Writing to v_mean.pkl....')
    with open('data/v_mean.pkl', 'wb') as f:
        pickle.dump(v_mean, f, pickle.HIGHEST_PROTOCOL)

    return

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def split_data(R_indices, R):
    # training data takes up 64% of shuffled data
    # cv data 16% and
    # test data 20%
    
    N = R.shape[0]
    i0 = int(N*.64)
    i1 = int(N*.80)
    
    train_R_indices = R_indices[:i0]
    train_R = R[:i0]
    
    cv_R_indices = R_indices[i0:i1]
    cv_R = R[i0:i1]
    
    test_R_indices = R_indices[i1:]
    test_R = R[i1:]
    
    return train_R_indices, train_R, cv_R_indices, cv_R, test_R_indices, test_R

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
    def __init__(self, R_indices, R, train_u_mean, train_v_mean, BATCH_SIZE):
        self.R_indices = R_indices
        self.R_values = R
        self.train_u_mean = train_u_mean
        self.train_v_mean = train_v_mean
        self.BATCH_SIZE = BATCH_SIZE
        self.i0 = np.inf
        self.i1 = np.inf
        self.epoch = 0
        self.broken = False
        self.last_batch = False

    def next(self):
        if self.i1 >= len(self.R_values):
            # new epoch.
            self.epoch += 1
            self.broken = False
            self.last_batch = False
            # reset the counter. 
            self.i0 = 0
            self.i1 = self.i0 + self.BATCH_SIZE
            print('New epoch: %d %s' % (self.epoch, '*'*30))
        else:
            self.i0 = self.i0 + self.BATCH_SIZE
            self.i1 = min(self.i0 + self.BATCH_SIZE, len(self.R_values))
            if self.i1 - self.i0 < self.BATCH_SIZE:
                # broken batch.
                self.broken = True
            if self.i1 == len(self.R_values):
                self.last_batch = True
        #
        return self.R_indices[self.i0:self.i1], self.R_values[self.i0:self.i1], \
               self.train_u_mean[self.i0:self.i1], self.train_v_mean[self.i0:self.i1]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def evaluate_preds_n_mae(sess, cv_R, cv_R_indices, R_pred, R_indices, R, BATCH_SIZE):
    i0 = (-1) * BATCH_SIZE
    i1 = 0
    preds = np.zeros((len(cv_R) // BATCH_SIZE) * BATCH_SIZE)
    for step in range(len(cv_R) // BATCH_SIZE):
        i0 = i0 + BATCH_SIZE
        i1 = i0 + BATCH_SIZE
        preds[i0:i1] = sess.run(R_pred, feed_dict={R_indices: cv_R_indices[i0:i1], R: cv_R[i0:i1]})

    mae = np.sum(np.abs(cv_R[:i1] - preds)) / preds.shape[0]
    return preds, mae

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def npy_read_data():
    print('Reading train_*.npy , cv_*.npy , test_*.npy and and u_mean.pkl files....')
    
    with open('data/train_R_indices.npy', 'rb') as f:
        train_R_indices = np.load(f)
    with open('data/train_R.npy', 'rb') as f:
        train_R = np.load(f)

    with open('data/cv_R_indices.npy', 'rb') as f:
        cv_R_indices = np.load(f)
    with open('data/cv_R.npy', 'rb') as f:
        cv_R = np.load(f)

    with open('data/test_R_indices.npy', 'rb') as f:
        test_R_indices = np.load(f)
    with open('data/test_R.npy', 'rb') as f:
        test_R = np.load(f)
        
    with open('data/u_mean_dict.pkl', 'rb') as f:
        u_mean_dict = pickle.load(f)
    with open('data/v_mean_dict.pkl', 'rb') as f:
        v_mean_dict = pickle.load(f)

    return train_R_indices, train_R, cv_R_indices, cv_R, test_R_indices, test_R, u_mean_dict, v_mean_dict


def read_and_split_data(RATINGS_PATH='data_ml-20m/ratings.csv', MOVIES_PATH='data_ml-20m/movies.csv'):
    # Read the Lab41 dataset into 
    # train_R, cv_R, test_R
    # train_R_indices, cv_R_indices, test_R_indices
    start = time.time()
    ls = os.listdir('data')
    
    with open(RATINGS_PATH) as f:
        ratings = f.readlines()
    with open(MOVIES_PATH) as f:
        movies = f.readlines()

    n_users = 138493
    n_movies = len(movies[1:]) # n_movies = 27278

    if ('test_R_indices.npy' in ls) and ('train_R_indices.npy' in ls) and ('cv_R_indices.npy' in ls) and \
        ('test_R.npy' in ls) and ('train_R.npy' in ls) and ('cv_R.npy' in ls) and ('u_mean_dict.pkl' in ls) and \
        ('v_mean_dict.pkl' in ls):
        
        train_R_indices, train_R, cv_R_indices, cv_R, test_R_indices, test_R, u_mean_dict, v_mean_dict =  npy_read_data()
        print('n_movies {}'.format(n_movies))
        
    else:
        print('npy and pkl not found...')
        # mapping from movie ID (as described in 'movies.csv'), to index in matrix V
        from_mID_2_idx = {}
        for i, row in enumerate(movies[1:]):
            mID = int(row.split(',')[0])
            from_mID_2_idx[mID] = i

        #...................................................

        N = len(ratings[1:])
        R_indices = np.zeros((N,2), dtype=np.int32)
        R = np.zeros(N)

        # Read in R and R_indices which consists of all training, cv and test data
        for i, row in enumerate(ratings[1:]):
            uID, mID, rij = [e for e in row.split(',')[:-1]]
            u_idx = int(uID) - 1
            v_idx = from_mID_2_idx[int(mID)]
            rij = float(rij)

            R_indices[i] = [u_idx, v_idx]
            R[i] = rij

        #...................................................

        # when the indices are not shuffled, U and V matrices get skewed and best mae_cv comes out about 0.8
        R_indices, R = shuffler(R_indices, R)
        o = print_runtime(start, p_flag=False)
        print('R and R_indices read in and shuffled. ' + o)

        train_R_indices, train_R, cv_R_indices, cv_R, test_R_indices, test_R = split_data(R_indices, R)
        # # # calculate_u_mean(train_R_indices, train_R, cv_R_indices, cv_R)
        # # # calculate_v_mean(train_R_indices, train_R, cv_R_indices, cv_R)
        
        with open('data/u_mean.pkl', 'rb') as f:
            u_mean = pickle.load(f)
        with open('data/v_mean.pkl', 'rb') as f:
            v_mean = pickle.load(f)

        R = zero_out_the_mean(R_indices, R, u_mean)

        train_R_indices, train_R, cv_R_indices, cv_R, test_R_indices, test_R = split_data(R_indices, R)

        with open('data/train_R_indices.npy', 'wb+') as f:
            np.save(f, train_R_indices)
        with open('data/train_R.npy', 'wb+') as f:
            np.save(f, train_R)

        with open('data/cv_R_indices.npy', 'wb+') as f:
            np.save(f, cv_R_indices)
        with open('data/cv_R.npy', 'wb+') as f:
            np.save(f, cv_R)

        with open('data/test_R_indices.npy', 'wb+') as f:
            np.save(f, test_R_indices)
        with open('data/test_R.npy', 'wb+') as f:
            np.save(f, test_R)
       
    return train_R_indices, train_R, cv_R_indices, cv_R, test_R_indices, test_R, n_users, n_movies, u_mean_dict, v_mean_dict

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def get_stacked_UV(R_indices, R, U, V, k, u_mean, v_mean, BATCH_SIZE):
    u_idx = R_indices[:,0]
    v_idx = R_indices[:,1]
    rows_U = tf.transpose(np.ones((k,1), dtype=np.int32)*u_idx)
    rows_V = tf.transpose(np.ones((k,1), dtype=np.int32)*v_idx)
    cols = np.arange(k, dtype=np.int32).reshape((1,-1))
    cols = tf.tile(cols, [BATCH_SIZE,1])

    indices_U = tf.stack([rows_U, cols], -1)
    indices_V = tf.stack([rows_V, cols], -1)
    stacked_U = tf.gather_nd(U, indices_U)
    stacked_V = tf.gather_nd(V, indices_V)
    # .....................................
    
    stacked_u_mean = tf.gather_nd(u_mean, indices_U)
    stacked_v_mean = tf.gather_nd(v_mean, indices_V)
    
    return stacked_U, stacked_V, stacked_u_mean, stacked_v_mean

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def construct_graph(LAMBDA=0, k=10, lr=0.001, BATCH_SIZE=1024*16, n_users=138493, n_movies=27278):

    R = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE,))
    R_indices = tf.placeholder(dtype=tf.int32, shape=(BATCH_SIZE,2))
    u_mean = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE,1)) 
    v_mean = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE,1)) 
    
    # initialization of U and V is critical. 
    # set mean=np.sqrt(mu/k), where mu ~ 3 or 3.5
    U = tf.Variable(tf.truncated_normal(shape=(n_users,k), mean=np.sqrt(3.5/k), stddev=0.2), dtype=tf.float32)
    V = tf.Variable(tf.truncated_normal(shape=(n_movies,k), mean=np.sqrt(3.5/k), stddev=0.2), dtype=tf.float32)

    # weights for cross-features
    X_UV = tf.Variable(tf.truncated_normal(shape=(k,k), mean=0, stddev=0.2), dtype=tf.float32)
    
    #.............................................. 
    
    stacked_U, stacked_V, stacked_u_mean, stacked_v_mean = get_stacked_UV(R_indices, R, U, V, k, u_mean, v_mean, BATCH_SIZE)

    # the term `tf.reduce_sum(U**2)` without passing an axis parameter sums up all the elements of matrix U**2.
    # Return value is a scalar.
    reg = (tf.reduce_sum((stacked_U - stacked_u_mean)**2) + 
           tf.reduce_sum((stacked_V - stacked_v_mean)**2) + 
           tf.reduce_sum((X_UV**2))) / (BATCH_SIZE*k)
    
    # the term `tf.multiply(stacked_U, stacked_V)` is elementwise multiplication.
    # Applying tf.reduce_sum(M, axis=1)--where M is a matrix--will sum all rows and return a column vector.
    # R_pred is a column vector of ratings corresponding to R_indices
    
    lin = tf.reduce_sum(tf.multiply(stacked_U, stacked_V), axis=1) 
    
    # ...........................................................
    # non-linear terms: np.multiply(U[i,:], V[j,:])**2
    # u_cdot_v_square = tf.square(tf.multiply(stacked_U, stacked_V)) 
    # nonlin = tf.reduce_sum(u_cdot_v_square, axis=1)
    
    #xft = tf.transpose(stacked_X_UV, perm=[1,2,0])[0,1] * tf.multiply(tf.transpose(stacked_U)[0], tf.transpose(stacked_V)[1])
    xft = X_UV[0,0] * stacked_U[:,0] * stacked_V[:,0]
    for p in range(k):
        for q in range(k):
            #xft += tf.transpose(stacked_X_UV, perm=[1,2,0])[p,q] * tf.multiply(tf.transpose(stacked_U)[p], tf.transpose(stacked_V)[q])
            xft += X_UV[p,q] * stacked_U[:,p] * stacked_V[:,q]
    # ...........................................................

    R_pred = 0.5 + tf.sigmoid(lin + xft) * 4.5

    # loss: L2-norm of difference btw actual and predicted ratings
    loss = tf.sqrt(tf.reduce_sum((R-R_pred)**2)/BATCH_SIZE) + LAMBDA*reg

    # Define train op.
    train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    
    return train, loss, reg, R_indices, R, U, V, R_pred, X_UV, u_mean, v_mean


def train_the_model(R_indices, R, train_R_indices, train_R, BATCH_SIZE, 
               NUM_EPOCHS, LAMBDA, k, lr, 
               train, loss, reg, U, V, R_pred,
               cv_R, cv_R_indices, test_R, test_R_indices,
               X_UV, 
               train_u_mean, train_v_mean, u_mean, v_mean):
    
    n_batches = len(train_R) // BATCH_SIZE
    init = tf.global_variables_initializer()
    batch = Batch(train_R_indices, train_R, train_u_mean, train_v_mean, BATCH_SIZE=BATCH_SIZE)
    epoch_end = time.time()
    _loss, _reg, batch_no = 0, 0, 0
    mae_train_arr, mae_cv_arr , mae_test_arr, loss_arr = [], [], [], []
    
    f_out = open('out.txt', 'a+')
    dp = 'NUM_EPOCHS: {}\nLAMBDA: {}\nk: {}\nlr: {}\nn_batches: {}\nBATCH_SIZE: {}'.format(\
                                            NUM_EPOCHS, LAMBDA, k, lr, n_batches, BATCH_SIZE)
    f_out.write(dp)
    print('\n' + dp)
    f_out.close()

    with tf.Session() as sess:
        sess.run(init)
        while not (batch.epoch==NUM_EPOCHS and batch.last_batch==True):
            batch_R_indices, batch_R, batch_u_mean, batch_v_mean = batch.next()
            if not batch.broken:
                batch_no += 1
                # _bl: batch loss
                # _br: batch regularization term
                
                _, _bl, _br = sess.run([train, loss, reg], 
                                        feed_dict={R_indices: batch_R_indices, R: batch_R,\
                                                   u_mean: batch_u_mean, v_mean: batch_v_mean})
                
                _loss += _bl
                _reg += _br
                print("batch_no: {}, _loss estimate: {:6.4f}, t={:6.2f} sec".format(
                        batch_no, _loss/batch_no, time.time()-epoch_end), end='\r') 
            
            if batch.last_batch: 
                # fetch the state of U, V matrices at current epoch
                _U, _V = sess.run([U, V])
                _X_UV= sess.run([X_UV])
                # fetch the mae's
                _, _mae_train = evaluate_preds_n_mae(sess, train_R, train_R_indices, R_pred, R_indices, R, BATCH_SIZE)
                _, _mae_cv = evaluate_preds_n_mae(sess, cv_R, cv_R_indices, R_pred, R_indices, R, BATCH_SIZE)
                preds, _mae_test = evaluate_preds_n_mae(sess, test_R, test_R_indices, R_pred, R_indices, R, BATCH_SIZE)
                
                mae_train_arr.append(_mae_train)
                mae_cv_arr.append(_mae_cv)
                mae_test_arr.append(_mae_test)
                loss_arr.append(_loss/n_batches)
                mean_preds = np.mean(preds)
                
                # printing....
                f_out = open('out.txt', 'a+')
                dp = '\nmae_train: %6.4f, **mae_cv: %6.4f**, mae_test: %6.4f,  mean(preds): %6.4f' % \
                      (_mae_train, _mae_cv, _mae_test, np.mean(preds))
                f_out.write(dp)
                print(dp)
                
                #dp = '(_reg/_loss) fraction: %6.4f' % (_reg/_loss)
                #f_out.write(dp)
                #print(dp)
                f_out.close()
                
                # resetting some iteration variables....
                _loss_S = _loss
                batch_no, _loss, _reg = 0, 0, 0
                epoch_end = time.time()
                
    f_out.close()
    return mae_train_arr, mae_cv_arr, mae_test_arr, loss_arr, mean_preds, n_batches, preds, \
           _U, _V, _X_UV

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
