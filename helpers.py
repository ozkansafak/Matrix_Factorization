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

def print_runtime(start, p_flag=False):
    end = time.time()
    if p_flag:
        return 'Runtime: %d min %d sec' % ((end-start)//60, (end-start)%60)
    else:
        print('Runtime: %d min %d sec' % ((end-start)//60, (end-start)%60))
        return None
    
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
    
    ax2.set_title('Training Loss (RMS error)')
    ax1.set_xlabel('epochs')
    ax2.set_xlabel('epochs')
    ax2.set_ylim(0.7, 1)
    i_min = np.argmin(mae_cv_arr)
    ax1.plot(i_min+1, mae_cv_arr[i_min], 'ro', markersize=13,
                markeredgewidth=2, markerfacecolor='None')
    
    return ax1, ax2

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def calculate_and_pickle_u_mean_dict(train_Y_indices, train_Y, cv_Y_indices, cv_Y):
    start = time.time()
    all_user_indices = np.concatenate((train_Y_indices, cv_Y_indices))[:,0]
    users = set(all_user_indices)
    
    all_Y = np.concatenate((train_Y, cv_Y))
    u_mean_dict = dict() 
    print('Calculating u_mean_dict.... (it takes ~45 mins)')
    
    for u in users:
        u_mean_dict[u] = np.mean(all_Y[all_user_indices == u])
    
    print('Writing to u_mean_dict.pkl....')
    with open('data/u_mean_dict.pkl', 'wb') as f:
        pickle.dump(u_mean_dict, f, pickle.HIGHEST_PROTOCOL)

    o = print_runtime(start, True)
    print('Finished u_mean_dict... ' + o)
    return u_mean_dict

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def calculate_and_pickle_v_mean_dict(train_Y_indices, train_Y, cv_Y_indices, cv_Y):
    start = time.time()
    all_movie_indices = np.concatenate((train_Y_indices, cv_Y_indices))[:,1]
    movies = set(all_movie_indices)
    
    all_Y = np.concatenate((train_Y, cv_Y))
    v_mean_dict = dict() 
    print('Calculating v_mean_dict.... (it takes ~45 mins)')
    
    i = 0
    for v in movies:
        i += 1
        print('i:%d,  v:%d' % (i,v), end='\r')
        v_mean_dict[v] = np.mean(all_Y[all_movie_indices == v])
    
    print('Writing to v_mean_dict.pkl....')
    with open('data/v_mean_dict.pkl', 'wb') as f:
        pickle.dump(v_mean_dict, f, pickle.HIGHEST_PROTOCOL)

    o = print_runtime(start, True)
    print('Finished v_mean_dict... ' + o)
    return v_mean_dict

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
    def __init__(self, Y_indices, Y_values, train_u_mean, train_v_mean, BATCH_SIZE):
        # name u_mean and v_mean and have them have the same shape as and corresponding values to Y_indices and Y
        # the original u_mean.pkl and v_mean.pl should be named u_mean_dict.pkl and v_mean_dict.pkl
        self.Y_indices = Y_indices
        self.Y_values = Y_values
        self.train_u_mean = train_u_mean
        self.train_v_mean = train_v_mean
        self.BATCH_SIZE = BATCH_SIZE
        self.i0 = np.inf
        self.i1 = np.inf
        self.epoch = 0
        self.broken = False
        self.last_batch = False
    #
    def next(self):
        if self.i1 >= len(self.Y_values):
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
            self.i1 = min(self.i0 + self.BATCH_SIZE, len(self.Y_values))
            if self.i1 - self.i0 < self.BATCH_SIZE:
                # broken batch.
                self.broken = True
            if self.i1 == len(self.Y_values):
                self.last_batch = True
                
        return self.Y_indices[self.i0:self.i1], self.Y_values[self.i0:self.i1], \
               self.train_u_mean[self.i0:self.i1], self.train_v_mean[self.i0:self.i1]
    # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def get_stacked_UV(Y_indices, Y, U, V, k, u_mean, v_mean, BATCH_SIZE):
    u_idx = Y_indices[:,0]
    v_idx = Y_indices[:,1]
    rows_U = tf.transpose(np.ones((k,1), dtype=np.int32)*u_idx)
    rows_V = tf.transpose(np.ones((k,1), dtype=np.int32)*v_idx)
    cols = np.arange(k, dtype=np.int32).reshape((1,-1))
    cols = tf.tile(cols, [BATCH_SIZE,1])

    indices_U = tf.stack([rows_U, cols], -1)
    indices_V = tf.stack([rows_V, cols], -1)
    stacked_U = tf.gather_nd(U, indices_U)
    stacked_V = tf.gather_nd(V, indices_V)
    # .....................................
    
    rows_U = tf.transpose(np.ones((k,1), dtype=np.int32)*u_idx)
    rows_V = tf.transpose(np.ones((k,1), dtype=np.int32)*v_idx)
    cols = np.arange(k, dtype=np.int32).reshape((1,-1))
    cols = tf.tile(cols, [BATCH_SIZE,1])
    
    indices_U = tf.stack([rows_U, cols], -1)
    indices_V = tf.stack([rows_V, cols], -1)
    stacked_u_mean = tf.gather_nd(u_mean, indices_U)
    stacked_v_mean = tf.gather_nd(v_mean, indices_V)
    
    return stacked_U, stacked_V, stacked_u_mean, stacked_v_mean

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def evaluate_preds_n_mae(sess, cv_Y, cv_Y_indices, Y_pred, Y_indices, Y, BATCH_SIZE):
    i0 = (-1) * BATCH_SIZE
    i1 = 0
    preds = np.zeros((len(cv_Y) // BATCH_SIZE) * BATCH_SIZE)
    for step in range(len(cv_Y) // BATCH_SIZE):
        i0 = i0 + BATCH_SIZE
        i1 = i1 + BATCH_SIZE
        preds[i0:i1] = sess.run(Y_pred, feed_dict={Y_indices: cv_Y_indices[i0:i1], Y: cv_Y[i0:i1]})

    mae = np.sum(np.abs(cv_Y[:i1] - preds)) / preds.shape[0]
    return preds, mae

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def npy_read_data():
    
    print('Reading .npy and .pkl files....')
    
    with open('data/train_Y_indices.npy', 'rb') as f:
        train_Y_indices = np.load(f)
    with open('data/train_Y.npy', 'rb') as f:
        train_Y = np.load(f)

    with open('data/cv_Y_indices.npy', 'rb') as f:
        cv_Y_indices = np.load(f)
    with open('data/cv_Y.npy', 'rb') as f:
        cv_Y = np.load(f)

    with open('data/test_Y_indices.npy', 'rb') as f:
        test_Y_indices = np.load(f)
    with open('data/test_Y.npy', 'rb') as f:
        test_Y = np.load(f)
        
    with open('data/u_mean_dict.pkl', 'rb') as f:
        u_mean_dict = pickle.load(f)
    with open('data/v_mean_dict.pkl', 'rb') as f:
        v_mean_dict = pickle.load(f)

    return train_Y_indices, train_Y, cv_Y_indices, cv_Y, test_Y_indices, test_Y, u_mean_dict, v_mean_dict

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def read_and_split_data(RATINGS_PATH='data_ml-20m/ratings.csv', MOVIES_PATH='data_ml-20m/movies.csv'):
    # Read the Lab41 dataset into 
    # train_Y, cv_Y, test_Y
    # train_Y_indices, cv_Y_indices, test_Y_indices
    
    ls = os.listdir('data')

    if ('test_Y_indices.npy' in ls) and ('train_Y_indices.npy' in ls) and ('cv_Y_indices.npy' in ls) and \
        ('test_Y.npy' in ls) and ('train_Y.npy' in ls) and ('cv_Y.npy' in ls) and ('u_mean_dict.pkl' in ls) and \
        ('v_mean_dict.pkl' in ls):
        
        train_Y_indices, train_Y, cv_Y_indices, cv_Y, test_Y_indices, test_Y, u_mean_dict, v_mean_dict =  npy_read_data()
        
    else:
        print('*npy and/or *pkl files not found.')
        print('Read raw data directly from %s and %s.' % (RATINGS_PATH, MOVIES_PATH))

        with open(RATINGS_PATH) as f:
            ratings = f.readlines()
        with open(MOVIES_PATH) as f:
            movies = f.readlines()

        # mapping from movie ID (as described in 'movies.csv'), to index in matrix V
        from_mID_2_idx = {}
        for i, row in enumerate(movies[1:]):
            mID = int(row.split(',')[0])
            from_mID_2_idx[mID] = i

        #...................................................

        N = len(ratings[1:])
        Y_indices = np.zeros((N,2), dtype=np.int32)
        Y = np.zeros(N)

        # Read in Y and Y_indices which consists of all training, cv and test data
        for i, row in enumerate(ratings[1:]):
            uID, mID, yij = [e for e in row.split(',')[:-1]]
            u_idx = int(uID) - 1
            v_idx = from_mID_2_idx[int(mID)]
            yij = float(yij)

            Y_indices[i] = [u_idx, v_idx]
            Y[i] = yij

        #...................................................

        # when the indices are not shuffled, U and V matrices get skewed and best mae_cv comes out about 0.8
        Y_indices, Y = shuffler(Y_indices, Y)
        print('Y and Y_indices read in and shuffled. ')
        
        train_Y_indices, train_Y, cv_Y_indices, cv_Y, test_Y_indices, test_Y = split_data(Y_indices, Y)
        
        if ('u_mean_dict.pkl' not in ls):
            u_mean_dict = calculate_and_pickle_u_mean_dict(train_Y_indices, train_Y, cv_Y_indices, cv_Y)
        else:
            with open('data/u_mean_dict.pkl', 'rb') as f:
                u_mean_dict = pickle.load(f)

        if ('v_mean_dict.pkl' not in ls):
            v_mean_dict = calculate_and_pickle_v_mean_dict(train_Y_indices, train_Y, cv_Y_indices, cv_Y)
        else:
            with open('data/v_mean_dict.pkl', 'rb') as f:
                v_mean_dict = pickle.load(f)

        # Y = zero_out_the_mean(Y_indices, Y, u_mean_dict)

        train_Y_indices, train_Y, cv_Y_indices, cv_Y, test_Y_indices, test_Y = split_data(Y_indices, Y)

        with open('data/train_Y_indices.npy', 'wb+') as f:
            np.save(f, train_Y_indices)
        with open('data/train_Y.npy', 'wb+') as f:
            np.save(f, train_Y)

        with open('data/cv_Y_indices.npy', 'wb+') as f:
            np.save(f, cv_Y_indices)
        with open('data/cv_Y.npy', 'wb+') as f:
            np.save(f, cv_Y)

        with open('data/test_Y_indices.npy', 'wb+') as f:
            np.save(f, test_Y_indices)
        with open('data/test_Y.npy', 'wb+') as f:
            np.save(f, test_Y)
    
    d = np.concatenate((train_Y_indices[:,0], cv_Y_indices[:,0], test_Y_indices[:,0]))
    n_users = len(set(d))

    d = np.concatenate((train_Y_indices[:,1], cv_Y_indices[:,1], test_Y_indices[:,1]))
    n_movies = len(set(d))
    
    # n_users = 138493
    # n_movies = len(movies[1:]) # n_movies = 27278

    return train_Y_indices, train_Y, cv_Y_indices, cv_Y, test_Y_indices, test_Y, n_users, n_movies, u_mean_dict, v_mean_dict

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def construct_graph(LAMBDA=0, k=10, lr=0.001, BATCH_SIZE=1024*16, n_users=138493, n_movies=26744):

    Y = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE,)) 
    Y_indices = tf.placeholder(dtype=tf.int32, shape=(BATCH_SIZE,2)) 
    u_mean = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE,1)) 
    v_mean = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE,1)) 
    
    # initialization of U and V is critical. 
    # set mean=np.sqrt(mu/k), where mu ~ 3 or 3.5
    U = tf.Variable(tf.truncated_normal(shape=(n_users,k), mean=np.sqrt(3.5/k), stddev=0.2), dtype=tf.float32)
    V = tf.Variable(tf.truncated_normal(shape=(n_movies,k), mean=np.sqrt(3.5/k), stddev=0.2), dtype=tf.float32)
        
    # weights for cross-features
    UV_xft = tf.Variable(tf.truncated_normal(shape=(k,k), mean=-np.sqrt(1./k), stddev=0.2), dtype=tf.float32)
    UU_xft = tf.Variable(tf.truncated_normal(shape=(k,k), mean=-np.sqrt(1./k), stddev=0.2), dtype=tf.float32)
    VV_xft = tf.Variable(tf.truncated_normal(shape=(k,k), mean=-np.sqrt(1./k), stddev=0.2), dtype=tf.float32)
    
    #.............................................. 
    
    stacked_U, stacked_V, stacked_u_mean, stacked_v_mean = get_stacked_UV(Y_indices, Y, U, V, k, u_mean, v_mean, BATCH_SIZE)

    # the term `tf.reduce_sum(U**2)` without passing an axis parameter sums up all the elements of matrix U**2.
    # Return value is a scalar.
    
    #reg = LAMBDA * (((((tf.reduce_sum((stacked_U - stacked_u_mean)**2) + 
    #                tf.reduce_sum((stacked_V - stacked_v_mean)**2)) + 
    #                tf.reduce_sum((UV_xft**2))) + 
    #                tf.reduce_sum((UU_xft**2))) + 
    #                tf.reduce_sum((VV_xft**2)))) / (BATCH_SIZE*k)
    
    reg = tf.constant(0.0, dtype=tf.float32)
    
    # the term `tf.multiply(stacked_U, stacked_V)` is elementwise multiplication.
    # Applying tf.reduce_sum(M, axis=1)--where M is a matrix--will sum all rows and return a column vector.
    # Y_pred is a column vector of ratings corresponding to Y_indices

    # ...........................................................
    lin = tf.reduce_sum(tf.multiply(stacked_U, stacked_V), axis=1) 

    xft = UV_xft[0,0] * tf.multiply(tf.transpose(stacked_U)[0], tf.transpose(stacked_V)[1])
    for p in range(k):
        for q in range(0, k):
            xft += UV_xft[p,q] * tf.multiply(tf.transpose(stacked_U)[p], tf.transpose(stacked_V)[q])

    for p in range(k):
        for q in range(p, k):
            xft += UU_xft[p,q] * tf.multiply(tf.transpose(stacked_U)[p], tf.transpose(stacked_U)[q])

    for p in range(k):
        for q in range(p, k):
            xft += VV_xft[p,q] * tf.multiply(tf.transpose(stacked_V)[p], tf.transpose(stacked_V)[q])
    #xft = tf.constant(np.zeros(BATCH_SIZE,), dtype=tf.float32)        
    print('xft.get_shape(): %s' % xft.get_shape())
    # ...........................................................
    
    Y_pred = tf.sigmoid(lin) * 5
    
    print('lin.get_shape(): %s ' % lin.get_shape())
    print('xft.get_shape(): %s' % xft.get_shape())
    print('Y_pred.get_shape(): %s ' % Y_pred.get_shape())
    
    # loss: L2-norm of difference btw actual and predicted ratings
    loss = tf.sqrt(tf.reduce_sum((Y - Y_pred)**2)/BATCH_SIZE)# + reg
    
    # Define train op.
    train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    print('loss.get_shape(): %s ' % loss.get_shape())
    print('tf.sqrt(tf.reduce_sum((Y - Y_pred)**2)/BATCH_SIZE): %a'% tf.sqrt(tf.reduce_sum((Y - Y_pred)**2)/BATCH_SIZE))
    print('reg.get_shape(): %s ' % reg.get_shape())
    print('UV_xft.get_shape(): %s ' % UV_xft.get_shape())
    print('UU_xft.get_shape(): %s ' % UU_xft.get_shape())
    print('VV_xft.get_shape(): %s ' % VV_xft.get_shape())
    print('stacked_U.get_shape(): %s ' % stacked_U.get_shape())
    print('stacked_V.get_shape(): %s ' % stacked_V.get_shape())
    print('tf.reduce_sum(tf.multiply(stacked_U, stacked_V), axis=1).get_shape(): %s ' % tf.reduce_sum(tf.multiply(stacked_U, stacked_V), axis=1).get_shape())

    
    return train, loss, reg, Y_indices, Y, U, V, Y_pred, UV_xft, UU_xft, VV_xft, u_mean, v_mean


def train_the_model(Y_indices, Y, train_Y_indices, train_Y, BATCH_SIZE, 
               NUM_EPOCHS, LAMBDA, k, lr, 
               train, loss, reg, U, V, Y_pred, 
               cv_Y, cv_Y_indices, test_Y, test_Y_indices, 
               UV_xft, UU_xft, VV_xft, 
               train_u_mean, train_v_mean, u_mean, v_mean):
    
    n_batches = len(train_Y) // BATCH_SIZE
    init = tf.global_variables_initializer()
    batch = Batch(train_Y_indices, train_Y, train_u_mean, train_v_mean, BATCH_SIZE=BATCH_SIZE)
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
            batch_Y_indices, batch_Y, batch_u_mean, batch_v_mean = batch.next()
            if not batch.broken:
                batch_no += 1
                # _bl: batch loss
                # _br: batch regularization term
                
                _, _bl, _br = sess.run([train, loss, reg], 
                                        feed_dict={Y_indices: batch_Y_indices, Y: batch_Y,\
                                                   u_mean: batch_u_mean, v_mean: batch_v_mean})
                
                _loss += _bl
                _reg += _br
                print("batch_no: {}, _loss estimate: {:6.4f}, t={:6.2f} sec".format(
                        batch_no, _loss/batch_no, time.time()-epoch_end), end='\r') 
            
            if batch.last_batch: 
                # fetch the state of U, V matrices at current epoch
                _U, _V = sess.run([U, V])
                _UV_xft, _UU_xft, _VV_xft = sess.run([UV_xft, UU_xft, VV_xft])
                # fetch the mae's
                _, _mae_train = evaluate_preds_n_mae(sess, train_Y, train_Y_indices, Y_pred, Y_indices, Y, BATCH_SIZE)
                _, _mae_cv = evaluate_preds_n_mae(sess, cv_Y, cv_Y_indices, Y_pred, Y_indices, Y, BATCH_SIZE)
                preds, _mae_test = evaluate_preds_n_mae(sess, test_Y, test_Y_indices, Y_pred, Y_indices, Y, BATCH_SIZE)
                
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
           _U, _V, _UV_xft, _UU_xft, _VV_xft

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
