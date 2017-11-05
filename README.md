### Difficulties on implementing Matrix Factorization on `tensorflow`

- Can't pass `tf.placeholder( ... ,shape=(None)` for input data `Y` and `Y_indices`. Every new user and new movie in `Y` will introduce new tunable variables.
In order to implement matrix factorization on `tensorflow`, I had to fix the `shape` of the `tf.placeholder`'s `Y` and `Y_indices` ahead of time. It's a small price to pay to be able to use GPU computation and backprop with automatic differentiation. 


- `Y` : sparse matrix of movie ratings. `shape == (n_users, n_movies)`
- `U` : users latent features matrix `(n_users, k)`
- `V` : movies latent features matrix `(n_movies, k)`
- `Y = np.dot(U,V)`

### Lab41 movie ratings data
- ratings were given at intervals of 0.5: {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}

### Basic Practical Methodology
- Shuffling the data before splitting it into train, CV and test sets was crucial.

    ##### Splitting the input data:
    -  training data takes up 64% of the input data, 
    - cv data 16% and
    - test data 20%.

### Trying non linear features
- `Y_pred = tf.sigmoid(Y_pred) * 5` dropped the `MAE_test` approximately from `.64` to `.62`. Firstly, I can't explain why sigmoid works better--although only by a tiny bit.
- However, adding squared dot product term along with the linear dot product, didn't prodcue any tangible improvement. 
```python 
u_cdot_v_square = tf.square(tf.multiply(sliced_U, sliced_V)) 
nl = tf.reduce_sum(u_cdot_v_square, axis=1)
Y_pred = Y_pred + alpha*nl
Y_pred = tf.sigmoid(Y_pred) * 5
```
##### Cross Features:
- I added some 2nd degree **cross-features** between user latent feature matrix `U` and movie latent feature matrix `V`. Allow me to explain... 
    


 

### Some things to consider:
- Matrix factorization is based on SVD which assumes `Y` .
 ``` Y = np.dot(U,V) ``` 
or
``` Y[i,j] = np.dot(U[i], V[j]) ```
- 
