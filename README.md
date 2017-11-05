### Difficulties on implementing Matrix Factorization on `tensorflow`

- Can't pass `tf.placeholder( ... ,shape=(None)` for input data `Y` and `Y_indices`. Every new user and new movie in `Y` will introduce new tunable variables.
In order to implement matrix factorization on `tensorflow`, I had to fix the `shape` of the `tf.placeholder`'s `Y` and `Y_indices` ahead of time. It's a small price to pay to be able to use GPU computation and backprop with automatic differentiation. 

-  matrix decompisition `Y = np.dot(U,V.T)`
- `Y` : movie ratings `shape == (n_users, n_movies)`
- `U` : users latent features matrix `(n_users, k)`
- `V` : movies latent features matrix `(n_movies, k)`


### Lab41 movie ratings data
- ratings were given at intervals of 0.5: {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}

### Basic Practical Methodology
- Shuffling the data before splitting it into train, CV and test sets was crucial.

    ##### Splitting the input data:
    -  training data takes up 64% of the input data, 
    - cv data 16% and
    - test data 20%.

### Linear vs Non-linear features
- `Y_pred = tf.sigmoid(Y_pred) * 5` dropped the `MAE_test` approximately from `.64` to `.62`. Firstly, I can't explain why sigmoid works better--although only by a tiny bit.
- However, adding squared dot product term along with the linear dot product, didn't prodcue any tangible improvement. 
```python 
u_cdot_v_square = tf.square(tf.multiply(sliced_U, sliced_V)) 
nl = tf.reduce_sum(u_cdot_v_square, axis=1)
Y_pred = Y_pred + alpha*nl
Y_pred = tf.sigmoid(Y_pred) * 5
```

##### Linear features:
Matrix factorization is based on a low-rank singular calue decomposition (SVD).  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  $$Y=U \cdot V^{T}$$

An individual rating of user $$i$$ on movie $$j$$  is given by 
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  $$y_{ij} = u_{i} \cdot v_{j}^{T}$$

Here, each user feature vector $$u_i$$ and movie feature vector $$v_j$$ is of length $$k$$. and the classical matrix factorization multiplies $$p^{th}$$ feature of $$u_{i}$$ with  $$p^{th}$$ feature of $$v_{j}$$. Here, one can assume the feature $$p$$ corresponds to how much of a specific genre is present in movie $$j$$ and how much a user $$i$$ likes that specific genre. When the rating $$y_{ij}$$ is modeled by a dot product between $$u_i$$ and $$v_j$$.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Linear Model: MAE (CV) = 0.6237**

##### Nonlinear Cross-Features:
The rating prediction with cross-features 
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   $$y_{ij} = u_{i} \cdot v_{j}^{T} + \sum_{p=0}^{k}\sum_{q=0}^{k} w_{pq} (u_{ip} \cdot v_{jq}^{T})$$

Here, we're multiplying $$p^{th}$$ feature of user $$i$$ with $$q^{th}$$ feature of movie $$j$$. This allows the model to learn cross interactions as, say, if a user likes the actor Tom Cruise (the $$p^{th}$$ feature--high  value for $$u_{ip}$$), and she doesn't like dark and suspenseful thrillers ($$q^{th}$$ feature--low value for $$u_{iq}$$), however, she likes the movie Eyes Wide Shut (even though it has a high value for $$v_{jq}$$), because an underlying reason that makes her not like dark suspenseful movies perhaps disappears if Tom Cruise is in the movie. For a model to capture such a pattern, it has to allow some sort of **nonlinear interactions** between feature $$p$$ and feature $$q$$.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Non-linear Model: MAE (CV) = 0.6160**

`Y_pred = np.dot(U,V) + alpha1*(xft) + alpha2*(uv_sq)`

The runtime for one epoch went from $$31$$ sec for linear model to $$60$$ sec when considering all 3 types of nonlinear feature crosssings ($$u_{ip}$$ ~ $$v_{jq}$$), ($$u_{ip}$$ ~ $$u_{iq}$$), ($$v_{jp}$$ ~ $$v_{jq}$$)
