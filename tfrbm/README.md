# tensorflow-rbm

Tensorflow implementation of Restricted Boltzmann Machine for layerwise pretraining of deep autoencoders.


rbm = BBRBM(n_visible, n_hidden, learning_rate=0.01, momentum=0.95, err_function='mse', use_tqdm=False)
or
rbm = GBRBM(n_visible, n_hidden, learning_rate=0.01, momentum=0.95, err_function='mse', use_tqdm=False, sample_visible=False, sigma=1)
Initialization.
* `n_visible` — number of neurons on visible layer
* `n_hidden` — number of neurons on hidden layer
* `use_tqdm` — use tqdm package for progress indication or not
* `err_function` — error function (it's **not used** in training process, just in `get_err` function), should be `mse` or `cosine`
Only for `GBRBM`:
* `sample_visible` — sample reconstructed data with Gaussian distribution (with reconstructed value as a mean and a `sigma` parameter as deviation) or not (if not, every gaussoid will be projected into a single point)
* `sigma` — standard deviation of the input data


rbm.fit(data_x, n_epoches=10, batch_size=10, shuffle=True, verbose=True)
Fit the model.
* `data_x` — data of shape `(n_data, n_visible)`
* `n_epoches` — number of epoches
* `batch_size` — batch size, should be as small as possible
* `shuffle` — shuffle data or not
* `verbose` — output to stdout
Returns errors array.


rbm.transform(batch_x)
Transform data. Input shape is `(n_data, n_visible)`, output shape is `(n_data, n_hidden)`.


rbm.transform_inv(batch_y)
Inverse transform data. Input shape is `(n_data, n_hidden)`, output shape is `(n_data, n_visible)`.
