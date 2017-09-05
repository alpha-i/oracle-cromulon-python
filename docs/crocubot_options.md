# Corcubot options

This page describes the keys required to define the Crocubot oracle. 
```yaml
oracle_arguments:
  data_transformation:
    feature_config_list:
      -
        name: close
        order: 'log-return'
        normalization: standard
        nbins: 12
        is_target: True
    exchange_name: 'NYSE'
    features_ndays: 10
    features_resample_minutes: 15
    features_start_market_minute: 60
    prediction_frequency_ndays: 1
    prediction_market_minute: 60
    target_delta_ndays: 1
    target_market_minute: 60
  train_path: 'D:\Zipline\20100101_20150101_10S\train'
  covariance_method: 'NERCOME'
  covariance_ndays: 9
  model_save_path: 'D:\Zipline\20100101_20150101_10S\model'
  d_type: float32
  tf_type: 32
  random_seed: 0
  n_epochs: 10
  n_training_samples: 1000
  learning_rate: 2e-3
  batch_size: 100
  cost_type: 'bayes'
  n_train_passes: 30
  n_eval_passes: 100
  resume_training: False
  n_series: 10
  n_features_per_series: 271
  n_forecasts: 10
  n_classification_bins: 12
  layer_heights: [3, 271]
  layer_widths: [3, 3]
  activation_functions: ["relu", "relu"]
  INITIAL_ALPHA: 0.2
  INITIAL_WEIGHT_UNCERTAINTY: 0.4
  INITIAL_BIAS_UNCERTAINTY: 0.4
  INITIAL_WEIGHT_DISPLACEMENT: 0.1
  INITIAL_BIAS_DISPLACEMENT: 0.4
  USE_PERFECT_NOISE: True
  double_gaussian_weights_prior: False
  wide_prior_std: 1.2
  narrow_prior_std: 0.05
  spike_slab_weighting: 0.5
```

| key | description |
| --- | --- |
| `data_transformation` | describes the data transformation  |
| `train_path` | path where the training related files will be stored |
| `covariance_method` | method for covariance calculation (`'NERCOME'` or `'Ledoit'`) |
| `model_save_path` | path where the trained models are saved ?? |
| `d_type` | floating point type for data analysis (`float32` or `float64`) |
| `tf_type` | floating point type for `TensorFlow` (`32` or `64`). should correspond to the one specified in `d_type` | 
| `random_seed` | a seed for the random variate generator (integer) |
| `n_epochs` | number of epochs for training |
| `n_training_samples` | number of training samples to be used. |
| `learning_rate` | learning rate of the training process. |
| `batch_size` | size of each batch for training the network. |
| `cost_type` | the model for the cost function. only accepts `'bayes'` for now. |
| `n_train_passes` | number of forward passes for computing the mean and the covariance in the train stage |
| `n_eval_passes` | number of forward passes for computing the mean and the covariance in the inference stage |
| `resume_training` | whether we should resume training from a previously saved position? |
| `n_series` | number of time series in the data. *this should be identical to the zipline data passed* :exclamation: NEED TO INFER THIS FROM THE DATA |
| `n_features_per_series` | number of data points/features per time series |
| `n_forecasts` | number of points to forecast per series in the future. only accepts `1` now. :exclamation: THIS SHOULD BE RENAMED AS `n_forecasts_per_series`|
| `n_classification_bins` | number of classification bins in the data |
| `layer_heights` | a list of numbers indicating the heights of the layers |
| `layer_widths` | a list of numbers indicating the widths of the layers |
| `activation_functions` | a list of strings indicating the activation function of the layers |
| `INITIAL_ALPHA` | initial value of alpha |
| `INITIAL_WEIGHT_UNCERTAINTY` | initial value for the std-dvn of weights |
| `INITIAL_BIAS_UNCERTAINTY` | initial value for the std-dvn of biases |
| `INITIAL_WEIGHT_DISPLACEMENT` | initial value of the mean displacement of weights from zero|
| `INITIAL_BIAS_DISPLACEMENT` | initial value of the mean displacement of biases from zero|
| `USE_PERFECT_NOISE` | whether we should use a perfect Gaussian noise or not? |
| `double_gaussian_weights_prior` | whether we should use a double Gaussian noise prior or not? |
| `wide_prior_std` | standard deviation of the *slab-prior* in the *Bayes-by-backprop* method |
| `narrow_prior_std` | standard deviation of the *spike-prior* in the *Bayes-by-backprop* method |
| `spike_slab_weighting` | the ratio or slab/spike in the prior. |


## `data_transformation`
The data transformation needs to be specified as a subsection with the following keys:

| key | description |
| --- | --- |
| `feature_config_list` | specified as a subsection containing `name`, `order`, `normalization`, `nbins` and `is_target` |
| `exchange_name` | name of the stock-exchange |
| `features_ndays` | :exclamation: THIS SHOULD BE IDENTICAL TO `trade_history_ndays` ? OR `train_history_ndays` |
| `features_resample_minutes` | re-sample frequency of the data :exclamation: THIS SHOULD BE IDENTICAL TO `trade_resample_rule` AND `train_resample_rule` |
| `features_start_market_minute` | the minute at which features start :exclamation: IS THERE ANY RELATION TO `train_minutes_offset`? |
| `prediction_frequency_ndays` | at what point in future we try to predict ? :exclamation: IS THERE ANY RELATION TO `trade_frequency`, `train_frequency`, AND `trade_horizon_ncycles` |
| `prediction_market_minute` | the minute at which the prediction is done. :exclamation: IS THERE ANY RELATION TO `trade_minutes_offset`? |
| `target_delta_ndays` | the number of days in the future the prediction is made aimed for. :exclamation: IS THERE ANY RELATION TO `trade_horizon_ncycles`? |
| `target_market_minute` | the minute at which the prediction in the future is made :exclamation: IS THERE ANY RELATION TO `trade_minutes_offset` | 

