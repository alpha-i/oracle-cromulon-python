# Configuration Options

We will explain the options to configure a backtest.

## `logging` section
+ `version`: ??
+ `formatters`: specifies the formatting related parameters
+ `handlers`: keys for specifying the logging handlers
+ `root`: ??

The `formatters` lists a set of different formatters. These are defined as subsections. Then, each of these formatter 
subsections can be defined using the following keys
+ `format`: Specify the format in which the logging is done. 
e.g. `'%(asctime)s - %(levelname)s [%(name)s:%(module)s]: %(message)s'` 
+ `datefmt`: the format of the time stamps of the logs. e.g. `'%Y/%m/%d %H:%M:%S'`

The `handlers` has the following subsection. 
+ `console`: specifies the logging parameters for the console output
+ `file`: specifies the logging parameters for the file output

`console` parameter itself is another section with the following parameters:
+ `class`: specify the python class which handles the logging. e.g. `'logging.StreamHandler'`.
+ `formatter`: Specify the `format`. The value of this key should correspond to one of the formatters defined under the
section `formatters`.
+ `level`: the level of logging required. The possible values are:
    + `INFO`
    + `DEBUG`
    + `ERROR`
+ `stream`: ?? e.g. `'ext://sys.stdout'`.

`file`  parameter itself is another section with the following parameters:
+ `class`: specify the class that handles the file output. e.g `'logging.FileHandler'`
+ `formatter`: Specify the `format`. The value of this key should correspond to one of the formatters defined under the
section `formatters`.
+ `level`:  the level of logging required. The possible values are:
    + `INFO`
    + `DEBUG`
    + `ERROR`
+ `filename`: The file to which the logs are to be written. 
e.g. `'D:\Zipline\20100101_20150101_10S\logs\quant_workflow-debug.log'`. Note that the path should be writable for this
to properly work.

`root` subsection can be specified with the following keys:
+ `level`: the level of logging required. The possible values are:
    + `INFO`
    + `DEBUG`
    + `ERROR`
+ `handlers`: A list of handlers defined in the `handlers` subsection. e.g. `['file', 'console']`.

For example, a full definition of the logging section might look like.
```yaml
logging:
  version: 1
  formatters:
    formatter:
      format: '%(asctime)s - %(levelname)s [%(name)s:%(module)s]: %(message)s'
      datefmt: '%Y/%m/%d %H:%M:%S'
  handlers:
    console:
      class: 'logging.StreamHandler'
      formatter: 'formatter'
      level: 'DEBUG'
      stream: 'ext://sys.stdout'
    file:
      class : 'logging.FileHandler'
      formatter: 'formatter'
      level: 'DEBUG'
      filename: 'D:\Zipline\20100101_20150101_10S\logs\quant_workflow-debug.log'
  root:
    level: 'DEBUG'
    handlers: ['file', 'console']
```


## `quant_workflow` section

The main keys under `quant_workflow` are:

+ `run_mode`: This key specifies the mode in which the `quant_workflow` is run. The possible options are:
    + `'backtest'`: The backtesting mode. 
    + `'oracle'`: dry run or just run the oracle and no trading strategy.
    + `'live'`: run the workflow in the live trading mode.
+ `results_path`: specifies the path in which results files are to be written. 
e.g `'D:\Zipline\20100101_20150101_10S\results'`
+ `fill_limit`: specifies the maximum number of time stamps in which missing data can be filled using 
previous values. e.g. `5`
+ `trade_resample_rule`: the period with which the data is to be re-sampled before passing to the oracle. 
We use `pandas` notation. .e.g `'15T'` 
+ `trade_history_ndays`: the number days of historical data provided for the inference part. .e.g `31`.  
+ `trade_frequency`: how often we trade. options are:
    + `'weekly'`: trade every week
    + `'daily'`: trade every day
+ `trade_days_offset`: *ONLY IF* `trade_frequency: 'weekly'`. Specifies the day we are trading. 
`0` implies Monday and so on. 
+ `trade_minutes_offset`: specifies the time at which the trade happens after the market opens. 
e.g. `60` implies an hour after. 
+ `trade_horizon_ncycles`: The number units of `trade_frequency` at which the oracle should do the prediction. 
e.g. `trade_frequency:'weekly'` and  `trade_horizon_ncycles: 1` and implies we are predicting one week ahead.
+ `train_resample_rule`: *THIS SHOULD BE IDENTICAL TO* `trade_resample_rule`. e.g `'15T'`
+ `train_history_ndays`: the number days of historical data provided for the training part. .e.g `100`.
+ `train_frequency`: specifies how often we train the network. This can be different from the `trade_frequency`.
    + `'weekly'`: trade every week
    + `'daily'`: trade every day
+ `train_days_offset`: *ONLY IF* `train_frequency: 'weekly'`. Specifies the day we are trading. 
`0` implies Monday and so on. 
+ `train_minutes_offset`: specifies the time at which the training happens after the market opens. 
e.g. `60` implies an hour after. 
+ `alert_level`: specifies the level of alert messages. Possible values are
    + `'NONE'` No alert
+ `execution_timeout`: specifies the time out in *seconds* after which the execution will be killed. e.g. `180.`
+ `open_order_timeout`: specifies the time out in *seconds* after which the open orders will be killed. e.g. `3600.`
+ `oracle:` This key specifies the configuration parameters for oracle. Parameters are specified as sub dictionary.
+ `portfolio`: This key specifies the parameters relating to the to portfolio creation. 
Parameters are specified as sub dictionary.
+ `universe`: This key is used to specify the parameters relating to the universe of stocks.

For example, if we are using the `http` this section will look like the one below:
```yaml
quant_workflow:
  run_mode: 'backtest'
  results_path: 'D:\Zipline\20100101_20150101_10S\results'
  fill_limit: 5
  trade_resample_rule: '15T'
  trade_history_ndays: 31
  trade_frequency: 'weekly'
  trade_days_offset: 1
  trade_minutes_offset: 60
  trade_horizon_ncycles: 1
  train_resample_rule: '15T'
  train_history_ndays: 100
  train_frequency: 'weekly'
  train_days_offset: 0
  train_minutes_offset: 60
  alert_level: 'NONE'
  execution_timeout: 180.
  open_order_timeout: 3600.
  oracle:
    method: 'http'
    protocol: 'http'
    host: '0.0.0.0'
    port: 8080
  portfolio:
    max_abs_individual_weight: 0.2
    max_abs_pos_gross_exposure: 0.75
    max_abs_neg_gross_exposure: 0.75
    margin_ratio: 1.0
    max_annualised_std: 0.3
  universe:
    method: 'fixed'
    symbol_list: ['AAPL', 'GOOGL', 'XOM', 'MSFT', 'JNJ', 'JPM', 'IBM', 'PG', 'BAC', 'T']
```

The `oracle` section will have the following sections:
+ `method`: specifies whether we are using a library or an http based communication. possible value are
    + `'library'`
    + `'http'`

If `'http'` is specified as the `method` we need further keys 
+ `protocol: 'http'`
+ `host: '0.0.0.0'`
+ `port: 8080`

For example, if using the `http`, the `oracle` section will look like:
```yaml
oracle:
  method: 'http'
  protocol: 'http'
  host: '0.0.0.0'
  port: 8080
```

If on the other hand we are using the `library` option for the `oracle` we need to specify the following keys.
+ `module_path`: a path to the module that is being used as the oracle. e.g. `alphai_crocubot_oracle.oracle`
+ `oracle_class_name`: name of the oracle python class. e.g. `CrocubotOracle`.
+ `oracle_arguments`: This subsection will specify *all* the arguments required to create a `oracle` class.

For example, if we are using `alphai_crocubot_oracle.oracle`, the `oracle` section will look like:
```yaml
oracle:
  method: library
  module_path: alphai_crocubot_oracle.oracle
  oracle_class_name: CrocubotOracle
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

The `portfolio` section can be used to specify the portfolio creation. The following keys are valid.
+ `max_abs_individual_weight`: ??
+ `max_abs_pos_gross_exposure`: ??
+ `max_abs_neg_gross_exposure`: ??
+ `margin_ratio`: ??
+ `max_annualised_std`: ??

The `universe` section deals with the universe creation. The valid keys are below:
+ `method`: This specifies the method for universe creation. Possible values are
    + `'fixed'`
    + `'liquidity'`

Depending on the method we will need to specify more keys to complete the `universe` section. If `method: 'fixed'` 
is set, then we need to give a list ot stocks to specify the universe. So the section will look like
```yaml
  universe:
    method: 'fixed'
    symbol_list: ['AAPL', 'GOOGL', 'XOM', 'MSFT', 'JNJ', 'JPM', 'IBM', 'PG', 'BAC', 'T']
```
*Note that your ingested data bundle should contain these stocks for this specification to work!*

## `zipline`

This section defines the parameters related to `zipline` library. The following keys are required.
+ `zipline_root`: Tha path to `zipline` root where the `extension.py` and the `data` folder resides.
e.g. `'D:\Zipline\20100101_20150101_10S\zipline_root'`.
+ `start_date`: start date of the run. e.g. `'20110401'`.
+ `end_date`: end date of the run. e.g. `20110430`.
+ `capital_base`: The amount of capital available. e.g. `1000000.`
+ `data_frequency`: ?? The data frequency of the ingested data. e.g. `'minute'`.
+ `data_bundle`: the name of the data bundle. This should be defined in the `extension.py` in the `zipline_root`.
+ `slippage_type`: the type of slippage to be used in the backtest. The possible options are:
    + `'TradeAtTheOpenSlippageModel'` ??
+ `spread`: ?? e.g. `0.`
+ `open_close_fraction`: ?? e.g. `0.`
+ `volume_limit`: ?? e.g. `0.`
+ `price_impact`: ?? `0.`
+ `commission_type`: The type of commission model. Possible options are:
    + `'PerShare'`: ??
+ `cost`: Defines the cost of each transaction. e.g. `0.0005` ??
+ `min_trade_cost`: ?? e.g. `1.`.

Thus a fully specified `zipline` section will look like
```yaml
zipline:
  zipline_root: 'D:\Zipline\20100101_20150101_10S\zipline_root'
  start_date: '20110401'
  end_date: '20110430'
  capital_base: 1000000.
  data_frequency: 'minute'
  data_bundle: 'test_bundle'
  slippage_type: 'TradeAtTheOpenSlippageModel'
  spread: 0.
  open_close_fraction: 0.
  volume_limit: 0.
  price_impact: 0.
  commission_type: 'PerShare'
  cost: 0.0005
  min_trade_cost: 1.
```

## `live_clock_configuration`
This section is used to define the details about the live clock.
+ `host`: defines the host. e.g. `localhost`
+ `port`: port for communication. e.g. `45672`
+ `queue_name`: name of the queue. e.g. `'clock-pulse'`. 