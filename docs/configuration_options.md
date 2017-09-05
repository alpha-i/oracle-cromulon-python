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

For example, if we are using the ``