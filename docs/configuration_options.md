# Configuration Options

We will explain the options to configure a backtest.

## `quant_workflow` section

The main keys under `quant_workflow` are:

+ `run_mode`: This key specifies the mode in which the `quant_workflow` is run. The possible options are:
    + `'backtest'`: The backtesting mode. 
    + `'oracle'`: dry run or just run the oracle and no trading strategy.
    + `'live'`: run the workflow in the live trading mode.
+ `results_path`: specifies the path in which results files are to be written. e.g `'D:\Zipline\20100101_20150101_10S\results'`
+ `fill_limit`: specifies the maximum number of time stamps in which missing data can be filled using previous values. e.g. `5`
+ `trade_resample_rule`: the period with which the data is to be re-sampled before passing to the oracle. We use `pandas` notation. .e.g `'15T'` 
+ `trade_history_ndays`: the number days of historical data provided for the inference part. .e.g `31`.  
+ `trade_frequency`: how often we trade. options are:
    + `'weekly'`: trade every week
    + `'daily'`: trade every day
+ `trade_days_offset`: *ONLY IF* `trade_frequency: 'weekly'`. Specifies the day we are trading. `0` implies Monday and so on. 
+ `trade_minutes_offset`: specifies the time at which the trade happens after the market opens. e.g. `60` implies an hour after. 
+ `trade_horizon_ncycles`: The number units of `trade_frequency` at which the oracle should do the prediction. 
e.g. `trade_frequency:'weekly'` and  `trade_horizon_ncycles: 1` and implies we are predicting one week ahead.