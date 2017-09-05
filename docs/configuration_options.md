# Configuration Options

The main sections of the configuration file are:
+ [`logging`](#logging)
+ [`quant_workflow`](#quant_workflow)
+ [`zipline`](#zipline)
+ [`live_clock_configuration`](#live_clock_configuration)

The `yaml` will look like:
```yaml
logging:
quant_workflow:
zipline:
live_clock_configuration:
```

Defining the crocubot-oracle is separately discussed in the [crocubot_options](crocubot_options.md)

We will explain the options to configure a backtest.

## `logging`

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

The following table explains the keys.

| key | description |
| --- | --- |
| `version` | :exclamation: ?? |
| `formatters` | specifies the formatting related parameters |
| `handlers` | keys for specifying the logging handlers |
| `root` | :exclamation: ? |
Each fo these keys are subsections themselves. They are explained below.

### `formatters`
The `formatters` lists a set of different formatters. These are defined as subsections. Then, each of these formatter 
subsections can be defined using the following keys

| key | description |
| --- | --- |
| `format` | Specify the format in which the logging is done. e.g. `'%(asctime)s - %(levelname)s [%(name)s:%(module)s]: %(message)s'`  |
| `datefmt` | the format of the time stamps of the logs. e.g. `'%Y/%m/%d %H:%M:%S'` |

### `handlers`
The `handlers` has the following subsection specifiction. 

| key | description |
| --- | --- |
| `console` | specifies the logging parameters for the console output |
| `file` | specifies the logging parameters for the file output |

#### `console`

| key | description |
| --- | --- |
| `class` | specify the python class which handles the logging. e.g. `'logging.StreamHandler'`. |
| `formatter` | Specify the `format`. The value of this key should correspond to one of the formatters defined under the section `formatters`. |
| `level` | the level of logging required. The possible values are: `INFO`, `DEBUG` and `ERROR` |


#### `file`

| key | description |
| --- | --- |
| `class` | specify the class that handles the file output. e.g `'logging.FileHandler'` |
| `formatter` | specify the `format`. The value of this key should correspond to one of the formatters defined under the section `formatters`. |
| `level` |  the level of logging required. The possible values are: `INFO`, `DEBUG` and `ERROR` |
| `filename` | The file to which the logs are to be written. Note that the path should be writable for this to properly work.|

### `root`

| key | description |
| --- | --- |
| `level` | the level of logging required. The possible values are: `INFO`, `DEBUG`, `ERROR` |
| `handlers` | a list of handlers defined in the `handlers` subsection. e.g. `['file', 'console']`.|

## `quant_workflow`
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
The main keys under `quant_workflow` are:

| key | description |
| --- | --- |
| `run_mode` | This key specifies the mode in which the `quant_workflow` is run. The possible options are `'backtest'`, `'oracle'` and `'live'`.
| `results_path` | specifies the path in which results files are to be written. e.g `'D:\Zipline\20100101_20150101_10S\results'` |
| `fill_limit` | specifies the maximum number of time stamps in which missing data can be filled using previous values. |
| `trade_resample_rule` | the period with which the data is to be re-sampled before passing to the oracle. We use `pandas` notation. |
| `trade_history_ndays` | the number days of historical data provided for the inference part. |
| `trade_frequency` | how often we trade. options are `'weekly'` and `'daily'` |
| `trade_days_offset` | Specifies the day we are trading. `0` implies Monday and so on. :exclamation: ONLY IF `trade_frequency: 'weekly'`. |
| `trade_minutes_offset` | specifies the time at which the trade happens after the market opens. e.g. `60` implies an hour after. |
| `trade_horizon_ncycles` | The number units of `trade_frequency` at which the oracle should do the prediction.  |
| `train_resample_rule` | :exclamation: THIS SHOULD BE IDENTICAL TO `trade_resample_rule`. e.g `'15T'` |
| `train_history_ndays` | the number days of historical data provided for the training part. .e.g `100`. |
| `train_frequency` | specifies how often we train the network. This can be different from the `trade_frequency`. `'weekly'` and `'daily'` |
| `train_days_offset` | :exclamation: ONLY IF `train_frequency: 'weekly'`. Specifies the day we are trading |
| `train_minutes_offset` | specifies the time at which the training happens after the market opens. |
| `alert_level` | specifies the level of alert messages. Possible values are `'NONE'` :exclamation: OTHERS? |
| `execution_timeout` | specifies the time out in *seconds* after which the execution will be killed. |
| `open_order_timeout` | specifies the time out in *seconds* after which the open orders will be killed. |
| `oracle` | This key specifies the configuration parameters for oracle. Parameters are specified as sub dictionary. See the section on `oracle`. |
| `portfolio` | This key specifies the parameters relating to the to portfolio creation. See the section on `portfolio` for more details. |
| `universe` | This key is used to specify the parameters relating to the universe of stocks. |

## `oracle`
The `oracle` section should contain a key called `method` which can either be `'http'` or `'library'`.
If `'http'` is specified as the `method` we need further keys as follows:
```yaml
oracle:
  method: 'http'
  protocol: 'http'
  host: '0.0.0.0'
  port: 8080
```

| key | description |
| --- | --- |
| `protocol` | what protocol to use for communication. e.g. `'http'` |
| `host` | hostname e.g. `'0.0.0.0'` |
| `port` | the port for communication `8080` |


If we are using the `library` option for the `oracle` we need to specify the following keys.

| key | description |
| --- | --- |
| `module_path` | a path to the module that is being used as the oracle. e.g. `alphai_crocubot_oracle.oracle` |
| `oracle_class_name` | name of the oracle python class. e.g. `CrocubotOracle`. |
| `oracle_arguments` | This subsection will specify *all* the arguments required to create a `oracle` class. |

For example, if we are using `alphai_crocubot_oracle.oracle`, the `oracle` section will look like:
```yaml
oracle:
  method: library
  module_path: alphai_crocubot_oracle.oracle
  oracle_class_name: CrocubotOracle
  oracle_arguments: ***
```
For more details on the oracle arguments see the [crocubot options](crocubot_options.md) page.

### `portfolio`
The `portfolio` section can be used to specify the portfolio creation. The following keys are valid.

| key | description |
| --- | --- |
| `max_abs_individual_weight` | :exclamation: ?? |
| `max_abs_pos_gross_exposure` | :exclamation: ?? |
| `max_abs_neg_gross_exposure` | :exclamation: ?? |
| `margin_ratio` |:exclamation: ?? |
| `max_annualised_std` | :exclamation: ?? |

### `universe` 
The `universe` section deals with the universe creation. This section requires the key `method` to be specified. It 
can either be `'fixed'` or `'liquidity'`. If `method: 'fixed'` is set, then we need to give a list ot stocks to specify the universe. 
So the section will look like
```yaml
  universe:
    method: 'fixed'
    symbol_list: ['AAPL', 'GOOGL', 'XOM', 'MSFT', 'JNJ', 'JPM', 'IBM', 'PG', 'BAC', 'T']
```
*Note that your ingested data bundle should contain these stocks for this specification to work!*

## `zipline`
This section defines the parameters related to `zipline` library. The following keys are required.

| key | description |
| --- | --- |
|  `zipline_root` | Tha path to `zipline` root where the `extension.py` and the `data` folder resides. e.g. `'D:\Zipline\20100101_20150101_10S\zipline_root'`. |
| `start_date` | start date of the run. e.g. `'20110401'`.|
| `end_date` | end date of the run. |
| `capital_base` | The amount of capital available. |
| `data_frequency` |  :exclamation: ?? The data frequency of the ingested data. e.g. `'minute'`. |
| `data_bundle` | the name of the data bundle. This should be defined in the `extension.py` in the `zipline_root`.
| `slippage_type` | the type of slippage to be used in the backtest. The possible options are `'TradeAtTheOpenSlippageModel'` :exclamation: ??|
| `spread` | :exclamation: ?? e.g. `0.` |
| `open_close_fraction` | :exclamation ?? e.g. `0.` |
| `volume_limit`| :exclamation: ?? e.g. `0.` |
| `price_impact`| :exclamation: ?? e.g. `0.` |
| `commission_type` | The type of commission model. Possible options are `'PerShare'` :exclamation: ?? |
| `cost` | Defines the cost of each transaction. e.g. `0.0005` :exclamation: ?? |
| `min_trade_cost` | :exclamation: ?? e.g. `1.`. |

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

For example, using the above values, this section can be written as
```yaml
live_clock_configuration:
  host: 'localhost'
  port: 45672
  queue_name: 'clock-pulse'
```