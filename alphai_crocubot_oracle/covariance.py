import nercome
import alphai_covariance.ledoit as ledoit
from alphai_finance.metrics.returns import returns_minutes_after_market_open_data_frame
from alphai_mvp_oracle.constants import COVARIANCE_METHOD_LEDOIT, COVARIANCE_METHOD_NERCOME

DEFAULT_N_ESTIMATES = 100
DEFAULT_SPLIT_STEPS = 1
DEFAULT_NUM_REALISATIONS_MULTIPLICATION_FACTOR = 3


def call_nercome_with_default_params(data, split_steps=DEFAULT_SPLIT_STEPS, n_estimates=DEFAULT_N_ESTIMATES):
    """
    :param data: daily returns. 2-d numpy array with shape [number of data points x number of dimensions (or assets)]
    :param sampling_days: sampling for nercome
    :param split_steps: splitting steps for nercome
    :param n_estimates: number of estimates in nercome
    :return:
    """
    assert not data.isnull().any().any()
    nr = data.shape[0]
    nd = data.shape[1]
    if nr < nd * DEFAULT_NUM_REALISATIONS_MULTIPLICATION_FACTOR:
        raise ValueError('Insufficient sampling data available.', 'nr = ', nr, 'nd = ', nd)
    sampling_days = nd * DEFAULT_NUM_REALISATIONS_MULTIPLICATION_FACTOR

    data_points = data.values[-sampling_days:, :]
    return nercome.estimate_covar(data_points.T, split_steps, n_estimates)


def estimate_covariance(data, ndays, minutes_after_open, estimation_method, exchange_calendar, forecast_interval):
    """
    :param data: OHLCV data
    :param ndays: number of historical days expected for the covariance estimate
    :param minutes_after_open: minutes after the covariance should be calculated
    :param estimation_method: covariance estimation method either NERCOME or Ledoit
    :param exchange_calendar: pandas_market_calendars
    :param forecast_interval: how many days ahead we should predict?
    :return:
    """
    if forecast_interval != 1:
        raise ValueError('This method is currently hardcoded for 1-day forecasting intervals.')

    data = returns_minutes_after_market_open_data_frame(data['close'], exchange_calendar, minutes_after_open)

    if len(data) < ndays:
        raise ValueError('The length of the data provided ({} days) is inconsistent with the model config ({} days)'
                         .format(len(data), ndays))

    if estimation_method == COVARIANCE_METHOD_NERCOME:
        covar_mat = call_nercome_with_default_params(data)
    elif estimation_method == COVARIANCE_METHOD_LEDOIT:
        covar_mat, shrinkage = ledoit.estimate_covariance(data)
    else:
        raise Exception("covariance method not supported: ", estimation_method)

    return covar_mat
