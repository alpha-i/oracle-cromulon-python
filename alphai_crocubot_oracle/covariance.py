from alphai_finance.metrics.returns import returns_minutes_after_market_open_data_frame
from alphai_covariance.dynamic_cov import estimate_cov

DEFAULT_N_ESTIMATES = 100
DEFAULT_SPLIT_STEPS = 1
DEFAULT_NUM_REALISATIONS_MULTIPLICATION_FACTOR = 3


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

    assert not data.isnull().any().any()

    nd = data.shape[1]
    sampling_days = nd*DEFAULT_NUM_REALISATIONS_MULTIPLICATION_FACTOR
    data_points = data.values[-sampling_days:, :]

    if len(data_points) < ndays:
        raise ValueError('The length of the data provided ({} days) is inconsistent with the model config ({} days)'
                         .format(len(data_points), ndays))

    covariance_matrix, _ = estimate_cov(data_points, method=estimation_method, is_dynamic=False)

    return covariance_matrix
