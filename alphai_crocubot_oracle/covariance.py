from alphai_crocubot_oracle.metrics.returns import returns_minutes_after_market_open_data_frame
from alphai_covariance.dynamic_cov import estimate_cov

DEFAULT_N_ESTIMATES = 100
DEFAULT_SPLIT_STEPS = 1
DEFAULT_NUM_REALISATIONS_MULTIPLICATION_FACTOR = 3


def estimate_covariance(data, ndays, minutes_after_open, estimation_method, exchange_calendar, forecast_interval_in_days):
    """
    :param data: OHLCV data
    :param ndays: number of historical days expected for the covariance estimate
    :param minutes_after_open: minutes after the covariance should be calculated
    :param estimation_method: covariance estimation method either NERCOME or Ledoit
    :param exchange_calendar: pandas_market_calendars
    :param forecast_interval_in_days: how many days ahead we should predict?
    :return: The covariance matrix of the data.
    """

    data = returns_minutes_after_market_open_data_frame(data['close'], exchange_calendar, minutes_after_open)

    assert not data.isnull().any().any()  # FIXME What are the conditions in which this fails?

    nd = data.shape[1]
    sampling_days = nd * DEFAULT_NUM_REALISATIONS_MULTIPLICATION_FACTOR
    data_points = data.values[-sampling_days:, :]

    covariance_matrix, _ = estimate_cov(data_points, method=estimation_method, is_dynamic=False)

    # Rescale for longer horizons
    return covariance_matrix * forecast_interval_in_days
