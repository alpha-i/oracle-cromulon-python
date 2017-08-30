from unittest import TestCase
from tests.helpers import (create_fixtures, destroy_fixtures, read_hdf5_into_dict_of_data_frames, FIXTURE_DATA_FULLPATH)
import pandas as pd
from alphai_crocubot_oracle.covariance import estimate_covariance, DEFAULT_NUM_REALISATIONS_MULTIPLICATION_FACTOR
from alphai_finance.data.transformation import FinancialDataTransformation
from alphai_finance.metrics.returns import returns_minutes_after_market_open_data_frame
from sklearn.covariance import GraphLassoCV
import numpy as np


class TestCrocubot(TestCase):
    def setUp(self):
        create_fixtures()

    def tearDown(self):
        destroy_fixtures()

    def _prepare_data_for_test(self):
        start_date = '20140102'  # these are values for the resources/sample_hdf5.h5
        end_date = '20140228'
        symbols = ['AAPL', 'INTC', 'MSFT']
        exchange_name = 'NYSE'
        fill_limit = 10
        resample_rule = '15T'

        historical_universes = pd.DataFrame(columns=['start_date', 'end_date', 'assets'])
        historical_universes.loc[0] = [pd.Timestamp(start_date), pd.Timestamp(end_date), symbols]
        data = read_hdf5_into_dict_of_data_frames(start_date, end_date, symbols, FIXTURE_DATA_FULLPATH, exchange_name,
                                                  fill_limit, resample_rule)
        return historical_universes, data

    def test_estimate_covariance(self):
        configuration = {
            'features_dict': {
                'close': {
                    'order': 'log-return',
                    'normalization': 'standard',
                    'resample_minutes': 15,
                    'ndays': 10,
                    'start_min_after_market_open': 60,
                    'is_target': False,
                },
            },
            'exchange_name': 'NYSE',
            'prediction_frequency_ndays': 1,
            'prediction_min_after_market_open': 60,
            'target_delta_ndays': 1,
            'target_min_after_market_open': 60,
        }

        data_transformation = FinancialDataTransformation(configuration)
        universe, data = self._prepare_data_for_test()
        ndays = 9  # FIXME this is the only value that works now.
        minutes_after_open = data_transformation.target_min_after_market_open
        estimation_method = "Ledoit"
        exchange_calendar = data_transformation.exchange_calendar
        forecast_interval = data_transformation.target_delta_ndays
        covariance_matrix = estimate_covariance(data, ndays, minutes_after_open, estimation_method, exchange_calendar,
                                                forecast_interval)

        ret_data = returns_minutes_after_market_open_data_frame(data['close'], exchange_calendar, minutes_after_open)
        print(ret_data.shape)
        nd = ret_data.shape[1]
        sampling_days = nd * DEFAULT_NUM_REALISATIONS_MULTIPLICATION_FACTOR
        data_points = ret_data.values[-sampling_days:, :]
        glass_model = GraphLassoCV()
        glass_model.fit(data_points)
        cov_mat = glass_model.covariance_
        self.assertTrue(np.allclose(covariance_matrix.diagonal(), cov_mat.diagonal()))
