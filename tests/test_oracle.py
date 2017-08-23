import os
from datetime import datetime, timedelta
from unittest import TestCase

import pandas as pd

from alphai_crocubot_oracle.constants import DATETIME_FORMAT_COMPACT
from alphai_crocubot_oracle.oracle import TRAIN_FILE_NAME_TEMPLATE

from tests.helpers import (
    FIXTURE_DESTINATION_DIR, FIXTURE_DATA_FULLPATH,
    create_fixtures, destroy_fixtures, read_hdf5_into_dict_of_data_frames,
    DummyMvpOracle
)


class TestMvp(TestCase):

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
        historical_universes.loc[0] = [
            pd.Timestamp(start_date),
            pd.Timestamp(end_date),
            symbols,
        ]
        data = read_hdf5_into_dict_of_data_frames(start_date,
                                                  end_date,
                                                  symbols,
                                                  FIXTURE_DATA_FULLPATH,
                                                  exchange_name,
                                                  fill_limit,
                                                  resample_rule
                                                  )
        return historical_universes, data

    def test_mvp_model_init(self):

        historical_universes, data = self._prepare_data_for_test()

        configuration = {
            'data_transformation': {
                'features_dict': {
                    'close': {
                        'order': 'log-return',
                        'normalization': 'standard',
                        'resample_minutes': 15,
                        'ndays': 10,
                        'start_min_after_market_open': 60,
                        'is_target': True,
                    },
                },
                'exchange_name': 'NYSE',
                'prediction_frequency_ndays': 1,
                'prediction_min_after_market_open': 60,
                'target_delta_ndays': 1,
                'target_min_after_market_open': 60,
            },
            'train_path': FIXTURE_DESTINATION_DIR,
            'covariance_method': 'NERCOME',
            'covariance_ndays': 30,
            'epochs': 10,
            'learning_rate': 0.001,
            'verbose': False,
            'batch_size': 32,
            'drop_out': 0.5,
            'l2': 0.00001,
            'n_hidden': 100,
            'save_model': False
        }

        model = DummyMvpOracle(configuration)

        train_time = datetime.now() - timedelta(minutes=1)
        model.train(historical_universes, data, train_time)

    def test_mvp_train_and_save_file(self):

        train_time = datetime.now()
        train_filename = TRAIN_FILE_NAME_TEMPLATE.format(train_time.strftime(DATETIME_FORMAT_COMPACT))

        expected_train_path = os.path.join(FIXTURE_DESTINATION_DIR, train_filename)

        historical_universes, data = self._prepare_data_for_test()

        configuration = {
            'data_transformation': {
                'features_dict': {
                    'close': {
                        'order': 'log-return',
                        'normalization': 'standard',
                        'resample_minutes': 15,
                        'ndays': 10,
                        'start_min_after_market_open': 60,
                        'is_target': True,
                    },
                },
                'exchange_name': 'NYSE',
                'prediction_frequency_ndays': 1,
                'prediction_min_after_market_open': 60,
                'target_delta_ndays': 1,
                'target_min_after_market_open': 60,
            },
            'train_path': FIXTURE_DESTINATION_DIR,
            'covariance_method': 'NERCOME',
            'covariance_ndays': 30,
            'epochs': 10,
            'learning_rate': 0.001,
            'verbose': False,
            'batch_size': 32,
            'drop_out': 0.5,
            'l2': 0.00001,
            'n_hidden': 100,
            'save_model': True
        }

        model = DummyMvpOracle(configuration)
        model.train(historical_universes, data, train_time)
        self.assertEqual(
            expected_train_path,
            model._current_train
        )

        self.assertTrue(os.path.exists(expected_train_path))
        self.assertEqual(
            model.get_current_train(), expected_train_path
        )

    def test_mvp_predict_without_train_file(self):

        configuration = {
            'data_transformation': {
                'features_dict': {
                    'close': {
                        'order': 'log-return',
                        'normalization': 'standard',
                        'resample_minutes': 15,
                        'ndays': 10,
                        'start_min_after_market_open': 60,
                        'is_target': True,
                    },
                },
                'exchange_name': 'NYSE',
                'prediction_frequency_ndays': 1,
                'prediction_min_after_market_open': 60,
                'target_delta_ndays': 1,
                'target_min_after_market_open': 60,
            },
            'train_path': FIXTURE_DESTINATION_DIR,
            'covariance_method': 'NERCOME',
            'covariance_ndays': 30,
            'epochs': 10,
            'learning_rate': 0.001,
            'verbose': False,
            'batch_size': 32,
            'drop_out': 0.5,
            'l2': 0.00001,
            'n_hidden': 100,
            'save_model': True
        }

        model = DummyMvpOracle(configuration)
        prediction_time = datetime(2017, 6, 7, 9) + timedelta(minutes=60)

        _, data = self._prepare_data_for_test()
        self.assertRaises(
            ValueError,
            model.predict,
            data,
            prediction_time
        )
