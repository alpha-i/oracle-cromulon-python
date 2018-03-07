import os
import shutil
import tempfile

import pandas_market_calendars as mcal

from alphai_feature_generation.cleaning import (
    convert_to_utc, select_trading_hours, fill_gaps_data_dict, resample_ohlcv)
from alphai_cromulon_oracle.flags import build_tensorflow_flags
from alphai_cromulon_oracle.oracle import CromulonOracle
from tests.hdf5_reader import read_feature_data_dict_from_hdf5

DATA_FILENAME = 'sample_hdf5.h5'

FIXTURES_SOURCE_DIR = os.path.join(os.path.dirname(__file__), 'resources')
FIXTURE_DESTINATION_DIR = tempfile.TemporaryDirectory().name

FIXTURE_DATA_FULLPATH = os.path.join(FIXTURE_DESTINATION_DIR, DATA_FILENAME)


def create_fixtures():

    if not os.path.exists(FIXTURE_DESTINATION_DIR):
        os.makedirs(FIXTURE_DESTINATION_DIR)

    shutil.copy(
        os.path.join(FIXTURES_SOURCE_DIR, DATA_FILENAME),
        FIXTURE_DESTINATION_DIR
    )

    os.chmod(FIXTURE_DATA_FULLPATH, 0o777)


def destroy_fixtures():
    shutil.rmtree(FIXTURE_DESTINATION_DIR)


def read_hdf5_into_dict_of_data_frames(start_date, end_date, symbols, file_path, exchange, fill_limit, resample_rule):
    """

    :param start_date: start date of the data
    :param end_date: end date of the data
    :param symbols: symbols of assets to read
    :param file_path: the path in which we can find the data
    :param exchange: name of the exchange
    :param fill_limit: maximum gap we are allowed to fill in the data
    :param resample_rule: frequency
    :return: a dictionary of DataFrames for open, high, low, close, volume
    """
    exchange_calendar = mcal.get_calendar(exchange)
    data_dict = read_feature_data_dict_from_hdf5(symbols, start_date, end_date, file_path)
    data_dict = convert_to_utc(data_dict)
    data_dict = select_trading_hours(data_dict, exchange_calendar)
    data_dict = fill_gaps_data_dict(data_dict, fill_limit, dropna=False)
    data_dict = resample_ohlcv(data_dict, resample_rule)
    return data_dict


class DummyCromulonOracle(CromulonOracle):
    def __init__(self, configuration):
        super().__init__(configuration)

    def get_train_file_manager(self):
        return self._train_file_manager


def default_oracle_config():
    configuration = {
        'nassets': 1,
        'data_transformation': {
            'fill_limit': 5,
            'holiday_calendar': 'NYSE',
            'feature_config_list': [
                {
                    'name': 'close',
                    'transformation': {
                        'name': 'log-return'
                    },
                    'normalization': 'standard',
                    'is_target': True,
                    'local': False,
                },
            ],
            'data_name': 'GYM',
            'features_ndays': 10,
            'features_resample_minutes': 15,
            'target_delta_hours': 1,
        },
        'train_path': FIXTURE_DESTINATION_DIR,
        'covariance_method': 'NERCOME',
        'covariance_ndays': 9,
        'model_save_path': FIXTURE_DESTINATION_DIR,
        'tensorboard_log_path': FIXTURE_DESTINATION_DIR,
        'd_type': 'float32',
        'tf_type': 32,
        'random_seed': 0,
        'predict_single_shares': False,
        'classify_per_series': True,
        'normalise_per_series': True,

        # Training specific
        'n_epochs': 200,
        'n_retrain_epochs': 20,
        'learning_rate': 1e-3,
        'batch_size': 200,
        'cost_type': 'bayes',
        'n_train_passes': 32,
        'n_eval_passes': 32,
        'resume_training': True,
        'use_gpu': False,

        # Topology
        'n_series': 1,
        'do_kernel_regularisation': True,
        'do_batch_norm': False,
        'n_res_blocks': 6,
        'n_features_per_series': 271,
        'n_forecasts': 84,
        'n_classification_bins': 6,
        'layer_heights': [400, 400, 400, 400, 400],
        'layer_widths': [1, 1, 1, 1, 1],
        'layer_types': ['conv', 'res', 'full', 'full', 'full'],
        'activation_functions': ['relu', 'relu', 'relu', 'linear', 'linear'],

        # Initial conditions
        'INITIAL_WEIGHT_UNCERTAINTY': 0.02,
        'INITIAL_BIAS_UNCERTAINTY': 0.02,
        'INITIAL_WEIGHT_DISPLACEMENT': 0.1,
        'INITIAL_BIAS_DISPLACEMENT': 0.1,
        'USE_PERFECT_NOISE': False,

        # Priors
        'double_gaussian_weights_prior': True,
        'wide_prior_std': 1.0,
        'narrow_prior_std': 0.001,
        'spike_slab_weighting': 0.6
    }
    # configuration = {
    #     'data_transformation': {
    #         'feature_config_list': [
    #             {
    #                 'name': 'close',
    #                 'transformation': {
    #                     'name': 'log-return'
    #                 },
    #                 'normalization': 'standard',
    #                 'is_target': True,
    #                 'local': False,
    #             },
    #         ],
    #         'holiday_calendar': 'NYSE',
    #         'fill_limit': 0,
    #         'features_ndays': 10,
    #         'features_resample_minutes': 15,
    #         'features_start_market_minute': 60,
    #         'prediction_frequency_ndays': 1,
    #         'prediction_market_minute': 60,
    #         'target_delta_ndays': 1,
    #         'target_market_minute': 60,
    #     },
    #     'train_path': FIXTURE_DESTINATION_DIR,
    #     'covariance_method': 'NERCOME',
    #     'covariance_ndays': 9,
    #     'model_save_path': FIXTURE_DESTINATION_DIR,
    #     'tensorboard_log_path': FIXTURE_DESTINATION_DIR,
    #     'd_type': 'float32',
    #     'tf_type': 32,
    #     'random_seed': 0,
    #
    #     # Training specific
    #     'predict_single_shares': True,
    #     'n_epochs': 1,
    #     'n_retrain_epochs': 1,
    #     'learning_rate': 2e-3,
    #     'batch_size': 100,
    #     'cost_type': 'bayes',
    #     'n_train_passes': 30,
    #     'n_eval_passes': 100,
    #     'resume_training': False,
    #     'classify_per_series': False,
    #     'normalise_per_series': False,
    #
    #     # Topology
    #     'n_series': 3,
    #     'nassets': 3,
    #     'n_features_per_series': 271,
    #     'n_forecasts': 1,
    #     'n_classification_bins': 12,
    #     'layer_heights': [270, 270],
    #     'layer_widths': [3, 3],
    #     'activation_functions': ['relu', 'relu'],
    #
    #     # Initial conditions
    #     'INITIAL_ALPHA': 0.2,
    #     'INITIAL_WEIGHT_UNCERTAINTY': 0.4,
    #     'INITIAL_BIAS_UNCERTAINTY': 0.4,
    #     'INITIAL_WEIGHT_DISPLACEMENT': 0.1,
    #     'INITIAL_BIAS_DISPLACEMENT': 0.4,
    #     'USE_PERFECT_NOISE': True,
    #
    #     # Priors
    #     'double_gaussian_weights_prior': False,
    #     'wide_prior_std': 1.2,
    #     'narrow_prior_std': 0.05,
    #     'spike_slab_weighting': 0.5,
    #
    #     'use_gpu': False,
    #     'n_res_blocks': 6,
    #     'do_kernel_regularisation': True
    # }

    return configuration


def default_scheduling_config():
    return {
        'prediction_horizon': 24,
        'prediction_frequency':
            {
                'frequency_type': 'DAILY',
                'days_offset': 0,
                'minutes_offset': 60
            },
        'prediction_delta': 10,

        'training_frequency':
            {
                'frequency_type': 'DAILY',
                'days_offset': 0,
                'minutes_offset': 60
            },
        'training_delta': 20,
    }


def get_default_flags():
    default_config = default_oracle_config()
    return build_tensorflow_flags(default_config)

