import os
from datetime import datetime, timedelta
from unittest import TestCase

import pandas as pd

from alphai_cromulon_oracle import DATETIME_FORMAT_COMPACT
from alphai_cromulon_oracle.oracle import TRAIN_FILE_NAME_TEMPLATE

from tests.helpers import (
    FIXTURE_DESTINATION_DIR,
    FIXTURE_DATA_FULLPATH,
    create_fixtures,
    destroy_fixtures,
    read_hdf5_into_dict_of_data_frames,
    DummyCromulonOracle,
    default_oracle_config,
    default_scheduling_config,
    DEFAULT_CALENDAR_NAME)


class TestCromulon(TestCase):
    def setUp(self):
        create_fixtures()

    def tearDown(self):
        destroy_fixtures()

    @staticmethod
    def _prepare_data_for_test():
        start_date = '20140102'  # these are values for the resources/sample_hdf5.h5
        end_date = '20140228'
        symbols = ['AAPL', 'INTC', 'MSFT']
        exchange_name = 'NYSE'
        fill_limit = 10
        resample_rule = '15T'

        data = read_hdf5_into_dict_of_data_frames(start_date,
                                                  end_date,
                                                  symbols,
                                                  FIXTURE_DATA_FULLPATH,
                                                  exchange_name,
                                                  fill_limit,
                                                  resample_rule
                                                  )
        return data

    def test_crocubot_train_and_predict(self):
        data = self._prepare_data_for_test()

        oracle_configuration = default_oracle_config()
        oracle_configuration['model']['n_correlated_series'] = 1

        oracle = DummyCromulonOracle(
            DEFAULT_CALENDAR_NAME,
            oracle_configuration=oracle_configuration,
            scheduling_configuration=default_scheduling_config()

        )

        train_time = datetime(2014, 1, 20, 9) + timedelta(minutes=60)
        target_timestamp = train_time + oracle.prediction_horizon

        oracle.train(data, train_time)

        predict_data = self._prepare_data_for_test()
        oracle.predict(predict_data, train_time, target_timestamp)

    def test_crocubot_train_and_save_file(self):
        train_time = datetime(2017, 6, 7, 9) + timedelta(minutes=60)
        train_filename = TRAIN_FILE_NAME_TEMPLATE.format(train_time.strftime(DATETIME_FORMAT_COMPACT))

        expected_train_path = os.path.join(FIXTURE_DESTINATION_DIR, train_filename)

        data = self._prepare_data_for_test()

        oracle = DummyCromulonOracle(
            DEFAULT_CALENDAR_NAME,
            oracle_configuration=default_oracle_config(),
            scheduling_configuration=default_scheduling_config()

        )

        oracle.train(data, train_time)

        tf_suffix = '.index'  # TF adds stuff to the end of its save files
        full_tensorflow_path = expected_train_path + tf_suffix
        self.assertTrue(os.path.exists(full_tensorflow_path))

    def test_crocubot_predict_without_train_file(self):
        oracle = DummyCromulonOracle(
            DEFAULT_CALENDAR_NAME,
            oracle_configuration=default_oracle_config(),
            scheduling_configuration=default_scheduling_config()

        )

        execution_time = datetime(2017, 6, 7, 9) + timedelta(minutes=60)

        _, predict_data = self._prepare_data_for_test()
        self.assertRaises(
            ValueError,
            oracle.predict,
            predict_data,
            execution_time
        )
