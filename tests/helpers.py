import os
import shutil

import pandas_market_calendars as mcal
from alphai_finance.data.cleaning import convert_to_utc, select_trading_hours, fill_gaps_data_dict, resample_ohlcv
from alphai_finance.data.read_from_hdf5 import read_feature_data_dict_from_hdf5

from alphai_crocubot_oracle.oracle import CrocubotOracle

DATA_FILENAME = 'sample_hdf5.h5'

FIXTURES_SOURCE_DIR = os.path.join(os.path.dirname(__file__), 'resources')
FIXTURE_DESTINATION_DIR = '/tmp/crocubot/'

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


class DummyCrocubotOracle(CrocubotOracle):
    def __init__(self, configuration):
        super().__init__(configuration)

    def get_train_file_manager(self):
        return self._train_file_manager

    def get_current_train(self):
        return self._current_train
