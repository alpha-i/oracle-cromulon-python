import os
import logging

from keras.models import load_model
import pandas as pd
import numpy as np

import mrpb as pb

from alphai_finance.data.transformation import FinancialDataTransformation

from alphai_mvp_oracle.covariance import estimate_covariance
from alphai_mvp_oracle.constants import DATETIME_FORMAT_COMPACT
from alphai_mvp_oracle.helpers import TrainFileManager

TRAIN_FILE_NAME_TEMPLATE = "{}_train_mrpb.hd5"

logging.getLogger(__name__).addHandler(logging.NullHandler())


class MvpOracle:
    def __init__(self, configuration):
        """
        :param configuration: dictionary containing all the parameters
            data_transformation: Dictionary containing the financial-data-transformation configuration:
                features_dict: Dictionary containing the financial-features configuration, with feature names as keys:
                    order: ['value', 'log-return']
                    normalization: [None, 'robust', 'min_max', 'standard']
                    resample_minutes: resample frequencuy of feature data_x in miinutes.
                    ndays: number of days of feature data_x.
                    start_min_after_market_open: start time of feature data_x in minutes after market open.
                    is_target: boolean to define if this feature is a target (y). The feature is always consider as x.
                exchange_name: name of the exchange to create the market calendar 
                prediction_frequency_ndays: frequency of the prediction in days
                prediction_min_after_market_open: prediction time in number of minutes after market open 
                target_delta_ndays: days difference between prediction and target
                target_min_after_market_open: target time in number of minutes after market open 
            train_path: The path to a folder in which the train data is to be stored.
            covariance_method: The name of the covariance estimation method.
            covariance_ndays: The number of previous days those are needed for the covariance estimate (int).
            epochs: The number of epochs in the model training as an integer.
            learning_rate: The learning rate of the model as a float.
            verbose: Is a verbose output required? (bool)
            batch_size:  The batch size in training as an integer
            drop_out: The dropout rate as a float.
            l2: The value of the l2 regularisation constant.
            n_hidden: The number of hidden layers in the model.
            save_model: If true, save every trained model.
        """

        logging.info('Initialising MVP Oracle.')

        self._data_transformation = FinancialDataTransformation(configuration['data_transformation'])
        self._train_path = configuration['train_path']
        self._covariance_method = configuration['covariance_method']
        self._covariance_ndays = configuration['covariance_ndays']
        self._epochs = configuration['epochs']
        self._learning_rate = configuration['learning_rate']
        self._verbose = configuration['verbose']
        self._batch_size = configuration['batch_size']
        self._drop_out = configuration['drop_out']
        self._l2 = configuration['l2']
        self._n_hidden = configuration['n_hidden']
        self._save_model = configuration['save_model']
        self._ml_library = configuration['ml_library']

        self._train_file_manager = TrainFileManager(
            self._train_path,
            TRAIN_FILE_NAME_TEMPLATE,
            DATETIME_FORMAT_COMPACT
        )

        self._train_file_manager.ensure_path_exists()

        self._ml_model = None
        self._est_cov = None
        self._current_train = None

        assert self._ml_library in ['keras', 'TF'], 'ml_library needs to be in [keras, TF]'

    def train(self, historical_universes, train_data, execution_time):
        """
        Trains the model

        :param pd.DataFrame historical_universes: dates and symbols of historical universes
        :param dict train_data: OHLCV data as dictionary of pandas DataFrame.
        :param datetime.datetime execution_time: time of execution of training
        :return: 
        """
        logging.info('Training model on {}.'.format(
            execution_time,
        ))
        train_x, train_y = self._data_transformation.create_train_data(train_data, historical_universes)

        assert train_y.shape[1] == 1
        train_x = train_x.swapaxes(2, 3).swapaxes(1, 3)
        train_y = train_y.reshape(train_y.shape[0], train_y.shape[2])

        if self._ml_library == 'keras':
            model, history = pb.train_model(train_x, train_y, epochs=self._epochs, lr=self._learning_rate,
                                            verbose=self._verbose, batch_size=self._batch_size, do=self._drop_out,
                                            l2=self._l2, n_hidden=self._n_hidden)
            self._ml_model = model

            if self._save_model:
                train_path = self._train_file_manager.new_filename(execution_time)
                self._current_train = train_path
                model.save(train_path)

        elif self._ml_library == 'TF':  # TODO Implement TF model and save
            pass

    def predict(self, predict_data, execution_time):
        """

        :param dict predict_data: OHLCV data as dictionary of pandas DataFrame
        :param datetime.datetime execution_time: time of execution of prediction

        :return tuple : mean vector (pd.Series) and covariance matrix (np.matrix)
        """
        if self._save_model:
            self._load_latest_train(execution_time)

        logging.info('MVP Oracle prediction on {}.'.format(
            execution_time,
        ))
        if self._ml_model is None:
            raise ValueError('No trained ML model available for prediction.')

        # call the covariance library
        logging.info('Estimating covariance matrix.')
        cov = estimate_covariance(
            predict_data,
            self._covariance_ndays,
            self._data_transformation.target_min_after_market_open,
            self._covariance_method,
            self._data_transformation.exchange_calendar,
            self._data_transformation.target_delta_ndays
        )
        if not np.isfinite(cov).all():
            raise ValueError('Covariance matrix computation failed. Contains non-finite values.')
        # convert the array into a dataframe
        cov = pd.DataFrame(data=cov, columns=predict_data['close'].columns, index=predict_data['close'].columns)

        predict_x = self._data_transformation.create_predict_data(predict_data)

        logging.info('Predicting mean values.')
        if self._ml_library == 'keras':
            means = self._ml_model.predict(predict_x)
        elif self._ml_library == 'TF':  # TODO Implement TF predict ot get means  and cov
            pass

        if not np.isfinite(means).all():
            raise ValueError('prediction of means failed. Contains non-finite values.')

        means = pd.Series(np.squeeze(means), index=predict_data['close'].columns)
        return means, cov

    def _load_latest_train(self, execution_time):

        latest_train = self._train_file_manager.latest_train_filename(
            execution_time
        )

        if latest_train != self._current_train:
            if self._ml_library == 'keras':
                self._ml_model = load_model(
                    os.path.join(
                        self._train_path,
                        latest_train
                    )
                )
                self._current_train = latest_train
            elif self._ml_library == 'TF':  # TODO Implement TF load latest model
                pass


