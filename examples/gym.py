# Uses the oracle to train and predict the gym data.

from datetime import date, timedelta
from datetime import datetime as dt
from copy import deepcopy
import logging
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateFormatter, AutoDateLocator

from alphai_cromulon_oracle.oracle import CromulonOracle
from alphai_cromulon_oracle.oracle import OraclePrediction
from alphai_delphi.oracle.oracle_configuration import OracleConfiguration

import examples.gym_iotools as io

logger = logging.getLogger('tipper')
logger.addHandler(logging.StreamHandler())
logging.basicConfig(level=logging.DEBUG)

EXECUTION_TIME = date(2016, 12, 7)  # beware - chunks missing from data in 2017
D_TYPE = 'float32'


def run_oracle():

    config = load_gym_config()
    gym_df = io.load_gym_dataframe()

    cut_gym_df = deepcopy(gym_df)
    cut_time = EXECUTION_TIME + timedelta(days=1)
    cut_gym_df = truncate_dataframe(cut_gym_df, cut_time)

    full_data_dict = make_dict_from_dataframe(gym_df)
    data_dict = make_dict_from_dataframe(cut_gym_df)

    oracle = CromulonOracle(config)
    oracle.train(full_data_dict, EXECUTION_TIME)

    prediction = oracle.predict(data_dict, EXECUTION_TIME, number_of_iterations=1)
    actuals = extract_actuals(full_data_dict, prediction.lower_bound.index)

    return prediction, actuals


def truncate_dataframe(gym_df, execution_time):

    return gym_df.ix[:execution_time]    # gym_df[(gym_df['date'] < EXECUTION_TIME)]


def extract_actuals(data_dict, index):

    target = data_dict['number_people']
    actuals = target.loc[index]

    return actuals


def make_dummy_prediction():
    """ Generates a mock prediction result (2 gyms; 3 forecasts each), useful for testing purposes. """

    gym_names = ['UCB', 'Santa Cruz']

    current_timestamp = dt(2008, 1, 1, 2, 2, 2)
    time1 = dt(2008, 1, 2, 2, 2, 2)
    time2 = dt(2008, 1, 3, 2, 2, 2)
    time3 = dt(2008, 1, 4, 2, 2, 2)
    target_timestamps = [time1, time2, time3]

    n_gyms = len(gym_names)
    n_timestamps = len(target_timestamps)
    data_shape = (n_timestamps, n_gyms)
    means = np.random.normal(10, 3, data_shape)
    conf_low = means - 1.5
    conf_high = means + 1.5

    means_pd = pd.DataFrame(data=means, columns=gym_names, index=target_timestamps)
    conf_low_pd = pd.DataFrame(data=conf_low, columns=gym_names, index=target_timestamps)
    conf_high_pd = pd.DataFrame(data=conf_high, columns=gym_names, index=target_timestamps)

    return OraclePrediction(means_pd, conf_low_pd, conf_high_pd, current_timestamp)


def make_dict_from_dataframe(df):
    """ Takes the csv-derived dataframe and splits into dict where each key is a column from the ."""

    cols = df.columns
    gym_names = ['UCBerkeley']
    data_dict = {}

    for col in cols:
        values = getattr(df, col).values
        data_dict[col] = pd.DataFrame(data=values, index=df.index, columns=gym_names, dtype=D_TYPE)

    return data_dict


def load_gym_config():

    n_forecasts = 84  # 7x24

    configuration = {
        'nassets': 1,
        'data_transformation': {
            'fill_limit': 5,
            'holiday_calendar': 'NYSE',
            'feature_config_list': [
                {
                    'name': 'number_people',
                    'normalization': 'min_max',
                    'length': 20,
                    'is_target': True,
                    'resolution': 60
                },
                {
                    'name': 'number_people',
                    'normalization': 'min_max',
                    'length': 20,
                    'resolution': 1440
                },
                ],
            # TODO ASK FERGUS ABOUT THIS
            # 'target_config_list': [
            #                            {
            #                                'name': 'number_people',
            #                                'length': n_forecasts,
            #                                'resolution': 4
            #                            },
            # ],
            'data_name': 'GYM',
            'features_ndays': 10,
            'features_resample_minutes': 15
        },
        'train_path': '/tmp/cromulon/',
        'model_save_path': '/tmp/cromulon/',
        'tensorboard_log_path': '/tmp/cromulon/',
        'd_type': D_TYPE,
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
        'n_forecasts': n_forecasts,
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

    oracle_config = OracleConfiguration(
        {
            "scheduling": {
                "prediction_horizon": 1/24,
                "prediction_frequency":
                    {
                        "frequency_type": "MINUTE",
                        "days_offset": 0,
                        "minutes_offset": 15
                    },
                "prediction_delta": 10,

                "training_frequency":
                    {
                        "frequency_type": "WEEKLY",
                        "days_offset": 0,
                        "minutes_offset": 15
                    },
                "training_delta": 20,
            },
            "oracle": configuration
        })

    return oracle_config

# prediction = make_dummy_prediction()
#
# with open('./dummy_prediction.pickle', 'wb') as handle:
#     pickle.dump(prediction, handle)
predictions, actuals = run_oracle()

print("Predictions: ", predictions)
print("actuals: ", actuals)

plt.figure(num=1)

ax = plt.axes()

t_predict = predictions.lower_bound.index.values

ax.plot(t_predict, predictions.lower_bound.values)
ax.plot(t_predict, predictions.upper_bound.values)

# Need actuals too
t_actuals = t_predict
ax.scatter(t_actuals.tolist(), list(actuals.values))

plt.ylim(0, 45)
plt.show()
