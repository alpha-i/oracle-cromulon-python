# Uses the oracle to train and predict the gym data.
import tempfile
import logging
from datetime import timedelta, datetime
from datetime import datetime as dt
from copy import deepcopy

import pandas as pd
import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from alphai_cromulon_oracle.oracle import CromulonOracle
from alphai_cromulon_oracle.oracle import OraclePrediction

import examples.gym_iotools as io

RUNTIME_PATH = tempfile.TemporaryDirectory().name

logger = logging.getLogger('tipper')
logger.addHandler(logging.StreamHandler())
logging.basicConfig(level=logging.DEBUG)

EXECUTION_TIME = datetime(2016, 12, 7)  # beware - chunks missing from data in 2017
D_TYPE = 'float32'

N_FORECASTS = 84  # 7x24

CALENDAR_NAME = 'GYMUK'

oracle_configuration = {
    'prediction_horizon': {
        'unit': 'hours',
        'value': 1
    },
    "prediction_delta": {
        'unit': 'days',
        'value': 10
    },
    "training_delta": {'unit': 'days', 'value': 20},
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
        'data_name': 'GYM',
        'features_ndays': 10,
        'features_resample_minutes': 15
    },
    'model': {
        'n_series': 1,
        'n_assets': 1,
        'train_path': RUNTIME_PATH,
        'model_save_path': RUNTIME_PATH,
        'tensorboard_log_path': RUNTIME_PATH,
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
        'do_kernel_regularisation': True,
        'do_batch_norm': False,
        'n_res_blocks': 6,
        'n_features_per_series': 271,
        'n_forecasts': N_FORECASTS,
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

}
scheduling_config = {
    "prediction_frequency":
        {
            "frequency_type": "MINUTE",
            "days_offset": 0,
            "minutes_offset": 15
        },
    "training_frequency":
        {
            "frequency_type": "WEEKLY",
            "days_offset": 0,
            "minutes_offset": 15
        },
}


def run_oracle():

    gym_df = io.load_gym_dataframe()

    cut_gym_df = deepcopy(gym_df)
    cut_time = EXECUTION_TIME + timedelta(days=1)

    cut_gym_df = truncate_dataframe(cut_gym_df, cut_time)

    full_data_dict = make_dict_from_dataframe(gym_df)
    data_dict = make_dict_from_dataframe(cut_gym_df)

    oracle = CromulonOracle(CALENDAR_NAME,
                            oracle_configuration=oracle_configuration,
                            scheduling_configuration=scheduling_config
                            )
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

    current_timestamp = datetime(2008, 1, 1, 2, 2, 2)
    time1 = datetime(2008, 1, 2, 2, 2, 2)
    time2 = datetime(2008, 1, 3, 2, 2, 2)
    time3 = datetime(2008, 1, 4, 2, 2, 2)
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


def make_dict_from_dataframe(csv_dataframe):
    """ Takes the csv-derived dataframe and splits into dict where each key is a column from the ."""

    cols = csv_dataframe.columns
    gym_names = ['UCBerkeley']
    data_dict = {}

    for col in cols:
        values = getattr(csv_dataframe, col).values
        data_dict[col] = pd.DataFrame(data=values, index=csv_dataframe.index, columns=gym_names, dtype=D_TYPE)

    return data_dict

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
