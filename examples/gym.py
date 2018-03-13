# Uses the oracle to train and predict the gym data.
import tempfile
import logging
from datetime import timedelta, datetime
from copy import deepcopy

import pandas as pd

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from alphai_cromulon_oracle.oracle import CromulonOracle
from examples.gym_iotools import load_gym_dataframe


logging.basicConfig(level=logging.DEBUG)
logging.getLogger('tipper').addHandler(logging.StreamHandler())


RUNTIME_PATH = tempfile.TemporaryDirectory().name
TARGET_FEATURE = 'number_people'

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
    "training_delta": {
        'unit': 'days',
        'value': 20
    },
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


def make_dict_from_dataframe(csv_dataframe):
    """ Takes the csv-derived dataframe and splits into dict where each key is a column from the ."""

    cols = csv_dataframe.columns
    gym_names = ['UCBerkeley']
    data_dict = {}

    for col in cols:
        values = getattr(csv_dataframe, col).values
        data_dict[col] = pd.DataFrame(data=values, index=csv_dataframe.index, columns=gym_names, dtype=D_TYPE)

    return data_dict


if __name__ == '__main__':

    gym_df = load_gym_dataframe()
    cut_gym_df = deepcopy(gym_df)
    cut_time = EXECUTION_TIME + timedelta(days=1)
    cut_gym_df = cut_gym_df.ix[:cut_time]

    complete_dataset = make_dict_from_dataframe(gym_df)
    prediction_dataset = make_dict_from_dataframe(cut_gym_df)

    oracle = CromulonOracle(CALENDAR_NAME,
                            oracle_configuration=oracle_configuration,
                            scheduling_configuration=scheduling_config
                            )
    oracle.train(complete_dataset, EXECUTION_TIME)

    prediction = oracle.predict(prediction_dataset, EXECUTION_TIME, number_of_iterations=1)

    actuals = complete_dataset[TARGET_FEATURE].loc[prediction.lower_bound.index]

    print("Predictions: ", prediction)
    print("actuals: ", actuals)

    plt.figure(num=1)

    ax = plt.axes()

    t_predict = prediction.lower_bound.index.values

    ax.plot(t_predict, prediction.lower_bound.values)
    ax.plot(t_predict, prediction.upper_bound.values)

    t_actuals = t_predict
    ax.scatter(t_actuals.tolist(), list(actuals.values))

    plt.ylim(0, 45)
    plt.show()
