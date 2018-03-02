# Used to read and process the gym data, with entries as follows:
# 'number_people', 'date', 'timestamp', 'is_weekend', 'is_holiday',
#        'temperature', 'is_start_of_semester', 'is_during_semester',
#        'day_of_week_0', 'day_of_week_1', 'day_of_week_2', 'day_of_week_3',
#        'day_of_week_4', 'day_of_week_5', 'day_of_week_6', 'month_1', 'month_2',
#        'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8',
#        'month_9', 'month_10', 'month_11', 'month_12', 'hour_0', 'hour_1',
#        'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8',
#        'hour_9', 'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14',
#        'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20',
#        'hour_21', 'hour_22', 'hour_23'],
import os
import numpy as np
import pandas as pd
import pytz
from datetime import timedelta, date

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import alphai_feature_generation.classifier  as cl

GYM_DATA_FILE = os.path.join(os.path.dirname(__file__), 'data.csv')

DEFAULT_TEST_FRACTION = 0.05
DROP_MONTHS = False  # set True for random forest, as it
COARSE_TEMP = False
DROP_DAYS = False
DEFAULT_TIME_RESOLUTION = '15T'


def load_scaled_gym_data(do_random_split, test_fraction=DEFAULT_TEST_FRACTION, start_test_date=None):

    X_train, X_test, y_train, y_test = load_x_y_gym_data(do_random_split, test_fraction, start_test_date)

    # Scale the data to be between -1 and 1
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def load_classified_gym_data(n_bins, test_fraction=DEFAULT_TEST_FRACTION):
    """ Returns training/test data with classified y

    :param n_bins:
    :return:
    """

    X_train, X_test, y_train, y_test = load_scaled_gym_data(do_random_split=False, test_fraction=test_fraction)

    y_train_bins, y_test_bins = classify_training_data(y_train, y_test, n_bins)

    return X_train, X_test,y_train_bins, y_test_bins


def rms(x):
    return np.sqrt(x.dot(x)/x.size)


def load_gym_dates():

    df = load_gym_dataframe()

    return df.index


def extract_time_range(df, start_date, end_date):
    """ Returns portion of a dataframe falling between two dates. """

    mask = (df['date'] > start_date) & (df['date'] <= end_date)
    return df.loc[mask]


def load_single_attendance(do_daily=True):
    """ Returns time-averaged attendance data

    :param do_daily:
    :return:
    """
    df = load_raw_gym_data()
    df.index = pd.to_datetime(df.date)
    df.date = pd.to_datetime(df.date)
    start_date = date(2016, 2, 4)

    delta_days = 1 if do_daily else 7
    end_date = start_date + timedelta(days=delta_days)

    df = extract_time_range(df, start_date, end_date)

    if not do_daily:
        df['weekday'] = [ts.dayofweek for ts in df.date]
        df = df.groupby('weekday').mean()

    time_blocks = df.index.values
    attendance = df['number_people'].values

    return time_blocks, attendance


def load_average_attendance(do_daily=True):
    """ Returns time-averaged attendance data

    :param do_daily:
    :return:
    """
    df = load_raw_gym_data()
    df.index = pd.to_datetime(df.date)

    if do_daily:
        df['hour'] = [ts.hour for ts in df.index]
        df = df.groupby('hour').mean()
    else:
        df['weekday'] = [ts.dayofweek for ts in df.index]
        df = df.groupby('weekday').mean()

    time_blocks = df.index.values
    average_attendance = df['number_people'].values

    return time_blocks, average_attendance


def load_x_y_gym_data(do_random_split=False, test_fraction=DEFAULT_TEST_FRACTION, start_test_date=None):
    """  Load gym data

    :param do_random_split:
    :return: Train and test data as numpy arrays
    """

    df = load_gym_dataframe()

    if DROP_MONTHS:
        df = df.drop(list(df.filter(regex='month')), axis=1)

    if DROP_DAYS:
        df = df.drop(list(df.filter(regex='day_of_week_2')), axis=1)

    if COARSE_TEMP:
        temp_vals = df.temperature
        temp_vals = np.floor(temp_vals)
        df.temperature = temp_vals

    # Drop data to avoid interpolation of training data on test
    unwanted_labels = ['date', 'timestamp']
    df = df.drop(unwanted_labels, axis=1)

    # # Add yesterday's attendance levels
    # entries_per_day = 100
    # df['old_attendance'] = df.number_people.shift(entries_per_day).fillna(value=0)

    feature_list = df.columns.tolist()
    print("Utilising ", len(feature_list), "features:", feature_list)

    data = df.values  # 62184 x 50

    if do_random_split:
        X = data[:, 1:]  # All data except number of people
        y = data[:, 0]  # Number of people
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=42)
    else:
        # Either split by date or fraction
        n_datapoints = data.shape[0]
        if start_test_date is None:
            split_index = int(n_datapoints * (1 - test_fraction))
        else:
            start_train_date = pd.datetime(1900, 1, 1)
            df['date'] = df.index
            df = extract_time_range(df, start_train_date, start_test_date)
            n_train_points = df.shape[0]
            split_index = n_train_points
            n_test_points = n_datapoints - n_train_points
            print("Selected ", n_train_points, " train samples and", n_test_points, " test samples")

        X_train = data[0:split_index, 1:]
        X_test = data[split_index:, 1:]
        y_train = data[0:split_index, 0]
        y_test = data[split_index:, 0]

    return X_train, X_test, y_train, y_test


def load_gym_dataframe():
    """

    :return: DataFrame of gym attendance
    """

    df = load_raw_gym_data()
    df = df.set_index(df.date)
    df.index = pd.to_datetime(df.index, utc=True).round(DEFAULT_TIME_RESOLUTION)
    df.index = df.index.tz_convert(pytz.timezone('US/Pacific'))

    df = df.loc[~df.index.duplicated(keep='first')]  # Remove duplicate entries

    df['date'] = df.index
    # One hot encode categorical columns
    columns = ["day_of_week", "month", "hour"]
    df = pd.get_dummies(df, columns=columns)

    return df


def classify_training_data(y_train, y_test, n_bins):
    """ Takes 1D numpy arrays and assigns them to equally populated bins

    :param y_train:
    :param y_test:
    :param n_bins:
    :return:
    """

    classifier = cl.BinDistribution(y_train, n_bins)

    y_train_bins = classifier.classify_labels(y_train)
    y_test_bins = classifier.classify_labels(y_test)

    return y_train_bins, y_test_bins


def declassify_bins(y_continuous, y_binned, n_bins):

    classifier = cl.BinDistribution(y_continuous, n_bins)

    return classifier.extract_point_estimates( y_binned, use_median=True)


def declassify_bins_with_errors(y_continuous, y_binned, n_bins):

    classifier = cl.BinDistribution(y_continuous, n_bins)

    n_pdfs = y_binned.shape[0]
    mean = np.zeros(n_pdfs)
    variance = np.zeros(n_pdfs)

    for i in range(n_pdfs):
        ybins = y_binned[i, :]
        temp_mean, temp_variance = classifier.declassify_single_pdf(ybins)
        mean[i] = temp_mean
        variance[i] = temp_variance

    uncertainty = np.sqrt(variance)

    return mean, uncertainty


def time_to_seconds(time):
    return time.hour * 3600 + time.minute * 60 + time.second


def load_raw_gym_data():
    return pd.read_csv(GYM_DATA_FILE)


def mean_absolute_percentage_error(y_true, y_pred):
    """  mean_absolute_percentage_error

    :param nparray y_true:
    :param nparray y_pred:
    :return:
    """

    abs_errors = (np.abs((y_true - y_pred) / y_true)) * 100
    finite_apes = abs_errors[np.isfinite(abs_errors)]

    return np.mean(finite_apes)


def convert_stupid_date_format_for_plotting(dates):
    """

    :param dates:
    :return:
    """

    dates_for_plt = []
    for i in range(len(dates)):
        forecast_date = dates[i].to_pydatetime()
        dates_for_plt.append(forecast_date)

    return dates_for_plt


