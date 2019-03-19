from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

buss_dataframe = pd.read_csv("data.csv", sep=",")

buss_dataframe = buss_dataframe.reindex(
    np.random.permutation(buss_dataframe.index))


def preprocess_features(buss_dataframe):
    """Prepares input features from buss data set.

    Args:
      buss_dataframe: A Pandas DataFrame expected to contain data
        from the buss data set.
    Returns:
      A DataFrame that contains the features to be used for the model, including
      synthetic features.
    """
    selected_features = buss_dataframe[
        ["idFrom",
         "idTo",
         "vehicleType",
         "month",
         "day",
         "hour",
         "minute",
         "holiday",
         "vacation",
         "temperature",
         "pType"]]
    processed_features = selected_features.copy()
    return processed_features


def preprocess_targets(buss_dataframe):
    """Prepares target features (i.e., labels) from buss data set.

    Args:
      buss_dataframe: A Pandas DataFrame expected to contain data
        from the buss data set.
    Returns:
      A DataFrame that contains the target feature.
    """
    output_targets = pd.DataFrame()
    output_targets["secondsDelay"] = (
        buss_dataframe["secondsDelay"])
    return output_targets


def linear_scale(series):
    min_val = series.min()
    max_val = series.max()
    scale = (max_val - min_val) / 2.0
    return series.apply(lambda x: ((x - min_val) / scale) - 1.0)


def normalize_linear_scale(examples_dataframe):
    """Returns a version of the input `DataFrame` that has all its features normalized linearly."""
    processed_features = pd.DataFrame()
    processed_features["vehicleType"] = linear_scale(examples_dataframe["vehicleType"])
    processed_features["month"] = linear_scale(examples_dataframe["month"])
    processed_features["day"] = linear_scale(examples_dataframe["day"])
    processed_features["hour"] = linear_scale(examples_dataframe["hour"])
    processed_features["minute"] = linear_scale(examples_dataframe["minute"])
    processed_features["temperature"] = linear_scale(examples_dataframe["temperature"])
    processed_features["pType"] = linear_scale(examples_dataframe["pType"])
    return processed_features


normalized_dataframe = normalize_linear_scale(preprocess_features(buss_dataframe))
training_examples = normalized_dataframe.head(15000)
print(training_examples.describe())

training_targets = preprocess_targets(buss_dataframe.head(15000))
print(training_targets.describe())

validation_examples = normalized_dataframe.tail(5000)
print(validation_examples.describe())

validation_targets = preprocess_targets(buss_dataframe.tail(5000))
print(validation_targets.describe())


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of multiple features.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def construct_feature_columns(input_features):
    """Construct the TensorFlow Feature Columns.

    Args:
      input_features: The names of the numerical input features to use.
    Returns:
      A set of feature columns
    """
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features])


def train_model(
        my_optimizer,
        steps,
        batch_size,
        hidden_units,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a linear regression model of multiple features.

    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.

    Args:
      learning_rate: A `float`, the learning rate.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      training_examples: A `DataFrame` containing one or more columns from
        `buss_dataframe` to use as input features for training.
      training_targets: A `DataFrame` containing exactly one column from
        `buss_dataframe` to use as target for training.
      validation_examples: A `DataFrame` containing one or more columns from
        `buss_dataframe` to use as input features for validation.
      validation_targets: A `DataFrame` containing exactly one column from
        `buss_dataframe` to use as target for validation.

    Returns:
      A `LinearRegressor` object trained on the training data.
    """

    global validation_root_mean_squared_error, training_root_mean_squared_error
    periods = 10
    steps_per_period = steps / periods

    # # Create a linear regressor object.
    # my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    # linear_regressor = tf.estimator.LinearRegressor(
    #     feature_columns=construct_feature_columns(training_examples),
    #     optimizer=my_optimizer
    # )

    # Create a DNNRegressor object.
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    dnn_regressor = tf.estimator.DNNRegressor(
        feature_columns=construct_feature_columns(training_examples),
        hidden_units=hidden_units,
        optimizer=my_optimizer,
    )

    # Create input functions.
    training_input_fn = lambda: my_input_fn(
        training_examples,
        training_targets["secondsDelay"],
        batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(
        training_examples,
        training_targets["secondsDelay"],
        num_epochs=1,
        shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(
        validation_examples, validation_targets["secondsDelay"],
        num_epochs=1,
        shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        dnn_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period,
        )
        # Take a break and compute predictions.
        training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        # Compute training and validation loss.
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    plt.show()

    print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
    print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)

    return dnn_regressor


dnn_regressor = train_model(
    my_optimizer=tf.train.AdagradOptimizer(learning_rate=0.3),
    steps=10000,
    batch_size=1000,
    hidden_units=[20, 20, 10, 5],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

# _ = training_examples.hist(bins=40, figsize=(18, 12), xlabelsize=10)
# plt.show()
