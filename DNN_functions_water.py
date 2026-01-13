import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

'''
Useful functions for training and evaluation of DNN ensemble for water permeability prediction.
Modified from DNN_functions.py to handle single output (water) instead of 6 gases.
'''

# User-defined mean squared error that ignores NaN values
# Used as the loss function when training on non-imputed data with missing entries
def nanmean_squared_error(y_true, Y_pred):
    """
    Custom loss function that ignores NaN values in the target variable.

    Args:
        y_true: True target values (may contain NaN)
        Y_pred: Predicted values

    Returns:
        Mean squared error calculated only on non-NaN values
    """
    Y_pred = ops.convert_to_tensor_v2(Y_pred)
    y_true = math_ops.cast(y_true, Y_pred.dtype)
    residuals = (y_true - Y_pred)
    # Convert NaN values to 0 for residual computation
    residuals_no_nan = tf.where(tf.math.is_nan(residuals), tf.zeros_like(residuals), residuals)
    # Calculate MSE only on non-NaN values
    sum_residuals = tf.reduce_sum(math_ops.squared_difference(residuals_no_nan, 0), -1) / tf.reduce_sum(tf.cast(~tf.math.is_nan(y_true), tf.float64), -1)
    return sum_residuals


# Evaluate a single DNN for water permeability regression - MLP Model (Multi-Layer Perceptron)
def evaluate_model(trainX, trainy, testX, testy):
    """
    Train and evaluate a single DNN model for water permeability prediction.

    Architecture:
        - Input layer: variable size (depends on features)
        - Hidden layers: 64 -> 64 -> 32 -> 16 -> 8 neurons (all with ReLU activation)
        - Dropout: 0.1 (10% dropout to prevent overfitting)
        - Output layer: 1 neuron (for water permeability)

    Args:
        trainX: Training features
        trainy: Training targets (water permeability)
        testX: Test features
        testy: Test targets (water permeability)

    Returns:
        model: Trained Keras model
        test_r2: R² score on test set
    """
    model = keras.models.Sequential()
    model.add(Dense(units=64, activation='relu'))  # First hidden layer
    model.add(Dense(units=64, activation='relu'))  # Second hidden layer
    model.add(Dense(units=32, activation='relu'))  # Third hidden layer
    model.add(Dense(units=16, activation='relu'))  # Fourth hidden layer
    model.add(Dense(units=8, activation='relu'))   # Fifth hidden layer
    model.add(Dropout(0.1))  # Dropout layer to prevent overfitting
    model.add(Dense(units=1))  # Output layer - SINGLE output for water permeability

    # Compile model with custom loss function and Adam optimizer
    model.compile(loss=nanmean_squared_error, optimizer='adam')

    # Train the model
    history = model.fit(
        trainX, trainy,
        epochs=50,
        batch_size=64,
        validation_data=(testX, testy),
        verbose=0  # Silent training
    )

    # Evaluate the model on test set
    predy = model.predict(testX)

    # Calculate R² score (ignoring NaN values if present)
    flag1 = ~np.isnan(testy.flatten())  # Identify non-NaN values
    test_r2 = r2_score(testy.flatten()[flag1], predy.flatten()[flag1])

    return model, test_r2


# Make ensemble predictions for water permeability
def ensemble_predictions(members, testX):
    """
    Generate ensemble predictions by averaging predictions from multiple DNN models.

    Args:
        members: List of trained DNN models
        testX: Features to predict on

    Returns:
        predictions: Mean predictions across all ensemble members
        variances: Variance of predictions across ensemble members (uncertainty estimate)
    """
    # Make predictions with each ensemble member
    yhats = [model.predict(testX) for model in members]
    yhats = np.array(yhats)

    # Calculate mean and variance across ensemble
    predictions = np.mean(yhats, axis=0)  # Mean prediction
    variances = np.var(yhats, axis=0)     # Prediction uncertainty

    return predictions, variances


# Evaluate a specific number of members in an ensemble
def evaluate_n_members(members, n_members, testX, testy):
    """
    Evaluate ensemble performance using a subset of ensemble members.

    Args:
        members: List of all trained DNN models
        n_members: Number of ensemble members to use
        testX: Test features
        testy: Test targets

    Returns:
        test_r2: R² score on test set
        avg_var: Average prediction variance (uncertainty)
    """
    # Select subset of ensemble members
    subset = members[:n_members]

    # Make ensemble predictions
    yhat, variances = ensemble_predictions(subset, testX)
    avg_var = np.mean(variances)

    # Calculate R² score (ignoring NaN values if present)
    flag1 = ~np.isnan(testy.flatten())
    test_r2 = r2_score(testy.flatten()[flag1], yhat.flatten()[flag1])

    return test_r2, avg_var
