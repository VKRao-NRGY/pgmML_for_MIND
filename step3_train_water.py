import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
tf.keras.backend.set_floatx('float64')
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import pickle
import argparse
import os
from DNN_functions_water import nanmean_squared_error, evaluate_model, ensemble_predictions, evaluate_n_members


'''
Script to train DNN model for water permeability prediction.
Uses Morgan fingerprints ('fing') as input features.
Outputs the saved model into the models folder, along with Y_train, Y_test, Y_pred_train, and Y_pred_test as .csv files.

Key differences from step3_train.py:
- Single output (water permeability) instead of 6 gases
- Optional imputation support (--imputation 'BLR' or 'none')
- Focus on DNN with fingerprint features
'''


def train(args):
    """
    Train DNN ensemble for water permeability prediction.

    Args:
        args: Command-line arguments containing:
            - dataset: Path to water permeability dataset CSV
            - features: Path to feature CSV file (fingerprints or descriptors)
            - imputation: Imputation method ('BLR', 'ERT', or 'none')
            - target_column: Name of the target column in dataset (default: 'log10_Water')
    """

    print('='*60)
    print('Water Permeability DNN Training')
    print('='*60)
    print(f'Dataset: {args.dataset}')
    print(f'Features: {args.features}')
    print(f'Imputation: {args.imputation}')
    print(f'Target column: {args.target_column}')
    print('='*60)

    # Read the training data
    try:
        DatasetA_Smiles_P = pd.read_csv(args.dataset)
        print(f'✓ Loaded dataset: {DatasetA_Smiles_P.shape[0]} rows')
    except FileNotFoundError:
        print(f'ERROR: Dataset file not found: {args.dataset}')
        print('Please provide a valid water permeability dataset CSV file.')
        return

    # Group by SMILES to average multiple measurements of the same polymer
    DatasetA_grouped = DatasetA_Smiles_P.groupby('Smiles').mean().reset_index()
    print(f'✓ Grouped by SMILES: {DatasetA_grouped.shape[0]} unique polymers')

    # Extract target variable (water permeability)
    if args.imputation == 'none':
        # Use raw target column without imputation
        if args.target_column not in DatasetA_grouped.columns:
            print(f'ERROR: Target column "{args.target_column}" not found in dataset.')
            print(f'Available columns: {list(DatasetA_grouped.columns)}')
            return
        Y = DatasetA_grouped[[args.target_column]].values
        imputation_suffix = 'none'

    elif args.imputation == 'BLR':
        # Use Bayesian Linear Regression imputed values
        imputed_col = args.target_column + '_Bayesian'
        if imputed_col not in DatasetA_grouped.columns:
            # Fallback to non-imputed column
            print(f'WARNING: Imputed column "{imputed_col}" not found. Using "{args.target_column}" instead.')
            if args.target_column not in DatasetA_grouped.columns:
                print(f'ERROR: Target column "{args.target_column}" not found in dataset.')
                return
            Y = DatasetA_grouped[[args.target_column]].values
        else:
            Y = DatasetA_grouped[[imputed_col]].values
            print(f'✓ Using BLR-imputed column: {imputed_col}')
        imputation_suffix = 'BLR'

    elif args.imputation == 'ERT':
        # Use Extremely Randomized Trees imputed values
        imputed_col = args.target_column + '_Etree'
        if imputed_col not in DatasetA_grouped.columns:
            # Fallback to non-imputed column
            print(f'WARNING: Imputed column "{imputed_col}" not found. Using "{args.target_column}" instead.')
            if args.target_column not in DatasetA_grouped.columns:
                print(f'ERROR: Target column "{args.target_column}" not found in dataset.')
                return
            Y = DatasetA_grouped[[args.target_column]].values
        else:
            Y = DatasetA_grouped[[imputed_col]].values
            print(f'✓ Using ERT-imputed column: {imputed_col}')
        imputation_suffix = 'ERT'

    else:
        print(f'ERROR: Invalid imputation method "{args.imputation}". Choose "BLR", "ERT", or "none".')
        return

    # Check for missing values
    nan_count = np.isnan(Y).sum()
    if nan_count > 0:
        print(f'WARNING: Found {nan_count} missing values in target variable.')
        print('The model will handle these using the custom nanmean_squared_error loss function.')
    else:
        print(f'✓ No missing values in target variable.')

    # Load input features
    try:
        X_features = pd.read_csv(args.features)
        print(f'✓ Loaded features: {X_features.shape}')
    except FileNotFoundError:
        print(f'ERROR: Feature file not found: {args.features}')
        return

    # Check if features need to be grouped by SMILES
    # If features have more rows than grouped targets, we need to group features too
    # This handles cases where the dataset has multiple measurements per polymer
    if X_features.shape[0] != DatasetA_grouped.shape[0]:
        print(f'⚠ Features ({X_features.shape[0]} rows) don\'t match grouped dataset ({DatasetA_grouped.shape[0]} rows)')
        print(f'  Grouping features by SMILES to match...')

        # Add SMILES column to features for grouping
        if 'Smiles' in DatasetA_Smiles_P.columns:
            X_features['Smiles'] = DatasetA_Smiles_P['Smiles'].values
            # Group features by SMILES (take mean of features for each unique polymer)
            # This averages feature values for all measurements of the same polymer
            X_features = X_features.groupby('Smiles').mean().reset_index()
            # Ensure order matches DatasetA_grouped and filter to only valid polymers
            # This handles cases where some SMILES may be invalid or filtered out
            X_features = X_features.set_index('Smiles').loc[DatasetA_grouped['Smiles']].reset_index()
            # Drop the Smiles column to get back to numeric features only
            X_features = X_features.drop('Smiles', axis=1)
            print(f'✓ Grouped features by SMILES: {X_features.shape}')
        else:
            print('ERROR: Cannot group features - SMILES column not found in dataset.')
            return

    # Normalize Y (standardize to mean=0, std=1)
    Y = np.array(Y)
    scaler = StandardScaler()
    Y = scaler.fit_transform(Y)
    print(f'✓ Standardized target variable (shape: {Y.shape})')

    # Determine feature type from filename
    if 'desc' in args.features:
        feature_type = 'desc'
        # Normalize X for descriptors
        X = np.array(X_features)
        Xscaler = StandardScaler()
        X = Xscaler.fit_transform(X)
        print(f'✓ Using chemical descriptors (standardized)')
    elif 'fing' in args.features:
        feature_type = 'fing'
        # Fingerprints are already binary, no normalization needed
        X = np.array(X_features)
        print(f'✓ Using Morgan fingerprints (no standardization)')
    else:
        print('WARNING: Could not determine feature type from filename. Assuming fingerprints.')
        feature_type = 'fing'
        X = np.array(X_features)

    # Ensure X and Y have matching number of samples
    if X.shape[0] != Y.shape[0]:
        print(f'ERROR: Mismatch between features ({X.shape[0]}) and targets ({Y.shape[0]})')
        print('Make sure the feature file corresponds to the same dataset.')
        return

    # Split into train and test sets (80/20 split)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(f'✓ Train/test split: {X_train.shape[0]} train, {X_test.shape[0]} test')

    # Train DNN ensemble
    print('\n' + '='*60)
    print('Training DNN Ensemble (16 models with bootstrap sampling)')
    print('='*60)

    newX = X_test
    newy = Y_test

    n_splits = 16  # Number of ensemble members
    scores, members = list(), list()

    for i in range(n_splits):
        print(f'\nTraining model {i+1}/{n_splits}...')

        # Bootstrap sampling: randomly sample 80% of training data with replacement
        ix = [j for j in range(len(X))]
        train_ix = resample(ix, replace=True, n_samples=round(X.shape[0]*0.8))
        test_ix = [x for x in ix if x not in train_ix]

        # Select data
        trainX, trainy = X[train_ix], Y[train_ix]
        testX, testy = X[test_ix], Y[test_ix]

        # Evaluate model
        model, test_r2 = evaluate_model(trainX, trainy, testX, testy)
        print(f'  Model {i+1} test R²: {test_r2:.4f}')

        scores.append(test_r2)
        members.append(model)

    # Summarize ensemble performance
    print('\n' + '='*60)
    print('Ensemble Training Complete')
    print('='*60)
    print(f'Mean R² across 16 models: {np.mean(scores):.4f}')
    print(f'Std of R² across 16 models: {np.std(scores):.4f}')

    # Evaluate full ensemble on held-out test set
    print('\nEvaluating full ensemble on test set...')
    ensemble_score, ensemble_variance = evaluate_n_members(members, n_splits, newX, newy)
    print(f'Ensemble R² on test set: {ensemble_score:.4f}')
    print(f'Ensemble prediction variance: {ensemble_variance:.6f}')

    # Generate predictions
    Y_pred_train, var_train = ensemble_predictions(members, X_train)
    Y_pred_train = scaler.inverse_transform(Y_pred_train)

    Y_pred_test, var_test = ensemble_predictions(members, newX)
    Y_pred_test = scaler.inverse_transform(Y_pred_test)

    Y_train = scaler.inverse_transform(Y_train)
    Y_test = scaler.inverse_transform(newy)

    # Save models and results
    print('\n' + '='*60)
    print('Saving Models and Results')
    print('='*60)

    model_name = f'DNN_{imputation_suffix}_{feature_type}_water'
    maindirectory = os.getcwd() + '/models/' + model_name

    if not os.path.exists(maindirectory):
        os.makedirs(maindirectory)

    # Save each ensemble member
    for count, model in enumerate(members):
        directory = maindirectory + '/DNN_' + str(count)
        model.save(directory)
        print(f'✓ Saved model {count+1}/{len(members)}')

    # Save scaler for future predictions
    scaler_file = maindirectory + '/Yscaler.pkl'
    pickle.dump(scaler, open(scaler_file, 'wb'))
    print(f'✓ Saved Y scaler')

    if feature_type == 'desc':
        Xscaler_file = maindirectory + '/Xscaler.pkl'
        pickle.dump(Xscaler, open(Xscaler_file, 'wb'))
        print(f'✓ Saved X scaler')

    # Save predictions and targets
    os.chdir(maindirectory)
    np.savetxt('Y_train.csv', Y_train, delimiter=",")
    np.savetxt('Y_test.csv', Y_test, delimiter=",")
    np.savetxt('Y_pred_train.csv', Y_pred_train, delimiter=",")
    np.savetxt('Y_pred_test.csv', Y_pred_test, delimiter=",")
    print(f'✓ Saved predictions and targets')

    # Calculate and save performance metrics
    train_r2 = r2_score(Y_train[~np.isnan(Y_train)], Y_pred_train[~np.isnan(Y_train)])
    test_r2 = r2_score(Y_test[~np.isnan(Y_test)], Y_pred_test[~np.isnan(Y_test)])
    train_rmse = np.sqrt(mean_squared_error(Y_train[~np.isnan(Y_train)], Y_pred_train[~np.isnan(Y_train)]))
    test_rmse = np.sqrt(mean_squared_error(Y_test[~np.isnan(Y_test)], Y_pred_test[~np.isnan(Y_test)]))

    metrics = {
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Ensemble size': n_splits
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('metrics.csv', index=False)
    print(f'✓ Saved performance metrics')

    print('\n' + '='*60)
    print('Training Complete!')
    print('='*60)
    print(f'Model directory: {maindirectory}')
    print(f'Train R²: {train_r2:.4f}')
    print(f'Test R²: {test_r2:.4f}')
    print(f'Train RMSE: {train_rmse:.4f}')
    print(f'Test RMSE: {test_rmse:.4f}')
    print('='*60)


# Command-line argument parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train DNN ensemble for water permeability prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Train with fingerprints and no imputation (complete dataset)
  python step3_train_water.py --dataset datasets/water_permeability.csv \\
                               --features datasets/water_permeability_X_fing.csv \\
                               --imputation none

  # Train with fingerprints and BLR imputation
  python step3_train_water.py --dataset datasets/water_permeability.csv \\
                               --features datasets/water_permeability_X_fing.csv \\
                               --imputation BLR

  # Train with descriptors and ERT imputation
  python step3_train_water.py --dataset datasets/water_permeability.csv \\
                               --features datasets/water_permeability_X_desc.csv \\
                               --imputation ERT \\
                               --target-column log10_Water_Flux
        '''
    )

    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to water permeability dataset CSV file (must contain "Smiles" column)')

    parser.add_argument('--features', type=str, required=True,
                        help='Path to feature CSV file (fingerprints or descriptors)')

    parser.add_argument('--imputation', type=str, default='none',
                        choices=['none', 'BLR', 'ERT'],
                        help='Imputation method: "none" (default, no imputation), "BLR" (Bayesian Linear Regression), or "ERT" (Extremely Randomized Trees)')

    parser.add_argument('--target-column', type=str, default='log10_Water',
                        help='Name of target column in dataset (default: log10_Water)')

    parsed_args = parser.parse_args()

    train(parsed_args)
