import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
tf.keras.backend.set_floatx('float64')

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import pickle
import argparse
import os
from DNN_functions_water import nanmean_squared_error, ensemble_predictions

'''
Script to predict water permeability of polymers with known SMILES strings but unknown permeability.
Requires trained DNN models as saved by step3_train_water.py.
Relevant chemical features of the screening dataset must be provided.
Outputs .csv file with predicted water permeability values.

Key differences from step4_screen.py:
- Single output (water permeability) instead of 6 gases
- Simplified prediction process for single target
- Support for uncertainty quantification via ensemble variance
'''


def test(args):
    """
    Predict water permeability for new polymers using trained DNN ensemble.

    Args:
        args: Command-line arguments containing:
            - model_dir: Directory containing trained model
            - features: Path to feature CSV file for prediction
            - output: Output filename for predictions
    """

    print('='*60)
    print('Water Permeability Prediction')
    print('='*60)
    print(f'Model directory: {args.model_dir}')
    print(f'Feature file: {args.features}')
    print(f'Output file: {args.output}')
    print('='*60)

    # Check if model directory exists
    if not os.path.exists(args.model_dir):
        print(f'ERROR: Model directory not found: {args.model_dir}')
        print('Please train a model first using step3_train_water.py')
        return

    # Parse model directory name to get feature type
    model_dir_name = os.path.basename(args.model_dir)
    if 'desc' in model_dir_name:
        feature_type = 'desc'
    elif 'fing' in model_dir_name:
        feature_type = 'fing'
    else:
        print('WARNING: Could not determine feature type from model directory name.')
        print('Assuming fingerprints (no X scaling).')
        feature_type = 'fing'

    # Load prediction features
    try:
        X_pred = pd.read_csv(args.features)
        print(f'✓ Loaded features: {X_pred.shape}')
    except FileNotFoundError:
        print(f'ERROR: Feature file not found: {args.features}')
        return

    # Convert to numpy array
    X_pred = np.array(X_pred)

    # Load X scaler if using descriptors
    if feature_type == 'desc':
        Xscaler_file = args.model_dir + '/Xscaler.pkl'
        if os.path.exists(Xscaler_file):
            Xscaler = pickle.load(open(Xscaler_file, 'rb'))
            X_pred = Xscaler.transform(X_pred)
            print(f'✓ Applied X standardization (descriptors)')
        else:
            print('WARNING: X scaler not found. Predictions may be inaccurate.')

    # Load Y scaler
    Yscaler_file = args.model_dir + '/Yscaler.pkl'
    if not os.path.exists(Yscaler_file):
        print('ERROR: Y scaler not found in model directory.')
        print('The model directory may be incomplete or corrupted.')
        return

    Yscaler = pickle.load(open(Yscaler_file, 'rb'))
    print(f'✓ Loaded Y scaler')

    # Load all ensemble models
    print('\nLoading DNN ensemble models...')
    folders = os.listdir(args.model_dir)
    indices = []
    for name in folders:
        full_path = os.path.join(args.model_dir, name)
        if os.path.isdir(full_path) and name.startswith('DNN_'):
            try:
                idx = int(name.split('_')[1])
                indices.append(idx)
            except (IndexError, ValueError):
                continue

    if len(indices) == 0:
        print('ERROR: No DNN models found in model directory.')
        print('The model directory may be incomplete or corrupted.')
        return

    max_index = max(indices)
    print(f'Found {len(indices)} ensemble members (indices: {min(indices)}-{max_index})')

    # Load each model
    models = list()
    for i in range(max_index + 1):
        model_path = os.path.join(args.model_dir, f'DNN_{i}')
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={'nanmean_squared_error': nanmean_squared_error}
            )
            models.append(model)
            print(f'✓ Loaded model {i+1}/{max_index+1}')
        else:
            print(f'WARNING: Model DNN_{i} not found, skipping...')

    if len(models) == 0:
        print('ERROR: No models could be loaded.')
        return

    print(f'✓ Successfully loaded {len(models)} models')

    # Make ensemble predictions
    print('\n' + '='*60)
    print('Making Predictions')
    print('='*60)

    Y_pred, Y_var = ensemble_predictions(models, X_pred)

    # Inverse transform to original scale
    Y_pred = Yscaler.inverse_transform(Y_pred)
    print(f'✓ Generated predictions for {Y_pred.shape[0]} samples')

    # Calculate prediction uncertainty (standard deviation from variance)
    Y_std = np.sqrt(Y_var)
    # Note: Y_std is in standardized scale, need to scale it back
    # For standard deviation, multiply by the scale (std) of the original Y
    Y_std_original_scale = Y_std * Yscaler.scale_[0]

    # Save predictions
    output_path = os.path.join(args.model_dir, args.output)

    # Create output dataframe with predictions and uncertainty
    output_df = pd.DataFrame({
        'Predicted_Water_Permeability': Y_pred.flatten(),
        'Prediction_Uncertainty': Y_std_original_scale.flatten()
    })

    output_df.to_csv(output_path, index=False)
    print(f'✓ Saved predictions to: {output_path}')

    # Print summary statistics
    print('\n' + '='*60)
    print('Prediction Summary')
    print('='*60)
    print(f'Number of predictions: {len(Y_pred)}')
    print(f'Mean predicted value: {np.mean(Y_pred):.4f}')
    print(f'Std of predictions: {np.std(Y_pred):.4f}')
    print(f'Min predicted value: {np.min(Y_pred):.4f}')
    print(f'Max predicted value: {np.max(Y_pred):.4f}')
    print(f'Mean prediction uncertainty: {np.mean(Y_std_original_scale):.4f}')
    print('='*60)

    # Optionally, identify top candidates (polymers with highest water permeability)
    if args.top_n > 0:
        print(f'\n' + '='*60)
        print(f'Top {args.top_n} Candidates (Highest Predicted Water Permeability)')
        print('='*60)

        # Sort by predicted value (descending)
        sorted_indices = np.argsort(Y_pred.flatten())[::-1]
        top_indices = sorted_indices[:args.top_n]

        top_df = pd.DataFrame({
            'Rank': range(1, len(top_indices) + 1),
            'Sample_Index': top_indices,
            'Predicted_Water_Permeability': Y_pred.flatten()[top_indices],
            'Prediction_Uncertainty': Y_std_original_scale.flatten()[top_indices]
        })

        print(top_df.to_string(index=False))

        # Save top candidates
        top_output_path = os.path.join(args.model_dir, 'top_candidates_' + args.output)
        top_df.to_csv(top_output_path, index=False)
        print(f'\n✓ Saved top candidates to: {top_output_path}')

    print('\n' + '='*60)
    print('Prediction Complete!')
    print('='*60)


# Command-line argument parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict water permeability using trained DNN ensemble',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Predict water permeability for new polymers
  python step4_test_water.py --model-dir models/DNN_none_fing_water \\
                              --features datasets/new_polymers_X_fing.csv \\
                              --output predictions.csv

  # Predict and show top 10 candidates
  python step4_test_water.py --model-dir models/DNN_BLR_fing_water \\
                              --features datasets/screening_X_fing.csv \\
                              --output screening_predictions.csv \\
                              --top-n 10
        '''
    )

    parser.add_argument('--model-dir', type=str, required=True,
                        help='Path to trained model directory (e.g., models/DNN_none_fing_water)')

    parser.add_argument('--features', type=str, required=True,
                        help='Path to feature CSV file for prediction (must match model feature type)')

    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output filename for predictions (default: predictions.csv)')

    parser.add_argument('--top-n', type=int, default=0,
                        help='Number of top candidates to display and save (default: 0, no top candidates)')

    parsed_args = parser.parse_args()

    test(parsed_args)
