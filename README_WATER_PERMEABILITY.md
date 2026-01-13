# Water Permeability Prediction with DNN

This directory contains specialized scripts for predicting **water permeability** of polymeric membranes using Deep Neural Networks (DNN) with Morgan fingerprint features.

## Overview

These scripts are adapted from the original `step3_train.py` and `step4_screen.py` to focus specifically on water permeability prediction rather than multiple gas permeabilities. Key differences:

- **Single Output**: Predicts only water permeability (not 6 separate gases)
- **DNN with Fingerprints**: Uses DNN framework with Morgan fingerprints ('fing') as default
- **Optional Imputation**: Supports Bayesian Linear Regression (BLR) or Extremely Randomized Trees (ERT) imputation, or no imputation for complete datasets
- **Uncertainty Quantification**: Provides prediction uncertainty estimates via ensemble variance

## Files

### Core Scripts

| File | Description |
|------|-------------|
| `DNN_functions_water.py` | DNN utility functions adapted for single output (water) |
| `step3_train_water.py` | Training script for water permeability DNN ensemble |
| `step4_test_water.py` | Prediction/screening script using trained models |

### Architecture

The DNN architecture for water permeability:

```
Input Layer (variable size - depends on features)
    ↓
Dense(64 units, ReLU)
    ↓
Dense(64 units, ReLU)
    ↓
Dense(32 units, ReLU)
    ↓
Dense(16 units, ReLU)
    ↓
Dense(8 units, ReLU)
    ↓
Dropout(0.1)
    ↓
Dense(1 unit, Linear) → Water Permeability
```

**Ensemble Strategy**: 16 bootstrap DNN models trained on 80% of data each, with predictions averaged across all models.

## Usage

### Step 1: Prepare Your Dataset

Your water permeability dataset CSV should contain:

**Required columns:**
- `Smiles`: Polymer SMILES strings
- `log10_Water` (or custom target column): Log10-transformed water permeability values

**Optional columns (for imputation):**
- `log10_Water_Bayesian`: Bayesian Linear Regression imputed values
- `log10_Water_Etree`: Extremely Randomized Trees imputed values

**Example dataset structure:**

```csv
Smiles,log10_Water,log10_Water_Bayesian,log10_Water_Etree
*C*,-3.52,-3.52,-3.52
*CC(*)C(=O)O,-2.89,-2.89,-2.89
*CC(*)c1ccccc1,-4.12,,-4.10
```

### Step 2: Generate Features

Use `step2_generate_Xfeatures.py` to compute Morgan fingerprints or chemical descriptors:

```bash
# Generate Morgan fingerprints (recommended)
python step2_generate_Xfeatures.py \
    --input datasets/water_permeability.csv \
    --output datasets/water_permeability_X_fing.csv \
    --feature-type fing

# Generate chemical descriptors (alternative)
python step2_generate_Xfeatures.py \
    --input datasets/water_permeability.csv \
    --output datasets/water_permeability_X_desc.csv \
    --feature-type desc
```

### Step 3: Train the Model

#### Option A: Complete Dataset (No Imputation)

If your dataset has no missing values:

```bash
python step3_train_water.py \
    --dataset datasets/water_permeability.csv \
    --features datasets/water_permeability_X_fing.csv \
    --imputation none
```

#### Option B: With Bayesian Linear Regression Imputation

If you have missing values and Bayesian imputed columns:

```bash
python step3_train_water.py \
    --dataset datasets/water_permeability.csv \
    --features datasets/water_permeability_X_fing.csv \
    --imputation BLR
```

#### Option C: With Extremely Randomized Trees Imputation

```bash
python step3_train_water.py \
    --dataset datasets/water_permeability.csv \
    --features datasets/water_permeability_X_fing.csv \
    --imputation ERT
```

#### Option D: Custom Target Column

If your target column has a different name:

```bash
python step3_train_water.py \
    --dataset datasets/water_permeability.csv \
    --features datasets/water_permeability_X_fing.csv \
    --imputation none \
    --target-column log10_Water_Flux
```

### Step 4: Make Predictions

Use the trained model to predict water permeability for new polymers:

```bash
python step4_test_water.py \
    --model-dir models/DNN_none_fing_water \
    --features datasets/new_polymers_X_fing.csv \
    --output predictions.csv
```

**To identify top candidates:**

```bash
python step4_test_water.py \
    --model-dir models/DNN_none_fing_water \
    --features datasets/screening_X_fing.csv \
    --output screening_predictions.csv \
    --top-n 10
```

This will display and save the top 10 polymers with highest predicted water permeability.

## Output Files

### Training Output

After training, the model directory will contain:

```
models/DNN_none_fing_water/
├── DNN_0/              # Ensemble member 1
├── DNN_1/              # Ensemble member 2
├── ...
├── DNN_15/             # Ensemble member 16
├── Yscaler.pkl         # Target scaler (for inverse transform)
├── Xscaler.pkl         # Feature scaler (only for descriptors)
├── Y_train.csv         # Training targets
├── Y_test.csv          # Test targets
├── Y_pred_train.csv    # Training predictions
├── Y_pred_test.csv     # Test predictions
└── metrics.csv         # Performance metrics (R², RMSE)
```

### Prediction Output

Prediction CSV contains:

| Column | Description |
|--------|-------------|
| `Predicted_Water_Permeability` | Predicted log10 water permeability |
| `Prediction_Uncertainty` | Uncertainty estimate (ensemble std deviation) |

**Example:**

```csv
Predicted_Water_Permeability,Prediction_Uncertainty
-3.52,0.12
-2.89,0.08
-4.12,0.15
```

**Top candidates file** (when using `--top-n`):

```csv
Rank,Sample_Index,Predicted_Water_Permeability,Prediction_Uncertainty
1,234,-2.15,0.09
2,567,-2.23,0.11
3,891,-2.31,0.08
```

## Command-Line Arguments

### step3_train_water.py

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--dataset` | Yes | - | Path to water permeability CSV |
| `--features` | Yes | - | Path to feature CSV (fingerprints or descriptors) |
| `--imputation` | No | `none` | Imputation method: `none`, `BLR`, or `ERT` |
| `--target-column` | No | `log10_Water` | Name of target column in dataset |

### step4_test_water.py

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model-dir` | Yes | - | Path to trained model directory |
| `--features` | Yes | - | Path to feature CSV for prediction |
| `--output` | No | `predictions.csv` | Output filename |
| `--top-n` | No | `0` | Number of top candidates to display (0 = none) |

## Key Features

### 1. Single Output Focus

Unlike the original scripts that predict 6 gas permeabilities, these scripts focus on a single target (water permeability). This simplifies:
- Model architecture (1 output unit vs 6)
- Data handling
- Interpretation of results

### 2. Flexible Imputation

Three imputation options:

- **`none`**: Use for complete datasets (no missing values)
- **`BLR`**: Bayesian Linear Regression (requires `*_Bayesian` columns)
- **`ERT`**: Extremely Randomized Trees (requires `*_Etree` columns)

If imputed columns are not found, the script automatically falls back to the raw column.

### 3. Uncertainty Quantification

The ensemble approach provides:
- **Point predictions**: Mean prediction across 16 models
- **Uncertainty estimates**: Standard deviation across ensemble (higher = less certain)

Use uncertainty to:
- Filter out unreliable predictions
- Prioritize experimental validation
- Understand model confidence

### 4. Detailed Output

Both scripts provide:
- Progress indicators during training/prediction
- Performance metrics (R², RMSE)
- Summary statistics
- Clear error messages with troubleshooting guidance

## Performance Expectations

Typical performance on water permeability datasets:

| Metric | Expected Range |
|--------|----------------|
| Test R² | 0.75 - 0.95 |
| Test RMSE | 0.15 - 0.40 (log10 units) |
| Training time | 2-5 minutes (16 models on ~400 samples) |
| Prediction time | < 10 seconds per 1000 samples |

Performance depends on:
- Dataset size and quality
- Feature representation (fingerprints vs descriptors)
- Chemical diversity of polymers
- Measurement consistency

## Troubleshooting

### Issue: "Target column not found"

**Solution**: Ensure your dataset has the correct target column name. Use `--target-column` to specify a custom name:

```bash
python step3_train_water.py \
    --dataset datasets/water_permeability.csv \
    --features datasets/water_permeability_X_fing.csv \
    --imputation none \
    --target-column log10_Water_Permeability
```

### Issue: "Mismatch between features and targets"

**Solution**: Ensure the feature file corresponds to the same dataset. The number of rows in both files must match after grouping by SMILES.

### Issue: "Model directory not found"

**Solution**: Train a model first using `step3_train_water.py`. The model directory is created in `models/DNN_{imputation}_{feature_type}_water/`.

### Issue: Poor model performance (low R²)

**Possible causes and solutions:**
1. **Insufficient data**: Water permeability datasets typically need 200+ samples for good performance
2. **High measurement noise**: Check experimental consistency
3. **Extrapolation**: Model struggles with polymers very different from training set
4. **Wrong feature type**: Try descriptors if fingerprints don't work well

### Issue: High prediction uncertainty

**Interpretation**: High uncertainty indicates:
- Polymer is unlike training data (extrapolation)
- Ambiguous structure-property relationship
- Need more similar training examples

**Action**: Validate high-uncertainty predictions experimentally or gather more training data in that chemical space.

## Comparison with Original Scripts

| Feature | Original (step3/step4) | Water-Specific |
|---------|----------------------|----------------|
| **Outputs** | 6 gases | 1 fluid (water) |
| **Model architecture** | 6 output units | 1 output unit |
| **Imputation** | Required | Optional |
| **Feature types** | desc or fing | fing (recommended) |
| **Uncertainty** | Yes (ensemble) | Yes (ensemble) |
| **Top candidates** | Manual filtering | Built-in `--top-n` |

## Best Practices

1. **Use log10-transformed targets**: Water permeability spans many orders of magnitude; log transformation improves model performance

2. **Standardize targets**: The script automatically standardizes targets for training and inverse-transforms for predictions

3. **Bootstrap ensemble**: 16 models with 80% bootstrap sampling provides good balance between performance and computational cost

4. **Validate extrapolation**: Check if prediction uncertainty is high for new polymers; if so, the model may be extrapolating

5. **Feature selection**: Morgan fingerprints (`fing`) typically work well for membrane permeability; descriptors (`desc`) are more general but may need feature selection

## Citation

If you use these scripts, please cite the original pgmML work:

```
[Original paper citation here]
```

## Support

For issues or questions:
1. Check this README first
2. Review error messages (they provide troubleshooting guidance)
3. Check that dataset and feature files are correctly formatted
4. Ensure model was trained successfully before making predictions

## License

[Same as original pgmML project]
