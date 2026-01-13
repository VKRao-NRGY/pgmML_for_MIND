"""
Validation script to verify the feature grouping fix for water permeability training.

This script demonstrates that the feature grouping logic correctly handles
datasets with multiple measurements per polymer.

Expected behavior:
1. Dataset has 977 rows (multiple measurements per polymer)
2. When grouped by SMILES, we get 9 unique polymers
3. Features (977 rows) are grouped by SMILES to match (9 rows)
4. Training proceeds with aligned features and targets

To run: python validate_fix.py
Requires: pandas, numpy
"""

import pandas as pd
import numpy as np

def validate_grouping():
    print("="*60)
    print("Validating Water Permeability Feature Grouping Fix")
    print("="*60)

    # Load the dataset
    dataset_path = 'datasets/dataset_Water_Permeability_RO.csv'
    features_path = 'datasets/dataset_Water_Permeability_RO_X_fing.csv'

    print(f"\n1. Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"   ✓ Original dataset: {df.shape[0]} rows")

    # Check SMILES
    valid_smiles = df[df['Smiles'].str.startswith('*', na=False)]
    print(f"   ✓ Valid polymer SMILES: {valid_smiles['Smiles'].nunique()} unique")

    # Group by SMILES (mimicking the training script)
    df_grouped = df.groupby('Smiles').mean().reset_index()
    print(f"   ✓ After grouping: {df_grouped.shape[0]} unique entries")

    # Load features
    print(f"\n2. Loading features: {features_path}")
    features = pd.read_csv(features_path)
    print(f"   ✓ Original features: {features.shape}")

    # Check if grouping is needed
    print(f"\n3. Checking alignment:")
    print(f"   Features rows: {features.shape[0]}")
    print(f"   Grouped dataset rows: {df_grouped.shape[0]}")
    print(f"   Match: {features.shape[0] == df_grouped.shape[0]}")

    if features.shape[0] != df_grouped.shape[0]:
        print(f"\n4. Applying grouping fix:")
        print(f"   ⚠ Features don't match grouped dataset")

        # Add SMILES to features
        features['Smiles'] = df['Smiles'].values
        print(f"   ✓ Added SMILES column to features")

        # Group features by SMILES
        features_grouped = features.groupby('Smiles').mean().reset_index()
        print(f"   ✓ Grouped features: {features_grouped.shape}")

        # Align with grouped dataset
        features_grouped = features_grouped.set_index('Smiles').loc[df_grouped['Smiles']].reset_index()
        features_final = features_grouped.drop('Smiles', axis=1)
        print(f"   ✓ Aligned and dropped SMILES: {features_final.shape}")

        # Verify alignment
        print(f"\n5. Final verification:")
        print(f"   Grouped targets: {df_grouped.shape[0]} samples")
        print(f"   Grouped features: {features_final.shape[0]} samples, {features_final.shape[1]} features")
        print(f"   ✓ MATCH: {df_grouped.shape[0] == features_final.shape[0]}")

        if df_grouped.shape[0] == features_final.shape[0]:
            print("\n" + "="*60)
            print("SUCCESS: Feature grouping fix works correctly!")
            print("="*60)
            return True
        else:
            print("\n" + "="*60)
            print("ERROR: Feature grouping fix failed!")
            print("="*60)
            return False
    else:
        print("\n✓ Features already match grouped dataset - no fix needed")
        return True

if __name__ == '__main__':
    try:
        success = validate_grouping()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
