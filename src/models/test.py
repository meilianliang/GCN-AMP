import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
import model
from model import ProDataset, test_model, collate_fn, plot_roc_single
from torch.utils.data import Dataset, DataLoader

def main():
    parser = argparse.ArgumentParser(description='Test')

    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model weights file (.pth or .pt file)')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data file (CSV format)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file does not exist: {args.model_path}")
        sys.exit(1)
        
    if not os.path.exists(args.test_data):
        print(f"Error: Test data file does not exist: {args.test_data}")
        sys.exit(1)
    
    print("="*60)
    print("Model Testing Configuration")
    print("="*60)
    print(f"Model file: {args.model_path}")
    print(f"Test data: {args.test_data}")
    print("="*60)
    
    try:
        print(f"\nLoading test data...")
        test_df = pd.read_csv(args.test_data)
        dataset = ProDataset(test_df, max_len=64)        
        test_loader = DataLoader(
            dataset, 
            batch_size=4, 
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=True
        )
        print(f"Number of test samples: {len(test_loader.dataset)}")
        
        print(f"\nTesting model...")
        test_metrics, results_df = test_model(
            model_path=args.model_path,
            test_loader=test_loader
        )        
        print("\n✓ Testing completed!")
        plot_roc_single('../../outputs/results/test_results.csv', 
                model_name='GCN-AMP',
                save_path='../../outputs/results/roc_curve_single.png')
        
    except Exception as e:
        print(f"\n✗ Error occurred during testing:  {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()