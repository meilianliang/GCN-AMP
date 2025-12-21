from model import five_fold_cross_validation
import pandas as pd
import argparse

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description='read CSV')
    parser.add_argument('--input', type=str, default="../../train.csv",
                    help='Please input the path of CSV (default：../../train.csv)')
    args = parser.parse_args()    
    full_train_df = pd.read_csv(args.input) 
    
    cv_results = five_fold_cross_validation(
        full_train_df, 
        num_epochs=30, 
        lr=0.00001, 
        k_folds=5
    )   
    print("\nComplete! The results have been saved to: outputs/results/cv_results/ ")
    print(f"Average AUC: {cv_results['average_metrics']['auc']:.4f} ± {cv_results['std_metrics']['auc']:.4f}")
    print(f"Average MCC: {cv_results['average_metrics']['mcc']:.4f} ± {cv_results['std_metrics']['mcc']:.4f}")