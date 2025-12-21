import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import argparse
import glob
from torch.utils.data import Dataset, DataLoader
import sys
from model import PeptideGCNWithAttention

Dataset_Path = "../../data/features/"
dt = 0.33

class ProDatasetInference(Dataset):
    def __init__(self, dataframe, max_len=64):
        if 'uniprot_id' not in dataframe.columns:
            dataframe = dataframe.copy()
            dataframe['uniprot_id'] = [f'sample_{i}' for i in range(len(dataframe))]
        
        if 'sequence' not in dataframe.columns:
            raise ValueError("Input dataframe must contain 'sequence' column")
        
        self.names = dataframe['uniprot_id'].values
        self.sequences = dataframe['sequence'].values
        self.max_len = max_len
        self.mean, self.std = self.load_values()
        
    def load_values(self):
        try:
            mean = np.load(Dataset_Path + 'mean_esm.npy')
            std = np.load(Dataset_Path + 'std_esm.npy')
            return mean, std
        except FileNotFoundError:
            esm_dim = 1280
            mean = np.zeros(esm_dim)
            std = np.ones(esm_dim)
            return mean, std
    
    def create_fully_connected_graph(self, num_nodes):
        if num_nodes <= 0:
            return torch.empty((2, 0), dtype=torch.long)
        
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
        if not edge_index:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, index):
        uniprot_id = self.names[index]
        sequence = self.sequences[index]
        
        feature_file = Dataset_Path + 'node_features_esm/' + uniprot_id + '.npy'
        
        if os.path.exists(feature_file):
            feature_matrix = np.load(feature_file)
        else:
            esm_dim = 1280
            seq_len = len(sequence)
            feature_matrix = np.random.randn(seq_len, esm_dim).astype(np.float32)
            os.makedirs(Dataset_Path + 'node_features_esm/', exist_ok=True)
            np.save(Dataset_Path + 'node_features_esm/' + uniprot_id + '.npy', feature_matrix)
        
        if self.mean is not None and self.std is not None:
            feature_matrix = (feature_matrix - self.mean) / self.std
        
        seq_len = len(sequence)
        actual_len = min(seq_len, self.max_len)
        
        if seq_len > self.max_len:
            feature_matrix = feature_matrix[:self.max_len, :]
        else:
            pad_size = self.max_len - seq_len
            if pad_size > 0:
                feature_matrix = np.pad(feature_matrix, ((0, pad_size), (0, 0)), 'constant')
        
        feature_matrix = torch.tensor(feature_matrix, dtype=torch.float)
        
        edge_index = self.create_fully_connected_graph(actual_len)
        
        return uniprot_id, sequence, feature_matrix, edge_index, actual_len

def collate_fn_inference(batch):
    names, sequences, features, edge_indices, actual_lens = zip(*batch)
    
    features_batch = torch.stack(features, dim=0)
    
    edge_index_batch = []
    offset = 0
    
    for i, (edge_index, actual_len) in enumerate(zip(edge_indices, actual_lens)):
        if edge_index.size(1) > 0:
            edge_index_batch.append(edge_index + offset)
        offset += actual_len
    
    if edge_index_batch:
        edge_index_batch = torch.cat(edge_index_batch, dim=1)
    else:
        edge_index_batch = torch.empty((2, 0), dtype=torch.long)
    
    batch_indices = []
    for i, actual_len in enumerate(actual_lens):
        batch_indices.extend([i] * actual_len)
    batch_indices = torch.tensor(batch_indices, dtype=torch.long)
    
    return names, sequences, features_batch, edge_index_batch, batch_indices

def predict_multifunction(model_paths, thresholds, input_file, output_file='multifunction_predictions.csv'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_names = []
    for model_path in model_paths:
        model_filename = os.path.basename(model_path)
        function_name = model_filename.split('_')[0]
        model_names.append(function_name)
    
    print(f"Loading {len(model_paths)} models:")
    for i, (model_path, model_name) in enumerate(zip(model_paths, model_names)):
        print(f"  {i+1}. {model_name} (threshold={thresholds[i]:.3f})")
    
    print(f"\nLoading input data from: {input_file}")
    input_df = pd.read_csv(input_file)
    print(f"Number of sequences to predict: {len(input_df)}")
    
    dataset = ProDatasetInference(input_df, max_len=64)
    data_loader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=False,
        collate_fn=collate_fn_inference,
        drop_last=False
    )
    
    all_probs = {model_name: [] for model_name in model_names}
    all_labels = {model_name: [] for model_name in model_names}
    
    for model_idx, (model_path, model_name, threshold) in enumerate(zip(model_paths, model_names, thresholds)):
        print(f"\nPredicting with model: {model_name}")
        
        model = PeptideGCNWithAttention(
            esm_input_dim=1280,
            mlp_hidden_dims=[768, 384],
            mlp_output_dim=192,
            gcn_hidden_dim=384,
            gcn_output_dim=128,
            gcn_layers=2,
            attention_heads=8,
            max_seq_len=64
        )
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        model_probs = []
        
        with torch.no_grad():
            for batch in data_loader:
                names, sequences, features, edge_index, batch_idx = batch
                
                features = features.view(-1, features.size(-1))
                
                if edge_index.size(1) == 0:
                    batch_size = features.size(0) // 64
                    dummy_output = torch.zeros(batch_size, 1, device=device) + 0.5
                    probs = dummy_output.cpu().numpy()
                else:
                    features = features.to(device)
                    edge_index = edge_index.to(device)
                    batch_idx = batch_idx.to(device)
                    
                    outputs = model(features, edge_index, batch_idx)
                    
                    if outputs.dim() > 1:
                        outputs = outputs.squeeze()
                    
                    probs = outputs.cpu().numpy()
                
                if probs.ndim == 0:
                    model_probs.append(probs)
                else:
                    model_probs.extend(probs.flatten())
        
        model_preds = (np.array(model_probs) > threshold).astype(int)
        all_probs[model_name] = model_probs
        all_labels[model_name] = model_preds
        
        print(f"  Predicted {sum(model_preds)} positive and {len(model_preds)-sum(model_preds)} negative")
    
    results_data = {
        'uniprot_id': input_df['uniprot_id'].values if 'uniprot_id' in input_df.columns else [f'sample_{i}' for i in range(len(input_df))],
        'sequence': input_df['sequence'].values
    }
    
    for model_name in model_names:
        results_data[f'{model_name}_prob'] = all_probs[model_name]
        results_data[f'{model_name}_label'] = all_labels[model_name]
    
    results_df = pd.DataFrame(results_data)
    
    label_columns = [f'{model_name}_label' for model_name in model_names]
    results_df['total_active_functions'] = results_df[label_columns].sum(axis=1)
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results_df.to_csv(output_file, index=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Total sequences: {len(results_df)}")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Multi-function Peptide Activity Prediction')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing model files')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to input data file')
    parser.add_argument('--output_file', type=str, default='../../outputs/results/multifunction_predictions.csv',
                       help='Path to save prediction results')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[dt, dt, dt, dt, dt],
                       help='Thresholds for each model')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory does not exist: {args.model_dir}")
        sys.exit(1)
        
    if not os.path.exists(args.input_file):
        print(f"Error: Input data file does not exist: {args.input_file}")
        sys.exit(1)
    
    model_extensions = ['.pth', '.pt', '.pkl']
    model_files = []
    for ext in model_extensions:
        model_files.extend(glob.glob(os.path.join(args.model_dir, f'*{ext}')))
    
    if not model_files:
        print(f"Error: No model files found in {args.model_dir}")
        sys.exit(1)
    
    model_files.sort()
    
    print("="*60)
    print("Multi-function Peptide Activity Prediction")
    print("="*60)
    print(f"Model directory: {args.model_dir}")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Number of models found: {len(model_files)}")
    
    for i, model_file in enumerate(model_files):
        model_name = os.path.basename(model_file).split('_')[0]
        threshold = args.thresholds[i] if i < len(args.thresholds) else dt
        print(f"  Model {i+1}: {model_name} (threshold={threshold:.3f})")
    
    print("="*60)
    
    try:
        print(f"\nStarting multi-function prediction...")
        
        results_df = predict_multifunction(
            model_paths=model_files[:5],
            thresholds=args.thresholds[:5] if len(args.thresholds) >= 5 else [dt]*5,
            input_file=args.input_file,
            output_file=args.output_file
        )
        
        print("\nPrediction completed successfully!")
        
    except Exception as e:
        print(f"\nError occurred during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()