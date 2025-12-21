import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import MultiheadAttention
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, matthews_corrcoef, roc_auc_score, 
                             confusion_matrix, roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')
import random


Dataset_Path = "../../data/features/"
threshold = 0.33

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'      
    random.seed(seed)    
    np.random.seed(seed)    
    torch.manual_seed(seed)    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False        
        torch.use_deterministic_algorithms(True, warn_only=True)


SEED = 25
set_seed(SEED)

def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


class ESMFeatureReducer(nn.Module):
    def __init__(self, input_dim=1280, hidden_dims=[512, 256], output_dim=128):
        super(ESMFeatureReducer, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.mlp(x)


class GCNLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super(GCNLayer, self).__init__()
        self.convs = nn.ModuleList()        
        self.convs.append(GCNConv(in_channels, hidden_channels))        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))        
        self.convs.append(GCNConv(hidden_channels, out_channels))        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1: 
                x = F.relu(x)
                x = self.dropout(x)
        return x


class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(SelfAttentionLayer, self).__init__()
        self.attention = MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_dim]
        x = x.transpose(0, 1)  # [batch_size, seq_len, embed_dim] -> [seq_len, batch_size, embed_dim]        
        attn_output, _ = self.attention(x, x, x)        
        x = x + self.dropout(attn_output)
        x = self.norm(x)        
        x = x.transpose(0, 1)  # [seq_len, batch_size, embed_dim] -> [batch_size, seq_len, embed_dim]        
        return x


class PeptideGCNWithAttention(nn.Module):
    def __init__(self, esm_input_dim=1280,  
                 mlp_hidden_dims=[512, 256], mlp_output_dim=128,
                 gcn_hidden_dim=64, gcn_output_dim=32, gcn_layers=2,
                 attention_heads=4, attention_dropout=0.1,
                 max_seq_len=64):
        super(PeptideGCNWithAttention, self).__init__()
        
        self.max_seq_len = max_seq_len
        
        self.esm_reducer = ESMFeatureReducer(
            esm_input_dim, mlp_hidden_dims, mlp_output_dim
        )
        
        self.combined_feature_dim = mlp_output_dim
        
        self.gcn = GCNLayer(
            self.combined_feature_dim, 
            gcn_hidden_dim, 
            gcn_output_dim, 
            gcn_layers
        )
        
        self.attention = SelfAttentionLayer(
            gcn_output_dim, 
            attention_heads, 
            attention_dropout
        )
        
        self.fc = nn.Sequential(
            nn.Linear(gcn_output_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()  
        )
        
    def forward(self, x, edge_index, batch=None):
        # x shape: [batch_size * seq_len, feature_dim]        
        reduced_esm = self.esm_reducer(x)        
        gcn_output = self.gcn(reduced_esm, edge_index)        
        if batch is not None:
            batch_size = torch.max(batch) + 1
        else:
            batch_size = 1            
        gcn_output_reshaped = gcn_output.view(batch_size, self.max_seq_len, -1)        
        attention_output = self.attention(gcn_output_reshaped)        
        pooled = torch.mean(attention_output, dim=1)        
        output = self.fc(pooled)        
        return output


class ProDataset(Dataset):
    def __init__(self, dataframe, max_len=64):
        self.names = np.array([str(n) for n in dataframe['uniprot_id'].values])
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values.astype(float) 
        self.max_len = max_len
        self.mean, self.std = self.load_values()
        self.edge_index = self.create_fully_connected_graph(max_len)
        
    def load_values(self):
        mean = np.load(Dataset_Path + 'mean_esm.npy')
        std = np.load(Dataset_Path + 'std_esm.npy')
        return mean, std
    
    def create_fully_connected_graph(self, num_nodes):
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  
                    edge_index.append([i, j])
        
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, index):
        uniprot_id = self.names[index]
        sequence = self.sequences[index]
        label = self.labels[index]
        
        feature_matrix = np.load(Dataset_Path + 'node_features_esm/' + uniprot_id + '.npy')
        feature_matrix = (feature_matrix - self.mean) / self.std
        
        seq_len = len(sequence)
        if seq_len > self.max_len:
            feature_matrix = feature_matrix[:self.max_len, :]
        else:
            pad_size = self.max_len - seq_len
            feature_matrix = np.pad(feature_matrix, ((0, pad_size), (0, 0)), 'constant')
        
        feature_matrix = torch.tensor(feature_matrix, dtype=torch.float)
        label = torch.tensor([label], dtype=torch.float)  
        # print(uniprot_id, feature_matrix.shape)        
        return uniprot_id, sequence, label, feature_matrix, self.edge_index


def collate_fn(batch):
    names, sequences, labels, features, edge_indices = zip(*batch)    
    features = torch.stack(features, dim=0)
    labels = torch.stack(labels, dim=0)    
    batch_size = len(batch)
    max_nodes = features.shape[1]      
    batch_edge_index = []
    for i, edge_index in enumerate(edge_indices):
        offset = i * max_nodes
        batch_edge_index.append(edge_index + offset)    
    batch_edge_index = torch.cat(batch_edge_index, dim=1)    
    batch_idx = torch.arange(batch_size).repeat_interleave(max_nodes)    
    features = features.view(-1, features.shape[-1])    
    return names, sequences, labels, features, batch_edge_index, batch_idx


def calculate_metrics(y_true, y_pred, y_prob):
    y_pred_binary = (y_pred > threshold).astype(int)
    y_true = y_true.astype(int)
    
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    specificity = recall_score(1 - y_true, 1 - y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred_binary)
    
    try:
        auc_score = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc_score = 0.5  
    
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'mcc': mcc,
        'auc': auc_score,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }


def test_model(model_path, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    test_preds = []
    test_probs = []
    test_targets = []
    test_names = []
    test_sequences = []
    
    criterion = nn.BCELoss()
    test_loss = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            names, sequences, labels, features, edge_index, batch_idx = batch
            
            features = features.to(device)
            edge_index = edge_index.to(device)
            batch_idx = batch_idx.to(device)
            labels = labels.to(device)
            labels = labels.squeeze(dim=1)
            
            outputs = model(features, edge_index, batch_idx)
            
            if outputs.dim() > 1:
                outputs = outputs.squeeze()
            
            loss = criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)
            
            probs_np = outputs.cpu().numpy()
            preds_np = (probs_np > threshold ).astype(float)
            targets_np = labels.cpu().numpy()
            
            if probs_np.ndim == 0:  
                test_probs.append(probs_np)
                test_preds.append(preds_np)
                test_targets.append(targets_np)
                test_names.append(names[0])
                test_sequences.append(sequences[0])
            else:  
                test_probs.extend(probs_np.flatten())
                test_preds.extend(preds_np.flatten())
                test_targets.extend(targets_np.flatten())
                test_names.extend(names)
                test_sequences.extend(sequences)
    
    test_loss /= len(test_loader.dataset)
    
    test_metrics = calculate_metrics(
        np.array(test_targets), 
        np.array(test_preds), 
        np.array(test_probs)
    )
    
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"Specificity: {test_metrics['specificity']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"MCC: {test_metrics['mcc']:.4f}")
    print(f"AUC: {test_metrics['auc']:.4f}")
    print(f"Confusion Matrix:")
    print(f"  TN: {test_metrics['tn']}, FP: {test_metrics['fp']}")
    print(f"  FN: {test_metrics['fn']}, TP: {test_metrics['tp']}")
    
    results_df = pd.DataFrame({
        'uniprot_id': test_names,
        'sequence': test_sequences,
        'true_label': test_targets,
        'predicted_prob': test_probs,
        'predicted_label': test_preds
    })
    results_df.to_csv('../../outputs/results/test_results.csv', index=False)
    
    return test_metrics, results_df


def plot_roc_single(results_file, model_name='Our Model', save_path=None):
    results_df = pd.read_csv(results_file)
    
    y_true = results_df['true_label'].values
    y_pred_probs = results_df['predicted_prob'].values
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    
    plt.plot(fpr, tpr, lw=2, label=f'{model_name}(AUC={roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return roc_auc


def check_data_distribution(dataset):
    all_labels = []
    
    for i in range(len(dataset)):
        _, _, label, _, _ = dataset[i]
        all_labels.append(label.numpy())
    
    all_labels = np.array(all_labels)
    
    print("Distribution:")
    print(f"  Class 0: {np.sum(all_labels == 0)}")
    print(f"  Class 1: {np.sum(all_labels == 1)}")
    print(f"  Proportion: {np.sum(all_labels == 1) / len(all_labels):.4f}")


def five_fold_cross_validation(full_train_df, num_epochs=30, lr=0.00001, k_folds=5):

    os.makedirs('../../outputs/results/cv_results', exist_ok=True)
    os.makedirs('../../outputs/models/cv_models', exist_ok=True)

    fold_results = []
    best_val_metrics_all_folds = []

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    
    X = full_train_df['uniprot_id'].values
    y = full_train_df['label'].values
    
    print("=" * 60)
    print(f"Begin {k_folds}-fold cross-validation:")
    print("=" * 60)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*50}")
        print(f"Fold {fold+1}/{k_folds} ")
        print(f"{'='*50}")
        
        train_fold_df = full_train_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = full_train_df.iloc[val_idx].reset_index(drop=True)
        
        print(f"Training Set Distribution - Positive Samples: {train_fold_df['label'].sum()}/{len(train_fold_df)}")
        print(f"Validation Set Distribution - Positive Samples: {val_fold_df['label'].sum()}/{len(val_fold_df)}")
        
        train_dataset = ProDataset(train_fold_df, max_len=64)
        val_dataset = ProDataset(val_fold_df, max_len=64)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=4, 
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=4, 
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=True
        )
        
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
        
        best_val_metrics, history = train_model_fold_val_only(
            model, train_loader, val_loader, 
            num_epochs=num_epochs, lr=lr, fold=fold
        )
        
        fold_result = {
            'fold': fold,
            'train_size': len(train_fold_df),
            'val_size': len(val_fold_df),
            'best_val_metrics': best_val_metrics         
        }
        
        fold_results.append(fold_result)
        best_val_metrics_all_folds.append(best_val_metrics)
        
        with open(f'../../outputs/results/cv_results/fold_{fold}_results.json', 'w') as f:
            json.dump(convert_to_serializable(fold_result), f, indent=4)
        
        print(f"\nBest validation set results for fold {fold+1}")
        print(f"  AUC: {best_val_metrics['auc']:.4f}")
        print(f"  MCC: {best_val_metrics['mcc']:.4f}")
        print(f"  F1: {best_val_metrics['f1']:.4f}")
        print(f"  Accuracy: {best_val_metrics['accuracy']:.4f}")

    average_metrics = calculate_average_metrics(best_val_metrics_all_folds)
    std_metrics = calculate_std_metrics(best_val_metrics_all_folds)
    
    cv_summary = {
        'fold_results': fold_results,
        'average_metrics': average_metrics,
        'std_metrics': std_metrics
    }
    
    with open('../../outputs/results/cv_results/cross_validation_summary.json', 'w') as f:
        json.dump(convert_to_serializable(cv_summary), f, indent=4)
    
    print_final_results_val_only(fold_results, average_metrics, std_metrics)
    
    return cv_summary

def train_model_fold_val_only(model, train_loader, val_loader, num_epochs=100, lr=0.001, fold=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()
    
    best_val_auc = 0
    best_val_metrics = None
    best_epoch = 0
    
    train_loss_history = []
    val_loss_history = []
    train_metrics_history = []
    val_metrics_history = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_probs = []
        train_targets = []
        
        for batch in train_loader:
            names, sequences, labels, features, edge_index, batch_idx = batch
            
            features = features.to(device)
            edge_index = edge_index.to(device)
            batch_idx = batch_idx.to(device)
            labels = labels.to(device)
            labels = labels.squeeze(dim=1)
            
            optimizer.zero_grad()
            outputs = model(features, edge_index, batch_idx)
            
            if outputs.dim() > 1:
                outputs = outputs.squeeze()
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * labels.size(0)
            
            probs_np = outputs.cpu().detach().numpy()
            preds_np = (probs_np > threshold).astype(float)
            targets_np = labels.cpu().numpy()
            
            if probs_np.ndim == 0:
                train_probs.append(probs_np)
                train_preds.append(preds_np)
                train_targets.append(targets_np)
            else:
                train_probs.extend(probs_np.flatten())
                train_preds.extend(preds_np.flatten())
                train_targets.extend(targets_np.flatten())
        
        train_loss /= len(train_loader.dataset)
        train_loss_history.append(train_loss)
        
        train_metrics = calculate_metrics(
            np.array(train_targets), 
            np.array(train_preds), 
            np.array(train_probs)
        )
        train_metrics_history.append(train_metrics)
        
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_probs = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                names, sequences, labels, features, edge_index, batch_idx = batch
                
                features = features.to(device)
                edge_index = edge_index.to(device)
                batch_idx = batch_idx.to(device)
                labels = labels.to(device)
                labels = labels.squeeze(dim=1)
                
                outputs = model(features, edge_index, batch_idx)
                
                if outputs.dim() > 1:
                    outputs = outputs.squeeze()
                
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                
                probs_np = outputs.cpu().numpy()
                preds_np = (probs_np > threshold).astype(float)
                targets_np = labels.cpu().numpy()
                
                if probs_np.ndim == 0:
                    val_probs.append(probs_np)
                    val_preds.append(preds_np)
                    val_targets.append(targets_np)
                else:
                    val_probs.extend(probs_np.flatten())
                    val_preds.extend(preds_np.flatten())
                    val_targets.extend(targets_np.flatten())
        
        val_loss /= len(val_loader.dataset)
        val_loss_history.append(val_loss)
        
        val_metrics = calculate_metrics(
            np.array(val_targets), 
            np.array(val_preds), 
            np.array(val_probs)
        )
        val_metrics_history.append(val_metrics)
        
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_val_metrics = val_metrics.copy()
            best_epoch = epoch
            torch.save(model.state_dict(), f'../../outputs/models/cv_models/best_model_fold{fold}.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f'Fold {fold+1}, Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            print(f'  Val AUC: {val_metrics["auc"]:.4f}, Val MCC: {val_metrics["mcc"]:.4f}')
    
    print(f'Fold {fold+1} Training completed!')
    print(f'Best AUC: {best_val_auc:.4f} (Epoch {best_epoch+1} )')
    
    return best_val_metrics, {
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'train_metrics': train_metrics_history,
        'val_metrics': val_metrics_history
    }

def calculate_average_metrics(metrics_list):
    avg_metrics = {}
    for key in metrics_list[0].keys():
        if key not in ['tn', 'fp', 'fn', 'tp']:
            values = [metrics[key] for metrics in metrics_list]
            avg_metrics[key] = np.mean(values)
    
    avg_metrics['tn'] = np.mean([metrics['tn'] for metrics in metrics_list])
    avg_metrics['fp'] = np.mean([metrics['fp'] for metrics in metrics_list])
    avg_metrics['fn'] = np.mean([metrics['fn'] for metrics in metrics_list])
    avg_metrics['tp'] = np.mean([metrics['tp'] for metrics in metrics_list])
    
    return avg_metrics

def calculate_std_metrics(metrics_list):
    std_metrics = {}
    for key in metrics_list[0].keys():
        if key not in ['tn', 'fp', 'fn', 'tp']:
            values = [metrics[key] for metrics in metrics_list]
            std_metrics[key] = np.std(values)
    
    return std_metrics

def print_final_results_val_only(fold_results, average_metrics, std_metrics):
    print("\n" + "="*70)
    print("Final results of 5-fold cross-validation (validation set performance)")
    print("="*70)
    
    for result in fold_results:
        metrics = result['best_val_metrics']
        print(f"Fold {result['fold']+1}:")
        print(f"  Size of training set: {result['train_size']}, Size of validation set: {result['val_size']}")
        print(f"  AUC: {metrics['auc']:.4f}, MCC: {metrics['mcc']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        print("-" * 50)

    print("\nAverage Validation Set Metrics (Mean ± Std):")
    print(f"  Accuracy:  {average_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}")
    print(f"  Precision: {average_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    print(f"  Recall:    {average_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    print(f"  Specificity: {average_metrics['specificity']:.4f} ± {std_metrics['specificity']:.4f}")
    print(f"  F1-score:  {average_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")
    print(f"  MCC:       {average_metrics['mcc']:.4f} ± {std_metrics['mcc']:.4f}")
    print(f"  AUC:       {average_metrics['auc']:.4f} ± {std_metrics['auc']:.4f}")
    
    with open('../../outputs/results/cv_results/average_metrics_val_only.txt', 'w') as f:
        f.write("Average results of 5-fold cross-validation on the validation set\n")
        f.write("="*50 + "\n")
        for key, value in average_metrics.items():
            if key not in ['tn', 'fp', 'fn', 'tp']:
                std = std_metrics[key]
                f.write(f"{key}: {value:.4f} ± {std:.4f}\n")