import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import torch
import argparse

Pt_Path = '../../data/processed/peptide_pt/'
out_dir = '../../data/features/node_features_esm/'
Node_Feature_num = 1280

def get_matrix(df):
    for i in range(0,len(df)):
        uniprot_id = str(df.loc[i,"uniprot_id"])
        sequence = df.loc[i,"sequence"]               
        # L * 1280
        esm_feature_path = Pt_Path +  uniprot_id + '.pt'           
        esm_feat = torch.load(esm_feature_path)["representations"][33].numpy()          
        esm_feat_truncated = esm_feat[:, 1:-1, :]  # [1, L, 1280]
        matrix = esm_feat_truncated[0]        
        print(uniprot_id, matrix.shape)
        np.save(out_dir + uniprot_id + '.npy',matrix)


def cal_mean_std(fastalist):
    total_length = 0
    oneD_mean = np.zeros(Node_Feature_num)
    oneD_mean_square = np.zeros(Node_Feature_num)
    for name in tqdm(fastalist):
        matrix = np.load(out_dir + name+'.npy')
        total_length += matrix.shape[0]
        oneD_mean += np.sum(matrix, axis=0)
        oneD_mean_square += np.sum(np.square(matrix),axis=0)
    oneD_mean /= total_length  # EX
    oneD_mean_square /= total_length  # E(X^2)
    oneD_std = np.sqrt(np.subtract(oneD_mean_square, np.square(oneD_mean)))  # DX = E(X^2)-(EX)^2, std = sqrt(DX)
    np.save('../../data/features/mean_esm.npy', oneD_mean)
    np.save('../../data/features/std_esm.npy', oneD_std)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='read CSV')
    parser.add_argument('--input', type=str, default="../../train.csv",
                    help='Please input the path of CSV (default：../../train.csv)')
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    fastalist = df["uniprot_id"].astype(str).to_list()
    os.makedirs(out_dir, exist_ok=True)
    get_matrix(df)
