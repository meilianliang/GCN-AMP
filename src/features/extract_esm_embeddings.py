import torch
from esm.data import Alphabet
import esm
import torch.nn as nn
from typing import Dict, Tuple
import os
import pandas as pd
import argparse

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features(
    sequence: str,
    model: nn.Module,
    alphabet,
    repr_layer: int = 33,
    device: torch.device = None
) -> Dict[str, torch.Tensor]:
    """
    Extract ESM features (node features and contact map) for a peptide.
    
    Args:
        sequence: peptide sequence
        model: ESM model (e.g., esm1b or esm2)
        alphabet: ESM alphabet object
        repr_layer: Representation layer to extract (default 33, corresponds to ESM-2)
        device: Computing device (e.g., 'cuda:0' or 'cpu')
    Returns:
        Dict containing:
        - "representations": Representations per layer (as a dictionary)
        - "contacts": Contact map (attention weights)
    """
    if device is None:
        device = next(model.parameters()).device  
    
    # Preprocess sequence
    data = [("protein", sequence)]
    batch_converter = alphabet.get_batch_converter()
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    
    # Extract features
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=True)
    
    # Obtain node features and contact map
    node_features = results["representations"][repr_layer].cpu()  # [1, L, dim]
    contacts = results["contacts"].cpu()  # [L, L]
    
    # print("Token indices:", batch_tokens[0].tolist())
    # decoded = ''.join([alphabet.get_tok(tok) for tok in batch_tokens[0].tolist()])
    # print("Decoded sequence:", decoded)
    return {
        "representations": {repr_layer: node_features},
        "contacts": contacts
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='read CSV')
    parser.add_argument('--input', type=str, default="../../train.csv",
                    help='Please input the path of CSV (default：../../train.csv)')
    args = parser.parse_args()
    output_dir = "../../data/processed/peptide_pt/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    df = pd.read_csv(args.input)
    id_list = df["uniprot_id"].tolist()
    for i in range(0,len(df)):
        filename = df.loc[i,"uniprot_id"]
        print(filename)
        sequence = df.loc[i,"sequence"]
        # print(len(sequence))
        features = extract_features(sequence, model, alphabet)
        torch.save(features, f"{output_dir}{filename}.pt")