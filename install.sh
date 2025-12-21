#!/bin/bash

# GCN-AMP One-Command Installation (GPU version)
echo "Installing GCN-AMP dependencies..."

# Install PyTorch with CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric dependencies
pip install torch-scatter==2.1.1 torch-sparse==0.6.17 torch-cluster==1.6.1 torch-spline-conv==1.2.2 -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# Install remaining packages including ESM
pip install numpy==1.23.5 pandas==1.5.3 matplotlib==3.7.0 seaborn==0.12.2 scikit-learn==1.2.0 tqdm==4.65.0 torch-geometric==2.3.1 fair-esm==2.0.0

echo "Installation complete!"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}'); import esm; print(f'ESM {esm.__version__}')"