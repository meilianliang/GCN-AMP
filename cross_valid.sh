#!/bin/bash

# Check if input file parameter is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <CSV file path>"
    echo "Example: $0 /path/to/your/data.csv"
    echo "Note: CSV file should be located in the project root directory or its subdirectories"
    exit 1
fi

# Get input file path
INPUT_FILE="$1"

# Check if file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' does not exist!"
    exit 1
fi

# Get absolute path of input file
if [[ "$INPUT_FILE" != /* ]]; then
    # If it's a relative path, convert to absolute path
    INPUT_FILE="$(cd "$(dirname "$INPUT_FILE")"; pwd)/$(basename "$INPUT_FILE")"
fi

echo "The training data file: $INPUT_FILE"
echo "========================================="

# Get project root directory (assuming script runs from project root)
PROJECT_ROOT="$(pwd)"
echo "Project root directory: $PROJECT_ROOT"

# Step 1: Run ESM-2 embedding extraction - extract_esm_embeddings.py
echo "Step 1: Running ESM-2 embedding extraction (extract_esm_embeddings.py)..."
cd "$PROJECT_ROOT/src/features" || { echo "Cannot enter src/features directory"; exit 1; }
python extract_esm_embeddings.py --input "$INPUT_FILE"

# Check if previous step succeeded
if [ $? -ne 0 ]; then
    echo "Error: ESM-2 embedding extraction step failed!"
    exit 1
fi

echo "ESM-2 embedding extraction completed!"
echo "========================================="

# Step 2: Run node feature creation - create_node_esm.py
echo "Step 2: Running node feature creation (create_node_esm.py)..."
# Already in src/features directory
python create_node_esm.py --input "$INPUT_FILE"

# Check if previous step succeeded
if [ $? -ne 0 ]; then
    echo "Error: Node feature creation step failed!"
    exit 1
fi

echo "Node feature creation completed!"
echo "========================================="

# Step 3: Run model evaluation - eval.py
echo "Step 3: Running 5-fold cross-validation (eval.py)..."
cd "$PROJECT_ROOT/src/models" || { echo "Cannot enter src/models directory"; exit 1; }
python eval.py --input "$INPUT_FILE"

# Check if previous step succeeded
if [ $? -ne 0 ]; then
    echo "Error: Five-fold cross-validation step failed!"
    exit 1
fi

echo "========================================="
echo "All steps completed! Processing finished."