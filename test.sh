#!/bin/bash

# Check if input file parameter is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <CSV file path> [model weight path]"
    echo "Example: $0 /path/to/your/data.csv"
    echo "         $0 /path/to/your/data.csv /path/to/model/weights.pth"
    echo "Note: CSV file should be located in the project root directory or its subdirectories"
    echo "      Model weight path is optional (default: outputs/models/cv_models/best_model_fold4.pth)"
    exit 1
fi

# Get input file path
INPUT_FILE="$1"

# Get model weight path (if provided)
MODEL_PATH="${2:-outputs/models/cv_models/best_model_fold4.pth}"

# Check if file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' does not exist!"
    exit 1
fi

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file '$MODEL_PATH' does not exist!"
    exit 1
fi

# Get absolute paths of files
if [[ "$INPUT_FILE" != /* ]]; then
    # If it's a relative path, convert to absolute path
    INPUT_FILE="$(cd "$(dirname "$INPUT_FILE")"; pwd)/$(basename "$INPUT_FILE")"
fi
if [[ "$MODEL_PATH" != /* ]]; then
    # If it's a relative path, convert to absolute path
    MODEL_PATH="$(cd "$(dirname "$MODEL_PATH")"; pwd)/$(basename "$MODEL_PATH")"
fi

echo "The test data file: $INPUT_FILE"
echo "Using model weights: $MODEL_PATH"
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

# Step 2: Run node feature creation - create_node_for_test.py
echo "Step 2: Running node feature creation (create_node_for_test.py)..."
# Already in src/features directory
python create_node_for_test.py --input "$INPUT_FILE"

# Check if previous step succeeded
if [ $? -ne 0 ]; then
    echo "Error: Node feature creation step failed!"
    exit 1
fi

echo "Node feature creation completed!"
echo "========================================="

# Step 3: Run model testing - test.py
echo "Step 3: Running model testing (test.py)..."
cd "$PROJECT_ROOT/src/models" || { echo "Cannot enter src/models directory"; exit 1; }
python test.py --model_path "$MODEL_PATH" --test_data "$INPUT_FILE"

# Check if previous step succeeded
if [ $? -ne 0 ]; then
    echo "Error: Model testing step failed!"
    exit 1
fi

echo "========================================="
echo "All steps completed! Testing finished."