#!/bin/bash

# Check if input file parameter is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <CSV file path> [model directory path]"
    echo "Example: $0 /path/to/your/data.csv"
    echo "         $0 /path/to/your/data.csv /path/to/model/directory"
    echo "Note: CSV file should be located in the project root directory or its subdirectories"
    echo "      Model directory path is optional (default: outputs/models/multi_functions)"
    exit 1
fi

# Get input file path
INPUT_FILE="$1"

# Get model directory path (if provided)
MODEL_DIR="${2:-outputs/models/multi_functions}"

# Check if CSV file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: CSV file '$INPUT_FILE' does not exist!"
    exit 1
fi

# Check if model dir exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model file '$MODEL_DIR' does not exist!"
    exit 1
fi

# Get absolute path of input file
if [[ "$INPUT_FILE" != /* ]]; then
    # If it's a relative path, convert to absolute path
    INPUT_FILE_ABS="$(cd "$(dirname "$INPUT_FILE")"; pwd)/$(basename "$INPUT_FILE")"
else
    INPUT_FILE_ABS="$INPUT_FILE"
fi

# Get absolute path of model directory
if [[ "$MODEL_DIR" != /* ]]; then
    # If it's a relative path, convert to absolute path
    if [ -d "$MODEL_DIR" ]; then
        MODEL_DIR_ABS="$(cd "$MODEL_DIR"; pwd)"
    else
        # Try to resolve relative to project root
        PROJECT_ROOT="$(pwd)"
        if [ -d "$PROJECT_ROOT/$MODEL_DIR" ]; then
            MODEL_DIR_ABS="$(cd "$PROJECT_ROOT/$MODEL_DIR"; pwd)"
        else
            # If directory doesn't exist yet, just convert the path
            MODEL_DIR_ABS="$(cd "$(dirname "$MODEL_DIR")"; pwd)/$(basename "$MODEL_DIR")"
        fi
    fi
else
    MODEL_DIR_ABS="$MODEL_DIR"
fi

echo "The data file: $INPUT_FILE_ABS"
echo "Using model directory: $MODEL_DIR_ABS"
echo "========================================="

# Get project root directory (assuming script runs from project root)
PROJECT_ROOT="$(pwd)"
echo "Project root directory: $PROJECT_ROOT"

# Step 1: Run ESM-2 embedding extraction - extract_esm_embeddings.py
echo "Step 1: Running ESM-2 embedding extraction (extract_esm_embeddings.py)..."
cd "$PROJECT_ROOT/src/features" || { echo "Cannot enter src/features directory"; exit 1; }
python extract_esm_embeddings.py --input "$INPUT_FILE_ABS"

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
python create_node_for_test.py --input "$INPUT_FILE_ABS"

# Check if previous step succeeded
if [ $? -ne 0 ]; then
    echo "Error: Node feature creation step failed!"
    exit 1
fi

echo "Node feature creation completed!"
echo "========================================="

# Step 3: Run model prediction - predict.py
echo "Step 3: Running model prediction (predict.py)..."
cd "$PROJECT_ROOT/src/models" || { echo "Cannot enter src/models directory"; exit 1; }
python predict.py --input_file "$INPUT_FILE_ABS" --model_dir "$MODEL_DIR_ABS"

# Check if previous step succeeded
if [ $? -ne 0 ]; then
    echo "Error: Model prediction step failed!"
    exit 1
fi

echo "========================================="
echo "All steps completed! Prediction finished."