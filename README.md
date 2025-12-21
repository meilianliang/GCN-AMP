Installation

Ensure Python 3.8+ is installed, then run:

chmod +x install.sh
./install.sh

==============================================

Usage

1. Cross-validation (training):

./cross_valid.sh /path/to/data.csv

2. Testing:

./test.sh /path/to/test_data.csv [optional_model_weights.pth]

3. Prediction:

./predict.sh /path/to/new_data.csv [optional_model_directory]

==============================================

Notes

1. Input CSV files should contain protein sequences

2. Outputs are saved to the outputs/ directory

3. Run scripts from the project root directory
