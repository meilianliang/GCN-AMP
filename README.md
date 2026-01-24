# GCN-AMP: Integrating Protein Language Models with Graph Convolution and Self-Attention for Peptide Functional Activity Prediction
## Abstract:

Accurate prediction of peptide functional activities is critical for combating antimicrobial resistance. Existing deep-learning approaches---even those incorporating protein language models---fail to adequately capture residue-level structural relationships essential for function prediction. We introduce GCN-AMP, a novel deep-learning framework that transcends sequence-only representations by integrating evolutionary embeddings with explicit graph-based relational modeling. Each peptide is encoded via ESM-2 and represented as a fully connected graph, where nodes correspond to residue embeddings. A graph convolutional network (GCN) models local residue interactions, while a self-attention mechanism captures long-range dependencies, forming a synergistic graph-topology and attention-filtering framework for multi-scale feature learning. Evaluated across five functional activities (antibacterial, antifungal, antiviral, anticancer, antihiv), GCN-AMP outperforms state-of-the-art methods on most tasks, with rigorous ablation studies confirming the critical roles of the fully connected graph prior and the GCN-attention synergy. 



## Installation

Ensure Python 3.8+ is installed, then run:

```sh
chmod +x *.sh
./install.sh
```



## Usage

### 1.Cross-validation (training):

Use the  `cross_valid.sh`   script to run cross-validation.

**Command:**

```sh
./cross_valid.sh </path/to/data.csv>
```

For example:
* using the example training file (train.csv):

  ```sh
  ./cross_valid.sh train.csv
  ```

  

### 2.Testing:

Use the  `test.sh`  script to test a trained model.

**Command:**

```sh
./test.sh </path/to/test_data.csv> [optional_model_weights.pth]
```

For example:

* using the example test file (Antiviral_test.csv) and the trained model ( Antiviral_best_model.pth ) :

  ```sh
  ./test.sh Antiviral_test.csv outputs/models/multi_functions/Antiviral_best_model.pth
  ```



### 3.Prediction: 

Use the  `predict.sh`  script to make predictions on new data.

**Command:**

```
./predict.sh </path/to/new_data.csv> [optional_model_directory]
```

For example:

* using the example file (.csv) and the model directory ( outputs/models/multi_functions/ ) :

  ```sh
  ./predict.sh samples.csv outputs/models/multi_functions/
  ```



###  Notes

1. Input CSV files should contain protein sequences
2. Outputs are saved to the outputs/ directory
3. Run scripts from the project root directory



## External Datasets 

To evaluate the model's generalization performance on unseen data, we provide two independent external datasets:

* AMP Dataset: [external_set_AMP.csv](https://github.com/meilianliang/GCN-AMP/blob/main/external_set_AMP.csv)
* Non-AMP Dataset: [external_set_NonAMP.csv](https://github.com/meilianliang/GCN-AMP/blob/main/external_set_NonAMP.csv)



## Model Checkpoints

The trained weight files for the five functional prediction models are saved in the following directory: 

outputs/models/multi_functions/

**Available Models:**
The directory contains checkpoints for all five functional predictions:

* Antibacterial model

* Antiviral model

* Antifungal model

* Anticancer model

* Anti-HIV model

**Usage Notes:**
By default, `predict.sh` uses all five model files simultaneously to generate comprehensive predictions. For targeted evaluation, you can specify a single model file when running `test.sh`.

