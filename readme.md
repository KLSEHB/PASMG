# PASMG - Implementation

## Pre-trained Model-based Vulnerability Detection Using Acyclic Simplified Control Flow Graph and Multi-objective Greedy Path Selection

## 🎯 Abstract

Fine-tuning pre-trained models has become the mainstream paradigm in software vulnerability detection. Compared with traditional methods, pre-trained models can acquire general programming knowledge from large-scale corpora, exhibit stronger generalization in transfer learning scenarios, and achieve competitive performance even in the absence of extensive labeled data. However, existing vulnerability detection methods based on pre-trained models still suffer from performance bottlenecks when dealing with complex control flows and long code snippets, mainly due to two reasons: (1) Most pre-trained models have input length limitations, which severely hinder their effectiveness on lengthy code. (2) For code with intricate control flows and data dependencies, pre-trained models often struggle to capture the complex internal logic, which impedes their ability to recognize vulnerability-related patterns. To address these issues, we propose PASMG, a novel vulnerability detection method based on Acyclic Simplified Control Flow Graph (ASCFG) path encoding and a multi-objective greedy selection strategy. Specifically, PASMG first parses the Abstract Syntax Tree (AST) to construct an ASCFG, which preserves only the core control flow of the program and simplifies control flow modeling. Then, all feasible paths in the ASCFG are enumerated, and a multi-objective greedy strategy is introduced to select a representative subset of paths by balancing node coverage reward and path length penalty. These selected paths are individually encoded using a pre-trained model, concatenated, and finally fed into a multilayer perceptron for classification. PASMG outperforms existing state-of-the-art baselines in terms of F1 score and recall, as demonstrated by experiments conducted on three widely used public datasets: Reveal, Devign, and VulCNN. Notably, when handling long code snippets (more than 510 tokens), PASMG achieves significant improvements of 27.15% and 61.90%, respectively, showcasing its superiority in analyzing complex code structures.

## 🏗️ Project Structure

```
PASMG
├───code             # Core source code for preprocessing, path extraction, and model training
│   ├───config.json             # Configuration file (e.g., path settings, hyperparameters)
│   ├───Construct_ASCFG.py      # Build ASCFG from AST
│   ├───ExtractExecutionPath.py # Main script to preprocess code, build AST, construct ASCFG, extract paths, and generate training dataset
│   ├───model4.py               # Define the classification model (encoder + classifier)
│   ├───PathFinder.py           # Enumerate  all paths from ASCFG and select paths
│   └───run.py                  # Main script to train and evaluate the PASMG model
├───dataset          # Datasets used in the experiments
│   ├───2PathsDataset           # Sample data with top-2 paths per function
│   ├───...
│   ├───6PathsDataset
│   │   ├───2
│   │   ├───...
│   │   └───34
│   └───FuncDataset             # Original function-level dataset before path processing
├───parserTool
├───pretrained-model  # Pretrained models (e.g., CodeBERT, PDBERT) for path encoding
│   ├───codebert
│   └───pdbert
├───saved_model       # Trained model checkpoints
├───readme.md         # Project description and usage instructions
└───requirements.txt  # List of Python packages and their versions required to run this project.
```

## 📂 Dataset

To evaluate the performance of PASMG against other models, we utilized the following three publicly available datasets: 
* Reveal [1]: https://drive.google.com/drive/folders/1KuIYgFcvWUXheDhT--cBALsfy1I4utOy
* Devign [2]: https://drive.google.com/drive/folders/1RqtDcOKKnIWxYAmkOTULYeJhuw_KTNys
* VulCNN [3]: https://github.com/CGCL-codes/VulCNN/blob/main/dataset/Dataset-sard.zip

## 📄 Data Format

All datasets are stored in `.jsonl` format, where each line is a JSON object representing one function.

### 1. Raw Function-Level Dataset

This dataset contains the full source code of each function, along with its vulnerability label. It is used as the input to `ExtractExecutionPath.py`.

**Format:**
```json
{"func": "void foo(int x) { if (x > 0) bar(); }", "target": 1}
```
* func: The full source code of the function.
* target: The label indicating whether the function is vulnerable (1) or not (0).

### 2. Path-Based Dataset
This is the output of ExtractExecutionPath.py, and is directly used for training the PASMG model. Each function is represented by a fixed number of representative execution paths extracted from its ASCFG.

**Format (example with 4 paths):**
```json
{"path1": "void foo(int x) { # path1 }", "path2": "void foo(int x) { # path2 }", "path3": "void foo(int x) { # path3 }", "path4": "void foo(int x) { # path4 }", "target": 1}

```
* path1, path2, ..., pathK: K representative paths selected from the ASCFG of the function.
* target: The same vulnerability label as in the raw dataset.

## 🔗 Pre-trained Model Download

Due to file size limitations, the pre-trained model weights used in PASMG and ablation experiments are not included in this repository. Please download them manually from the following links:

- **PDBERT**: [Download Link (Zenodo)](https://zenodo.org/records/10140638/files/PDBERT_data.zip?)  
- **CodeBERT**: [Download Link (huggingface)](https://huggingface.co/microsoft/codebert-base/tree/main)

After downloading, extract the model folders and place them under:
```
/PASMG/pretrained-model/
```
The final structure should look like:
```
pretrained-model/
├───codebert
│   ├───config.json
│   ├───merges.txt
│   ├───pytorch_model.bin
│   ├───special_tokens_map.json
│   ├───tokenizer_config.json
│   └───vocab.json
└───pdbert
    ├───config.json
    ├───merges.txt
    ├───pytorch_model.bin
    ├───special_tokens_map.json
    ├───tokenizer_config.json
    └───vocab.json
```

## 🔧 Environment Setup

We recommend using a virtual environment to avoid package conflicts.

### Step 1: Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Usage
### Step 1: Preprocess and Generate Path-Based Dataset
Before training, you need to extract representative paths from source code using the ASCFG-based method
```shell
cd ./PASMG/code
python ExtractExecutionPath.py --training_set=../Dataset/FuncDataset/Devign/train.jsonl --validation_set=../Dataset/FuncDataset/Devign/valid.jsonl --test_set=../Dataset/FuncDataset/Devign/test.jsonl --ExecutionPaths_training_set=../Dataset/4PathsDataset/14/Devign/train.jsonl --ExecutionPaths_validation_set=../Dataset/4PathsDataset/14/Devign/valid.jsonl --ExecutionPaths_test_set=../Dataset/4PathsDataset/14/Devign/test.jsonl --pretrained_model_path=../pretrained-model/pdbert --PathNum=4 --alpha=14 --beta=1
```
This script will:

* Normalize code and parse it into AST using tree-sitter.
* Construct the ASCFG for each function.
* Enumerate execution paths and apply multi-objective greedy selection.
* Save the resulting path-level dataset under the Dataset/4PathsDataset/ directory.
  
The output will be used as training data for PASMG.

### Step 2: Train and Evaluate the Model
Once the path-based dataset is ready, you can train and evaluate the PASMG model using either of the following two methods:
#### 🔹 Option 1: Specify All Parameters via Command Line
You can directly pass all configuration parameters as command-line arguments when running the training script:
```shell
python run.py ExecutionPaths_train_set=../Dataset/4PathsDataset/14/Devign/train.jsonl ExecutionPaths_valid_set=../Dataset/4PathsDataset/14/Devign/valid.jsonl ExecutionPaths_test_set=../Dataset/4PathsDataset/14/Devign/test.jsonl output_dir=../saved_model/4_14 pretrained_model_path=../pretrain-model/pdbert block_size=512 train_batch_size=16 eval_batch_size=16 epochs=7 gradient_accumulation_steps=2 seed=619 PathNum=4 weight_decay=0.0001 learning_rate=1.5e-05 adam_epsilon=1e-08 max_grad_norm=1.0 threshold=null do_train=true do_test=true write_results=false
```
#### 🔹 Option 2: Use Configuration File
Alternatively, you can define all parameters in config.json, and simply run:
```shell
python run.py
```
If a parameter is not specified on the command line, the script will automatically fall back to its value in config.json.


## 🏆 Result
We evaluate the performance of our PASMG model on three datasets. The results are summarized in the table below:
<table align="center">
<tr>
    <td></td>
    <td>Precision</td>
    <td>Recall</td>
    <td>F1 Score</td>
</tr>
<tr>
    <td>Reveal</td>
    <td>43.33</td>
    <td>63.41</td>
    <td>51.49</td>
</tr>
<tr>
    <td>Devign</td>
    <td>54.84</td>
    <td>90.3</td>
    <td>68.24</td>
</tr>
<tr>
    <td>VulCNN</td>
    <td>99.61</td>
    <td>99.3</td>
    <td>99.45</b></td>
</tr>
</table>