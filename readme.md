# PASMG - Implementation

## Pre-trained Code Model-based Vulnerability Detection via Statement-Level Simplified Control Flow Graph and Multi-objective Greedy Path Selection

## 🎯 Abstract

Fine-tuning pre-trained code models is an effective paradigm for vulnerability detection, but fixed input-length limits and insufficient structural modeling remain challenging for long functions with complex control flow. PASMG constructs a Statement-level Simplified Control Flow Graph (SSCFG) from the Abstract Syntax Tree (AST), enumerates candidate graph paths, and uses a multi-objective greedy strategy to select representative paths by balancing node coverage and path length. These paths are linearized into coverage-oriented statement sequences, encoded by a pre-trained code model, and aggregated for function-level vulnerability prediction. We evaluate PASMG on the two real-world datasets Chromium+Debian (ReVeal) and FFmpeg+Qemu (Devign), and the synthetic SARD dataset, covering 27,003 vulnerable and 56,409 non-vulnerable samples. PASMG is compared with 19 baseline methods: 2 static analysis tools, 5 non-pretrained deep learning methods, 7 fine-tuning-based pre-trained code models, and 5 LLM-based methods. It achieves relative F1 improvements of 3.83%, 3.76%, and 2.49% over the strongest baseline on the three datasets, respectively. On long-code samples, PASMG achieves a relative F1 improvement of up to 11.81%.

## 🏗️ Project Structure

```
PASMG
├───code             # Core source code for preprocessing, path extraction, and model training
│   ├───config.json             # Configuration file (e.g., path settings, hyperparameters)
│   ├───Construct_ASCFG.py      # Legacy filename; constructs the SSCFG from the AST
│   ├───ExtractExecutionPath.py # Main script to preprocess code, build the SSCFG, extract paths, and generate training data
│   ├───model4.py               # Define the classification model (encoder + classifier)
│   ├───PathFinder.py           # Enumerate candidate graph paths and select representative paths
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
* Chromium+Debian (ReVeal) [1]: https://drive.google.com/drive/folders/1KuIYgFcvWUXheDhT--cBALsfy1I4utOy
* FFmpeg+Qemu (Devign) [2]: https://drive.google.com/drive/folders/1RqtDcOKKnIWxYAmkOTULYeJhuw_KTNys
* SARD [3]: https://github.com/CGCL-codes/VulCNN/blob/main/dataset/Dataset-sard.zip

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
This is the output of ExtractExecutionPath.py, and is directly used for training the PASMG model. Each function is represented by a fixed number of coverage-oriented statement sequences derived from representative paths in the SSCFG.

**Format (example with 4 paths):**
```json
{"path1": "void foo(int x) { # path1 }", "path2": "void foo(int x) { # path2 }", "path3": "void foo(int x) { # path3 }", "path4": "void foo(int x) { # path4 }", "target": 1}

```
* path1, path2, ..., pathK: K coverage-oriented statement sequences derived from representative SSCFG paths.
* target: The same vulnerability label as in the raw dataset.

## 🔗 Pre-trained Model Download

Due to file size limitations, the pre-trained model weights used in PASMG and ablation experiments are not included in this repository. Please download them manually from the following links:

- **CodeBERT**: [Download Link (huggingface)](https://huggingface.co/microsoft/codebert-base/tree/main)
- **VulBERTa**: [Download Link](https://1drv.ms/u/s!AueKnGqzBuIVkq4CynZHsF8Mv-en1g?e=3gg60p)
- **CodeT5+**: [Download Link (huggingface)](https://huggingface.co/Salesforce/codet5p-110m-embedding)
- **UnixCoder-base**: [Download Link (huggingface)](https://huggingface.co/microsoft/unixcoder-base)
- **PDBERT**: [Download Link (Zenodo)](https://zenodo.org/records/10140638/files/PDBERT_data.zip?)  

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
Before training, you need to extract representative paths from source code using the SSCFG-based method.
```shell
cd ./PASMG/code
python ExtractExecutionPath.py --training_set=../Dataset/FuncDataset/Devign/train.jsonl --validation_set=../Dataset/FuncDataset/Devign/valid.jsonl --test_set=../Dataset/FuncDataset/Devign/test.jsonl --ExecutionPaths_training_set=../Dataset/4PathsDataset/14/Devign/train.jsonl --ExecutionPaths_validation_set=../Dataset/4PathsDataset/14/Devign/valid.jsonl --ExecutionPaths_test_set=../Dataset/4PathsDataset/14/Devign/test.jsonl --pretrained_model_path=../pretrained-model/pdbert --PathNum=4 --alpha=14 --beta=1
```
This script will:

* Normalize code and parse it into AST using tree-sitter.
* Construct the SSCFG for each function.
* Enumerate candidate graph paths and apply multi-objective greedy selection.
* Save the resulting path-level dataset under the Dataset/4PathsDataset/ directory.
  
The output will be used as training data for PASMG.

### Step 2: Train and Evaluate the Model
Once the path-based dataset is ready, you can train and evaluate the PASMG model using either of the following two methods:
#### 🔹 Option 1: Specify All Parameters via Command Line
You can directly pass all configuration parameters as command-line arguments when running the training script:
```shell
python run.py --ExecutionPaths_train_set=../Dataset/4PathsDataset/14/Devign/train.jsonl --ExecutionPaths_valid_set=../Dataset/4PathsDataset/14/Devign/valid.jsonl --ExecutionPaths_test_set=../Dataset/4PathsDataset/14/Devign/test.jsonl --output_dir=../saved_model/4_14 --pretrained_model_path=../pretrained-model/pdbert --block_size=512 --train_batch_size=16 --eval_batch_size=16 --epochs=7 --gradient_accumulation_steps=2 --seed=619 --PathNum=4 --weight_decay=0.0001 --learning_rate=1.5e-04 --adam_epsilon=1e-08 --max_grad_norm=1.0
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
    <td>Chromium+Debian</td>
    <td>43.33</td>
    <td>63.41</td>
    <td>51.49</td>
</tr>
<tr>
    <td>FFmpeg+Qemu</td>
    <td>54.84</td>
    <td>90.3</td>
    <td>68.24</td>
</tr>
<tr>
    <td>SARD</td>
    <td>99.61</td>
    <td>99.3</td>
    <td>99.45</td>
</tr>
</table>
