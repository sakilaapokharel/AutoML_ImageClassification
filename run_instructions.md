# Running the Pipeline

## Download the Datasets

Before running any of the scripts, download all supported datasets.

### Usage

```bash
python download_datasets.py
```

This script downloads the following datasets into the `data/` directory:

* Emotions
* Flowers
* Fashion
* Skin Cancer

Datasets are downloaded only once and reused in subsequent runs.

---

The project provides four entry points depending on the stage of the workflow you want to execute.

## 1. End-to-End Pipeline (Search + Training + Evaluation)

Runs the complete pipeline:

1. Performs AutoML hyperparameter search.
2. Trains the image classifier using the best configuration found during the search.
3. Evaluates the trained model on the test set.

### Usage

```bash
python run_pipeline.py --dataset <dataset_name>
```

### Example

```bash
python run_pipeline.py --dataset skin_cancer
```

### Optional Arguments

| Argument            | Default      | Description                                                      |
| ------------------- | ------------ | ---------------------------------------------------------------- |
| `--dataset`         | **Required** | Dataset to use (`emotions`, `flowers`, `fashion`, `skin_cancer`) |
| `--seed`            | `42`         | Random seed                                                      |
| `--tabpfn_mode`     | `local`      | TabPFN backend (`local` or `client`)                             |
| `--search_strategy` | `bohb`       | AutoML search strategy (`bohb` or `successive_halving`)          |
| `--batch_size`      | `64`         | Student model inference batch size                               |

Example with all options:

```bash
python run_pipeline.py \
    --dataset skin_cancer \
    --seed 42 \
    --tabpfn_mode local \
    --search_strategy bohb \
    --batch_size 64
```

---

## 2. Run AutoML Search Only

Runs only the AutoML search to identify the best hyperparameter configuration.

The best configuration is saved and can later be used for model training.

### Usage

```bash
python run_only_search.py --dataset <dataset_name>
```

### Example

```bash
python run_only_search.py --dataset skin_cancer
```

### Optional Arguments

| Argument            | Default      | Description                                                      |
| ------------------- | ------------ | ---------------------------------------------------------------- |
| `--dataset`         | **Required** | Dataset to use (`emotions`, `flowers`, `fashion`, `skin_cancer`) |
| `--seed`            | `42`         | Random seed                                                      |
| `--tabpfn_mode`     | `local`      | TabPFN backend (`local` or `client`)                             |
| `--search_strategy` | `bohb`       | Search strategy (`bohb` or `successive_halving`)                 |

Example:

```bash
python run_only_search.py \
    --dataset skin_cancer \
    --search_strategy bohb
```

---

## 3. Train Model Using Existing Search Results

Uses the latest saved AutoML configuration to train the image classifier and evaluate it on the test set.

> **Note:** AutoML search must be completed before running this script.

### Usage

```bash
python run_train_only.py --dataset <dataset_name>
```

### Example

```bash
python run_train_only.py --dataset skin_cancer
```

---

## 4. Evaluate a Trained Model

Loads the saved model checkpoints and evaluates the trained classifier without retraining.

> **Note:** A trained model checkpoint must already exist.

### Usage

```bash
python run_prediction.py --dataset <dataset_name>
```

### Example

```bash
python run_prediction.py --dataset skin_cancer
```

This script:

1. Loads the saved checkpoints.
2. Restores the trained model.
3. Evaluates the model on the test dataset.
