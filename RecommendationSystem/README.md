# Recommendation System

## Introduction


## Features


## Quick Start

```python
pip install -r requirements.txt
```

**Prerequisite**: 

**Usage**:

```sh

```

### DataSet






## Project Structure
```
RecommendationSystem/
│
├── data/
│   ├── train.txt              # Training data
│   ├── test.txt               # Test data
│   ├── itemAtrribute.txt     # Item attributes data
│   ├── ResultForm.txt         # Result format for submission
│   └── DataFormatExplanation.txt # Explanation of the dataset format
│
├── preprocessing/
│   ├── __init__.py           # Init file to make this a package
│   ├── preprocess.py         # Script for data preprocessing functions
│   └── cache.py              # Script for caching intermediate data
│
├── models/.
│   ├── __init__.py           # Init file to make this a package
│   ├── svd.py                # Implementation of SVD model
│   ├── ncf.py                # Implementation of Neural Collaborative Filtering model
│   ├── fm.py                 # Implementation of Factorization Machine model
│   ├── hybrid.py             # Implementation of the hybrid model (NCF + FM)
|   ├── lightgbm.py           # Implementation of LightGBM model
|   |── two_tower.py          # Implementation of Two-Tower model
│   └── gnn.py                # Implementation of Graph Neural Network model (future work)
│
├── utils/
│   ├── __init__.py           # Init file to make this a package
│   ├── evaluation.py         # Script for evaluation metrics (RMSE, etc.)
│   ├── config.py             # Configuration file for hyperparameters and paths
│   └── helpers.py            # Helper functions 
│
|── __init__.py               # Init file to make this a package
├── train.py                  # Main training script
├── predict.py                # Script to generate predictions for test data
├── requirements.txt          # Dependencies
├── README.md                 # Project description and instructions
└── run.sh                    # Shell script to run the entire pipeline
```