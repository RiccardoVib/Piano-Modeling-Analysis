# A Comparative Computational Approach to Piano Modeling Analysis

This code repository for the article _A Comparative Computational Approach to Piano Modeling Analysis_, Proceedings of the SMC Conferences. SMC Network, 2023.

This repository contains all the necessary utilities to use our code. Find the code located inside the "./src" folder

### Contents

1. [Datasets](#datasets)
2. [How to Train and Run Inference](#how-to-train-and-run-inference)

<br/>

# Datasets
Datsets is available at the following link:
[Piano Recordings](https://doi.org/10.5281/zenodo.7620389)

# How To run the analysis

First, install Python dependencies:
```
cd ./src
pip install -r requirements.txt
```

Ensure you have loaded the dataset into the chosen datasets folder

Available options: 
* --data_dir - Folder directory in which the datasets are stored [str] (default="./datasets")
* --type - If consider single notes (single) or chord (chord) and re-triggered notes (rep) [str] (default="Single")

Example: 
```
cd ./src/

python starter.py --type 'Single' 
```

