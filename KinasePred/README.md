# KinasePred


KinasePred is a ML tool that allows the prediction of the potential activity of small molecules against kinases. This repository provides a Python script for kinase prediction using a predefined environment and dependencies. 

## Installation

To set up the required environment, use the provided YAML file:

```sh
conda env create -f kinasepred_env.yml
```

Then, activate the environment:

```sh
conda activate kinase_pred
```

## Usage

Run the prediction script with Python:

```sh
python predict.py -in input.csv -o output.csv
```

### Input
- The input file must be a CSV containing a column named `SMILES`, which represents the molecular structures.
- A sample CSV file (input.csv) is included in the repository for testing purposes.

### Output
- The script will generate a CSV file with the prediction results indicating whether each molecule is predicted to have kinase activity.

## Dependencies

All necessary dependencies are included in `kinase_pred.yml`.

