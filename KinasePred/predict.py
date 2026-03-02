#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print("""
 _  ___                   ___            _ 
| |/ (_)_ _  __ _ ___ ___| _ \_ _ ___ __| |
| ' <| | ' \/ _` (_-</ -_)  _/ '_/ -_) _` |
|_|\_\_|_||_\__,_/__/\___|_| |_| \___\__,_|


University of Pisa - Department of Pharmacy - MMVSL https://www.mmvsl.it/wp/
""")

import os
import joblib
import argparse
import warnings
import logging
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow.compiler").setLevel(logging.ERROR)
logging.getLogger("rdkit").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
sys.stderr = open(os.devnull, 'w')

try:
    import tensorflow as tf
    from tensorflow import keras
except RuntimeError as e:
    if "module compiled against API version" in str(e):
        sys.exit(0)  
    else:
        raise e  
finally:
    sys.stderr = sys.__stderr__  

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.set_visible_devices([], 'GPU')
        print("[INFO] Using CPU only.")
    except RuntimeError:
        pass

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

parser = argparse.ArgumentParser(description="KinasePred API")
parser.add_argument('-in', '--input_file', required=True, type=str, help="Input CSV file (must contain a 'SMILES' column)")
parser.add_argument('-o', '--output', required=True, type=str, help="Output file")
args = parser.parse_args()

output_name = args.output if args.output.endswith(".csv") else f"{args.output}.csv"

def fp_as_array(mol):
    try:
        generator = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((1,), int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except:
        return np.zeros((2048,), int)

model_path = "model"
if not os.path.exists(model_path):
    print("[ERROR] Model file not found.")
    sys.exit(1)

try:
    model = keras.models.load_model(model_path, compile=True)
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    sys.exit(1)

def models_to_proba(X):
    prob = model.predict(X, verbose=0)
    return np.hstack([1 - prob, prob])

def get_prediction(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "ERROR"
    
    fp = fp_as_array(mol).reshape(1, -1)
    prob = models_to_proba(fp)[0][1]  # Probabilità di attività
    return "YES" if prob > 0.5 else "NO"

def make_report(data):
    data["Kinase Activity Prediction"] = data["SMILES"].apply(get_prediction)
    data.to_csv(output_name, index=False)
    print(f"[INFO] Predictions saved as {output_name}")

try:
    df = pd.read_csv(args.input_file)
    if "SMILES" not in df.columns:
        raise ValueError("Column 'SMILES' not found in input file.")
except Exception as e:
    print(f"[ERROR] {e}")
    sys.exit(1)

make_report(df)
