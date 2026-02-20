import os
import joblib
import warnings
import logging
import numpy as np
import pandas as pd
import sys
import tensorflow as tf
from tensorflow import keras
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from KinasePred import model

class KinasePredHanlder:
    """
    Handles interactions of small molecules with Kinases
    """

    AVAILABLE_PROPERTIES = ["Kinase Activity Prediction"]

    def __init__(self):
        self.model = {"Kinase Activity Prediction": None}

    def _load_models(self) -> None:
        try:
            self.models[] = keras.models.load_model(model_path, compile=True)
        except Exception as e:
            print(f"ERROR: Failed to load KinasePred models: {e}")