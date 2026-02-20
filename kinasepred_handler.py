import os
import joblib
import warnings
import logging
import numpy as np
import pandas as pd
import sys
import tensorflow as tf
from typing import List, Dict, Any, Optional
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
        self.model_path = "./KinasePred/model"
        self.device = tf.config.set_visible_devices('CPU')
        self._load_models()

    def _load_models(self) -> None:
        try:
            self.models["Kinase Activity Prediction"] = keras.models.load_model(self.model_path, compile=True)
        except Exception as e:
            print(f"ERROR: Failed to load KinasePred models: {e}")

    def _smi_to_array(self, smi):
        mol = Chem.MolFromSmiles(smi)
        try:
            generator = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            fp = generator.GetFingerprint(mol)
            arr = np.zeros((1,), int)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        except:
            return np.zeros((2048,), int)
        
    def process_multiple_properties(self, smiles:str, property_list: List[str]) -> Dict[str, Any]:
        """Process a single SMILES."""
        if not smiles:
            return {}
        
        valid_props = [p for p in property_list if p in self.AVAILABLE_PROPERTIES]
        batch_preds = {}
        try:
            if "Kinase Activity Prediction" in valid_props:
                arr = self._smi_to_array(smiles)
                prob = self.models["Kinase Activity Prediction"].predict(arr, verbose=0)
                batch_preds["Kinase Activity Prediction"] = np.hstack([1 - prob, prob]) # I do not love getting rid of the probability, but this is the way KinasePred Authors designed it
            res_entry = {
                    "smiles": smiles,
                    "status": "success",
                    "results": {},
                    "error": None
                }
            for prop in valid_props:
                res_entry["results"][prop] = {"smiles": smiles, "status": "success", "results": float(batch_preds[prop])} #Overkill for this model. Done to keep conformity of strucutre
            
            return res_entry
        
        except Exception as e:
            # Return error objects for the batch
            return {"smiles": smiles, "status": "error", "results": {}, "error": str(e)} 
    
    def process_multiple_properties_batch(self, smiles_list: List[str], property_list: List[str]) -> List[Dict[str, Any]]:
        res_ls = []
        for smi in smiles_list:
            one_smi_res_obj = self.process_multiple_properties(smi, property_list)
            res_ls.append(one_smi_res_obj)
        return res_ls
    
if __name__ == '__main__':
    handler = KinasePredHanlder()
    print(handler.process_multiple_properties("CCCOCCCOCN", ["Kinase Activity Prediction"]))
   