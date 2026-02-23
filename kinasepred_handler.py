import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from typing import List, Dict, Any

# Suppress annoying TF logs as the original script did
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)

class KinasePredHanlder:
    """
    Handles interactions of small molecules with Kinases using TensorFlow 2.13+
    """

    AVAILABLE_PROPERTIES = ["Kinase Activity Prediction"]
    
    def __init__(self):
        self.models = {"Kinase Activity Prediction": None}
        self.model_path = "./KinasePred/model"
        
        tf.config.set_visible_devices([], 'GPU')
        
        self._load_models()

    def _load_models(self) -> None:
        if not os.path.exists(self.model_path):
            print(f"ERROR: Model path not found at {self.model_path}")
            return

        try:
            # Loading the model (SavedModel format)
            self.models["Kinase Activity Prediction"] = keras.models.load_model(self.model_path, compile=True)
        except Exception as e:
            print(f"ERROR: Failed to load KinasePred models: {e}")

    def _smi_to_array(self, smi: str) -> np.ndarray:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return np.zeros((2048,), dtype=np.float32)
        
        try:
            # Modern RDKit Morgan Fingerprint Generation
            generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
            fp = generator.GetFingerprint(mol)
            # Fix: Initialize to correct size (2048) and float32 for TF compatibility
            arr = np.zeros((2048,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        except Exception:
            return np.zeros((2048,), dtype=np.float32)
        
    def process_multiple_properties(self, smiles: str, property_list: List[str]) -> Dict[str, Any]:
        """Process a single SMILES string and return structured results."""
        if not smiles:
            return {"smiles": smiles, "status": "error", "results": {}, "error": "Empty SMILES"}
        
        valid_props = [p for p in property_list if p in self.AVAILABLE_PROPERTIES]
        res_entry = {
            "smiles": smiles,
            "status": "success",
            "results": {}, 
            "error": None
        }

        try:
            if "Kinase Activity Prediction" in valid_props:
                arr = self._smi_to_array(smiles).reshape(1, -1)
                prob = self.models["Kinase Activity Prediction"].predict(arr, verbose=0)
                
                result_stack = np.hstack([1 - prob, prob])
                final_val = float(result_stack[0][1])
                
                # FIXED: Nest the result under the property name key
                res_entry["results"]["Kinase Activity Prediction"] = {
                    "property": "Kinase Activity Prediction",
                    "status": "success", 
                    "results": final_val,
                    "error": None
                }
            
            return res_entry
        
        except Exception as e:
            return {"smiles": smiles, "status": "error", "results": {}, "error": str(e)} 
    
    def process_multiple_properties_batch(self, smiles_list: List[str], property_list: List[str]) -> List[Dict[str, Any]]:
        """Process a list of SMILES."""
        return [self.process_multiple_properties(smi, property_list) for smi in smiles_list]
    
if __name__ == '__main__':
    handler = KinasePredHanlder()
    # Test with a sample SMILES
    test_smi = "CCCOCCCOCN"
    results = handler.process_multiple_properties(test_smi, ["Kinase Activity Prediction"])
    print(results)
   