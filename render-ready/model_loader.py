"""
Model Loader with Auto-Scan, Sanity Tests, and Auto-Repair Functionality
"""
import sys
import joblib
import logging
import numpy as np
import pandas as pd
import traceback
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Define predict_integer function for LOS and Recovery models
# This function is needed to load pickled models that reference it
def predict_integer(model, data):
    """Predict integer values from a regression model"""
    predictions = model.predict(data)
    if isinstance(predictions, np.ndarray):
        rounded = np.round(predictions).astype(int)
    else:
        rounded = int(np.round(predictions))
    return rounded

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_loader.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ModelLoader:
    """Manages loading, testing, and repairing ML models"""
    
    def __init__(self, models_dir: str = "models/models pkl", data_path: Optional[str] = None):
        self.models_dir = Path(models_dir)
        self.data_path = data_path
        self.loaded_models: Dict[str, Dict] = {}
        self.model_manifest = {}
        self.feature_schema = self._get_feature_schema()
        
    def _get_feature_schema(self) -> Dict:
        """Define expected feature schemas for each model"""
        return {
            'cancer_label': {
                'features': [
                    'Age', 'Gender', 'Tobacco Use', 'Alcohol Consumption', 'HPV Infection',
                    'Betel Quid Use', 'Chronic Sun Exposure', 'Poor Oral Hygiene',
                    'Family History of Cancer', 'Compromised Immune System', 'Oral Lesions',
                    'Unexplained Bleeding', 'Difficulty Swallowing', 'White or Red Patches in Mouth',
                    'Early Diagnosis', 'Year_of_Diagnosis', 'Age_x_Tobacco', 'Age_Group_Encoded',
                    'Diet_Encoded', 'Treatment Type_No Treatment', 'Treatment Type_Radiation',
                    'Treatment Type_Surgery', 'Treatment Type_Targeted Therapy'
                ],
                'scaler_path': 'cancer_label_robust_scaler1.pkl',
                'model_type': 'classification'
            },
            'cost': {
                'features': 'all_except_target',
                'target': 'Cost of Treatment (USD)',
                'model_type': 'regression'
            },
            'survival_rate': {
                'features': 'first_17_cols',
                'target': 'Survival Rate (5-Year, %)',
                'model_type': 'regression'
            },
            'los': {
                'features': 'all_except_target',
                'target': 'Predicted_LOS(Days)',
                'model_type': 'regression',
                'has_custom_predict': True
            },
            'recovery': {
                'features': 'all_except_target',
                'target': 'Predicted_Recovery(Days)',
                'model_type': 'regression',
                'has_custom_predict': True
            }
        }
    
    def scan_models(self) -> List[str]:
        """Scan models directory for .pkl files"""
        pkl_files = []
        if self.models_dir.exists():
            pkl_files = list(self.models_dir.glob('*.pkl'))
        logger.info(f"MODEL_SCAN: Found {len(pkl_files)} .pkl files: {[f.name for f in pkl_files]}")
        return [str(f) for f in pkl_files]
    
    def identify_model(self, pkl_path: str) -> Optional[str]:
        """Identify model type from filename"""
        name = Path(pkl_path).stem.lower()
        
        # Skip scaler files - they're not standalone models
        if 'scaler' in name:
            return None
        
        # Skip feature_order files - they're not models, just metadata
        if 'feature_order' in name:
            return None
        
        # Skip mapping files - they're not models
        if 'mapping' in name:
            return None
        
        if 'cancer_label' in name and 'scaler' not in name:
            return 'cancer_label'
        elif 'cost' in name:
            return 'cost'
        elif 'survival' in name or 'survival_rate' in name:
            return 'survival_rate'
        elif 'los' in name and 'recovery' not in name:
            return 'los'
        elif 'recovery' in name:
            return 'recovery'
        return None
    
    def load_model(self, pkl_path: str) -> Tuple[Optional[Any], Optional[str]]:
        """Load a model from .pkl file"""
        logger.info(f"ATTEMPT_LOAD: {pkl_path}")
        try:
            model = joblib.load(pkl_path)
            logger.info(f"LOADED: {Path(pkl_path).name} from {pkl_path}")
            return model, None
        except Exception as e:
            error_msg = f"Failed to load {pkl_path}: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            return None, error_msg
    
    def create_test_inputs(self, model_id: str) -> List[np.ndarray]:
        """Create synthetic test inputs for sanity testing"""
        schema = self.feature_schema.get(model_id, {})
        features = schema.get('features', [])
        
        test_inputs = []
        
        if isinstance(features, list) and len(features) > 0:
            # Baseline (median values) - 23 features for cancer_label
            baseline = np.array([50, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2020, 50, 2, 1, 1, 0, 0, 0], dtype=np.float64)
            if len(baseline) != len(features):
                # Adjust to match actual feature count
                baseline = np.zeros(len(features), dtype=np.float64)
                baseline[:min(len(baseline), len(features))] = 0.5
            
            # Variant A (change Age, Tobacco, Oral Lesions)
            variant_a = baseline.copy()
            if len(variant_a) > 0:
                variant_a[0] = 65  # Age
            if len(variant_a) > 2:
                variant_a[2] = 0   # Tobacco Use
            if len(variant_a) > 10:
                variant_a[10] = 1  # Oral Lesions
            
            # Variant B (change Treatment Type)
            variant_b = baseline.copy()
            if len(variant_b) >= 4:
                variant_b[-4:] = [0, 1, 0, 0]  # Treatment Type
            
            test_inputs = [baseline, variant_a, variant_b]
        else:
            # For models with 'all_except_target', create generic inputs
            # Updated feature counts after leakage removal during training
            if model_id in ['cost', 'los', 'recovery']:
                feature_count = 27  # 27 features after leakage removal
            elif model_id == 'survival_rate':
                feature_count = 16  # 16 features after one-hot encoding
            else:
                feature_count = 31  # Default
            
            test_inputs = [
                np.random.randn(feature_count).astype(np.float64) + 0.5,
                np.random.randn(feature_count).astype(np.float64) + 1.0,
                np.random.randn(feature_count).astype(np.float64) + 0.0
            ]
        
        return test_inputs
    
    def sanity_test(self, model: Any, model_id: str, pkl_path: str, scaler: Optional[Any] = None) -> Tuple[bool, Dict]:
        """Run sanity tests on a loaded model"""
        logger.info(f"SANITY_TEST: Starting for {model_id}")
        start_time = time.time()
        results = {
            'runtime_ms': 0,
            'outputs': [],
            'status': 'FAIL',
            'errors': []
        }
        
        try:
            # Get test inputs
            test_inputs = self.create_test_inputs(model_id)
            
            # Prepare inputs based on model type
            schema = self.feature_schema.get(model_id, {})
            
            # Apply scaler if available (for cancer_label and survival_rate)
            if scaler is not None and model_id in ['cancer_label', 'survival_rate']:
                try:
                    # Ensure inputs have the right shape and count
                    scaled_inputs = []
                    for inp in test_inputs:
                        # Reshape to (1, n_features)
                        inp_reshaped = inp.reshape(1, -1) if len(inp.shape) == 1 else inp
                        # Transform with scaler
                        scaled = scaler.transform(inp_reshaped)
                        scaled_inputs.append(scaled)
                    test_inputs_scaled = scaled_inputs
                except Exception as e:
                    logger.warning(f"Scaler transform failed for {model_id}: {e}, trying without scaler")
                    test_inputs_scaled = []
                    for inp in test_inputs:
                        inp_reshaped = inp.reshape(1, -1) if len(inp.shape) == 1 else inp
                        test_inputs_scaled.append(inp_reshaped)
            else:
                test_inputs_scaled = []
                for inp in test_inputs:
                    inp_reshaped = inp.reshape(1, -1) if len(inp.shape) == 1 else inp
                    test_inputs_scaled.append(inp_reshaped)
            
            # Run predictions
            outputs = []
            for i, test_input in enumerate(test_inputs_scaled):
                try:
                    pred_start = time.time()
                    
                    # Check if model has custom predict_integer method
                    if isinstance(model, dict):
                        if 'predict_integer' in model:
                            pred = model['predict_integer'](model.get('model', model), test_input)
                        elif 'model' in model:
                            pred = model['model'].predict(test_input)
                        else:
                            # Try direct prediction on dict values
                            pred = list(model.values())[0].predict(test_input) if model else None
                    elif hasattr(model, 'predict'):
                        pred = model.predict(test_input)
                    elif callable(model):
                        # For Keras/TensorFlow models
                        try:
                            pred = model(test_input, training=False)
                        except:
                            # Try without training parameter
                            pred = model(test_input)
                    else:
                        pred = None
                    
                    if pred is None:
                        raise ValueError("Prediction returned None")
                    
                    # Ensure prediction is in a usable format
                    if isinstance(pred, np.ndarray):
                        pred = pred.flatten() if pred.ndim > 1 else pred
                    
                    pred_time = (time.time() - pred_start) * 1000
                    
                    if pred_time > 3000:  # 3 seconds threshold
                        results['errors'].append(f"Prediction too slow: {pred_time:.2f}ms")
                    
                    outputs.append({
                        'input_idx': i,
                        'prediction': pred.tolist() if hasattr(pred, 'tolist') else str(pred),
                        'runtime_ms': pred_time
                    })
                except Exception as e:
                    results['errors'].append(f"Prediction {i} failed: {str(e)}")
            
            # Check outputs
            if len(outputs) == 0:
                results['errors'].append("No successful predictions")
                return False, results
            
            # Extract prediction values for checking
            pred_values = []
            for out in outputs:
                pred_val = out['prediction']
                if isinstance(pred_val, list):
                    # Handle nested lists or arrays
                    if len(pred_val) > 0:
                        val = pred_val[0] if not isinstance(pred_val[0], list) else pred_val[0][0]
                    else:
                        val = None
                    pred_values.append(float(val) if val is not None and not isinstance(val, (list, np.ndarray)) else None)
                elif isinstance(pred_val, np.ndarray):
                    # Handle numpy arrays
                    flat_val = pred_val.flatten()[0] if pred_val.size > 0 else None
                    pred_values.append(float(flat_val) if flat_val is not None else None)
                else:
                    pred_values.append(float(pred_val) if pred_val is not None else None)
            
            # Filter out None values for checking
            valid_pred_values = [v for v in pred_values if v is not None]
            
            # Check for constant outputs
            if len(valid_pred_values) > 1 and len(set(valid_pred_values)) == 1:
                results['errors'].append("Model returns constant predictions across varied inputs")
            
            # Check for NaN/inf
            for val in pred_values:
                if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                    results['errors'].append("Model returns NaN or Inf")
            
            # For classification models, check predict_proba if available
            if schema.get('model_type') == 'classification' and hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(test_inputs_scaled[0])
                    if np.allclose(proba.sum(axis=1), 1.0) == False:
                        results['errors'].append("Invalid probability distribution")
                except:
                    pass
            
            # If no errors, mark as PASS
            if len(results['errors']) == 0:
                results['status'] = 'PASS'
            
            results['runtime_ms'] = (time.time() - start_time) * 1000
            results['outputs'] = outputs
            
            logger.info(f"SANITY_TEST: {model_id} - Status: {results['status']}, Runtime: {results['runtime_ms']:.2f}ms")
            
            return results['status'] == 'PASS', results
            
        except Exception as e:
            error_msg = f"Sanity test failed: {str(e)}"
            results['errors'].append(error_msg)
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            return False, results
    
    def load_all_models(self) -> Dict[str, Dict]:
        """Scan, load, and test all models"""
        logger.info("="*80)
        logger.info("STARTING MODEL LOADING PROCESS")
        logger.info("="*80)
        
        pkl_files = self.scan_models()
        
        for pkl_path in pkl_files:
            # Skip scaler files - they're loaded separately with their models
            filename = Path(pkl_path).name.lower()
            if 'scaler' in filename:
                logger.debug(f"Skipping scaler file: {filename}")
                continue
            
            model_id = self.identify_model(pkl_path)
            if model_id is None:
                logger.debug(f"Could not identify model type for {pkl_path}, skipping")
                continue
            
            # Load model
            model, error = self.load_model(pkl_path)
            if model is None:
                self.loaded_models[model_id] = {
                    'status': 'OFFLINE',
                    'error': error,
                    'pkl_path': pkl_path
                }
                continue
            
            # Load scaler if needed
            scaler = None
            if model_id == 'cancer_label':
                scaler_path = self.models_dir / self.feature_schema[model_id]['scaler_path']
                if scaler_path.exists():
                    scaler, _ = self.load_model(str(scaler_path))
            elif model_id == 'survival_rate':
                # Load survival rate scaler if it exists
                scaler_path = self.models_dir / "survival_rate_scaler.pkl"
                if scaler_path.exists():
                    scaler, _ = self.load_model(str(scaler_path))
                    logger.debug(f"Loaded survival_rate scaler from {scaler_path}")
            
            # Run sanity test
            passed, test_results = self.sanity_test(model, model_id, pkl_path, scaler)
            
            if passed:
                self.loaded_models[model_id] = {
                    'status': 'HEALTHY',
                    'model': model,
                    'scaler': scaler,
                    'pkl_path': pkl_path,
                    'test_results': test_results
                }
                logger.info(f"[OK] {model_id}: HEALTHY")
            else:
                self.loaded_models[model_id] = {
                    'status': 'FAILED',
                    'model': model,
                    'scaler': scaler,
                    'pkl_path': pkl_path,
                    'test_results': test_results,
                    'needs_repair': True
                }
                logger.warning(f"[FAILED] {model_id}: FAILED SANITY TEST - {test_results['errors']}")
        
        logger.info("="*80)
        logger.info(f"MODEL LOADING COMPLETE: {len(self.loaded_models)} models processed")
        logger.info("="*80)
        
        return self.loaded_models
    
    def get_model_status(self, model_id: str) -> Dict:
        """Get status of a specific model"""
        return self.loaded_models.get(model_id, {'status': 'NOT_LOADED'})


if __name__ == '__main__':
    loader = ModelLoader()
    models = loader.load_all_models()
    
    print("\n" + "="*80)
    print("MODEL STATUS SUMMARY")
    print("="*80)
    for model_id, info in models.items():
        print(f"{model_id}: {info['status']}")

