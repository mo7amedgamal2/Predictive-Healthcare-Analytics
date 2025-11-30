"""
Flask Web Application for Oral Cancer Prediction Models
"""
from flask import Flask, render_template, request, jsonify
import logging
import sys
from model_loader import ModelLoader
from dashboard_helper import load_dashboard_data, filter_data, get_dashboard_charts, get_visualization_charts
import numpy as np
import pandas as pd
import traceback
import warnings
import threading
import joblib
from pathlib import Path
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
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'oral-cancer-prediction-secret-key'

# Initialize model loader
logger.info("Initializing model loader...")
model_loader = ModelLoader(models_dir="models/models pkl")

# Load Treatment Type encoder
treatment_encoder = None
treatment_encoder_path = Path("models/models pkl/treatment_type_encoder.pkl")
if treatment_encoder_path.exists():
    try:
        treatment_encoder = joblib.load(treatment_encoder_path)
        logger.info("Treatment Type encoder loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load Treatment Type encoder: {e}")
else:
    logger.warning("Treatment Type encoder not found. Please run scripts/create_treatment_encoder.py first.")

# Load feature order files for each model
feature_orders = {}
for model_name in ['cancer_label', 'cost', 'los', 'recovery', 'survival_rate']:
    feature_order_path = Path(f"models/models pkl/{model_name}_feature_order.pkl")
    if feature_order_path.exists():
        try:
            feature_orders[model_name] = joblib.load(feature_order_path)
            logger.info(f"Loaded feature order for {model_name}: {len(feature_orders[model_name])} features")
        except Exception as e:
            logger.warning(f"Failed to load feature order for {model_name}: {e}")

# Load all models at startup
loaded_models = {}
model_status = {}

# Cache for dataset and charts
_cached_dataset = None
_cached_dashboard_charts = None
_cached_visualization_data = None

def load_models():
    """Load all models at application startup"""
    global loaded_models, model_status
    logger.info("Loading all models at startup...")
    loaded_models = model_loader.load_all_models()
    model_status = {k: v.get('status', 'UNKNOWN') for k, v in loaded_models.items()}
    logger.info(f"Models loaded: {model_status}")

def get_cached_dataset():
    """Get cached dataset, load if not cached"""
    global _cached_dataset
    if _cached_dataset is None:
        logger.info("Loading dataset into cache...")
        _cached_dataset = load_dashboard_data()
        logger.info(f"Dataset cached: {_cached_dataset.shape}")
    return _cached_dataset


def get_cached_dashboard_charts():
    """Get cached dashboard charts, generate if not cached"""
    global _cached_dashboard_charts
    if _cached_dashboard_charts is None:
        logger.info("Generating dashboard charts into cache...")
        df = get_cached_dataset()
        _cached_dashboard_charts = get_dashboard_charts(df)
        logger.info("Dashboard charts cached")
    return _cached_dashboard_charts

# Load models at startup (before first request)
load_models()

# Pre-load dataset and initial charts in background
def initialize_cache():
    """Pre-load dataset and charts on startup"""
    try:
        logger.info("Pre-loading dataset and charts...")
        get_cached_dataset()
        get_cached_dashboard_charts()
        logger.info("Cache initialization complete")
    except Exception as e:
        logger.warning(f"Cache initialization failed: {e}")

# Initialize cache in background thread to not block startup
cache_thread = threading.Thread(target=initialize_cache, daemon=True)
cache_thread.start()

@app.before_request
def ensure_models_loaded():
    """Ensure models are loaded before each request"""
    global loaded_models, model_status
    if not loaded_models:
        load_models()

def normalize_input_data(data: dict) -> dict:
    """Normalize and compute derived features from unified form input"""
    # Helper function to get value with default
    def get_val(key, default=0):
        # Try exact key first, then common variations
        if key in data:
            return data[key]
        # Try with underscores
        key_underscore = key.replace(' ', '_').replace('(', '').replace(')', '')
        if key_underscore in data:
            return data[key_underscore]
        return default
    
    # Helper function to convert binary Yes/No to int
    def to_binary_int(val, default=0):
        if isinstance(val, str):
            val_lower = val.lower().strip()
            if val_lower in ['yes', '1', 'true']:
                return 1
            elif val_lower in ['no', '0', 'false']:
                return 0
        try:
            return int(val)
        except (ValueError, TypeError):
            return default
    
    # Helper function to convert Gender to int
    def gender_to_int(val, default=1):
        if isinstance(val, str):
            val_lower = val.lower().strip()
            if val_lower in ['male', 'm', '1']:
                return 0
            elif val_lower in ['female', 'f', '0']:
                return 1
        try:
            return int(val)
        except (ValueError, TypeError):
            return default
    
    # Extract base features
    age = float(get_val('Age', 50))
    gender = gender_to_int(get_val('Gender', 'Male'), 0)
    tobacco_use = to_binary_int(get_val('Tobacco Use', 'No'), 0)
    alcohol_consumption = to_binary_int(get_val('Alcohol Consumption', 'No'), 0)
    hpv_infection = to_binary_int(get_val('HPV Infection', 'No'), 0)
    betel_quid_use = to_binary_int(get_val('Betel Quid Use', 'No'), 0)
    chronic_sun_exposure = to_binary_int(get_val('Chronic Sun Exposure', 'No'), 0)
    poor_oral_hygiene = to_binary_int(get_val('Poor Oral Hygiene', 'No'), 0)
    family_history = to_binary_int(get_val('Family History of Cancer', 'No'), 0)
    compromised_immune = to_binary_int(get_val('Compromised Immune System', 'No'), 0)
    oral_lesions = to_binary_int(get_val('Oral Lesions', 'No'), 0)
    unexplained_bleeding = to_binary_int(get_val('Unexplained Bleeding', 'No'), 0)
    difficulty_swallowing = to_binary_int(get_val('Difficulty Swallowing', 'No'), 0)
    white_red_patches = to_binary_int(get_val('White or Red Patches in Mouth', 'No'), 0)
    tumor_size = float(get_val('Tumor Size (cm)', 2.0))
    cancer_stage = float(get_val('Cancer Stage', 1))
    early_diagnosis = to_binary_int(get_val('Early Diagnosis', 'Yes'), 1)
    year_of_diagnosis = int(get_val('Year_of_Diagnosis', 2023))
    
    # Get diet intake and encode it
    diet_intake = get_val('Diet (Fruits & Vegetables Intake)', 'Moderate')
    if isinstance(diet_intake, str):
        diet_encoded = {'Low': 0, 'Moderate': 1, 'High': 2}.get(diet_intake, 1)
    else:
        diet_encoded = int(diet_intake) if diet_intake in [0, 1, 2] else 1
    
    # Get treatment type and one-hot encode it using the saved encoder
    treatment_type = str(get_val('Treatment Type', 'Surgery'))
    
    # Use encoder if available, otherwise fall back to manual encoding
    if treatment_encoder is not None:
        try:
            # Transform using encoder
            treatment_array = np.array([[treatment_type]])
            encoded = treatment_encoder.transform(treatment_array)[0]
            feature_names = treatment_encoder.get_feature_names_out(['Treatment Type'])
            
            # Create dictionary from encoded values
            treatment_dict = {name: float(val) for name, val in zip(feature_names, encoded)}
            
            # Map to expected column names (matching pandas get_dummies format)
            treatment_no = treatment_dict.get('Treatment Type_No Treatment', 0.0)
            treatment_radiation = treatment_dict.get('Treatment Type_Radiation', 0.0)
            treatment_surgery = treatment_dict.get('Treatment Type_Surgery', 0.0)
            treatment_targeted = treatment_dict.get('Treatment Type_Targeted Therapy', 0.0)
            treatment_chemo = treatment_dict.get('Treatment Type_Chemotherapy', 0.0)
        except Exception as e:
            logger.warning(f"Encoder transform failed for '{treatment_type}': {e}, using fallback")
            # Fallback to manual encoding
            treatment_no = 1 if treatment_type == 'No Treatment' else 0
            treatment_radiation = 1 if treatment_type == 'Radiation' else 0
            treatment_surgery = 1 if treatment_type == 'Surgery' else 0
            treatment_targeted = 1 if treatment_type == 'Targeted Therapy' else 0
            treatment_chemo = 1 if treatment_type == 'Chemotherapy' else 0
    else:
        # Fallback to manual encoding if encoder not loaded
        treatment_no = 1 if treatment_type == 'No Treatment' else 0
        treatment_radiation = 1 if treatment_type == 'Radiation' else 0
        treatment_surgery = 1 if treatment_type == 'Surgery' else 0
        treatment_targeted = 1 if treatment_type == 'Targeted Therapy' else 0
        treatment_chemo = 1 if treatment_type == 'Chemotherapy' else 0
    
    # Compute derived features
    age_x_tobacco = age * tobacco_use
    tumor_size_x_stage = tumor_size * cancer_stage
    
    # Compute age group encoded
    if age < 20:
        age_group_encoded = 0  # '0-19'
    elif age < 40:
        age_group_encoded = 1  # '20-39'
    elif age < 60:
        age_group_encoded = 2  # '40-59'
    elif age < 80:
        age_group_encoded = 3  # '60-79'
    else:
        age_group_encoded = 4  # '80+'
    
    # Build normalized data dictionary
    normalized = {
        'Age': age,
        'Gender': gender,
        'Tobacco Use': tobacco_use,
        'Alcohol Consumption': alcohol_consumption,
        'HPV Infection': hpv_infection,
        'Betel Quid Use': betel_quid_use,
        'Chronic Sun Exposure': chronic_sun_exposure,
        'Poor Oral Hygiene': poor_oral_hygiene,
        'Family History of Cancer': family_history,
        'Compromised Immune System': compromised_immune,
        'Oral Lesions': oral_lesions,
        'Unexplained Bleeding': unexplained_bleeding,
        'Difficulty Swallowing': difficulty_swallowing,
        'White or Red Patches in Mouth': white_red_patches,
        'Tumor Size (cm)': tumor_size,
        'Cancer Stage': cancer_stage,
        'Early Diagnosis': early_diagnosis,
        'Year_of_Diagnosis': year_of_diagnosis,
        'Age_x_Tobacco': age_x_tobacco,
        'Age_Group_Encoded': age_group_encoded,
        'Diet_Encoded': diet_encoded,
        'Tumor_Size_x_Stage': tumor_size_x_stage,
        'Treatment Type_No Treatment': treatment_no,
        'Treatment Type_Radiation': treatment_radiation,
        'Treatment Type_Surgery': treatment_surgery,
        'Treatment Type_Targeted Therapy': treatment_targeted,
        'Treatment Type_Chemotherapy': treatment_chemo,
    }
    
    return normalized

def prepare_cancer_label_input(data: dict) -> np.ndarray:
    """Prepare input for cancer label prediction using saved feature order"""
    # Normalize input data
    normalized = normalize_input_data(data)
    
    # Use saved feature order if available, otherwise use hardcoded order
    if 'cancer_label' in feature_orders:
        feature_order = feature_orders['cancer_label']
        features = [normalized.get(col, 0.0) for col in feature_order]
    else:
        # Fallback to hardcoded order (23 features, NO Cancer Stage or Tumor Size)
        features = [
            normalized['Age'],
            normalized['Gender'],
            normalized['Tobacco Use'],
            normalized['Alcohol Consumption'],
            normalized['HPV Infection'],
            normalized['Betel Quid Use'],
            normalized['Chronic Sun Exposure'],
            normalized['Poor Oral Hygiene'],
            normalized['Family History of Cancer'],
            normalized['Compromised Immune System'],
            normalized['Oral Lesions'],
            normalized['Unexplained Bleeding'],
            normalized['Difficulty Swallowing'],
            normalized['White or Red Patches in Mouth'],
            normalized['Early Diagnosis'],
            normalized['Year_of_Diagnosis'],
            normalized['Age_x_Tobacco'],
            normalized['Age_Group_Encoded'],
            normalized['Diet_Encoded'],
            normalized['Treatment Type_No Treatment'],
            normalized['Treatment Type_Radiation'],
            normalized['Treatment Type_Surgery'],
            normalized['Treatment Type_Targeted Therapy']
        ]
    
    return np.array(features, dtype=np.float64).reshape(1, -1)

def prepare_regression_input(data: dict, feature_count: int = 27, model_name: str = 'cost') -> np.ndarray:
    """Prepare input for regression models using saved feature order"""
    # Normalize input data
    normalized = normalize_input_data(data)
    
    # Use saved feature order if available
    if model_name in feature_orders:
        feature_order = feature_orders[model_name]
        features = [normalized.get(col, 0.0) for col in feature_order]
    else:
        # Fallback to hardcoded order (27 features)
        features = [
            normalized['Age'],
            normalized['Gender'],
            normalized['Tobacco Use'],
            normalized['Alcohol Consumption'],
            normalized['HPV Infection'],
            normalized['Betel Quid Use'],
            normalized['Chronic Sun Exposure'],
            normalized['Poor Oral Hygiene'],
            normalized['Family History of Cancer'],
            normalized['Compromised Immune System'],
            normalized['Oral Lesions'],
            normalized['Unexplained Bleeding'],
            normalized['Difficulty Swallowing'],
            normalized['White or Red Patches in Mouth'],
            normalized['Tumor Size (cm)'],
            normalized['Cancer Stage'],
            normalized['Early Diagnosis'],
            normalized['Year_of_Diagnosis'],
            normalized['Tumor_Size_x_Stage'],
            normalized['Age_x_Tobacco'],
            normalized['Age_Group_Encoded'],
            normalized['Diet_Encoded'],
            normalized['Treatment Type_No Treatment'],
            normalized['Treatment Type_Radiation'],
            normalized['Treatment Type_Surgery'],
            normalized['Treatment Type_Targeted Therapy'],
            normalized['Treatment Type_Chemotherapy']  # 27th feature
        ]
        
        # Ensure exactly feature_count features
        if len(features) != feature_count:
            if len(features) > feature_count:
                features = features[:feature_count]
            else:
                while len(features) < feature_count:
                    features.append(0.0)
    
    return np.array(features, dtype=np.float64).reshape(1, -1)

def prepare_survival_rate_input(data: dict) -> np.ndarray:
    """Prepare input for survival rate prediction (16 features after one-hot encoding)"""
    # Normalize input data
    normalized = normalize_input_data(data)
    
    # Survival rate model uses first 17 columns (including target "Survival Rate (5-Year, %)")
    # After dropping the target, we have 16 features
    # After pd.get_dummies with drop_first=True, we still have 16 features
    # (all binary columns stay as single columns)
    
    # The 16 features used during training (matching X_surv_encoded after get_dummies)
    # These are the 16 features from first 17 columns excluding the target
    features = [
        float(normalized['Age']),
        float(normalized['Gender']),
        float(normalized['Tobacco Use']),
        float(normalized['Alcohol Consumption']),
        float(normalized['HPV Infection']),
        float(normalized['Betel Quid Use']),
        float(normalized['Chronic Sun Exposure']),
        float(normalized['Poor Oral Hygiene']),
        float(normalized['Family History of Cancer']),
        float(normalized['Compromised Immune System']),
        float(normalized['Oral Lesions']),
        float(normalized['Unexplained Bleeding']),
        float(normalized['Difficulty Swallowing']),
        float(normalized['White or Red Patches in Mouth']),
        float(normalized['Tumor Size (cm)']),
        float(normalized['Cancer Stage']),
    ]
    
    # Ensure exactly 16 features
    if len(features) != 16:
        features = features[:16]
        while len(features) < 16:
            features.append(0.0)
    
    return np.array(features, dtype=np.float64).reshape(1, -1)

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', model_status=model_status)

@app.route('/predict')
def predict_page():
    """Predictions page"""
    return render_template('predict.html', model_status=model_status)

@app.route('/dashboard')
def dashboard():
    """Dashboard page with interactive charts"""
    try:
        # Use cached dataset
        df = get_cached_dataset()
        
        # Get filter options
        age_groups = sorted(df['Age_Group'].dropna().unique().tolist()) if 'Age_Group' in df.columns else []
        genders = sorted(df['Gender'].dropna().unique().tolist()) if 'Gender' in df.columns else []
        countries = sorted(df['Country'].dropna().unique().tolist()) if 'Country' in df.columns else []
        stages = sorted([int(s) for s in df['Cancer Stage'].dropna().unique() if pd.notna(s)]) if 'Cancer Stage' in df.columns else []
        treatments = sorted(df['Treatment Type'].dropna().unique().tolist()) if 'Treatment Type' in df.columns else []
        years = [int(df['Year_of_Diagnosis'].min()), int(df['Year_of_Diagnosis'].max())] if 'Year_of_Diagnosis' in df.columns else [2018, 2025]
        
        # Get cached initial charts (no filters applied)
        charts = get_cached_dashboard_charts()
        
        return render_template('dashboard.html', 
                             model_status=model_status,
                             charts=charts,
                             age_groups=age_groups,
                             genders=genders,
                             countries=countries,
                             stages=stages,
                             treatments=treatments,
                             years=years)
    except Exception as e:
        logger.error(f"Error loading dashboard: {e}")
        logger.debug(traceback.format_exc())
        return render_template('dashboard.html', model_status=model_status, error=str(e))

@app.route('/api/dashboard/filter', methods=['POST'])
def dashboard_filter():
    """API endpoint for filtering dashboard data"""
    try:
        data = request.json
        # Use cached dataset
        df = get_cached_dataset()
        
        # Apply filters
        filtered_df = filter_data(
            df,
            selected_age_groups=data.get('age_groups'),
            selected_genders=data.get('genders'),
            selected_countries=data.get('countries'),
            selected_stages=data.get('stages'),
            selected_treatments=data.get('treatments'),
            selected_years=data.get('years')
        )
        
        # Generate charts (only when filtering, not cached)
        charts = get_dashboard_charts(filtered_df)
        
        return jsonify(charts)
    except Exception as e:
        logger.error(f"Error filtering dashboard: {e}")
        logger.debug(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/visualizations')
def visualizations():
    """Visualizations page with Chart.js charts"""
    return render_template('visualizations.html', model_status=model_status)

@app.route('/api/visualizations/data', methods=['GET'])
def get_visualization_data():
    """API endpoint to get data for visualizations - optimized with caching"""
    try:
        global _cached_visualization_data
        
        # Return cached data if available
        if _cached_visualization_data is not None:
            logger.debug("Returning cached visualization data")
            return jsonify(_cached_visualization_data)
        
        # Use cached dataset instead of loading from file
        df = get_cached_dataset()
        
        # Create Cancer_Label if it doesn't exist
        if 'Cancer_Label' not in df.columns and 'Oral Cancer (Diagnosis)' in df.columns:
            df['Cancer_Label'] = df['Oral Cancer (Diagnosis)'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
        
        # Stats
        total_patients = len(df)
        if 'Cancer_Label' in df.columns:
            cancer_cases = int(df['Cancer_Label'].sum())
            no_cancer_cases = total_patients - cancer_cases
        elif 'Oral Cancer (Diagnosis)' in df.columns:
            cancer_cases = int((df['Oral Cancer (Diagnosis)'] == 'Yes').sum())
            no_cancer_cases = total_patients - cancer_cases
        else:
            cancer_cases = 0
            no_cancer_cases = total_patients
        
        # Diagnosis distribution
        if 'Oral Cancer (Diagnosis)' in df.columns:
            diagnosis_counts = df['Oral Cancer (Diagnosis)'].value_counts()
            diagnosis_data = {
                'labels': diagnosis_counts.index.tolist(),
                'values': diagnosis_counts.values.tolist()
            }
        else:
            diagnosis_data = {'labels': ['No', 'Yes'], 'values': [no_cancer_cases, cancer_cases]}
        
        # Age Group vs Cancer Diagnosis
        if 'Age_Group' in df.columns and 'Oral Cancer (Diagnosis)' in df.columns:
            age_cross = pd.crosstab(df['Age_Group'], df['Oral Cancer (Diagnosis)'])
            age_group_data = {
                'labels': age_cross.index.tolist(),
                'no_cancer': age_cross.get('No', pd.Series([0]*len(age_cross))).tolist(),
                'cancer': age_cross.get('Yes', pd.Series([0]*len(age_cross))).tolist()
            }
        else:
            age_group_data = {'labels': [], 'no_cancer': [], 'cancer': []}
        
        # Cancer Stage Distribution
        if 'Cancer Stage' in df.columns:
            stage_counts = df['Cancer Stage'].value_counts().sort_index()
            stage_data = {
                'labels': [f'Stage {int(x)}' for x in stage_counts.index],
                'values': stage_counts.values.tolist()
            }
        else:
            stage_data = {'labels': [], 'values': []}
        
        # Treatment Type Distribution
        if 'Treatment Type' in df.columns and 'Oral Cancer (Diagnosis)' in df.columns:
            treatment_cross = pd.crosstab(df['Treatment Type'], df['Oral Cancer (Diagnosis)'])
            treatment_data = {
                'labels': treatment_cross.index.tolist(),
                'no_cancer': treatment_cross.get('No', pd.Series([0]*len(treatment_cross))).tolist(),
                'cancer': treatment_cross.get('Yes', pd.Series([0]*len(treatment_cross))).tolist()
            }
        else:
            treatment_data = {'labels': [], 'no_cancer': [], 'cancer': []}
        
        # Treatment by Cancer Stage
        if 'Treatment Type' in df.columns and 'Cancer Stage' in df.columns:
            treatment_stage_cross = pd.crosstab(df['Cancer Stage'], df['Treatment Type'])
            treatments_list = []
            for treatment in treatment_stage_cross.columns:
                treatments_list.append({
                    'name': str(treatment),
                    'values': treatment_stage_cross[treatment].tolist()
                })
            treatment_stage_data = {
                'stages': [f'Stage {int(x)}' for x in treatment_stage_cross.index],
                'treatments': treatments_list
            }
        else:
            treatment_stage_data = {'stages': [], 'treatments': []}
        
        # Yearly Diagnosis
        if 'Year_of_Diagnosis' in df.columns and 'Oral Cancer (Diagnosis)' in df.columns:
            yearly_cross = pd.crosstab(df['Year_of_Diagnosis'], df['Oral Cancer (Diagnosis)'])
            yearly_data = {
                'labels': [str(int(x)) for x in yearly_cross.index],
                'no_cancer': yearly_cross.get('No', pd.Series([0]*len(yearly_cross))).tolist(),
                'cancer': yearly_cross.get('Yes', pd.Series([0]*len(yearly_cross))).tolist()
            }
        else:
            yearly_data = {'labels': [], 'no_cancer': [], 'cancer': []}
        
        # Country Distribution (Top 10)
        if 'Country' in df.columns and 'Oral Cancer (Diagnosis)' in df.columns:
            country_cross = pd.crosstab(df['Country'], df['Oral Cancer (Diagnosis)'])
            country_cross = country_cross.sort_values(by='Yes' if 'Yes' in country_cross.columns else country_cross.columns[0], ascending=False).head(10)
            country_data = {
                'labels': country_cross.index.tolist(),
                'no_cancer': country_cross.get('No', pd.Series([0]*len(country_cross))).tolist(),
                'cancer': country_cross.get('Yes', pd.Series([0]*len(country_cross))).tolist()
            }
        else:
            country_data = {'labels': [], 'no_cancer': [], 'cancer': []}
        
        # Tumor Size by Cancer Stage
        if 'Tumor Size (cm)' in df.columns and 'Cancer Stage' in df.columns:
            tumor_size_avg = df.groupby('Cancer Stage')['Tumor Size (cm)'].mean().sort_index()
            tumor_size_data = {
                'labels': [f'Stage {int(x)}' for x in tumor_size_avg.index],
                'values': tumor_size_avg.values.tolist()
            }
        else:
            tumor_size_data = {'labels': [], 'values': []}
        
        # Cost vs Economic Burden (sample for scatter plot - limit to 1000 points for performance)
        if 'Cost of Treatment (USD)' in df.columns and 'Economic Burden (Lost Workdays per Year)' in df.columns:
            sample_size = min(1000, len(df))
            df_sample = df.sample(n=sample_size, random_state=42)
            
            if 'Cancer_Label' in df_sample.columns:
                cancer_mask = df_sample['Cancer_Label'] == 1
            elif 'Oral Cancer (Diagnosis)' in df_sample.columns:
                cancer_mask = df_sample['Oral Cancer (Diagnosis)'] == 'Yes'
            else:
                cancer_mask = pd.Series([False] * len(df_sample))
            
            cost_burden_data = {
                'cancer': [
                    {'x': float(x), 'y': float(y)} 
                    for x, y in zip(
                        df_sample[cancer_mask]['Cost of Treatment (USD)'],
                        df_sample[cancer_mask]['Economic Burden (Lost Workdays per Year)']
                    )
                ],
                'no_cancer': [
                    {'x': float(x), 'y': float(y)} 
                    for x, y in zip(
                        df_sample[~cancer_mask]['Cost of Treatment (USD)'],
                        df_sample[~cancer_mask]['Economic Burden (Lost Workdays per Year)']
                    )
                ]
            }
        else:
            cost_burden_data = {'cancer': [], 'no_cancer': []}
        
        # Cache the result
        _cached_visualization_data = {
            'stats': {
                'total_patients': total_patients,
                'cancer_cases': cancer_cases,
                'no_cancer_cases': no_cancer_cases
            },
            'diagnosis': diagnosis_data,
            'age_group': age_group_data,
            'cancer_stage': stage_data,
            'treatment': treatment_data,
            'treatment_stage': treatment_stage_data,
            'yearly': yearly_data,
            'country': country_data,
            'tumor_size': tumor_size_data,
            'cost_burden': cost_burden_data
        }
        
        return jsonify(_cached_visualization_data)
        
    except Exception as e:
        logger.error(f"Error generating visualization data: {str(e)}")
        logger.debug(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html', model_status=model_status)

@app.route('/contact')
def contact():
    """Contact page"""
    return render_template('contact.html', model_status=model_status)

@app.route('/api/predict/cancer_label', methods=['POST'])
def predict_cancer_label():
    """API endpoint for cancer label prediction"""
    try:
        data = request.json
        
        if 'cancer_label' not in loaded_models:
            return jsonify({'error': 'Cancer Label model not available'}), 503
        
        model_info = loaded_models['cancer_label']
        if model_info['status'] != 'HEALTHY':
            return jsonify({'error': f'Model status: {model_info["status"]}'}), 503
        
        model = model_info['model']
        scaler = model_info.get('scaler')
        
        # Prepare input
        input_data = prepare_cancer_label_input(data)
        
        # Apply scaler
        if scaler is not None:
            input_data = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0] if hasattr(model, 'predict_proba') else None
        
        result = {
            'prediction': int(prediction),
            'prediction_label': 'Cancer Detected' if prediction == 1 else 'No Cancer',
            'confidence': float(proba[1]) if proba is not None else None,
            'probabilities': {
                'no_cancer': float(proba[0]) if proba is not None else None,
                'cancer': float(proba[1]) if proba is not None else None
            }
        }
        
        logger.info(f"Cancer Label Prediction: {result}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in cancer label prediction: {str(e)}")
        logger.debug(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/cost', methods=['POST'])
def predict_cost():
    """API endpoint for cost prediction"""
    try:
        data = request.json
        
        if 'cost' not in loaded_models:
            return jsonify({'error': 'Cost model not available'}), 503
        
        model_info = loaded_models['cost']
        if model_info['status'] != 'HEALTHY':
            return jsonify({'error': f'Model status: {model_info["status"]}'}), 503
        
        model = model_info['model']
        
        # Prepare input using saved feature order
        input_data = prepare_regression_input(data, feature_count=27, model_name='cost')
        
        # Predict
        prediction = model.predict(input_data)[0]
        
        result = {
            'prediction': float(prediction),
            'prediction_usd': f"${prediction:,.2f}",
            'unit': 'USD'
        }
        
        logger.info(f"Cost Prediction: {result}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in cost prediction: {str(e)}")
        logger.debug(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/survival_rate', methods=['POST'])
def predict_survival_rate():
    """API endpoint for survival rate prediction"""
    try:
        data = request.json
        
        if 'survival_rate' not in loaded_models:
            return jsonify({'error': 'Survival Rate model not available'}), 503
        
        model_info = loaded_models['survival_rate']
        if model_info['status'] != 'HEALTHY':
            return jsonify({'error': f'Model status: {model_info["status"]}'}), 503
        
        model = model_info['model']
        scaler = model_info.get('scaler')  # Get scaler if available
        
        # For survival rate, prepare input (16 features after encoding)
        input_data = prepare_survival_rate_input(data)
        
        # Apply scaler if available
        if scaler is not None:
            try:
                input_data = scaler.transform(input_data)
            except Exception as e:
                logger.warning(f"Scaler transform failed for survival_rate: {e}")
        
        # Predict - handle Keras models
        if isinstance(model, dict):
            # If model is a dict, try to get the actual model
            actual_model = model.get('model', list(model.values())[0] if model else None)
            if actual_model and hasattr(actual_model, 'predict'):
                prediction = actual_model.predict(input_data)
            elif callable(model):
                prediction = model(input_data, training=False)
            else:
                prediction = np.array([50.0])
        elif hasattr(model, 'predict'):
            prediction = model.predict(input_data)
        elif callable(model):
            # Keras/TensorFlow model - expect 16 features
            prediction = model(input_data, training=False)
        else:
            prediction = np.array([50.0])  # fallback
        
        # Normalize prediction to float
        if hasattr(prediction, 'flatten'):
            prediction = float(prediction.flatten()[0])
        elif isinstance(prediction, (list, np.ndarray)):
            prediction = float(prediction[0] if len(prediction) > 0 else 50.0)
        else:
            prediction = float(prediction)
        
        result = {
            'prediction': float(prediction),
            'prediction_percent': f"{prediction:.2f}%",
            'unit': 'percentage'
        }
        
        logger.info(f"Survival Rate Prediction: {result}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in survival rate prediction: {str(e)}")
        logger.debug(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/los', methods=['POST'])
def predict_los():
    """API endpoint for Length of Stay prediction"""
    try:
        data = request.json
        
        if 'los' not in loaded_models:
            return jsonify({'error': 'LOS model not available'}), 503
        
        model_info = loaded_models['los']
        if model_info['status'] != 'HEALTHY':
            return jsonify({'error': f'Model status: {model_info["status"]}'}), 503
        
        model = model_info['model']
        
        # Prepare input using saved feature order
        input_data = prepare_regression_input(data, feature_count=27, model_name='los')
        
        # Check for custom predict_integer
        if isinstance(model, dict) and 'predict_integer' in model:
            prediction = model['predict_integer'](model['model'], input_data)[0]
        elif hasattr(model, 'predict_integer'):
            prediction = model.predict_integer(model, input_data)[0]
        else:
            prediction = int(np.round(model.predict(input_data)[0]))
        
        result = {
            'prediction': int(prediction),
            'prediction_days': f"{prediction} days",
            'unit': 'days'
        }
        
        logger.info(f"LOS Prediction: {result}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in LOS prediction: {str(e)}")
        logger.debug(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/recovery', methods=['POST'])
def predict_recovery():
    """API endpoint for recovery days prediction"""
    try:
        data = request.json
        
        if 'recovery' not in loaded_models:
            return jsonify({'error': 'Recovery model not available'}), 503
        
        model_info = loaded_models['recovery']
        if model_info['status'] != 'HEALTHY':
            return jsonify({'error': f'Model status: {model_info["status"]}'}), 503
        
        model = model_info['model']
        
        # Prepare input using saved feature order
        input_data = prepare_regression_input(data, feature_count=27, model_name='recovery')
        
        # Check for custom predict_integer
        if isinstance(model, dict):
            if 'predict_integer' in model:
                pred_result = model['predict_integer'](model.get('model', model), input_data)
                prediction = pred_result[0] if hasattr(pred_result, '__getitem__') else pred_result
            elif 'model' in model:
                pred_result = model['model'].predict(input_data)
                prediction = int(np.round(pred_result[0] if hasattr(pred_result, '__getitem__') else pred_result))
            else:
                # Try first value in dict
                pred_result = list(model.values())[0].predict(input_data)
                prediction = int(np.round(pred_result[0] if hasattr(pred_result, '__getitem__') else pred_result))
        elif hasattr(model, 'predict_integer'):
            pred_result = model.predict_integer(model, input_data)
            prediction = pred_result[0] if hasattr(pred_result, '__getitem__') else pred_result
        elif hasattr(model, 'predict'):
            pred_result = model.predict(input_data)
            prediction = int(np.round(pred_result[0] if hasattr(pred_result, '__getitem__') else pred_result))
        else:
            # Fallback
            prediction = 30
        
        result = {
            'prediction': int(prediction),
            'prediction_days': f"{prediction} days",
            'unit': 'days'
        }
        
        logger.info(f"Recovery Prediction: {result}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in recovery prediction: {str(e)}")
        logger.debug(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_status', methods=['GET'])
def get_model_status():
    """API endpoint to get status of all models"""
    status = {}
    for model_id, info in loaded_models.items():
        status[model_id] = {
            'status': info.get('status', 'UNKNOWN'),
            'pkl_path': info.get('pkl_path', 'N/A'),
            'test_results': info.get('test_results', {})
        }
    return jsonify(status)


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    logger.info("Starting Flask application...")
    logger.info(f"Access the app at: http://0.0.0.0:{port}")
    app.run(debug=debug, host='0.0.0.0', port=port)

