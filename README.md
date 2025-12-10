# 🦷 Oral Cancer Prediction System

A comprehensive machine learning web application for predicting oral cancer diagnosis, treatment costs, survival rates, length of stay, and recovery time. Built with Flask and powered by multiple ML models including Logistic Regression, XGBoost, and Neural Networks.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents

- [Features](#-features)
- [Models](#-models)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Project Structure](#-project-structure)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- **Multi-Model Predictions**: Five specialized ML models for comprehensive cancer analysis
- **Interactive Dashboard**: Real-time data visualization with Plotly charts
- **Data Analytics**: Advanced filtering and statistical analysis
- **RESTful API**: Clean API endpoints for all prediction models
- **Model Health Monitoring**: Real-time model status and health checks
- **Responsive Design**: Modern, user-friendly web interface
- **Production Ready**: Optimized for deployment with Docker and cloud platforms

## 🤖 Models

The system includes five trained machine learning models:

1. **Cancer Label Classifier** (Logistic Regression)
   - Predicts oral cancer diagnosis (Yes/No)
   - Input: 23 features including demographics, risk factors, and symptoms
   - Output: Binary classification with confidence scores

2. **Treatment Cost Predictor** (Regression)
   - Estimates treatment cost in USD
   - Input: 27 features including patient characteristics and treatment type
   - Output: Cost prediction in USD

3. **Survival Rate Predictor** (Neural Network)
   - Predicts 5-year survival rate percentage
   - Input: 16 features after feature engineering
   - Output: Survival rate percentage (0-100%)

4. **Length of Stay (LOS) Predictor** (Regression)
   - Predicts hospital stay duration in days
   - Input: 27 features
   - Output: Number of days

5. **Recovery Time Predictor** (Regression)
   - Predicts recovery period in days
   - Input: 27 features
   - Output: Number of recovery days

## 🛠 Tech Stack

### Backend
- **Flask 3.0.0** - Web framework
- **scikit-learn 1.3.0** - Machine learning models
- **TensorFlow 2.13.0** - Deep learning for survival rate prediction
- **XGBoost 2.0.0** - Gradient boosting models
- **Pandas 2.0.3** - Data manipulation
- **NumPy 1.24.3** - Numerical computing
- **Joblib 1.3.2** - Model serialization

### Frontend
- **HTML5/CSS3** - Structure and styling
- **JavaScript** - Interactive features
- **Plotly 5.17.0** - Interactive data visualizations
- **Chart.js** - Additional charting capabilities

### Deployment
- **Docker** - Containerization
- **Gunicorn 21.2.0** - WSGI HTTP server
- **Fly.io** - Cloud deployment platform

## 📦 Installation

### Prerequisites

- Python 3.8 or higher (Python 3.11 recommended)
- pip (Python package manager)
- Git (optional, for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd render-ready
```

### Step 2: Create Virtual Environment (Recommended)

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** TensorFlow installation may take several minutes. Ensure you have a stable internet connection.

### Step 4: Verify Model Files

Ensure all model files are present in `models/models pkl/`:
- `cancer_label_best_model1_logisticregression.pkl`
- `oral_cancer_cost_of_treatment_model.pkl`
- `Final_Survival_Rate.pkl`
- `oral_cancer_los_model_integer.pkl`
- `oral_cancer_recovery_model_integer.pkl`
- Supporting scalers and feature order files

## 🚀 Usage

### Running the Application

#### Method 1: Direct Execution

```bash
python app.py
```

#### Method 2: Using Flask CLI

```bash
flask run --host=0.0.0.0 --port=8080
```

### Accessing the Application

Once running, open your browser and navigate to:

```
http://localhost:8080
```

### Available Pages

- **Home**: `http://localhost:8080/` - Overview and model status
- **Predictions**: `http://localhost:8080/predict` - Make predictions
- **Dashboard**: `http://localhost:8080/dashboard` - Interactive analytics dashboard
- **Visualizations**: `http://localhost:8080/visualizations` - Data visualizations
- **About**: `http://localhost:8080/about` - Project information
- **Contact**: `http://localhost:8080/contact` - Contact information

### Development Mode

To run in development mode with auto-reload:

**Windows (PowerShell):**
```powershell
$env:FLASK_ENV="development"
python app.py
```

**Mac/Linux:**
```bash
export FLASK_ENV=development
python app.py
```

## 🔌 API Endpoints

### Prediction Endpoints

#### 1. Cancer Label Prediction
```http
POST /api/predict/cancer_label
Content-Type: application/json

{
  "Age": 55,
  "Gender": "Male",
  "Tobacco Use": "Yes",
  "Alcohol Consumption": "Yes",
  "HPV Infection": "No",
  "Tumor Size (cm)": 2.5,
  "Cancer Stage": 2,
  ...
}
```

**Response:**
```json
{
  "prediction": 1,
  "prediction_label": "Cancer Detected",
  "confidence": 0.85,
  "probabilities": {
    "no_cancer": 0.15,
    "cancer": 0.85
  }
}
```

#### 2. Treatment Cost Prediction
```http
POST /api/predict/cost
```

**Response:**
```json
{
  "prediction": 45000.50,
  "prediction_usd": "$45,000.50",
  "unit": "USD"
}
```

#### 3. Survival Rate Prediction
```http
POST /api/predict/survival_rate
```

**Response:**
```json
{
  "prediction": 72.5,
  "prediction_percent": "72.50%",
  "unit": "percentage"
}
```

#### 4. Length of Stay Prediction
```http
POST /api/predict/los
```

**Response:**
```json
{
  "prediction": 12,
  "prediction_days": "12 days",
  "unit": "days"
}
```

#### 5. Recovery Time Prediction
```http
POST /api/predict/recovery
```

**Response:**
```json
{
  "prediction": 45,
  "prediction_days": "45 days",
  "unit": "days"
}
```

### Utility Endpoints

#### Model Status
```http
GET /api/model_status
```

**Response:**
```json
{
  "cancer_label": {
    "status": "HEALTHY",
    "pkl_path": "models/models pkl/cancer_label_best_model1_logisticregression.pkl",
    "test_results": {...}
  },
  ...
}
```

#### Dashboard Filter
```http
POST /api/dashboard/filter
Content-Type: application/json

{
  "age_groups": ["40-59", "60-79"],
  "genders": ["Male"],
  "countries": ["USA"],
  "stages": [1, 2],
  "treatments": ["Surgery"],
  "years": [2020, 2023]
}
```

## 📁 Project Structure

```
render-ready/
├── app.py                      # Main Flask application
├── model_loader.py             # Model loading and management
├── dashboard_helper.py         # Dashboard data processing
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── fly.toml                    # Fly.io deployment config
├── Procfile                    # Process configuration
├── runtime.txt                 # Python version specification
├── render.yaml                 # Render deployment config
│
├── models/
│   └── models pkl/            # Trained model files (.pkl)
│       ├── cancer_label_best_model1_logisticregression.pkl
│       ├── oral_cancer_cost_of_treatment_model.pkl
│       ├── Final_Survival_Rate.pkl
│       ├── oral_cancer_los_model_integer.pkl
│       ├── oral_cancer_recovery_model_integer.pkl
│       └── [scalers and feature order files]
│
├── data/
│   └── oral_cancer_prediction_dataset.csv
│
├── templates/                  # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── dashboard.html
│   ├── visualizations.html
│   ├── about.html
│   └── contact.html
│
└── static/                     # Static files
    ├── css/
    │   └── style.css
    ├── js/
    │   └── main.js
    └── uploads/
        └── presentation/
```

## 🚢 Deployment

### Docker Deployment

#### Build Docker Image
```bash
docker build -t oral-cancer-app .
```

#### Run Container
```bash
docker run -p 8080:8080 oral-cancer-app
```

### Fly.io Deployment

1. **Install Fly CLI**
   ```bash
   # Windows (PowerShell)
   iwr https://fly.io/install.ps1 -useb | iex
   ```

2. **Login**
   ```bash
   fly auth login
   ```

3. **Deploy**
   ```bash
   fly deploy
   ```

4. **Monitor**
   ```bash
   fly logs
   fly status
   ```

### Render Deployment

The project includes `render.yaml` for easy deployment on Render.com. Simply connect your repository and Render will automatically detect the configuration.

## 🔧 Configuration

### Environment Variables

- `PORT`: Server port (default: 8080)
- `FLASK_ENV`: Flask environment (`development` or `production`)

### Model Loading

Models are loaded in a background thread to prevent startup timeouts. The first request may take up to 30 seconds while models initialize.

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   ```bash
   pip install -r requirements.txt
   ```

2. **Port Already in Use**
   - Change port in `app.py` or kill the process using the port

3. **Models Not Loading**
   - Verify model files exist in `models/models pkl/`
   - Check logs for specific errors
   - Ensure sufficient memory (TensorFlow requires significant RAM)

4. **TensorFlow Installation Issues**
   ```bash
   pip install --upgrade pip
   pip install tensorflow==2.13.0
   ```

5. **Memory Errors**
   - Reduce number of Gunicorn workers in production
   - Increase available system memory

## 📊 Performance

- **Model Loading**: ~30-60 seconds on first startup
- **Prediction Latency**: < 100ms per prediction
- **Dashboard Load Time**: < 2 seconds (with caching)
- **Memory Usage**: ~500MB - 1GB (depending on models)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **DEPI Data Science Team** - Initial work

## 🙏 Acknowledgments

- Dataset providers and medical research community
- Open-source ML library maintainers
- Flask and Python community

## 📞 Support

For issues, questions, or contributions, please open an issue on the GitHub repository.

---

**Note**: This application is for educational and research purposes. Medical predictions should always be verified by qualified healthcare professionals.

