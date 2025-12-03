# Fly.io Deployment Guide

## ✅ Project Structure

The project is now configured for Fly.io deployment with the following structure:

```
/render-ready
    app.py                    # Main Flask application
    model_loader.py           # Model loading utilities
    dashboard_helper.py       # Dashboard data helpers
    requirements.txt          # Python dependencies
    Dockerfile               # Docker configuration
    fly.toml                 # Fly.io configuration
    Procfile                 # Process file for gunicorn
    .dockerignore            # Docker ignore patterns
    /models/
        /models pkl/         # ML model pickle files
    /data/
        oral_cancer_prediction_dataset.csv
    /templates/              # HTML templates
    /static/                 # CSS, JS, uploads
```

## 🚀 Deployment Steps

### 1. Install Fly CLI

```bash
# Windows (PowerShell)
iwr https://fly.io/install.ps1 -useb | iex

# Mac/Linux
curl -L https://fly.io/install.sh | sh
```

### 2. Login to Fly.io

```bash
fly auth login
```

### 3. Initialize Fly.io App (if not already done)

```bash
fly launch
```

Or if you already have an app:

```bash
fly apps create oral-cancer-prediction
```

### 4. Update fly.toml App Name

Edit `fly.toml` and set the `app` name to your desired app name:

```toml
app = "your-app-name"
```

### 5. Deploy to Fly.io

```bash
fly deploy
```

### 6. Check Deployment Status

```bash
fly status
fly logs
```

## 🔧 Configuration Details

### Port Configuration
- Internal port: **8080** (configured in fly.toml)
- The app automatically uses PORT environment variable

### Memory & Resources
- **512MB RAM** (configured in fly.toml)
- **1 shared CPU**
- Auto-start/stop machines enabled

### Model Loading Optimization
- Models load in **background thread** to prevent startup timeout
- Models are cached after first load
- Dataset and charts are pre-loaded in background

### Health Checks
- Health check endpoint: `/api/model_status`
- Interval: 30 seconds
- Timeout: 5 seconds
- Grace period: 10 seconds

## 📝 Important Notes

### Path Configuration
All paths are now **relative** using `Path(__file__).resolve().parent.parent`:
- Models: `BASE_DIR / "models" / "models pkl"`
- Data: `BASE_DIR / "data" / "oral_cancer_prediction_dataset.csv"`
- No absolute Windows paths

### Model Loading
- Models load in background thread to prevent Fly.io startup timeout
- First request may wait up to 30 seconds for models to load
- Subsequent requests use cached models

### Logging
- Logs go to stdout (Fly.io captures these)
- Log level: INFO (reduced from DEBUG for production)

### Memory Considerations
- TensorFlow models can be memory-intensive
- If you encounter memory issues, consider:
  - Upgrading to 1GB RAM in fly.toml
  - Using lighter model alternatives
  - Optimizing model loading

## 🐛 Troubleshooting

### Check Logs
```bash
fly logs
```

### SSH into Container
```bash
fly ssh console
```

### Check App Status
```bash
fly status
fly info
```

### Restart App
```bash
fly apps restart oral-cancer-prediction
```

### View Metrics
```bash
fly metrics
```

## 🔄 Updating the Deployment

1. Make changes to your code
2. Test locally (see below)
3. Deploy:
   ```bash
   fly deploy
   ```

## 🧪 Testing Locally with Docker

### Build Docker Image
```bash
docker build -t oral-cancer-app .
```

### Run Container
```bash
docker run -p 8080:8080 oral-cancer-app
```

### Test in Browser
Open: http://localhost:8080

## 📊 Monitoring

### View Real-time Logs
```bash
fly logs
```

### Check App Health
```bash
curl https://your-app-name.fly.dev/api/model_status
```

### View Metrics
Visit: https://fly.io/dashboard

## 🔐 Environment Variables

You can set environment variables in fly.toml or via CLI:

```bash
fly secrets set SECRET_KEY=your-secret-key
```

Current environment variables:
- `PORT=8080` (set in fly.toml)
- `PYTHONUNBUFFERED=1` (set in fly.toml)

## ✅ Deployment Checklist

- [x] All paths are relative (no absolute Windows paths)
- [x] Models load in background thread
- [x] Dockerfile configured for production
- [x] fly.toml configured with correct port
- [x] Procfile uses gunicorn
- [x] Health checks configured
- [x] Logging to stdout only
- [x] Memory optimized (512MB)
- [x] Timeout prevention (background loading)

## 🎯 Next Steps

1. **Deploy**: Run `fly deploy`
2. **Monitor**: Watch logs with `fly logs`
3. **Test**: Visit your app URL
4. **Scale**: Adjust resources in fly.toml if needed

## 📞 Support

If you encounter issues:
1. Check `fly logs` for errors
2. Verify all model files are present
3. Check memory usage: `fly metrics`
4. Review Fly.io status: https://status.fly.io

---

**Ready to deploy!** 🚀

