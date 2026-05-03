# EUR/USD Daily Price Direction Predictor
## End-to-End MLOps Project

### Project Structure
```
eurusd_predictor/
├── configs/
│   └── config.yaml              # All project configuration
├── data/
│   └── fetch_data.py            # Data fetching & preprocessing
├── features/
│   └── feature_engineering.py  # Time series feature engineering
├── models/
│   └── model_registry.py        # All model definitions (sklearn + pytorch)
├── training/
│   ├── search_models.py         # Model search with MLflow tracking
│   ├── hyperparameter_tuning.py # Optuna HPO for best model
│   └── train_final.py           # Final training, validation & saving
├── utils/
│   ├── mlflow_utils.py          # MLflow/S3 setup helpers
│   └── plotting.py              # Validation & diagnostic plots
├── requirements.txt
└── README.md
```

### Pipeline Steps (run in order)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your settings
#    Edit configs/config.yaml — set your S3 bucket, MLflow URI, etc.

# 3. Fetch & preprocess data
python data/fetch_data.py

# 4. Run model search (finds best algorithm)
python training/search_models.py

# 5. Hyperparameter tuning on best model
#    First check MLflow UI to find your best model name, then:
python training/hyperparameter_tuning.py --model <best_model_name>

# 6. Train final model, validate & save to MLflow/S3
python training/train_final.py --model <best_model_name>
```

### Signal Classes
| Label | Signal | Meaning |
|-------|--------|---------|
| 0 | Strong Sell | Large downward move expected |
| 1 | Sell | Small downward move expected |
| 2 | Hold | Price roughly unchanged |
| 3 | Buy | Small upward move expected |
| 4 | Strong Buy | Large upward move expected |

### MLOps Stack
- **Experiment Tracking**: MLflow hosted on AWS S3
- **Model Search**: Scikit-learn models + PyTorch LSTM/Transformer
- **HPO**: Optuna with Bayesian optimization
- **Data Source**: yfinance (free, no API key needed)
- **Artifacts**: Models, plots, metrics all logged to MLflow

### Notes for Portfolio
- All experiments are tracked and reproducible
- Model is retrained on a rolling basis (add scheduler for production)
- Add a feature store (e.g. Feast) to take this further
- Add a REST API (FastAPI) + Docker to serve the model
- Add data drift monitoring (Evidently AI) for production
