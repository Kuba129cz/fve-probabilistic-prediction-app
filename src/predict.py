import json
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from src.preprocessing import Preprocessor
from src.models.model_attention import Model

def predict(x_past: pd.DataFrame, x_future: pd.DataFrame):
    artifacts_dir = Path(__file__).resolve().parent.parent / "artifacts"
    
    weights_dir = artifacts_dir / "model_weights"
    with open(weights_dir / "args.json", "r") as f:
        args_dict = json.load(f)
    args = argparse.Namespace(**args_dict)
    
    base_path = artifacts_dir / "scalers"
    preprocessor = Preprocessor.from_saved(str(base_path))
    
    x_past['temperature'] = x_past['tmp_amb']

    x_past.drop(columns=['temperature_2m', 'wind_speed_10m', 'wind_direction_10m', 'int_sol_irr', 'wind_vel', 'tmp_amb', 'time'], inplace=True, errors='ignore')
    x_future.drop(columns=['wind_speed_10m', 'wind_direction_10m', 'time', 'energy'], inplace=True, errors='ignore')

    x_past.rename(columns={
        'cloud_cover': 'cloud_cover.total',
        'relative_humidity_2m': 'humidity',
        'surface_pressure': 'pressure',
        'shortwave_radiation': 'irradiance',
        'pm10': 'openmeteo_pm10',
    }, inplace=True)

    x_future.rename(columns={
        'temperature_2m': 'temperature',
        'cloud_cover': 'cloud_cover.total',
        'relative_humidity_2m': 'humidity',
        'surface_pressure': 'pressure',
        'shortwave_radiation': 'irradiance',
        'pm10': 'openmeteo_pm10',
    }, inplace=True)
    
    x_future['energy'] = 0.0
    x_future['tmp_module'] = 0.0

    x_past_scaled = preprocessor.transform(x_past)
    x_future_scaled = preprocessor.transform(x_future)

    x_past_scaled.drop(columns=["temperature"], inplace=True)
    x_future_scaled.drop(columns=["energy", 'tmp_module'], inplace=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    past_features = args.lookback_cols  
    future_features = args.horizon_cols

    past_tensor = torch.tensor(x_past_scaled[past_features].values, dtype=torch.float32).unsqueeze(0).to(device)
    future_tensor = torch.tensor(x_future_scaled[future_features].values, dtype=torch.float32).unsqueeze(0).to(device)

    all_predictions = []
    
    for i in range(1, 6):
        model = Model(args=args).to(device)
        model_path = weights_dir / f"best_model_probability_preds_{i}.pth"
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() 
        
        with torch.no_grad():
            preds = model(past_tensor, future_tensor)
            all_predictions.append(preds.cpu()) 
            
    stacked_preds = torch.stack(all_predictions)
    ensemble_mean = torch.mean(stacked_preds, dim=0)
    
    final_prediction_scaled = ensemble_mean.squeeze(0).numpy()
    
    final_prediction_scaled = preprocessor.inverse_transform_targets(final_prediction_scaled)
    final_prediction_scaled = np.clip(final_prediction_scaled, a_min=0, a_max=None)
    
    return final_prediction_scaled