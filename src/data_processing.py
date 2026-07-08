# src/data_processing.py
from datetime import date, timedelta
import pandas as pd
from src.meteo import get_airconditions, get_archive_meteo, get_fve_data_utc_day, get_forecast_meteo
from typing import Tuple, Optional
import numpy as np
import pvlib
import os

USERNAME = os.environ.get("USERNAME")
PASSWORD = os.environ.get("PASSWORD")

LATITUDE=49.13114
LONGITUDE=15.18067

def get_meteo_dataset(predicted_day: date) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[list]]:
    today = date.today()
    fve_target = None
    prev_predicted_day = predicted_day - timedelta(days=1)

    if predicted_day > today:
        return None
    
    elif predicted_day == today:
        future_meteo = get_forecast_meteo(start_date=predicted_day, end_date=predicted_day)
    else:
        future_meteo = get_archive_meteo(start_date=predicted_day, end_date=predicted_day)
        fve_target = get_fve_data_utc_day(day=predicted_day, username=USERNAME, password=PASSWORD)

    if fve_target is not None and 'energy' in fve_target.columns:
        fve_target = fve_target.sort_values('time')['energy'].tolist()

    past_meteo = get_archive_meteo(start_date=prev_predicted_day, end_date=prev_predicted_day)
    fve_history = get_fve_data_utc_day(day=prev_predicted_day, username=USERNAME, password=PASSWORD)

    air_conditions_predicted_day = get_airconditions(start_date=predicted_day, end_date=predicted_day)
    air_conditions_prev = get_airconditions(start_date=prev_predicted_day, end_date=prev_predicted_day)

    past_dataset = pd.merge(past_meteo, fve_history, on="time", how="inner")
    past_dataset = pd.merge(past_dataset, air_conditions_prev, on="time", how="inner")

    past_dataset = add_harmonic_variables(past_dataset)
    past_dataset = add_wind_vectors(past_dataset)
    past_dataset = add_solar_position(past_dataset, LATITUDE, LONGITUDE)
    
    future_meteo = future_meteo.merge(air_conditions_predicted_day, on="time", how="inner")

    if future_meteo is not None and not future_meteo.empty:
        future_meteo = add_harmonic_variables(future_meteo)
        future_meteo = add_wind_vectors(future_meteo)
        future_meteo = add_solar_position(future_meteo, LATITUDE, LONGITUDE)

    return past_dataset, future_meteo, fve_target

def add_harmonic_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])

    hour = df["time"].dt.hour
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24)

    day_of_year = df["time"].dt.dayofyear
    df["sin_day_of_year"] = np.sin(2 * np.pi * day_of_year / 365)
    df["cos_day_of_year"] = np.cos(2 * np.pi * day_of_year / 365)

    return df


def add_wind_vectors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    speed_col = "wind_vel" if "wind_vel" in df.columns else "wind_speed_10m"
    dir_col = "wind_direction_10m"
    
    if speed_col in df.columns and dir_col in df.columns:
        df["wind_u"] = df[speed_col] * np.cos(np.radians(df[dir_col].astype(float)))
        df["wind_v"] = df[speed_col] * np.sin(np.radians(df[dir_col].astype(float)))
    
    return df


def add_solar_position(df: pd.DataFrame, latitude: float, longitude: float) -> pd.DataFrame:
    df = df.copy()
    df = df.set_index('time')
    
    solpos = pvlib.solarposition.get_solarposition(
        time=df.index, 
        latitude=latitude,
        longitude=longitude
    )

    df['solar_elevation'] = solpos['elevation']
    
    df = df.reset_index()
    return df