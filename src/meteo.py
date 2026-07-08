import requests
import pandas as pd
import json
from datetime import date, timedelta

def get_archive_meteo(start_date: date, end_date: date, lat: int =49.13114, lon: int = 15.18067) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ("temperature_2m", "cloud_cover", "wind_speed_10m", "wind_direction_10m", "relative_humidity_2m", "surface_pressure", "shortwave_radiation"),
        "wind_speed_unit": "ms",
        "cell_selection": "land",
        "timezone": "UTC"
    }

    response = requests.get(url=url, params=params)
    if response.status_code != 200:
        return None
    
    data = response.json()

    dataset = pd.DataFrame(data["hourly"])
    dataset["time"] = pd.to_datetime(dataset["time"], format="%Y-%m-%dT%H:%M")
    dataset["time"] = dataset["time"].dt.tz_localize("UTC")
    
    return dataset

def get_airconditions(start_date, end_date, lat=49.13114, lon=15.18067) -> pd.DataFrame:
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ("pm10", "ozone"),
        "wind_speed_unit": "ms",
        "cell_selection": "land",
        "timezone": "UTC"
    }

    response = requests.get(url=url, params=params)

    if response.status_code != 200:
        return None
    
    data = response.json()
    dataset = pd.DataFrame(data["hourly"])
    dataset["time"] = pd.to_datetime(dataset["time"], format="%Y-%m-%dT%H:%M")
    dataset["time"] = dataset["time"].dt.tz_localize("UTC")    

    return dataset

def get_forecast_meteo(start_date: date, end_date: date, lat: float = 49.13114, lon: float = 15.18067) -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast" 
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ("temperature_2m", "cloud_cover", "wind_speed_10m", "wind_direction_10m", "relative_humidity_2m", "surface_pressure", "shortwave_radiation"),
        "wind_speed_unit": "ms",
        "cell_selection": "land",
        "timezone": "UTC"
    }

    response = requests.get(url=url, params=params)

    if response.status_code != 200:
        return None
    
    data = response.json()
    
    dataset = pd.DataFrame(data["hourly"])
    dataset["time"] = pd.to_datetime(dataset["time"], format="%Y-%m-%dT%H:%M")
    dataset["time"] = dataset["time"].dt.tz_localize("UTC")
    
    return dataset

def get_fve_data_for_day(day: date, username: str, password: str) -> pd.DataFrame:
    """
    Downloads data from the API via a POST request for the given day.

    :param day: Date object of type `datetime.date`
    :param username: API username
    :param password: API password
    :return: A list of measurements (dicts) for the given day
    """
    url = f"https://aba.solarmon.eu/rest-server/?q=getDataPredMod&date={day.strftime('%Y-%m-%d')}"
    payload = {
        'username': username,
        'password': password
    }

    response = requests.post(url, data=payload)
    if response.status_code != 200:
        return None
        
    raw_data = json.loads(response.json())
    dataset = pd.DataFrame(raw_data["data"])

    dataset["time"] = pd.to_datetime(dataset["time"], format="%Y-%m-%d %H:%M:%S")
    dataset["time"] = dataset["time"].dt.tz_localize("Europe/Prague", ambiguous="NaT", nonexistent="NaT")
    dataset["time"] = dataset["time"].dt.tz_convert("UTC")

    dataset["energy"] = dataset["energy"].astype(float)

    dataset = dataset.set_index("time")

    dataset = dataset.groupby(pd.Grouper(freq="h")).agg({
        "int_sol_irr": "mean",
        "wind_vel": "mean",
        "tmp_amb": "mean",
        "tmp_module": "mean",
        "energy": "sum"
    })

    dataset["energy"] = dataset["energy"] / 1000

    dataset = dataset.reset_index()
    return dataset

def get_fve_data_utc_day(day: date, username: str, password: str) -> pd.DataFrame:
    df_curr = get_fve_data_for_day(day, username, password)
    next_day = day + timedelta(days=1)
    df_next = get_fve_data_for_day(next_day, username, password)
    combined_df = pd.concat([df_curr, df_next], ignore_index=True)

    start_utc = pd.Timestamp(f"{day} 00:00:00").tz_localize("UTC")
    end_utc = pd.Timestamp(f"{day} 23:59:59").tz_localize("UTC")

    utc_day_df = combined_df[(combined_df["time"] >= start_utc) & (combined_df["time"] <= end_utc)]

    return utc_day_df.sort_values("time").reset_index(drop=True)