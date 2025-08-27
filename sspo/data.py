import os
from stravalib import Client as StravaClient
from stravalib.model import DetailedAthlete, Segment, BaseEffort
import numpy as np
import pandas as pd
import openmeteo_requests
from pathlib import Path

ACCESS_TOKEN = os.environ["ACCESS_TOKEN"]
SEGMENT_LIST = os.environ["SEGMENT_LIST"].split(",")
client = StravaClient(access_token=ACCESS_TOKEN)
athlete = client.get_athlete()

def collect_data(client: StravaClient, segment_list: list) -> pd.DataFrame:
    athlete = client.get_athlete()
    data_df = pd.DataFrame()
    for segment_id in segment_list:
        segment = client.get_segment(segment_id)
        segment_efforts = client.get_segment_efforts(segment_id)
        powered_segment_efforts = []
        for effort in segment_efforts:
            powered_segment_efforts.append(effort) if effort.device_watts == True else True
        for powered_effort in powered_segment_efforts:
            single_df = effort_to_df(athlete, segment, powered_effort)
            data_df = pd.concat([data_df, single_df], axis=0)

    return data_df

def effort_to_df(athlete: DetailedAthlete, segment: Segment, effort: BaseEffort) -> pd.DataFrame:

    openmeteo = openmeteo_requests.Client()
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": segment.start_latlng.root[0],
        "longitude": segment.start_latlng.root[1],
        "start_date": f"{effort.start_date_local.year}-{effort.start_date_local.month:02d}-{effort.start_date_local.day:02d}",
        "end_date": f"{effort.start_date_local.year}-{effort.start_date_local.month:02d}-{effort.start_date_local.day:02d}",
        "hourly": ["temperature_2m", "wind_speed_10m", "wind_direction_10m"],
        "timezone": "auto",
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()

    effort_dict = {'id': [effort.id],
    'athlete_weight': [athlete.weight],
    'distance': [segment.distance],
    'avg_grade': [round(segment.average_grade, 1)],
    'max_grade': [round(segment.maximum_grade, 1)],
    'elevation_gain': [segment.total_elevation_gain],
    "start_latitude": [segment.start_latlng.root[0]],
    "start_longitude": [segment.start_latlng.root[1]],
    "end_latitude": [segment.end_latlng.root[0]],
    "end_longitude": [segment.end_latlng.root[1]],
    'avg_power': [effort.average_watts],
    'temperature': [hourly.Variables(0).Values(effort.start_date_local.hour)],
    'wind_speed': [hourly.Variables(1).Values(effort.start_date_local.hour)],
    'wind_direction': [hourly.Variables(2).Values(effort.start_date_local.hour)],
    'time': [effort.moving_time]
    }

    effort_dtypes = {'id': np.int64,
    'athlete_weight': np.int8,
    'distance': np.int32,
    'avg_grade': np.float16,
    'max_grade': np.float16,
    'elevation_gain': np.int16,
    "start_latitude": np.float16,
    "start_longitude": np.float16,
    "end_latitude": np.float16,
    "end_longitude": np.float16,
    'avg_power': np.int16,
    'temperature': np.int8,
    'wind_speed': np.int8,
    'wind_direction': np.int16,
    'time': np.int16
    }

    effort_df = pd.DataFrame(effort_dict)
    effort_df = effort_df.astype(effort_dtypes)

    return effort_df

def store_to_database(athlete: DetailedAthlete, data: pd.DataFrame) -> None:
    os.makedirs("database/old", exist_ok=True)
    os.makedirs("database/current", exist_ok=True)
    file_path = Path(f"database/current/{athlete.id}_data.parquet")
    old_file_path = Path(f"database/old/{athlete.id}_data.parquet")

    if file_path.exists():
        database_df = pd.read_parquet(file_path)
        database_df.to_parquet(old_file_path, engine='fastparquet')
        database_df.join(data, on="id", lsuffix="left", rsuffix="right", how="outer")
        database_df.to_parquet(file_path, engine='fastparquet')
    else:
        data.to_parquet(file_path, engine='fastparquet')


if __name__ == "__main__":
    data_df = collect_data(client, SEGMENT_LIST)
    store_to_database(data=data_df, athlete=athlete)
