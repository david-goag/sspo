# data_collection.py
import os
from stravalib import Client as StravaClient
from stravalib.model import DetailedAthlete, Segment, BaseEffort
import numpy as np
import pandas as pd
import openmeteo_requests
from pathlib import Path
import time # To add a small delay and be respectful to APIs

# --- INITIALIZATION ---
print("=============================================")
print("      Starting Strava Data Collection        ")
print("=============================================")
ACCESS_TOKEN = os.environ["ACCESS_TOKEN"]
SEGMENT_LIST = os.environ["SEGMENT_LIST"].split(",")
print("✅ Environment variables loaded successfully.")


client = StravaClient(access_token=ACCESS_TOKEN)
athlete = client.get_athlete()

def collect_data(client: StravaClient, segment_list: list) -> pd.DataFrame:
    """Iterates through a list of segments, collecting all powered efforts."""
    athlete = client.get_athlete()
    print(f"✅ Authenticated as {athlete.firstname} {athlete.lastname}.")

    data_df = pd.DataFrame()
    total_segments = len(segment_list)

    for i, segment_id in enumerate(segment_list):
        print(f"\n--- Processing Segment {i+1}/{total_segments} (ID: {segment_id}) ---")
        try:
            segment = client.get_segment(segment_id)
            print(f"   - Segment Name: '{segment.name}'")

            segment_efforts = client.get_segment_efforts(segment_id)

            powered_segment_efforts = []
            for effort in segment_efforts:
                powered_segment_efforts.append(effort) if effort.device_watts == True else True

            print(f"   - Found {len(list(client.get_segment_efforts(segment_id)))} total efforts, {len(powered_segment_efforts)} with power data.")

            if not powered_segment_efforts:
                print("   - ⚠️ No powered efforts found for this segment. Skipping.")
                continue

            for j, powered_effort in enumerate(powered_segment_efforts):
                #print(f"     - Fetching data for effort {j+1}/{len(powered_segment_efforts)}...")
                single_df = effort_to_df(athlete, segment, powered_effort)
                data_df = pd.concat([data_df, single_df], axis=0)
                time.sleep(0.5) # Small delay to be respectful to the weather API

        except Exception as e:
            print(f"   - ❌ An error occurred while processing segment {segment_id}: {e}")
            continue

    return data_df

def effort_to_df(athlete: DetailedAthlete, segment: Segment, effort: BaseEffort) -> pd.DataFrame:
    """Converts a single Strava effort into a structured DataFrame row, including weather data."""
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
    """Stores the collected data to a Parquet file, handling existing data."""
    print("\n--- Storing Data ---")

    # Define paths
    current_dir = Path("database/current")
    old_dir = Path("database/old")

    # Create directories if they don't exist
    current_dir.mkdir(parents=True, exist_ok=True)
    old_dir.mkdir(parents=True, exist_ok=True)

    file_path = current_dir / f"{athlete.id}_data.parquet"
    old_file_path = old_dir / f"{athlete.id}_data_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.parquet"

    if file_path.exists():
        print(f"   - Existing database found at '{file_path}'.")
        database_df = pd.read_parquet(file_path)

        # Backup the old file
        database_df.to_parquet(old_file_path, engine='fastparquet')
        print(f"   - Backup of old data saved to '{old_file_path}'.")

        # Combine and remove duplicates, keeping the new data
        database_df.join(data, on="id", lsuffix="left", rsuffix="right", how="outer")

        new_rows = len(database_df.join(data, on="id", lsuffix="left", rsuffix="right", how="outer")) - len(database_df)
        print(f"   - Merged data: {new_rows} new efforts added.")

        database_df.to_parquet(file_path, engine='fastparquet')
    else:
        print(f"   - No existing database found. Creating new file.")
        data.to_parquet(file_path, engine='fastparquet')

    print(f"💾 Data successfully saved to '{file_path}'. Total efforts: {len(pd.read_parquet(file_path))}")


if __name__ == "__main__":

    data_df = collect_data(client, SEGMENT_LIST)

    if not data_df.empty:
        store_to_database(data=data_df, athlete=athlete)
    else:
        print("\nNo new data collected. Database remains unchanged.")

    print("\n=============================================")
    print("          Script finished successfully       ")
    print("=============================================")
