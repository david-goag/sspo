# app.py
import streamlit as st
import pandas as pd
import numpy as np
import polyline
import requests
import pydeck as pdk
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date, time, timedelta
from urllib.parse import urlparse, parse_qs
import math
import time as time_sleep # Import the time module for sleeping
from bs4 import BeautifulSoup


# Import the helper classes/functions from your separate files
from strava_auth import StravaAuth
# The scrape_leaderboard function is no longer needed as we are implementing it directly
# from strava_scraper import scrape_leaderboard

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Strava Segment Pacing Optimizer",
    page_icon="🚴",
    layout="wide"
)

# --- SESSION STATE INITIALIZATION ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "athlete_info" not in st.session_state:
    st.session_state.athlete_info = None

# --- PREDICTION FUNCTION ---
def predict_time_from_api(segment_data, map_data, line_segments_df, weather_data, user_inputs):
    """Prepares the input data and returns a time prediction from the API."""

    api_url = "https://api-879488749692.europe-west1.run.app/predict"

    try:
        params = {
            'athlete_weight': np.int8(user_inputs['weight']),
            'distance': np.int32(segment_data['distance']),
            'avg_grade': np.float16((map_data['elevation'].iloc[-1] - map_data['elevation'].iloc[0]) / segment_data['distance'] * 100 if segment_data['distance'] > 0 else 0),
            'max_grade': np.float16(line_segments_df['gradient'].max()),
            'elevation_gain': np.int16(segment_data['elevation_gain']),
            'start_latitude': np.float16(segment_data['start_latlng'][0]),
            'start_longitude': np.float16(segment_data['start_latlng'][1]),
            'end_latitude': np.float16(map_data['lat'].iloc[-1]),
            'end_longitude': np.float16(map_data['lon'].iloc[-1]),
            'avg_power': np.int16(user_inputs['power']),
            'temperature': np.int8(weather_data['temperature']),
            'wind_speed': np.int8(weather_data['wind_speed']),
            'wind_direction': np.int16(weather_data['wind_direction']),
        }

        response = requests.get(api_url, params=params)
        response.raise_for_status()
        prediction_data = response.json()

        return int(prediction_data['Seconds']), params

    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while calling the prediction API: {e}")
        st.error(f"API Response: {e.response.text}")
        return None, params
    except (KeyError, IndexError):
        st.error("Received an unexpected response from the prediction API.")
        st.write("The API returned the following data, but the app could not find the expected key:")
        st.json(prediction_data)
        return None, params

def find_power_for_target_time(target_seconds, segment_data, map_data, line_segments_df, weather_data, user_inputs):
    """Iteratively finds the power required to meet a target time."""
    current_power = user_inputs['power']

    with st.spinner("Calculating power needed for Top 10..."):
        for _ in range(20): # Limit iterations to prevent infinite loops
            inputs_for_calc = user_inputs.copy()
            inputs_for_calc['power'] = current_power

            predicted_time, _ = predict_time_from_api(segment_data, map_data, line_segments_df, weather_data, inputs_for_calc)

            if predicted_time is None:
                return None # API error occurred

            if predicted_time <= target_seconds:
                return current_power # Success!

            time_diff = predicted_time - target_seconds
            power_adjustment = max(1, int(time_diff / 5))
            current_power += power_adjustment

            if current_power > 1000: # Safety break
                return None

    return None # Could not find a suitable power

# --- API HELPER FUNCTIONS ---
@st.cache_data
def get_segment_data(_segment_id, access_token):
    headers = {'Authorization': f'Bearer {access_token}'}
    url = f"https://www.strava.com/api/v3/segments/{_segment_id}"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        encoded_polyline = data.get('map', {}).get('polyline')
        if encoded_polyline:
            decoded_coords = polyline.decode(encoded_polyline)
            map_df = pd.DataFrame(decoded_coords, columns=['lat', 'lon'])
            return {
                "name": data.get("name", "N/A"),
                "map_data": map_df,
                "start_latlng": data.get("start_latlng", [0, 0]),
                "distance": data.get("distance", 0),
                "elevation_gain": data.get("total_elevation_gain", 0)
            }
        else:
            st.warning("Segment found, but it does not contain map data.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch segment data from Strava API: {e}")
        return None

@st.cache_data
def get_weather_forecast(lat, lon, ride_datetime):
    url = "https://api.open-meteo.com/v1/forecast"
    ride_date_str = ride_datetime.strftime('%Y-%m-%d')
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "ms", "timezone": "auto",
        "start_date": ride_date_str, "end_date": ride_date_str
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        target_hour_str = ride_datetime.strftime('%Y-%m-%dT%H:00')
        try:
            hour_index = data['hourly']['time'].index(target_hour_str)
            return {
                "temperature": data['hourly']['temperature_2m'][hour_index],
                "wind_speed": data['hourly']['wind_speed_10m'][hour_index] * 3.6,
                "wind_direction": data['hourly']['wind_direction_10m'][hour_index]
            }
        except (ValueError, IndexError):
            st.warning("Could not find weather for the exact hour. Using the first available forecast.")
            return {
                "temperature": data['hourly']['temperature_2m'][0],
                "wind_speed": data['hourly']['wind_speed_10m'][0] * 3.6,
                "wind_direction": data['hourly']['wind_direction_10m'][0]
            }
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch weather data: {e}")
        return None

@st.cache_data
def get_elevation_data(map_df):
    url = "https://api.open-meteo.com/v1/elevation"
    all_elevations = []
    for i in range(0, len(map_df), 100):
        batch = map_df.iloc[i:i+100]
        lat_str = ",".join(batch['lat'].astype(str))
        lon_str = ",".join(batch['lon'].astype(str))
        params = {"latitude": lat_str, "longitude": lon_str}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            all_elevations.extend(data['elevation'])
            time_sleep.sleep(1)
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch elevation data: {e}")
            return None
    map_df['elevation'] = all_elevations
    return map_df

def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c * 1000

def get_color_from_gradient(gradient):
    if gradient > 8: return [204, 0, 0, 200]
    elif gradient > 5: return [255, 0, 0, 200]
    elif gradient > 2: return [255, 128, 0, 200]
    elif gradient > -2: return [0, 255, 0, 200]
    else: return [0, 0, 255, 200]

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dLon = lon2 - lon1
    x = math.sin(dLon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    initial_bearing = math.atan2(x, y)
    return (math.degrees(initial_bearing) + 360) % 360

def get_wind_description(segment_bearing, wind_direction):
    diff = abs(segment_bearing - wind_direction)
    angle = min(diff, 360 - diff)
    if angle <= 45:
        return "Tailwind"
    elif angle >= 135:
        return "Headwind"
    else:
        return "Crosswind"

def degrees_to_cardinal(d):
    dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    ix = round(d / (360. / len(dirs)))
    return dirs[ix % len(dirs)]

def parse_time_to_seconds(time_str):
    """Converts a time string like '7m 27s', '7:27', or '48s' to seconds."""
    if pd.isna(time_str): return None
    time_str = str(time_str)

    total_seconds = 0
    if 'h' in time_str:
        parts = time_str.split('h')
        total_seconds += int(parts[0]) * 3600
        time_str = parts[1].strip()
    if 'm' in time_str:
        parts = time_str.split('m')
        total_seconds += int(parts[0]) * 60
        time_str = parts[1].strip()
    if 's' in time_str:
        total_seconds += int(time_str.replace('s', '').strip())
    elif ':' in time_str:
        parts = time_str.split(':')
        if len(parts) == 2:
            total_seconds += int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            total_seconds += int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

    return total_seconds if total_seconds > 0 else None


# --- UI PAGES ---
def set_bg_hack(url):
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("{url}");
             background-attachment: fixed;
             background-size: cover;
             background-position: center 20%;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def show_authentication_page(auth):
    set_bg_hack("https://trello-backgrounds.s3.amazonaws.com/SharedBackground/2560x1440/1609b2cc34793439f41f21b944076194/photo-1534787238916-9ba6764efd4f.webp")

    st.markdown("""
        <style>
        .glass-container {
            background: rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(3px);
            -webkit-backdrop-filter: blur(3px);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 30px;
            color: white;
        }
        .glass-container h1, .glass-container h2, .glass-container h3, .glass-container p, .glass-container li {
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

    auth_url = auth.get_authorization_url()

    login_html = f"""
    <div class="glass-container">
        <h1 style='text-align: center;'>🚴‍♂️ Strava Pacing Optimizer</h1>
        <h2 style='text-align: center; font-style: italic; font-weight: 400;'>Your personal pacing strategist for any Strava segment.</h2>
        <hr>
        <h3 style='text-align: center; font-weight: 400;'>🔐 Connect to Strava to begin</h3>
        <a href="{auth_url}" target="_blank" style="display: block; padding: 0.5em 1em; background-color: #FF4B4B; color: white; text-decoration: none; border-radius: 0.5rem; text-align: center; width: 100%; box-sizing: border-box;">🔗 Connect to Strava</a>
    </div>
    """

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown(login_html, unsafe_allow_html=True)


def show_main_app():
    st.markdown(
        """
        <style>
            .main .block-container {
                padding-top: 1rem;
                padding-bottom: 1rem;
            }
            [data-testid="stSidebar"] {
                width: 400px !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    athlete = st.session_state.athlete_info
    with st.sidebar:
        st.image(athlete["profile"], width=60)
        with st.popover(f"Welcome, {athlete['firstname']}!", use_container_width=True):
             if st.button("Logout", use_container_width=True):
                for key in st.session_state.keys():
                    del st.session_state[key]
                st.rerun()

        st.header("Pacing Optimizer")
        st.markdown("Enter the segment, check your weight, and input power to get a time estimate.")

        st.markdown("---")

        st.header("1. Your Details & Segment")
        segment_url = st.text_input("Strava Segment URL or ID:", value="https://www.strava.com/segments/13260861")

        default_weight = int(athlete.get('weight', 75) or 75)
        weight = st.number_input("Your Weight (kg):", min_value=40, max_value=150, value=default_weight, step=1)

        st.header("2. Your Goal & Ride Time")
        if 'ride_date' not in st.session_state:
            st.session_state.ride_date = date.today()
        if 'ride_time' not in st.session_state:
            st.session_state.ride_time = datetime.now().time()

        desired_power = st.number_input("Target Power (Watts):", min_value=0, max_value=2000, value=250, step=1)
        max_forecast_date = date.today() + timedelta(days=14)
        st.date_input("Date of Ride:", key="ride_date", max_value=max_forecast_date)
        st.time_input("Time of Ride (24h):", key="ride_time")

        st.markdown("---")

        if st.button("🚀 Generate Time Estimate", type="primary", use_container_width=True):
            if segment_url and weight and desired_power:
                st.cache_data.clear()
                st.session_state.prediction_inputs = {
                    "segment": segment_url, "weight": weight, "power": desired_power,
                    "ride_datetime": datetime.combine(st.session_state.ride_date, st.session_state.ride_time)
                }
            else:
                st.warning("Please fill in all fields.")

<<<<<<< HEAD
=======
        if st.button(":door: Logout"):
            if st.session_state.get("access_token"):
                auth = StravaAuth()
                with st.spinner("Logging out from Strava..."):
                    success = auth.deauthorize(st.session_state.access_token)
                    if success:
                        st.success("Successfully logged out from Strava!")
                    else:
                        st.warning(
                            "Could not revoke Strava access token, but clearing local session."
                        )

            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

>>>>>>> 3b3de1c64d90ec8134a4159029390c5235643707
    st.title("Pacing Optimizer Dashboard")

    if 'prediction_inputs' in st.session_state:
        show_results_page()
    else:
        st.info("⬅️ Enter your details in the sidebar to generate a pacing plan.")


def show_results_page():
    inputs = st.session_state.prediction_inputs
    segment_id = inputs['segment'].split('/')[-1] if '/' in inputs['segment'] else inputs['segment']

    with st.spinner("Fetching data and calculating your plan..."):
        segment_data = get_segment_data(segment_id, st.session_state.access_token)
        if not segment_data: return

        map_data = get_elevation_data(segment_data.get("map_data").copy())
        if map_data is None: return

        start_lat, start_lon = segment_data["start_latlng"]
        weather_data = get_weather_forecast(start_lat, start_lon, inputs["ride_datetime"])
        if not weather_data: return

        map_data['lat_next'] = map_data['lat'].shift(-1)
        map_data['lon_next'] = map_data['lon'].shift(-1)
        map_data['distance_segment'] = map_data.apply(lambda r: haversine_np(r['lon'], r['lat'], r['lon_next'], r['lat_next']) if pd.notna(r['lon_next']) else 0, axis=1)
        map_data['cumulative_distance'] = map_data['distance_segment'].cumsum()
        map_data['smoothed_elevation'] = map_data['elevation'].rolling(window=25, center=True, min_periods=1).mean()
        map_data['elevation_next'] = map_data['smoothed_elevation'].shift(-1)

        line_segments_df = map_data.dropna(subset=['lon_next', 'lat_next']).copy()
        line_segments_df['elevation_change'] = line_segments_df['elevation_next'] - line_segments_df['smoothed_elevation']
        line_segments_df['gradient'] = line_segments_df.apply(lambda r: (r['elevation_change'] / r['distance_segment']) * 100 if r['distance_segment'] > 0 else 0, axis=1)
        line_segments_df['color'] = line_segments_df['gradient'].apply(get_color_from_gradient)

        predicted_seconds, api_params = predict_time_from_api(segment_data, map_data, line_segments_df, weather_data, inputs)

        if predicted_seconds is None:
            return

        predicted_time_str = f"{predicted_seconds // 60}:{predicted_seconds % 60:02d}"
        avg_speed_kmh = (segment_data['distance'] / predicted_seconds) * 3.6 if predicted_seconds > 0 else 0

        variable_power = (inputs['power'] + (line_segments_df['gradient'] * 10)).clip(lower=0)

        segment_bearing = calculate_bearing(map_data['lat'].iloc[0], map_data['lon'].iloc[0], map_data['lat'].iloc[-1], map_data['lon'].iloc[-1])
        wind_desc = get_wind_description(segment_bearing, weather_data['wind_direction'])
        wind_cardinal = degrees_to_cardinal(weather_data['wind_direction'])

        with st.expander("API Parameters Sent for Prediction"):
            st.json(api_params)

        st.subheader(f"Segment: [{segment_data['name']}](https://www.strava.com/segments/{segment_id})")

        main_col, map_col = st.columns([1, 1])

        with main_col:
            st.subheader("📊 Prediction & Conditions")
            with st.container(border=True):
                pred_col, stats_col = st.columns([2,1.5])
                with pred_col:
                    st.markdown(f"<h3 style='text-align: center;'>Predicted Time</h3>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: center; color: #FF4B4B;'>{predicted_time_str}</h1>", unsafe_allow_html=True)
                    st.metric("Average Speed", f"{avg_speed_kmh:.1f} km/h", help="Based on the predicted time and segment distance.")
                    st.info(f"Estimate for an average power of {inputs['power']} W.")

                with stats_col:
                    avg_grade = (map_data['elevation'].iloc[-1] - map_data['elevation'].iloc[0]) / segment_data['distance'] * 100 if segment_data['distance'] > 0 else 0
                    st.metric("Distance", f"{segment_data['distance']/1000:.2f} km")
                    st.metric("Elevation Gain", f"{segment_data['elevation_gain']:.0f} m")
                    st.metric("Avg. Grade", f"{avg_grade:.1f}%")
                    predicted_rank_placeholder = st.empty()

                st.markdown("---")
                weather_cols = st.columns(3)
                weather_cols[0].metric("Temperature", f"{weather_data['temperature']}°C")
                weather_cols[1].metric("Wind Direction", f"{wind_cardinal} - {wind_desc}")
                weather_cols[2].metric("Wind Speed", f"{weather_data['wind_speed']:.1f} km/h")


        with map_col:
            st.subheader("🗺️ 3D Segment Map")
            if "MAPBOX_API_KEY" not in st.secrets:
                st.error("Mapbox API key not found.")
                return

            view_state = pdk.ViewState(latitude=map_data["lat"].mean(), longitude=map_data["lon"].mean(), zoom=13.5, pitch=60, bearing=0)

            line_layer = pdk.Layer("LineLayer", data=line_segments_df, get_source_position="[lon, lat, smoothed_elevation]", get_target_position="[lon_next, lat_next, elevation_next]", get_color="color", get_width=5, pickable=True)
            vertical_line_layer = pdk.Layer("LineLayer", data=map_data, get_source_position="[lon, lat, 0]", get_target_position="[lon, lat, smoothed_elevation]", get_color="color", get_width=1)
            start_point, end_point = map_data.iloc[[0]].copy(), map_data.iloc[[-1]].copy()
            start_point.loc[:, 'icon_data'] = [ {"url": "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-start.png", "width": 128, "height": 128, "anchorY": 128} ]
            end_point.loc[:, 'icon_data'] = [ {"url": "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-finish.png", "width": 128, "height": 128, "anchorY": 128} ]
            icon_layer = pdk.Layer("IconLayer", data=pd.concat([start_point, end_point]), get_icon="icon_data", get_position="[lon, lat, elevation]", get_size=4, size_scale=15)
            wind_arrow_data = pd.DataFrame([{"lon": map_data['lon'].mean(), "lat": map_data['lat'].mean(), "icon_data": {"url": "https://raw.githubusercontent.com/ajduberstein/wind-js/master/arrow.png", "width": 512, "height": 512, "anchorY": 256}, "angle": 450 - weather_data['wind_direction']}])
            wind_arrow_layer = pdk.Layer("IconLayer", data=wind_arrow_data, get_icon="icon_data", get_position="[lon, lat]", get_size=10, size_scale=30, get_angle="angle")

            st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/dark-v10", layers=[line_layer, vertical_line_layer, icon_layer, wind_arrow_layer], initial_view_state=view_state, tooltip={"html": "<b>Gradient:</b> {gradient:.1f}%"}))

            st.markdown("""
                **Effort Scale:**
                <span style="color:#0000FF; font-weight:bold;">● Recovery</span> |
                <span style="color:#00FF00; font-weight:bold;">● Steady</span> |
                <span style="color:#FF8000; font-weight:bold;">● Moderate</span> |
                <span style="color:#FF0000; font-weight:bold;">● Hard</span> |
                <span style="color:#CC0000; font-weight:bold;">● Max</span>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📈 Variable Power & Elevation Profile")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=map_data['cumulative_distance'] / 1000, y=map_data['smoothed_elevation'], name="Elevation", fill='tozeroy', line=dict(color='grey'), customdata=np.stack((line_segments_df['gradient'],), axis=-1), hovertemplate="<b>Distance:</b> %{x:.2f} km<br><b>Elevation:</b> %{y:.1f} m<br><b>Grade:</b> %{customdata[0]:.1f}%<extra></extra>"), secondary_y=False)
        fig.add_trace(go.Scatter(x=line_segments_df['cumulative_distance'] / 1000, y=variable_power, name="Power Plan", line=dict(color='red'), hovertemplate="<b>Power:</b> %{y:.0f} W<extra></extra>"), secondary_y=True)
        fig.update_layout(title_text="Pacing Plan vs. Elevation Profile", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), hovermode="x unified")
        fig.update_xaxes(title_text="Distance (km)")
        fig.update_yaxes(title_text="Elevation (m)", secondary_y=False)
        fig.update_yaxes(title_text="Power (W)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        # --- LEADERBOARD SECTION ---
        st.markdown("---")
        st.subheader("🏆 Segment Leaderboard Comparison")

        leaderboard_df = None # Initialize to handle potential errors
        try:
            response = requests.get(f"https://www.strava.com/segments/{segment_id}")
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            table = soup.find("table")
            if table:
                raw_df = pd.read_html(table.prettify())[0]

                # Robustly select and rename columns by position
                if raw_df.shape[1] >= 4:
                    leaderboard_df = raw_df.iloc[:, [0, 2, 3, -1]].copy()
                    leaderboard_df.columns = ['Rank', 'Athlete', 'Date', 'Time']

                    leaderboard_df['Time (s)'] = leaderboard_df['Time'].apply(parse_time_to_seconds)
                    leaderboard_df.dropna(subset=['Time (s)'], inplace=True)
                    leaderboard_df['Time (s)'] = leaderboard_df['Time (s)'].astype(int)
                else:
                    st.warning("Could not parse the leaderboard structure as expected.")
                    leaderboard_df = None
            else:
                st.info("Could not find a leaderboard table on the segment page.")
        except Exception as e:
            st.error(f"An error occurred while scraping the leaderboard: {e}")


        if leaderboard_df is not None and not leaderboard_df.empty:
            predicted_rank = leaderboard_df[leaderboard_df['Time (s)'] < predicted_seconds].shape[0] + 1
            predicted_rank_placeholder.metric("Predicted Rank", f"~{predicted_rank}", help="Estimated rank based on the public leaderboard.")

            if predicted_rank <= 10:
                user_effort = pd.DataFrame([{
                    "Rank": "★",
                    "Athlete": f"{st.session_state.athlete_info['firstname']} (Your Prediction)",
                    "Time": predicted_time_str,
                    "Time (s)": predicted_seconds,
                    "Date": "Forecast"
                }])
                display_df = pd.concat([leaderboard_df, user_effort]).sort_values(by="Time (s)").head(10)
            else:
                display_df = leaderboard_df.head(10)

            display_df = display_df.drop(columns=['Time (s)'], errors='ignore')
            display_df.reset_index(drop=True, inplace=True)
            display_df['Rank'] = display_df.index + 1

            def highlight_user(row):
                if "(Your Prediction)" in str(row['Athlete']):
                    return ['background-color: #FF4B4B; color: white'] * len(row)
                return [''] * len(row)

            st.dataframe(
                display_df[['Rank', 'Athlete', 'Time', 'Date']].style.apply(highlight_user, axis=1),
                use_container_width=True,
                hide_index=True
            )

            if predicted_rank > 10 and len(leaderboard_df) >= 10:
                time_to_beat = leaderboard_df.iloc[9]['Time (s)']
                power_for_top_10 = find_power_for_target_time(time_to_beat, segment_data, map_data, line_segments_df, weather_data, inputs)
                if power_for_top_10:
                    power_diff = power_for_top_10 - inputs['power']
                    st.warning(f"🎯 To break into the Top 10, you would need to hold an average of **{power_for_top_10} W** (+{power_diff} W).")
                else:
                    st.info("Could not calculate the power required for a Top 10 finish.")

        else:
            st.info("Could not display leaderboard at this time.")


# --- MAIN ROUTER ---
def main():
    auth = StravaAuth()
    query_params = st.query_params
    if "code" in query_params and not st.session_state.authenticated:
        authorization_code = query_params.get("code")
        st.query_params.clear()
        with st.spinner("Authenticating with Strava..."):
            token_data = auth.exchange_code_for_token(authorization_code)
            if token_data:
                st.session_state.access_token = token_data["access_token"]
                st.session_state.athlete_info = token_data["athlete"]
                st.session_state.authenticated = True
            else:
                st.error("Authentication failed. Please try logging in again.")
                st.stop()
    if st.session_state.authenticated:
        show_main_app()
    else:
        show_authentication_page(auth)

if __name__ == "__main__":
    main()
