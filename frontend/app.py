# app.py
import streamlit as st
import pandas as pd
import numpy as np
import polyline
import requests
import pydeck as pdk
import plotly.graph_objects as go
import joblib
import xgboost
from plotly.subplots import make_subplots
from datetime import datetime, date, time, timedelta
from urllib.parse import urlparse, parse_qs
import math
import time as time_sleep # Import the time module for sleeping

# Import the authentication logic from your separate file
from strava_auth import StravaAuth

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

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    """Loads the trained XGBoost model from a file."""
    try:
        # Note: This should be a model trained to predict 'time'
        model = joblib.load('model.joblib')
        return model
    except FileNotFoundError:
        st.error("Model file ('model.joblib') not found. Please make sure it's in the same directory as app.py.")
        return None

# --- PREDICTION FUNCTION ---
def predict_time(model, segment_data, map_data, line_segments_df, weather_data, user_inputs):
    """Prepares the input data and returns a time prediction from the model."""
    if model is None:
        st.error("Model could not be loaded. Cannot make a prediction.")
        return 720 # Return a default value

    try:
        input_data = pd.DataFrame({
            'athlete_weight': [user_inputs['weight']],
            'distance': [segment_data['distance']],
            'avg_grade': [(map_data['elevation'].iloc[-1] - map_data['elevation'].iloc[0]) / segment_data['distance'] * 100 if segment_data['distance'] > 0 else 0],
            'max_grade': [line_segments_df['gradient'].max()],
            'elevation_gain': [segment_data['elevation_gain']],
            'start_latitude': [segment_data['start_latlng'][0]],
            'start_longitude': [segment_data['start_latlng'][1]],
            'end_latitude': [map_data['lat'].iloc[-1]],
            'end_longitude': [map_data['lon'].iloc[-1]],
            'temperature': [weather_data['temperature']],
            'wind_speed': [weather_data['wind_speed']],
            'wind_direction': [weather_data['wind_direction']],
            'avg_power': [user_inputs['power']] # Use power as an input feature
        })

        prediction = model.predict(input_data)
        return int(prediction[0]) # Predicted time in seconds

    except ValueError as e:
        if "feature_names mismatch" in str(e):
            st.error("Model Mismatch Error: The loaded 'model.joblib' file is not trained to predict time.")
            st.info("Please run the 'train_time_model.py' script to generate a new model file that predicts time based on power.")
            return 720 # Return a default value
        else:
            st.error(f"An error occurred during prediction: {e}")
            return 720
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return 720


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
            time_sleep.sleep(1) # Add a 1-second delay to avoid hitting rate limits
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
    """Calculates the bearing between two points."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dLon = lon2 - lon1
    x = math.sin(dLon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    initial_bearing = math.atan2(x, y)
    return (math.degrees(initial_bearing) + 360) % 360

def get_wind_description(segment_bearing, wind_direction):
    """Determines if the wind is a headwind, tailwind, or crosswind."""
    diff = abs(segment_bearing - wind_direction)
    angle = min(diff, 360 - diff)
    if angle <= 45:
        return "Tailwind"
    elif angle >= 135:
        return "Headwind"
    else:
        return "Crosswind"

def degrees_to_cardinal(d):
    """Converts degrees to a cardinal direction."""
    dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    ix = round(d / (360. / len(dirs)))
    return dirs[ix % len(dirs)]

# --- UI PAGES ---
def set_bg_hack(url):
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("{url}");
             background-attachment: fixed;
             background-size: cover;
             background-position: center 20%; /* Move the background image up */
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def show_authentication_page(auth):
    set_bg_hack("https://trello-backgrounds.s3.amazonaws.com/SharedBackground/2560x1440/1609b2cc34793439f41f21b944076194/photo-1534787238916-9ba6764efd4f.webp")

    # Custom CSS for the glass effect and white text
    st.markdown("""
        <style>
        .glass-container {
            background: rgba(0, 0, 0, 0.1); /* Lighter semi-transparent background */
            backdrop-filter: blur(3px);
            -webkit-backdrop-filter: blur(3px); /* Smoother blur */
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 30px;
            color: white; /* Ensure text inside is white */
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
        <a href="{auth_url}" target="_self" style="display: block; padding: 0.5em 1em; background-color: #FF4B4B; color: white; text-decoration: none; border-radius: 0.5rem; text-align: center; width: 100%; box-sizing: border-box;">🔗 Connect to Strava</a>
    </div>
    """

    # Use columns to center the login card
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown(login_html, unsafe_allow_html=True)


def show_main_app():
    # Custom CSS to make the sidebar wider
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                width: 400px !important; # Set the width to your desired value
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    athlete = st.session_state.athlete_info
    with st.sidebar:
        st.header(f"Welcome, {athlete['firstname']}! 👋")
        st.image("https://aiguajoc.com/wp-content/uploads/Beneficios-de-practicar-el-cycling-AIGUAJOC.webp")
        st.markdown("Enter your segment, target power, and planned ride time to get a time estimate.")

        st.markdown("---")

        st.header("1. Your Details & Segment")
        segment_url = st.text_input("Strava Segment URL or ID:", value="https://www.strava.com/segments/13260861")
        weight = st.number_input("Your Weight (kg):", min_value=40.0, max_value=150.0, value=75.0, step=0.5)

        st.header("2. Your Goal & Ride Time")
        if 'ride_date' not in st.session_state:
            st.session_state.ride_date = date.today()
        if 'ride_time' not in st.session_state:
            st.session_state.ride_time = datetime.now().time()

        desired_power = st.number_input("Target Power (Watts):", min_value=100, max_value=500, value=250, step=5)
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

        if st.button("🚪 Logout"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

    st.title("Pacing Optimizer Dashboard")

    if 'prediction_inputs' in st.session_state:
        show_results_page()
    else:
        st.info("⬅️ Enter your details in the sidebar to generate a pacing plan.")


def show_results_page():
    inputs = st.session_state.prediction_inputs
    st.header("3. Your Pacing Plan")
    segment_id = inputs['segment'].split('/')[-1] if '/' in inputs['segment'] else inputs['segment']

    model = load_model()

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
        map_data['smoothed_elevation'] = map_data['elevation'].rolling(window=15, center=True, min_periods=1).mean()
        map_data['elevation_next'] = map_data['smoothed_elevation'].shift(-1)

        line_segments_df = map_data.dropna(subset=['lon_next', 'lat_next']).copy()
        line_segments_df['elevation_change'] = line_segments_df['elevation_next'] - line_segments_df['smoothed_elevation']
        line_segments_df['gradient'] = line_segments_df.apply(lambda r: (r['elevation_change'] / r['distance_segment']) * 100 if r['distance_segment'] > 0 else 0, axis=1)
        line_segments_df['color'] = line_segments_df['gradient'].apply(get_color_from_gradient)

        predicted_seconds = predict_time(model, segment_data, map_data, line_segments_df, weather_data, inputs)
        predicted_time_str = f"{predicted_seconds // 60}:{predicted_seconds % 60:02d}"

        variable_power = (inputs['power'] + (line_segments_df['gradient'] * 10)).clip(lower=0)

        segment_bearing = calculate_bearing(map_data['lat'].iloc[0], map_data['lon'].iloc[0], map_data['lat'].iloc[-1], map_data['lon'].iloc[-1])
        wind_desc = get_wind_description(segment_bearing, weather_data['wind_direction'])
        wind_cardinal = degrees_to_cardinal(weather_data['wind_direction'])

        st.subheader(f"Segment: [{segment_data['name']}](https://www.strava.com/segments/{segment_id})")

        avg_grade = (map_data['elevation'].iloc[-1] - map_data['elevation'].iloc[0]) / segment_data['distance'] * 100 if segment_data['distance'] > 0 else 0
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        stat_col1.metric("Distance", f"{segment_data['distance']/1000:.2f} km")
        stat_col2.metric("Elevation Gain", f"{segment_data['elevation_gain']:.0f} m")
        stat_col3.metric("Avg. Grade", f"{avg_grade:.1f}%")

        st.markdown("---")

        # --- Create a two-column layout for the results ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🗺️ 3D Segment Map")
            if "MAPBOX_API_KEY" not in st.secrets:
                st.error("Mapbox API key not found.")
                return

            mapbox_key = st.secrets["MAPBOX_API_KEY"]

            view_state = pdk.ViewState(latitude=map_data["lat"].mean(), longitude=map_data["lon"].mean(), zoom=13, pitch=60, bearing=0)

            # Layer for the 3D, color-coded elevation line
            line_layer = pdk.Layer(
                "LineLayer", data=line_segments_df, get_source_position="[lon, lat, smoothed_elevation]",
                get_target_position="[lon_next, lat_next, elevation_next]", get_color="color", get_width=5, pickable=True
            )

            # Create the data for the "wall" effect
            wall_data = []
            for i in range(len(map_data) - 1):
                wall_data.append({
                    "polygon": [
                        [map_data['lon'][i], map_data['lat'][i], 0],
                        [map_data['lon'][i+1], map_data['lat'][i+1], 0],
                        [map_data['lon'][i+1], map_data['lat'][i+1], map_data['smoothed_elevation'][i+1]],
                        [map_data['lon'][i], map_data['lat'][i], map_data['smoothed_elevation'][i]]
                    ],
                    "color": line_segments_df['color'].iloc[i]
                })
            wall_df = pd.DataFrame(wall_data)

            wall_layer = pdk.Layer(
                "PolygonLayer",
                data=wall_df,
                get_polygon="polygon",
                get_fill_color="color",
                stroked=False,
            )

            start_point, end_point = map_data.iloc[[0]], map_data.iloc[[-1]]
            start_icon = {"url": "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-start.png", "width": 128, "height": 128, "anchorY": 128}
            end_icon = {"url": "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-finish.png", "width": 128, "height": 128, "anchorY": 128}
            start_point['icon_data'] = [start_icon]
            end_point['icon_data'] = [end_icon]

            icon_layer = pdk.Layer("IconLayer", data=pd.concat([start_point, end_point]), get_icon="icon_data", get_position="[lon, lat, elevation]", get_size=4, size_scale=15)

            wind_arrow_data = pd.DataFrame([{
                "lon": map_data['lon'].mean(),
                "lat": map_data['lat'].mean(),
                "icon_data": {
                    "url": "https://raw.githubusercontent.com/ajduberstein/wind-js/master/arrow.png",
                    "width": 512, "height": 512, "anchorY": 256
                },
                "angle": 450 - weather_data['wind_direction']
            }])

            wind_arrow_layer = pdk.Layer(
                "IconLayer", data=wind_arrow_data, get_icon="icon_data", get_position="[lon, lat]",
                get_size=10, size_scale=30, get_angle="angle"
            )

            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/dark-v10",
                layers=[line_layer, wall_layer, icon_layer, wind_arrow_layer],
                initial_view_state=view_state,
                tooltip={"html": "<b>Gradient:</b> {gradient:.1f}%"}
            ))

            st.markdown("""
                **Effort Scale:**
                <span style="color:#0000FF; font-weight:bold;">● Recovery</span> |
                <span style="color:#00FF00; font-weight:bold;">● Steady</span> |
                <span style="color:#FF8000; font-weight:bold;">● Moderate</span> |
                <span style="color:#FF0000; font-weight:bold;">● Hard</span> |
                <span style="color:#CC0000; font-weight:bold;">● Max</span>
            """, unsafe_allow_html=True)

        with col2:
            st.subheader("📊 Time Prediction & Conditions")

            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                st.metric("Predicted Time", value=predicted_time_str)
                st.info(f"Estimate for an average power of {inputs['power']} W.")
            with sub_col2:
                st.write(f"**Forecast for {inputs['ride_datetime'].strftime('%Y-%m-%d %H:%M')}**")
                st.metric("Wind Speed", f"{weather_data['wind_speed']:.1f} km/h")
                st.metric("Wind Direction", f"{wind_cardinal} - {wind_desc}")
                st.metric("Temperature", f"{weather_data['temperature']}°C")

            st.markdown("---")

            st.subheader("📈 Variable Power & Elevation Profile")
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(
                    x=map_data['cumulative_distance'] / 1000, y=map_data['smoothed_elevation'], name="Elevation",
                    fill='tozeroy', line=dict(color='grey'),
                    customdata=np.stack((line_segments_df['gradient'],), axis=-1),
                    hovertemplate="<b>Distance:</b> %{x:.2f} km<br><b>Elevation:</b> %{y:.1f} m<br><b>Grade:</b> %{customdata[0]:.1f}%<extra></extra>"
                ), secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=line_segments_df['cumulative_distance'] / 1000, y=variable_power, name="Power Plan",
                    line=dict(color='red'), hovertemplate="<b>Power:</b> %{y:.0f} W<extra></extra>"
                ), secondary_y=True,
            )
            fig.update_layout(
                title_text="Pacing Plan vs. Elevation Profile",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )
            fig.update_xaxes(title_text="Distance (km)")
            fig.update_yaxes(title_text="Elevation (m)", secondary_y=False)
            fig.update_yaxes(title_text="Power (W)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

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
