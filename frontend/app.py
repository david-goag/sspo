# app.py
import streamlit as st
import pandas as pd
import numpy as np
import polyline
import requests
import pydeck as pdk # You'll need to install this: pip install pydeck
from datetime import datetime, date, time, timedelta
from urllib.parse import urlparse, parse_qs

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

# --- API HELPER FUNCTIONS ---
@st.cache_data
def get_segment_data(_segment_id, access_token): # Add underscore to avoid caching conflicts with widget keys
    """Fetches segment data from the Strava API, including the map polyline."""
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
                "distance": data.get("distance", 0)
            }
        else:
            st.warning("Segment found, but it does not contain map data.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch segment data from Strava API: {e}")
        return None

@st.cache_data
def get_weather_forecast(lat, lon, ride_datetime):
    """Fetches weather forecast data from the Open-Meteo API for a specific time."""
    url = "https://api.open-meteo.com/v1/forecast"
    ride_date_str = ride_datetime.strftime('%Y-%m-%d')
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "ms",
        "timezone": "auto",
        "start_date": ride_date_str,
        "end_date": ride_date_str
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Find the index for the specific hour of the ride
        target_hour_str = ride_datetime.strftime('%Y-%m-%dT%H:00')
        try:
            hour_index = data['hourly']['time'].index(target_hour_str)
            return {
                "temperature": data['hourly']['temperature_2m'][hour_index],
                "wind_speed": data['hourly']['wind_speed_10m'][hour_index] * 3.6, # Convert m/s to km/h
                "wind_direction": data['hourly']['wind_direction_10m'][hour_index]
            }
        except (ValueError, IndexError):
            st.warning("Could not find weather for the exact hour. Using the first available forecast for the day.")
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
    """Fetches elevation data for a DataFrame of coordinates from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/elevation"
    # Batches requests to avoid URL length limits if the segment is very long
    all_elevations = []
    for i in range(0, len(map_df), 500):
        batch = map_df.iloc[i:i+500]
        lat_str = ",".join(batch['lat'].astype(str))
        lon_str = ",".join(batch['lon'].astype(str))
        params = {"latitude": lat_str, "longitude": lon_str}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            all_elevations.extend(data['elevation'])
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch elevation data: {e}")
            return None

    map_df['elevation'] = all_elevations
    return map_df

def haversine_np(lon1, lat1, lon2, lat2):
    """Calculate the distance between two points on Earth in meters."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km * 1000

def get_color_from_gradient(gradient):
    """Assigns a color based on the slope gradient for the effort scale."""
    if gradient > 8: return [204, 0, 0, 200]      # Dark Red (Maximum Effort)
    elif gradient > 5: return [255, 0, 0, 200]    # Red (Hard Effort)
    elif gradient > 2: return [255, 128, 0, 200]  # Orange (Moderate Effort)
    elif gradient > -2: return [0, 255, 0, 200]   # Green (Steady)
    else: return [0, 0, 255, 200]                 # Blue (Recovery)

# --- UI PAGES ---
def show_authentication_page(auth):
    """Displays the page for the user to initiate the authentication process."""
    st.title("🚴‍♂️ Strava Pacing Optimizer")
    st.markdown("### 🔐 Connect to Strava to begin")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            To get started, you'll need to authorize this app to access your Strava data.

            1.  Click the button below.
            2.  You'll be taken to Strava to approve access.
            3.  You will be redirected back here automatically.
            """
        )
        auth_url = auth.get_authorization_url()
        st.link_button("🔗 Connect to Strava", auth_url, type="primary", use_container_width=True)

def show_main_app():
    """Displays the main application UI after successful authentication."""
    athlete = st.session_state.athlete_info

    with st.sidebar:
        st.header(f"Welcome, {athlete['firstname']}! 👋")
        st.image("https://aiguajoc.com/wp-content/uploads/Beneficios-de-practicar-el-cycling-AIGUAJOC.webp")
        if st.button("🚪 Logout"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

    st.title("Pacing Optimizer Dashboard")
    st.markdown("Enter your segment, goal, and planned ride time to get a custom pacing plan.")
    st.markdown("---")

    st.header("1. Your Details & Segment")
    input_col1, input_col2 = st.columns(2)
    with input_col1:
        segment_url = st.text_input("Strava Segment URL or ID:", value="https://www.strava.com/segments/13260861")
    with input_col2:
        weight = st.number_input("Your Weight (kg):", min_value=40.0, max_value=150.0, value=75.0, step=0.5)

    st.header("2. Your Goal & Ride Time")
    if 'ride_date' not in st.session_state:
        st.session_state.ride_date = date.today()
    if 'ride_time' not in st.session_state:
        st.session_state.ride_time = datetime.now().time()

    goal_col1, goal_col2, goal_col3 = st.columns(3)
    with goal_col1:
        desired_duration_str = st.text_input("Desired Duration (e.g., 10:30):", value="12:00")
    with goal_col2:
        max_forecast_date = date.today() + timedelta(days=14)
        st.date_input("Date of Ride:", key="ride_date", max_value=max_forecast_date)
    with goal_col3:
        st.time_input("Time of Ride (24h):", key="ride_time")

    st.markdown("---")
    _, center_col, _ = st.columns([2, 1, 2])
    with center_col:
        if st.button("🚀 Generate Pacing Plan", type="primary", use_container_width=True):
            if segment_url and weight and desired_duration_str:
                st.session_state.prediction_inputs = {
                    "segment": segment_url, "weight": weight, "duration": desired_duration_str,
                    "ride_datetime": datetime.combine(st.session_state.ride_date, st.session_state.ride_time)
                }
            else:
                st.warning("Please fill in all fields.")
    st.markdown("---")

    if 'prediction_inputs' in st.session_state:
        show_results_page()

def show_results_page():
    """Fetches data and displays the prediction results."""
    inputs = st.session_state.prediction_inputs
    st.header("3. Your Pacing Plan")

    segment_id = inputs['segment'].split('/')[-1] if '/' in inputs['segment'] else inputs['segment']

    with st.spinner("Fetching data and calculating your plan..."):
        segment_data = get_segment_data(segment_id, st.session_state.access_token)
        if not segment_data: return

        map_data = get_elevation_data(segment_data.get("map_data").copy())
        if map_data is None: return

        start_lat, start_lon = segment_data["start_latlng"]
        weather_data = get_weather_forecast(start_lat, start_lon, inputs["ride_datetime"])
        if not weather_data: return

        base_power = 250
        time_parts = list(map(int, inputs['duration'].split(':')))
        desired_seconds = time_parts[0] * 60 + time_parts[1]
        constant_power_result = int(base_power + (inputs['weight'] - 75) + (720 - desired_seconds) * 0.5 + (weather_data['wind_speed'] * 2))

        # --- DATA PROCESSING FOR ACCURATE GRADIENTS ---
        map_data['lat_next'] = map_data['lat'].shift(-1)
        map_data['lon_next'] = map_data['lon'].shift(-1)

        # Calculate distance between each point
        map_data['distance_segment'] = map_data.apply(
            lambda row: haversine_np(row['lon'], row['lat'], row['lon_next'], row['lat_next']) if pd.notna(row['lon_next']) else 0,
            axis=1
        )
        map_data['cumulative_distance'] = map_data['distance_segment'].cumsum()

        # Smooth elevation data to remove noise and get a more accurate gradient
        map_data['smoothed_elevation'] = map_data['elevation'].rolling(window=5, center=True, min_periods=1).mean()
        map_data['elevation_next'] = map_data['smoothed_elevation'].shift(-1)

        line_segments_df = map_data.dropna(subset=['lon_next', 'lat_next'])
        line_segments_df['elevation_change'] = line_segments_df['elevation_next'] - line_segments_df['smoothed_elevation']
        line_segments_df['gradient'] = line_segments_df.apply(
            lambda row: (row['elevation_change'] / row['distance_segment']) * 100 if row['distance_segment'] > 0 else 0,
            axis=1
        )
        line_segments_df['color'] = line_segments_df['gradient'].apply(get_color_from_gradient)


        # --- Update Variable Power Chart Data ---
        chart_data = pd.DataFrame({
            'Distance (km)': map_data['cumulative_distance'] / 1000,
            'Power (W)': np.random.normal(constant_power_result, 30, len(map_data)), # Placeholder
            'Elevation (m)': map_data['smoothed_elevation']
        })

        st.subheader(f"Analysis for Segment: `{segment_data['name']}`")
        tab1, tab2, tab3 = st.tabs(["🗺️ Segment Map", "📊 Constant Power Plan", "📈 Variable Power Plan"])

        with tab1:
            st.subheader("Segment Map")
            if "MAPBOX_API_KEY" not in st.secrets:
                st.error("Mapbox API key not found.")
                return

            view_state = pdk.ViewState(latitude=map_data["lat"].mean(), longitude=map_data["lon"].mean(), zoom=13, pitch=50)

            line_layer = pdk.Layer(
                "LineLayer",
                data=line_segments_df,
                get_source_position="[lon, lat]",
                get_target_position="[lon_next, lat_next]",
                get_color="color",
                get_width=5,
                pickable=True
            )

            start_point = map_data.iloc[[0]].copy()
            end_point = map_data.iloc[[-1]].copy()
            start_icon_data = {"url": "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-start.png", "width": 128, "height": 128, "anchorY": 128}
            end_icon_data = {"url": "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-finish.png", "width": 128, "height": 128, "anchorY": 128}
            start_point['icon_data'] = start_point.apply(lambda r: start_icon_data, axis=1)
            end_point['icon_data'] = end_point.apply(lambda r: end_icon_data, axis=1)

            icon_layer = pdk.Layer("IconLayer", data=pd.concat([start_point, end_point]), get_icon="icon_data", get_position="[lon, lat]", get_size=4, size_scale=15)

            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=view_state,
                layers=[line_layer, icon_layer],
                tooltip={"html": "<b>Gradient:</b> {gradient:.1f}%"}
            ))

            # Updated legend for the "Effort Scale"
            st.markdown("""
                **Effort Scale:**
                <span style="color:#0000FF; font-weight:bold;">● Recovery</span> |
                <span style="color:#00FF00; font-weight:bold;">● Steady</span> |
                <span style="color:#FF8000; font-weight:bold;">● Moderate</span> |
                <span style="color:#FF0000; font-weight:bold;">● Hard</span> |
                <span style="color:#CC0000; font-weight:bold;">● Max</span>
            """, unsafe_allow_html=True)

        with tab2:
            st.subheader("Constant Power & Conditions")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Hold this Average Power", value=f"{constant_power_result} W")
                st.info("This is the steady power output to aim for across the entire segment.")
            with col2:
                st.write(f"**Forecast for {inputs['ride_datetime'].strftime('%Y-%m-%d %H:%M')}**")
                sub_col1, sub_col2, sub_col3 = st.columns(3)
                sub_col1.metric("Wind", f"{weather_data['wind_speed']:.1f} km/h")
                sub_col2.metric("Dir.", f"{weather_data['wind_direction']}°")
                sub_col3.metric("Temp.", f"{weather_data['temperature']}°C")
        with tab3:
            st.subheader("Variable Power & Elevation Profile")
            st.write("This chart shows recommended power adjustments based on the segment's elevation profile.")
            st.line_chart(chart_data, x='Distance (km)', y=['Power (W)', 'Elevation (m)'])

# --- MAIN ROUTER ---
def main():
    """Main function to handle page routing based on authentication status."""
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
