# pages/0_Main_Dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import polyline
import requests
import pydeck as pdk
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date, time, timedelta
import math
import time as time_sleep  # Import the time module for sleeping
from bs4 import BeautifulSoup
from pathlib import Path

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Dashboard | Strava Pacing Optimizer", page_icon="🚴", layout="wide"
)

# --- Hide sidebar navigation and header anchor links ---
st.markdown(
    """
    <style>
        .st-emotion-cache-arp25b a {
    color: rgb(255, 75, 75);
    text-decoration: none;
}
        a {
            color: orange
        }
        .bestlink {
            text-color: orange;

        }
        [data-testid="stSidebarNav"] {
            display: none;
        }
        [data-testid="stHeaderActionElements"] {
            display: none;
        }
    </style>
""",
    unsafe_allow_html=True,
)


# --- SESSION STATE CHECK ---
if not st.session_state.get("authenticated"):
    st.error("Please log in to view this page.")
    st.link_button("Go to Login", "/")
    st.stop()

# Re-importing StravaAuth to be used within this page if needed
from strava_auth import StravaAuth

# --- TOKEN REFRESH LOGIC ---
if (
    "expires_at" in st.session_state
    and datetime.now().timestamp() > st.session_state.expires_at
):
    auth = StravaAuth()
    token_data = auth.refresh_token(st.session_state.refresh_token)
    if token_data:
        st.session_state.access_token = token_data["access_token"]
        st.session_state.refresh_token = token_data["refresh_token"]
        st.session_state.expires_at = token_data["expires_at"]
        st.rerun()
    else:
        st.stop()


# --- PREDICTION FUNCTION ---
def predict_time_from_api(
    segment_data: dict,
    map_data: pd.DataFrame,
    line_segments_df: pd.DataFrame,
    weather_data: dict,
    user_inputs: dict,
):
    api_url = "https://api-879488749692.europe-west1.run.app/predict_df"

    try:
        user_inputs_clean = {
            "weight": user_inputs["weight"],
            "power": user_inputs["power"],
            "entry_speed": user_inputs["entry_speed"],
        }

        payload = {
            "map_data": str(map_data.to_json(orient="records")),
            "line_segments_df": str(line_segments_df.to_json(orient="records")),
            "weather_data": weather_data,
            "user_inputs": user_inputs_clean,
        }

        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        prediction_data = response.json()

        return int(prediction_data["seconds"]), prediction_data.get("dataframe", [])

    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while calling the prediction API: {e}")
        if hasattr(e, "response") and e.response is not None:
            st.error(f"API Response: {e.response.text}")
        return None, {}
    except (KeyError, IndexError) as e:
        st.error(f"Received an unexpected response from the prediction API: {e}")
        return None, {}


def find_power_for_target_time(
    target_seconds, segment_data, map_data, line_segments_df, weather_data, user_inputs
):
    """Iteratively finds the power required to meet a target time."""
    current_power = user_inputs["power"]

    with st.spinner("Calculating power needed for Top 10..."):
        for _ in range(20):  # Limit iterations to prevent infinite loops
            inputs_for_calc = user_inputs.copy()
            inputs_for_calc["power"] = current_power

            predicted_time, _ = predict_time_from_api(
                segment_data, map_data, line_segments_df, weather_data, inputs_for_calc
            )

            if predicted_time is None:
                return None  # API error occurred

            if predicted_time <= target_seconds:
                return current_power  # Success!

            time_diff = predicted_time - target_seconds
            power_adjustment = max(1, int(time_diff / 5))
            current_power += power_adjustment

            if current_power > 1000:  # Safety break
                return None

    return None  # Could not find a suitable power


# --- API HELPER FUNCTIONS ---
@st.cache_data
def get_segment_data(_segment_id, access_token):
    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"https://www.strava.com/api/v3/segments/{_segment_id}"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        encoded_polyline = data.get("map", {}).get("polyline")
        if encoded_polyline:
            decoded_coords = polyline.decode(encoded_polyline)
            map_df = pd.DataFrame(decoded_coords, columns=["lat", "lon"])
            return {
                "name": data.get("name", "N/A"),
                "map_data": map_df,
                "start_latlng": data.get("start_latlng", [0, 0]),
                "distance": data.get("distance", 0),
                "elevation_gain": data.get("total_elevation_gain", 0),
            }
        else:
            st.error(f"Segment {_segment_id} has no route data available.")
            return None
    except requests.exceptions.RequestException as e:
        if hasattr(e, "response") and e.response is not None:
            if e.response.status_code == 404:
                st.error(
                    f"Segment {_segment_id} not found. Please check the segment ID."
                )
            elif e.response.status_code == 401:
                st.error(
                    f"Access denied to segment {_segment_id}. The segment might be private."
                )
            elif e.response.status_code == 403:
                st.error(
                    f"Forbidden access to segment {_segment_id}. You might not have permission to view this segment."
                )
            else:
                st.error(
                    f"Error accessing segment {_segment_id}: {e.response.status_code}"
                )
        else:
            st.error(f"Network error when trying to access segment {_segment_id}")
        return None


@st.cache_data
def get_weather_forecast(lat, lon, ride_datetime):
    url = "https://api.open-meteo.com/v1/forecast"
    ride_date_str = ride_datetime.strftime("%Y-%m-%d")
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "ms",
        "timezone": "auto",
        "start_date": ride_date_str,
        "end_date": ride_date_str,
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        target_hour_str = ride_datetime.strftime("%Y-%m-%dT%H:00")
        hour_index = data["hourly"]["time"].index(target_hour_str)
        return {
            "temperature": data["hourly"]["temperature_2m"][hour_index],
            "wind_speed": data["hourly"]["wind_speed_10m"][hour_index] * 3.6,
            "wind_direction": data["hourly"]["wind_direction_10m"][hour_index],
        }
    except (requests.exceptions.RequestException, ValueError, IndexError):
        return None


@st.cache_data
def get_elevation_data(map_df):
    url = "https://api.open-meteo.com/v1/elevation"

    # For large segments, sample points to reduce API calls and processing time
    original_length = len(map_df)
    if original_length > 1000:
        # Sample every nth point to get ~500-800 points max
        sample_rate = max(1, original_length // 600)
        sampled_df = map_df.iloc[::sample_rate].copy()
        st.info(
            f"📊 Large segment detected ({original_length} points). Sampling every {sample_rate} points ({len(sampled_df)} total) for faster processing."
        )
    else:
        sampled_df = map_df.copy()

    all_elevations = []
    total_batches = (len(sampled_df) + 99) // 100

    for i in range(0, len(sampled_df), 100):
        batch_num = (i // 100) + 1

        # Add delay between batches to avoid rate limiting
        if batch_num > 1:
            time_sleep.sleep(1)  # 1 second delay between batches

        batch = sampled_df.iloc[i : i + 100]
        params = {
            "latitude": ",".join(batch["lat"].astype(str)),
            "longitude": ",".join(batch["lon"].astype(str)),
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            elevation_data = response.json()["elevation"]
            all_elevations.extend(elevation_data)
        except requests.exceptions.RequestException as e:
            if "429" in str(e):
                st.error(
                    f"❌ Rate limit hit. Try again in a few minutes, or use a shorter segment."
                )
            else:
                st.error(f"❌ Elevation API failed: {e}")
            return None

    sampled_df["elevation"] = all_elevations

    # If we sampled, interpolate elevations back to all original points
    if original_length > 1000:
        # Simple linear interpolation based on position along the route
        original_indices = np.arange(len(map_df))
        sampled_indices = np.arange(0, len(map_df), sample_rate)[: len(sampled_df)]

        map_df["elevation"] = np.interp(
            original_indices, sampled_indices, sampled_df["elevation"]
        )
        st.success(
            f"✅ Elevation data processed for {len(map_df)} points (sampled from {len(sampled_df)} points)"
        )
    else:
        map_df = sampled_df

    return map_df


# --- HELPER & UI FUNCTIONS ---
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a)) * 1000


def get_color_from_gradient(gradient):
    if gradient > 9:
        return [15, 15, 15, 200]
    elif gradient > 6:
        return [255, 25, 0, 200]
    elif gradient > 3:
        return [0, 130, 255, 200]
    elif gradient > 0:
        return [0, 255, 0, 200]
    else:
        return [230, 230, 230, 200]


def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dLon = lon2 - lon1
    x = math.sin(dLon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(
        dLon
    )
    initial_bearing = math.atan2(x, y)
    return (math.degrees(initial_bearing) + 360) % 360


def get_wind_description(segment_bearing, wind_direction):
    diff = abs(segment_bearing - wind_direction)
    angle = min(diff, 360 - diff)
    if angle <= 45:
        return "Headwind"
    elif angle >= 135:
        return "Tailwind"
    else:
        return "Crosswind"


def degrees_to_cardinal(d):
    dirs = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]
    ix = round(d / (360.0 / len(dirs)))
    return dirs[ix % len(dirs)]


def parse_time_to_seconds(time_str):
    if pd.isna(time_str):
        return None
    time_str = str(time_str)
    total_seconds = 0
    if "h" in time_str:
        parts = time_str.split("h")
        total_seconds += int(parts[0]) * 3600
        time_str = parts[1].strip()
    if "m" in time_str:
        parts = time_str.split("m")
        total_seconds += int(parts[0]) * 60
        time_str = parts[1].strip()
    if "s" in time_str:
        total_seconds += int(time_str.replace("s", "").strip())
    elif ":" in time_str:
        parts = time_str.split(":")
        if len(parts) == 2:
            total_seconds += int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            total_seconds += int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return total_seconds if total_seconds > 0 else None


def show_main_app():
    st.markdown(
        """<style>.main .block-container {padding-top: 1rem; padding-bottom: 1rem;} [data-testid="stSidebar"] {width: 460px !important;} [data-testid="stSidebarCollapseButton"] {display: none;} </style>""",
        unsafe_allow_html=True,
    )
    athlete = st.session_state.athlete_info
    with st.sidebar:

        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(athlete["profile"], width=100)
        with col2:
            with st.popover(
                f"🚴 Welcome, {athlete['firstname']}!", use_container_width=True
            ):
                st.link_button(
                    "Strava Profile",
                    f"https://www.strava.com/athletes/{athlete['id']}",
                    use_container_width=True,
                )
                if st.button("Logout", use_container_width=True):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.switch_page("app.py")
            if st.button("🚀 Technical Showcase", use_container_width=True):
                st.switch_page("pages/1_Technical_Showcase.py")

        st.markdown("---")

        col_seg, col_weight = st.columns(2)
        with col_seg:
            segment_url = st.text_input("Strava Segment URL or ID:", value=38078350)
        with col_weight:
            default_weight = int(athlete.get("weight", 75) or 75)
            weight = st.number_input(
                "Your Weight (kg):",
                min_value=40,
                max_value=150,
                value=default_weight,
                step=1,
            )

        desired_power = st.number_input(
            "Target Power (Watts):", min_value=0, max_value=2000, value=250, step=1
        )
        entry_speed = st.slider(
            "Entry Speed (km/h):", min_value=1, max_value=50, value=20, step=1
        )

        col_date, col_time = st.columns(2)
        with col_date:
            st.date_input(
                "Date of Ride:",
                key="ride_date",
                max_value=date.today() + timedelta(days=14),
            )
        with col_time:
            st.time_input("Time of Ride (24h):", key="ride_time")

        st.markdown("")
        if st.button(
            "🚀 Generate Time Estimate", type="primary", use_container_width=True
        ):
            if segment_url:
                st.cache_data.clear()
                st.session_state.prediction_inputs = {
                    "segment": segment_url,
                    "weight": weight,
                    "power": desired_power,
                    "ride_datetime": datetime.combine(
                        st.session_state.ride_date, st.session_state.ride_time
                    ),
                    "entry_speed": entry_speed,
                }
                st.rerun()
            else:
                st.warning("Please enter a segment URL or ID.")

        st.markdown("")
        st.markdown("")
        st.markdown("")

        script_dir = Path(__file__).parent.parent
        logo_path = script_dir / "api_logo_cptblWith_strava_horiz_white.png"
        if logo_path.is_file():
            st.image(
                str(logo_path),
                width=250,
            )

    if "prediction_inputs" in st.session_state:
        show_results_page(st.session_state.prediction_inputs)
    else:
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.info("⬅️ Enter your details in the sidebar to generate a time estimate.")


def show_results_page(inputs):
    # This is the full, complete dashboard logic
    segment_id = (
        inputs["segment"].split("/")[-1]
        if "/" in inputs["segment"]
        else inputs["segment"]
    )

    with st.spinner("le wawa thinking..."):
        segment_data = get_segment_data(segment_id, st.session_state.access_token)

        if not segment_data:
            st.error(
                "❌ Unable to load segment data. Please try a different segment or check the error message above."
            )
            st.info(
                "💡 Try using a public segment like 38078350 (the default) or find a public segment on Strava."
            )
            return

        st.markdown(
            f"<h1 style='color: #FF4B4B;'>Time estimate for: <span class='bestlink'><a href='https://www.strava.com/segments/{segment_id}'>{segment_data['name']}</a></span></>",
            unsafe_allow_html=True,
        )

        map_data = get_elevation_data(segment_data.get("map_data").copy())
        if map_data is None:
            st.error("❌ Failed to get elevation data")
            return
        start_lat, start_lon = segment_data["start_latlng"]
        weather_data = get_weather_forecast(
            start_lat, start_lon, inputs["ride_datetime"]
        )
        if not weather_data:
            st.error("❌ Failed to get weather data")
            return

        map_data["lat_next"] = map_data["lat"].shift(-1)
        map_data["lon_next"] = map_data["lon"].shift(-1)
        map_data["distance_segment"] = map_data.apply(
            lambda r: (
                haversine_np(r["lon"], r["lat"], r["lon_next"], r["lat_next"])
                if pd.notna(r["lon_next"])
                else 0
            ),
            axis=1,
        )
        map_data["cumulative_distance"] = map_data["distance_segment"].cumsum()
        map_data["smoothed_elevation"] = (
            map_data["elevation"].rolling(window=25, center=True, min_periods=1).mean()
        )
        map_data["elevation_next"] = map_data["smoothed_elevation"].shift(-1)

        line_segments_df = map_data.dropna(subset=["lon_next", "lat_next"]).copy()

        if line_segments_df.empty:
            st.error(
                "Could not process segment path. The segment may be too short or have invalid GPS data."
            )
            return

        line_segments_df["elevation_change"] = (
            line_segments_df["elevation_next"] - line_segments_df["smoothed_elevation"]
        )
        line_segments_df["gradient"] = line_segments_df.apply(
            lambda r: (
                (r["elevation_change"] / r["distance_segment"]) * 100
                if r["distance_segment"] > 0
                else 0
            ),
            axis=1,
        )
        line_segments_df["color"] = line_segments_df["gradient"].apply(
            get_color_from_gradient
        )

        predicted_seconds, api_dataframe = predict_time_from_api(
            segment_data, map_data, line_segments_df, weather_data, inputs
        )

        if predicted_seconds is None:
            return

        predicted_time_str = f"{predicted_seconds // 60}:{predicted_seconds % 60:02d}"
        avg_speed_kmh = (
            (segment_data["distance"] / predicted_seconds) * 3.6
            if predicted_seconds > 0
            else 0
        )
        variable_power = (inputs["power"] + (line_segments_df["gradient"] * 10)).clip(
            lower=0
        )
        segment_bearing = calculate_bearing(
            map_data["lat"].iloc[0],
            map_data["lon"].iloc[0],
            map_data["lat"].iloc[-1],
            map_data["lon"].iloc[-1],
        )
        wind_desc = get_wind_description(
            segment_bearing, weather_data["wind_direction"]
        )
        wind_cardinal = degrees_to_cardinal(weather_data["wind_direction"])

        main_col, map_col = st.columns([1, 2])

        with main_col:
            with st.container(border=True, height=635):
                st.subheader("📊 Prediction & Conditions")
                st.markdown("")
                with st.container(border=True):
                    st.markdown(
                        "<h3 style='text-align: center; color: #FF4B4B; padding-top: 0; margin-bottom: 0;'>Predicted Time</>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<h1 style='text-align: center; color: #FF4B4B; padding-top: 0; margin-bottom: 0;'>{predicted_time_str}</>",
                        unsafe_allow_html=True,
                    )
                st.markdown("")
                pred_col, stats_col = st.columns([1, 1])
                with pred_col:
                    st.metric(
                        "Average Speed",
                        f"{avg_speed_kmh:.1f} km/h",
                        help="Based on the predicted time and segment distance.",
                    )
                    st.metric(
                        "Elevation Gain",
                        f"{segment_data['elevation_gain']:.0f} m",
                        help="Based on the predicted time and segment distance.",
                    )
                    st.metric(
                        "Wind Speed",
                        f"{weather_data['wind_speed']:.1f} km/h",
                        help="Based on the predicted time and segment distance.",
                    )

                with stats_col:
                    avg_grade = (
                        (map_data["elevation"].iloc[-1] - map_data["elevation"].iloc[0])
                        / segment_data["distance"]
                        * 100
                        if segment_data["distance"] > 0
                        else 0
                    )
                    st.metric(
                        "Distance",
                        f"{segment_data['distance']/1000:.2f} km",
                        help="Based on the predicted time and segment distance.",
                    )
                    st.metric(
                        "Avg. Grade",
                        f"{avg_grade:.1f}%",
                        help="Based on the predicted time and segment distance.",
                    )
                    st.metric(
                        "Temperature",
                        f"{weather_data['temperature']}°C",
                        help="Based on the predicted time and segment distance.",
                    )

                st.metric(
                    "Wind Direction",
                    f"{wind_cardinal} - {wind_desc}",
                    help="Headwind - Works against you. Tailwind - Works for you. Crosswind - Pushes you sideways.",
                )

        with map_col:
            with st.container(border=True, height=635):
                st.subheader("🗺️ 3D Segment Map")
                if "MAPBOX_API_KEY" not in st.secrets:
                    st.error("Mapbox API key not found.")
                    return

                view_state = pdk.ViewState(
                    latitude=map_data["lat"].mean(),
                    longitude=map_data["lon"].mean(),
                    zoom=13.5,
                    pitch=60,
                    bearing=0,
                )

                line_layer = pdk.Layer(
                    "LineLayer",
                    data=line_segments_df,
                    get_source_position="[lon, lat, smoothed_elevation]",
                    get_target_position="[lon_next, lat_next, elevation_next]",
                    get_color="color",
                    get_width=5,
                    pickable=True,
                )
                vertical_line_layer = pdk.Layer(
                    "LineLayer",
                    data=line_segments_df,
                    get_source_position="[lon, lat, 0]",
                    get_target_position="[lon, lat, smoothed_elevation]",
                    get_color=[100, 100, 100, 200],
                    get_width=1,
                )

                st.pydeck_chart(
                    pdk.Deck(
                        map_style="mapbox://styles/mapbox/dark-v10",
                        layers=[line_layer, vertical_line_layer],
                        initial_view_state=view_state,
                    )
                )

                st.markdown(
                    """
                    **Gradient Scale:**
                    <span style="color:#BFBFBF; font-weight:bold;">● <0%</span> |
                    <span style="color:#00CC00; font-weight:bold;">● 0-3%</span> |
                    <span style="color:#0099FF; font-weight:bold;">● 3-6%</span> |
                    <span style="color:#FF3300; font-weight:bold;">● 6-9%</span> |
                    <span style="color:#4D4D4D; font-weight:bold;">● >9%</span>
                """,
                    unsafe_allow_html=True,
                )

        power_col, lead_col = st.columns([1.5, 1])
        with power_col:
            with st.container(border=True, height=580):
                st.subheader("📈 Speed & Elevation Profile")
                st.markdown("")

                # Create subplot with secondary y-axis
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                # Add elevation trace
                fig.add_trace(
                    go.Scatter(
                        x=map_data["cumulative_distance"] / 1000,
                        y=map_data["smoothed_elevation"],
                        name="Elevation",
                        fill="tozeroy",
                        line=dict(color="lightgrey"),
                        customdata=np.stack((line_segments_df["gradient"],), axis=-1),
                        hovertemplate="<b>Distance:</b> %{x:.2f} km<br><b>Elevation:</b> %{y:.1f} m<br><b>Grade:</b> %{customdata[0]:.1f}%<extra></extra>",
                    ),
                    secondary_y=False,
                )

                if api_dataframe and len(api_dataframe) > 0:
                    speed_df = pd.DataFrame(api_dataframe)

                    if "distance" in speed_df.columns and "times" in speed_df.columns:
                        speed_df["speed_kmh"] = (
                            speed_df["distance"] / speed_df["times"]
                        ) * 3.6

                        speed_df["cumulative_distance"] = speed_df["distance"].cumsum()

                        entry_speed_kmh = inputs["entry_speed"]

                        x_values = [0] + list(speed_df["cumulative_distance"] / 1000)
                        y_values = [entry_speed_kmh] + list(speed_df["speed_kmh"])

                        fig.add_trace(
                            go.Scatter(
                                x=x_values,
                                y=y_values,
                                name="Predicted Speed",
                                line=dict(color="red", width=3),
                                hovertemplate="<b>Speed:</b> %{y:.1f} km/h<extra></extra>",
                            ),
                            secondary_y=True,
                        )

                fig.update_layout(
                    title_text="Speed vs. Elevation Profile",
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                    ),
                    hovermode="x unified",
                )
                fig.update_xaxes(title_text="Distance (km)")
                fig.update_yaxes(title_text="Elevation (m)", secondary_y=False)
                fig.update_yaxes(title_text="Speed (km/h)", secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)

        with lead_col:
            with st.container(border=True, height=580):

                st.subheader("🏆 Segment Leaderboard Comparison")

                st.markdown("")

                leaderboard_df = None
                try:
                    response = requests.get(
                        f"https://www.strava.com/segments/{segment_id}"
                    )
                    response.raise_for_status()
                    soup = BeautifulSoup(response.content, "html.parser")

                    table = soup.find("table")
                    if table:
                        raw_df = pd.read_html(table.prettify())[0]
                        raw_df["Power"] = raw_df["Power"].str.replace(
                            "  Power Meter", ""
                        )

                        if raw_df.shape[1] >= 4:
                            leaderboard_df = raw_df.iloc[:, [0, 1, 3, -1]].copy()
                            leaderboard_df.columns = [
                                "Rank",
                                "Athlete",
                                "Power",
                                "Time",
                            ]

                            leaderboard_df["Time (s)"] = leaderboard_df["Time"].apply(
                                parse_time_to_seconds
                            )
                            leaderboard_df.dropna(subset=["Time (s)"], inplace=True)
                            leaderboard_df["Time (s)"] = leaderboard_df[
                                "Time (s)"
                            ].astype(int)
                        else:
                            st.warning(
                                "Could not parse the leaderboard structure as expected."
                            )
                            leaderboard_df = None
                    else:
                        st.info(
                            "Could not find a leaderboard table on the segment page."
                        )
                except Exception as e:
                    st.error(f"An error occurred while scraping the leaderboard: {e}")

                if leaderboard_df is not None and not leaderboard_df.empty:
                    predicted_rank = (
                        leaderboard_df[
                            leaderboard_df["Time (s)"] < predicted_seconds
                        ].shape[0]
                        + 1
                    )

                    if predicted_rank <= 10:
                        st.info(
                            f"🎯 You would become Top {predicted_rank} with a power average of {inputs['power']} W."
                        )
                        user_effort = pd.DataFrame(
                            [
                                {
                                    "Rank": "★",
                                    "Athlete": f"{st.session_state.athlete_info['firstname']} (Your Prediction)",
                                    "Time": predicted_time_str,
                                    "Time (s)": predicted_seconds,
                                    "Power": f"{inputs['power']} W",
                                }
                            ]
                        )
                        display_df = (
                            pd.concat([leaderboard_df, user_effort])
                            .sort_values(by="Time (s)")
                            .head(10)
                        )
                    else:
                        display_df = leaderboard_df.head(10)

                    display_df = display_df.drop(columns=["Time (s)"], errors="ignore")
                    display_df.reset_index(drop=True, inplace=True)
                    display_df["Rank"] = display_df.index + 1

                    def highlight_user(row):
                        if "(Your Prediction)" in str(row["Athlete"]):
                            return ["background-color: #FF4B4B; color: white"] * len(
                                row
                            )
                        return [""] * len(row)

                    if predicted_rank > 10 and len(leaderboard_df) >= 10:
                        time_to_beat = leaderboard_df.iloc[9]["Time (s)"]
                        power_for_top_10 = find_power_for_target_time(
                            time_to_beat,
                            segment_data,
                            map_data,
                            line_segments_df,
                            weather_data,
                            inputs,
                        )
                        if power_for_top_10:
                            power_diff = power_for_top_10 - inputs["power"]
                            st.warning(
                                f"🎯 Break into the Top 10 with an average of **{power_for_top_10} W** (+{power_diff} W)."
                            )
                        else:
                            st.warning(
                                f"🎯 You would not break into the Top 10 with a power average of {inputs['power']} W."
                            )

                    st.dataframe(
                        display_df[["Rank", "Athlete", "Time", "Power"]].style.apply(
                            highlight_user, axis=1
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                else:
                    st.info("Could not display leaderboard at this time.")


if __name__ == "__main__":
    show_main_app()
