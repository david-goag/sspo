import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

app = FastAPI()
xgb_reg = pickle.load(open('xgb_reg.pkl', 'rb'))
app.state.model = xgb_reg

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get('/')
def index():
    return {'API loaded': True}

@app.get("/predict")
def predict(
    athlete_weight: int,
    distance: int,
    avg_grade: float,
    max_grade: float,
    elevation_gain: int,
    start_latitude: float,
    start_longitude: float,
    end_latitude: float,
    end_longitude: float,
    avg_power: int,
    temperature: int,
    wind_speed: int,
    wind_direction: int
    ):


    X_pred = pd.DataFrame(dict(
        athlete_weight=[np.int8(athlete_weight)],
        distance=[np.int32(distance)],
        avg_grade=[np.float16(avg_grade)],
        max_grade=[np.float16(max_grade)],
        elevation_gain=[np.int16(elevation_gain)],
        start_latitude=[np.float16(start_latitude)],
        start_longitude=[np.float16(start_longitude)],
        end_latitude=[np.float16(end_latitude)],
        end_longitude=[np.float16(end_longitude)],
        avg_power=[np.int16(avg_power)],
        temperature=[np.int8(temperature)],
        wind_speed=[np.int8(wind_speed)],
        wind_direction=[np.int16(wind_direction)]
    ))

    y_pred = app.state.model.predict(X_pred)
    y_pred = float(y_pred[0])
    return {'Seconds': y_pred}
